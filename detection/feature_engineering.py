"""Builds the 30+ feature vector consumed by the ensemble ML models.

Feature groups (see README):
  - Benford features (15): chi-square / Z-score / MAD across 5 windows
  - Trade pattern features
  - Volume and timing features
  - Wallet graph features
  - Cross-asset coordination features (6): synchrony, net flow, counterparty overlap, volume
    correlation, pair diversity, Benford MAD std
  - GNN embedding features (GNN_EMBEDDING_DIM, default 32): gnn_0 … gnn_31

Each `compute_*_features` function operates on the trade DataFrame produced
by `ingestion.historical_loader.trades_to_dataframe` (or the streamer,
buffered into a DataFrame) for a single wallet.
"""

from __future__ import annotations

import logging
import math

import networkx as nx
import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)
from detection.benford_engine import (
    BenfordMetrics,
    compute_benford_metrics_for_windows,
    cross_pair_benford_consistency,
)
from detection.streaming_benford import StreamingBenfordSketch
from detection.wallet_graph import (
    NO_RING,
    build_ring_statistics,
    compute_wallet_graph_metrics,
    detect_wash_trading_rings,
)
from ingestion.data_models import AccountActivity

FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Benford features — 5 windows (1h, 4h, 24h, 168h, 720h)
    **{
        f"benford_chi_square_{h}h": (
            f"Chi-square goodness-of-fit of trade amounts against Benford's Law "
            f"over the trailing {h}-hour window. High values indicate the digit "
            f"distribution is statistically inconsistent with natural trading."
        )
        for h in [1, 4, 24, 168, 720]
    },
    **{
        f"benford_mad_{h}h": (
            f"Mean Absolute Deviation between observed and expected Benford digit "
            f"frequencies over the trailing {h}-hour window. Values above 0.015 "
            f"indicate non-conformity (Nigrini, 2012)."
        )
        for h in [1, 4, 24, 168, 720]
    },
    **{
        f"benford_z_max_{h}h": (
            f"Maximum per-digit Z-score against the Benford expected proportion "
            f"over the trailing {h}-hour window. Highlights the single most "
            f"anomalous digit in the distribution."
        )
        for h in [1, 4, 24, 168, 720]
    },
    # Trade pattern features
    "counterparty_concentration_ratio": (
        "Fraction of total trade volume transacted with a single counterparty. "
        "Values near 1.0 indicate the wallet trades almost exclusively with one "
        "other account, a hallmark of wash-trading arrangements."
    ),
    "round_trip_frequency": (
        "Proportion of trades where the base and counter account are identical, "
        "indicating the wallet is trading with itself. Any non-zero value is a "
        "strong wash-trade signal."
    ),
    "self_matching_rate": (
        "Rate at which the wallet appears on both sides of a trade. Identical to "
        "round_trip_frequency in the current implementation; included as a "
        "separate signal for ensemble diversity."
    ),
    "order_cancellation_rate": (
        "Fraction of the wallet's manage-offer operations that were cancellations "
        "rather than fills or updates. High cancellation rates can indicate "
        "layering or spoofing strategies."
    ),
    # Volume and timing features
    "volume_per_counterparty_ratio": (
        "Total traded volume divided by the number of unique counterparties. "
        "Very high values suggest concentrated wash trading with few accounts."
    ),
    "intra_minute_clustering": (
        "Fraction of time-buckets (1-minute resolution) containing more than one "
        "trade. High clustering indicates burst activity consistent with "
        "automated wash-trade scripts."
    ),
    "off_hours_activity_ratio": (
        "Fraction of trades executed between UTC 00:00 and 05:00. Legitimate "
        "retail activity tends to follow business-hours patterns; sustained "
        "off-hours activity may indicate bot-driven manipulation."
    ),
    "volume_spike_frequency": (
        "Fraction of trades whose amount exceeds 3× the 10-trade rolling mean. "
        "Frequent spikes can indicate pump-and-dump volume inflation."
    ),
    # Wallet graph features
    "funding_source_similarity": (
        "Cosine similarity between this wallet's funding-source fingerprint and "
        "known wash-trade clusters in the funding graph. High values suggest the "
        "wallet shares infrastructure with flagged accounts."
    ),
    "network_centrality": (
        "Betweenness centrality of the wallet in the funding graph. High "
        "centrality indicates the wallet acts as a hub routing funds between "
        "multiple suspicious accounts."
    ),
    "account_age_days": (
        "Age of the Stellar account in days at the time of scoring. "
        "Very young accounts (< 7 days) combined with high risk scores are a "
        "strong indicator of throwaway wash-trade accounts."
    ),
    # Cross-asset coordination features
    "cross_pair_trade_synchrony": (
        "Fraction of trades where the wallet also transacted on a different asset "
        "pair within the synchrony window. Simultaneous multi-pair activity is "
        "difficult to explain by normal market-making behaviour."
    ),
    "net_asset_flow_deviation": (
        "Maximum absolute net asset flow (normalised by total volume) across all "
        "assets. Values near 0 indicate a fully closed cycle — the defining "
        "characteristic of wash trading where no real economic transfer occurs."
    ),
    "cross_pair_counterparty_overlap": (
        "Jaccard similarity of counterparty sets across asset pairs. High overlap "
        "means the wallet uses the same small set of counterparties on every pair, "
        "consistent with a coordinated wash-trade network."
    ),
    "cross_pair_volume_correlation": (
        "Pearson correlation of per-minute trade volumes across asset pairs. "
        "Strong positive correlation indicates the wallet inflates volume on "
        "multiple pairs simultaneously."
    ),
    "pair_diversity_score": (
        "Shannon entropy of volume distribution across traded asset pairs, "
        "normalised to [0, 1]. Low values indicate concentration on a single pair; "
        "high values indicate diversified (potentially synthetic) activity."
    ),
    "cross_pair_mad_std": (
        "Standard deviation of per-pair Benford MAD scores. Low values mean "
        "Benford non-conformity is equally distributed across all pairs — "
        "consistent with a systematic automated trading pattern."
    ),
    # Cross-chain features
    "solana_linked_wash_score": (
        "Risk score of a linked Solana address (via Wormhole bridge) if found. "
        "If the Stellar wallet has a linked Solana address and that address's "
        "Solana-chain risk score is available (from external provider, cached), "
        "this feature surfaces that signal. Value [0, 100]; 0 if no link found."
    ),
}


def compute_benford_features(
    wallet_trades: pd.DataFrame,
    decompose: bool = True,
    liquidity_profiler=None,
    asset: str | None = None,
    precomputed_metrics: dict[int, BenfordMetrics] | None = None,
) -> dict:
    """Flatten per-window Benford metrics into a feature row.

    Produces ``benford_chi_square_{h}h``, ``benford_mad_{h}h``, and
    ``benford_z_max_{h}h`` for each configured window (preserved for backward
    compatibility).  When ``liquidity_profiler`` and ``asset`` are provided,
    also adds calibrated variants:

    - ``benford_calibrated_chi_{h}h`` — chi-square vs. regime baseline
    - ``benford_calibrated_mad_{h}h`` — MAD vs. regime baseline
    - ``benford_regime_id`` — which liquidity cluster this asset belongs to
    - ``benford_regime_baseline_mad`` — the regime's expected MAD
    - ``benford_deviation_from_regime`` — calibrated MAD / regime baseline MAD

    When ``decompose=True``, also adds ``benford_residual_chi_square_{h}h``
    and ``benford_residual_mad_{h}h`` — Benford metrics on STL residuals.
    Residual features are set to ``NaN`` for insufficient-data windows.
    """
    per_window = precomputed_metrics or compute_benford_metrics_for_windows(wallet_trades)

    features: dict = {}
    for hours, metrics in per_window.items():
        features[f"benford_chi_square_{hours}h"] = metrics["chi_square"]
        features[f"benford_mad_{hours}h"] = metrics["mad"]
        features[f"benford_z_max_{hours}h"] = max(metrics["z_scores"].values(), default=0.0)

    if decompose and not wallet_trades.empty:
        for hours, res_metrics in _compute_residual_benford_for_windows(wallet_trades).items():
            features[f"benford_residual_chi_square_{hours}h"] = res_metrics.get("chi_square", 0.0)
            features[f"benford_residual_mad_{hours}h"] = res_metrics.get("mad", 0.0)

    if liquidity_profiler is not None and asset is not None:
        _add_calibrated_benford_features(
            features, wallet_trades, per_window, liquidity_profiler, asset
        )

    return features


def _add_calibrated_benford_features(
    features: dict,
    wallet_trades: pd.DataFrame,
    per_window: dict,
    liquidity_profiler,
    asset: str,
) -> None:
    """Mutate *features* in-place, adding calibrated Benford features."""
    from config import config

    regime_id = liquidity_profiler.get_regime_id(asset)
    baseline_mad = liquidity_profiler.get_baseline_mad(asset)
    features["benford_regime_id"] = regime_id
    features["benford_regime_baseline_mad"] = baseline_mad

    timestamps = (
        pd.to_datetime(wallet_trades["ledger_close_time"])
        if not wallet_trades.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    ref = timestamps.max() if not timestamps.empty else pd.Timestamp.now(tz="UTC")

    cal_mads: list[float] = []
    for hours in config.BENFORD_WINDOWS_HOURS:
        if not wallet_trades.empty:
            window_start = ref - pd.Timedelta(hours=hours)
            window_amounts = wallet_trades.loc[
                (timestamps > window_start) & (timestamps <= ref), "amount"
            ]
        else:
            window_amounts = pd.Series(dtype=float)

        cal_chi = liquidity_profiler.calibrated_chi_square(window_amounts, asset)
        cal_mad = liquidity_profiler.calibrated_mad(window_amounts, asset)
        features[f"benford_calibrated_chi_{hours}h"] = cal_chi
        features[f"benford_calibrated_mad_{hours}h"] = cal_mad
        cal_mads.append(cal_mad)

    mean_cal_mad = float(np.mean(cal_mads)) if cal_mads else 0.0
    features["benford_deviation_from_regime"] = (
        mean_cal_mad / baseline_mad if baseline_mad > 0 else 0.0
    )


def _compute_residual_benford_for_windows(
    wallet_trades: pd.DataFrame,
) -> dict[int, BenfordMetrics | dict[Any, Any]]:
    """Compute Benford metrics on STL residuals for each configured window.

    For each window, the trade sub-frame is decomposed via STL and Benford
    metrics are computed on the absolute residuals.  Returns NaN entries for
    windows where decomposition is not possible (insufficient data).
    """
    from detection.benford_engine import compute_benford_metrics
    from detection.ts_decomposition import decompose_trade_amounts

    windows_hours = config.BENFORD_WINDOWS_HOURS
    timestamps = pd.to_datetime(wallet_trades["ledger_close_time"])
    ref = timestamps.max()

    results: dict[int, BenfordMetrics | dict] = {}
    for hours in windows_hours:
        window_start = ref - pd.Timedelta(hours=hours)
        window_df = wallet_trades[(timestamps > window_start) & (timestamps <= ref)]

        residuals = decompose_trade_amounts(window_df)
        if residuals is None:
            results[hours] = {"chi_square": float("nan"), "mad": float("nan")}
        else:
            pos_residuals = residuals.abs()
            pos_residuals = pos_residuals[pos_residuals > 0]
            results[hours] = compute_benford_metrics(pos_residuals)

    return results


def compute_graph_embedding_features(
    wallet: str,
    funding_graph: nx.DiGraph,
    gnn_encoder: object,
) -> dict[str, float]:
    """Return GNN embedding features for a wallet.

    This function attempts to compute an embedding for ``wallet`` using the
    provided ``gnn_encoder`` and ``funding_graph``. If embedding computation
    fails (e.g., missing wallet node, encoder error), it returns a zero vector
    of length ``config.GNN_EMBEDDING_DIM``.

    Args:
        wallet: Stellar account id to compute embeddings for.
        funding_graph: Directed funding graph used as input to the encoder.
        gnn_encoder: Encoder instance that provides an ``encode(graph, wallet)``
            method returning an indexable embedding vector.

    Returns:
        A dictionary mapping feature names ``gnn_0``..``gnn_{N-1}`` to floats,
        where ``N`` is ``config.GNN_EMBEDDING_DIM``.

    Raises:
        Any exceptions raised by the encoder are caught and will not be
        propagated.
    """
    try:

        emb = gnn_encoder.encode(funding_graph, wallet)  # type: ignore[attr-defined]
        return {f"gnn_{i}": float(emb[i]) for i in range(len(emb))}
    except Exception:
        return {f"gnn_{i}": 0.0 for i in range(config.GNN_EMBEDDING_DIM)}


def compute_order_cancellation_rate(wallet: str, orderbook_events: pd.DataFrame | None) -> float:
    """Fraction of a wallet's manage-offer operations that were cancellations.

    `orderbook_events` is the output of
    `ingestion.orderbook_loader.load_accounts_orderbook_events` (or `None`/
    empty if order-book ingestion wasn't run), with an `account` and
    `action` ("created"/"cancelled"/"updated") column.
    """
    if orderbook_events is None or orderbook_events.empty:
        return 0.0

    wallet_events = orderbook_events[orderbook_events["account"] == wallet]
    if wallet_events.empty:
        return 0.0

    cancelled = (wallet_events["action"] == "cancelled").sum()
    return float(cancelled / len(wallet_events))


def compute_trade_pattern_features(
    wallet: str,
    wallet_trades: pd.DataFrame,
    orderbook_events: pd.DataFrame | None = None,
) -> dict:
    """Compute trade-pattern features for a wallet.

    Computes signals based on the wallet's trade history, and (optionally)
    augments them with order-cancellation statistics derived from order-book
    events.

    Args:
        wallet: Stellar account id to score.
        wallet_trades: Trades involving ``wallet``. Expected columns include
            ``base_account``, ``counter_account``, and ``amount``.
        orderbook_events: Optional order-book event DataFrame as produced by
            ``ingestion.orderbook_loader.load_accounts_orderbook_events``.
            When provided, it must include ``account`` and ``action``
            (with values such as ``created``, ``cancelled``, ``updated``).

    Returns:
        A dictionary with the following keys:

        - ``counterparty_concentration_ratio``: Fraction of volume attributed
          to the wallet's most-used counterparty.
        - ``round_trip_frequency``: Fraction of trades where
          ``base_account == counter_account``.
        - ``net_roundtrip_ratio``: Currently identical to
          ``round_trip_frequency``.
        - ``self_matching_rate``: Currently identical to
          ``round_trip_frequency``.
        - ``order_cancellation_rate``: Fraction of the wallet's manage-offer
          operations that were cancellations.

    Raises:
        KeyError: If required columns (e.g., ``base_account``,
            ``counter_account``, ``amount``) are missing from ``wallet_trades``.
    """
    order_cancellation_rate = compute_order_cancellation_rate(wallet, orderbook_events)

    if wallet_trades.empty:
        return {
            "counterparty_concentration_ratio": 0.0,
            "round_trip_frequency": 0.0,
            "net_roundtrip_ratio": 0.0,
            "self_matching_rate": 0.0,
            "order_cancellation_rate": order_cancellation_rate,
        }

    counterparty_col = wallet_trades["base_account"].where(
        wallet_trades["base_account"] != wallet, wallet_trades["counter_account"]
    )
    volume_by_counterparty = wallet_trades.groupby(counterparty_col)["amount"].sum()
    total_volume = volume_by_counterparty.sum()
    concentration = (volume_by_counterparty.max() / total_volume) if total_volume else 0.0

    # Round-trip: trade pairs where the asset sent comes back to the wallet
    # within the same trade set (proxy until full graph traversal is added).
    round_trips = (wallet_trades["base_account"] == wallet_trades["counter_account"]).sum()
    round_trip_frequency = round_trips / len(wallet_trades)

    self_matching_rate = round_trip_frequency  # same accounts trading with themselves

    return {
        "counterparty_concentration_ratio": float(concentration),
        "round_trip_frequency": float(round_trip_frequency),
        "net_roundtrip_ratio": float(round_trip_frequency),
        "self_matching_rate": float(self_matching_rate),
        "order_cancellation_rate": order_cancellation_rate,
    }


def compute_volume_timing_features(wallet_trades: pd.DataFrame) -> dict:
    """Compute volume and timing anomaly features for a wallet.

    Args:
        wallet_trades: Trades involving a single wallet. Expected columns
            include ``ledger_close_time`` (timestamp) and ``counter_account``.
            Must also contain ``amount`` for volume-based calculations.

    Returns:
        A dictionary containing:

        - ``volume_per_counterparty_ratio``: Total volume divided by the
          number of unique counterparties.
        - ``intra_minute_clustering``: Fraction of 1-minute buckets that
          contain more than one trade.
        - ``off_hours_activity_ratio``: Fraction of trades executed during
          off-hours (UTC hours 00:00-04:59).
        - ``volume_spike_frequency``: Fraction of trades whose amount exceeds
          3× the rolling mean (window size = 10 trades).

    Raises:
        KeyError: If required columns (e.g., ``ledger_close_time``,
            ``counter_account``, ``amount``) are missing.
    """
    if wallet_trades.empty:
        return {
            "volume_per_counterparty_ratio": 0.0,
            "intra_minute_clustering": 0.0,
            "off_hours_activity_ratio": 0.0,
            "volume_spike_frequency": 0.0,
        }

    timestamps = pd.to_datetime(wallet_trades["ledger_close_time"])
    n_unique_counterparties = wallet_trades["counter_account"].nunique() or 1
    volume_per_counterparty_ratio = wallet_trades["amount"].sum() / n_unique_counterparties

    minute_buckets = timestamps.dt.floor("min")
    intra_minute_clustering = (
        minute_buckets.value_counts().gt(1).sum() / minute_buckets.nunique()
        if minute_buckets.nunique()
        else 0.0
    )

    # "Off hours" defined as UTC 00:00-05:00, a simple proxy for unusual
    # ledger-time activity.
    off_hours_mask = timestamps.dt.hour < 5
    off_hours_activity_ratio = off_hours_mask.mean()

    rolling_volume = wallet_trades["amount"].rolling(window=10, min_periods=1).mean()
    spikes = (wallet_trades["amount"] > rolling_volume * 3).sum()
    volume_spike_frequency = spikes / len(wallet_trades)

    return {
        "volume_per_counterparty_ratio": float(volume_per_counterparty_ratio),
        "intra_minute_clustering": float(intra_minute_clustering),
        "off_hours_activity_ratio": float(off_hours_activity_ratio),
        "volume_spike_frequency": float(volume_spike_frequency),
    }


def compute_wallet_graph_features(
    wallet: str,
    activity: AccountActivity | None,
    reference_time: pd.Timestamp,
    funding_graph: nx.DiGraph | None = None,
    community_map: dict[str, int] | None = None,
    ring_stats: dict[int, dict] | None = None,
) -> dict:
    """Funding-source similarity, network centrality, account age, ring signals.

    `funding_source_similarity` and `network_centrality` are computed from
    `funding_graph` (see `detection.wallet_graph.build_funding_graph`) when
    provided, and default to `0.0` otherwise.

    When `community_map` (wallet -> community_id, from
    `detect_wash_trading_rings`) is supplied, three wash-trading ring features
    are added: `in_wash_trading_ring`, `ring_size`, and `ring_internal_density`.
    They are omitted entirely when no community map is available, so flows
    without a funding graph keep their existing feature schema unchanged.
    """
    account_age_days = 0.0
    if activity is not None:
        created_at = pd.to_datetime(activity.account_created_at, utc=True)
        account_age_days = (reference_time - created_at).total_seconds() / 86400

    graph_metrics = (
        compute_wallet_graph_metrics(wallet, funding_graph)
        if funding_graph is not None
        else {"funding_source_similarity": 0.0, "network_centrality": 0.0}
    )

    features = {
        **graph_metrics,
        "account_age_days": float(account_age_days),
    }

    if community_map is not None:
        features.update(_ring_features(wallet, community_map, ring_stats))

    return features


def _ring_features(
    wallet: str, community_map: dict[str, int], ring_stats: dict[int, dict] | None
) -> dict:
    """Build the three ring features for `wallet` from the community map."""
    community_id = community_map.get(wallet, NO_RING)
    in_ring = community_id != NO_RING

    ring_size = 0
    internal_density = 0.0
    if in_ring:
        stats = (ring_stats or {}).get(community_id)
        if stats is not None:
            ring_size = stats["ring_size"]
            internal_density = stats["internal_edge_density"]
        else:
            ring_size = sum(1 for cid in community_map.values() if cid == community_id)

    return {
        "in_wash_trading_ring": bool(in_ring),
        "ring_size": int(ring_size),
        "ring_internal_density": float(internal_density),
    }


def compute_cross_asset_features(
    wallet: str,
    all_pairs_df: pd.DataFrame,
    pair_benford_sketches: dict[str, dict[int, StreamingBenfordSketch]] | None = None,
) -> dict:
    """Compute cross-asset coordination features for a wallet.

    ``all_pairs_df`` should contain trades across multiple asset pairs for the
    same wallet. Each trade must relate to ``wallet`` either as ``base_account``
    or as ``counter_account``.

    This function returns six features describing coordination across asset
    pairs (synchrony, net flow closure, counterparty overlap, cross-pair volume
    correlation, pair diversity, and cross-pair Benford MAD consistency).

    Args:
        wallet: Stellar account id to score.
        all_pairs_df: DataFrame containing trades for multiple pairs.
            Expected columns include:

            - ``ledger_close_time``: trade timestamp.
            - ``base_account`` / ``counter_account``: wallet-related accounts.
            - ``amount``: trade amount.
            - ``pair_id``: identifier for the asset pair.

            If ``pair_id`` is not present, the function will attempt to infer it
            from ``base_asset`` and ``counter_asset``.
        pair_benford_sketches: Optional precomputed Benford sketch objects by
            pair and window hour. When provided, cross-pair MAD std is computed
            from these sketches instead of recomputing from trades.

    Returns:
        A dictionary with the following keys:

        - ``cross_pair_trade_synchrony``: fraction of wallet trades where the
          wallet also trades on another pair within the configured synchrony
          window (see ``config.CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS``).
        - ``net_asset_flow_deviation``: maximum absolute net flow (normalized
          by total volume). Values near 0 indicate a closed cycle.
        - ``cross_pair_counterparty_overlap``: Jaccard similarity of
          counterparties between pairs.
        - ``cross_pair_volume_correlation``: Pearson correlation of per-minute
          traded volumes across pairs.
        - ``pair_diversity_score``: normalized Shannon entropy of volume
          distribution across pairs.
        - ``cross_pair_mad_std``: standard deviation / consistency score of
          Benford MAD scores across pairs.

    Raises:
        KeyError: If required columns are missing (e.g., ``ledger_close_time``,
            ``base_account``, ``counter_account``, ``amount``), or if ``pair_id``
            is missing and required asset columns (``base_asset``,
            ``counter_asset``) are also absent.
    """

    # Default values for single pair or empty data
    default_features = {
        "cross_pair_trade_synchrony": 0.0,
        "net_asset_flow_deviation": 1.0,
        "cross_pair_counterparty_overlap": 0.0,
        "cross_pair_volume_correlation": 0.0,
        "pair_diversity_score": 0.0,
        "cross_pair_mad_std": 0.0,
    }

    if all_pairs_df.empty:
        return default_features

    # Ensure timestamp column exists and is datetime
    if "ledger_close_time" not in all_pairs_df.columns:
        return default_features

    # Filter to trades involving the wallet
    mask = (all_pairs_df["base_account"] == wallet) | (all_pairs_df["counter_account"] == wallet)
    wallet_trades = all_pairs_df[mask].copy()

    if wallet_trades.empty:
        return default_features

    # Ensure we have a pair_id column; if not, infer from base/counter assets
    if "pair_id" not in wallet_trades.columns:
        if (
            "base_asset" not in wallet_trades.columns
            or "counter_asset" not in wallet_trades.columns
        ):
            return default_features
        wallet_trades["pair_id"] = (
            wallet_trades["base_asset"].astype(str)
            + "/"
            + wallet_trades["counter_asset"].astype(str)
        )

    n_pairs = wallet_trades["pair_id"].nunique()
    if n_pairs < 2:
        # Less than 2 pairs: cross-pair features don't apply
        return default_features

    features = {}

    # Feature 1: cross_pair_trade_synchrony
    # Fraction of trades where wallet also trades on another pair within window
    wallet_times = pd.to_datetime(wallet_trades["ledger_close_time"], errors="coerce")
    window_seconds = config.CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS
    synchrony_count = 0
    for trade_time in wallet_times:
        if pd.isna(trade_time):
            continue
        other_trades = wallet_times[
            (wallet_times >= trade_time - pd.Timedelta(seconds=window_seconds))
            & (wallet_times <= trade_time + pd.Timedelta(seconds=window_seconds))
        ]
        other_pairs = wallet_trades.loc[other_trades.index, "pair_id"].unique()
        if len(other_pairs) > 1:
            synchrony_count += 1

    features["cross_pair_trade_synchrony"] = float(synchrony_count / len(wallet_trades))

    # Feature 2: net_asset_flow_deviation
    # Compute net flow for each asset; deviation = max(|net_flow|) / total_volume
    net_flows = {}
    total_volume = 0.0

    for _, trade in wallet_trades.iterrows():
        base_asset = trade.get("base_asset")
        counter_asset = trade.get("counter_asset")
        amount = float(trade.get("amount", 0.0))

        if trade["base_account"] == wallet:
            # Wallet sends base, receives counter
            if base_asset not in net_flows:
                net_flows[base_asset] = 0.0
            net_flows[base_asset] -= amount
            if counter_asset not in net_flows:
                net_flows[counter_asset] = 0.0
            net_flows[counter_asset] += amount
        else:
            # Wallet sends counter, receives base
            if counter_asset not in net_flows:
                net_flows[counter_asset] = 0.0
            net_flows[counter_asset] -= amount
            if base_asset not in net_flows:
                net_flows[base_asset] = 0.0
            net_flows[base_asset] += amount

        total_volume += amount

    max_net_flow = max((abs(flow) for flow in net_flows.values()), default=0.0)
    features["net_asset_flow_deviation"] = max_net_flow / total_volume if total_volume > 0 else 1.0

    # Feature 3: cross_pair_counterparty_overlap
    # Jaccard similarity of counterparty sets across pairs
    counterparties_by_pair = {}
    for pair_id in wallet_trades["pair_id"].unique():
        pair_trades = wallet_trades[wallet_trades["pair_id"] == pair_id]
        counterparties = set()
        for _, trade in pair_trades.iterrows():
            if trade["base_account"] == wallet:
                counterparties.add(trade["counter_account"])
            else:
                counterparties.add(trade["base_account"])
        counterparties_by_pair[pair_id] = counterparties

    pairs_list = list(counterparties_by_pair.keys())
    if len(pairs_list) >= 2:
        # Compute Jaccard between first and second pair (simplified; could do all pairs)
        set1 = counterparties_by_pair[pairs_list[0]]
        set2 = counterparties_by_pair[pairs_list[1]]
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        features["cross_pair_counterparty_overlap"] = (
            float(intersection / union) if union > 0 else 0.0
        )
    else:
        features["cross_pair_counterparty_overlap"] = 0.0

    # Feature 4: cross_pair_volume_correlation
    # Pearson correlation of trade volumes across pairs, bucketed by minute
    minute_volumes_by_pair = {}
    for pair_id in wallet_trades["pair_id"].unique():
        pair_trades = wallet_trades[wallet_trades["pair_id"] == pair_id].copy()
        pair_trades["minute"] = pd.to_datetime(pair_trades["ledger_close_time"]).dt.floor("min")
        minute_volumes = pair_trades.groupby("minute")["amount"].sum()
        minute_volumes_by_pair[pair_id] = minute_volumes

    if len(minute_volumes_by_pair) >= 2:
        pairs_list = list(minute_volumes_by_pair.keys())
        volumes_1 = minute_volumes_by_pair[pairs_list[0]]
        volumes_2 = minute_volumes_by_pair[pairs_list[1]]
        # Align by minute
        aligned_idx = volumes_1.index.intersection(volumes_2.index)
        if len(aligned_idx) > 1:
            correlation = float(volumes_1[aligned_idx].corr(volumes_2[aligned_idx]))
            features["cross_pair_volume_correlation"] = (
                correlation if not pd.isna(correlation) else 0.0
            )
        else:
            features["cross_pair_volume_correlation"] = 0.0
    else:
        features["cross_pair_volume_correlation"] = 0.0

    # Feature 5: pair_diversity_score
    # Shannon entropy of volume distribution across pairs
    pair_volumes = wallet_trades.groupby("pair_id")["amount"].sum()
    total_vol = pair_volumes.sum()
    if total_vol > 0:
        proportions = pair_volumes / total_vol
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in proportions)
        max_entropy = math.log2(len(proportions)) if len(proportions) > 0 else 1.0
        features["pair_diversity_score"] = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        features["pair_diversity_score"] = 0.0

    # Feature 6: cross_pair_mad_std
    # Standard deviation of Benford MAD scores across pairs
    per_pair_metrics = {}
    if pair_benford_sketches:
        for pair_id, sketches in pair_benford_sketches.items():
            metrics = {h: s.to_metrics() for h, s in sketches.items()}
            # Average MAD across all windows for this pair
            per_pair_metrics[pair_id] = BenfordMetrics(
                chi_square=0.0,
                mad=(sum(m.mad for m in metrics.values()) / len(metrics) if metrics else 0.0),
                mad_nonconforming=False,
                z_scores={},
                sample_size=0,
            )
    else:
        for pair_id in wallet_trades["pair_id"].unique():
            pair_trades = wallet_trades[wallet_trades["pair_id"] == pair_id]
            metrics = compute_benford_metrics_for_windows(pair_trades)
            # Average MAD across all windows for this pair
            per_pair_metrics[pair_id] = BenfordMetrics(
                chi_square=0.0,
                mad=(sum(m.mad for m in metrics.values()) / len(metrics) if metrics else 0.0),
                mad_nonconforming=False,
                z_scores={},
                sample_size=0,
            )

    features["cross_pair_mad_std"] = cross_pair_benford_consistency(per_pair_metrics)

    return features


def compute_cross_venue_features(
    wallet: str,
    sdex_trades: pd.DataFrame,
    amm_trades: pd.DataFrame,
) -> dict:
    """Compute cross-venue coordination features for a wallet.

    This function delegates the computation to
    ``detection.cross_venue_features.compute_cross_venue_features``.

    Args:
        wallet: Stellar account id to score.
        sdex_trades: Trades on the SDEX venue.
        amm_trades: Trades on the AMM venue.

    Returns:
        A dictionary containing 7 cross-venue feature values:

        - ``venue_trade_ratio``
        - ``cross_venue_volume_correlation``
        - ``cross_venue_timing_synchrony``
        - ``cross_venue_net_flow``
        - ``counterparty_venue_overlap``
        - ``simultaneous_order_pair``
        - ``cross_venue_cluster_score``

    Raises:
        Any exceptions raised by the delegated implementation.
    """

    from detection.cross_venue_features import compute_cross_venue_features as _cvf

    return _cvf(wallet, sdex_trades, amm_trades)


def compute_graph_embedding_features(
    wallet: str,
    graph: nx.DiGraph,
    encoder,
) -> dict:
    """Return GNN embedding features for a wallet as a flat dict.

    This is a convenience wrapper that returns ``gnn_0``..``gnn_{N-1}``
    features for ``wallet`` using ``encoder``. If the wallet node is not
    present in ``graph`` or embedding computation fails, it returns a zero
    vector.

    Args:
        wallet: Stellar account id to compute embeddings for.
        graph: Directed funding graph (or similar) used as encoder input.
        encoder: Encoder instance that provides an ``encode(graph, wallet)``
            method.

    Returns:
        A dictionary mapping ``gnn_{i}`` to float values.

    Raises:
        Any exceptions from the encoder are caught and not propagated.
    """

    dim = config.GNN_EMBEDDING_DIM
    zero_features = {f"gnn_{i}": 0.0 for i in range(dim)}

    try:
        if wallet not in graph:
            return zero_features
        embedding = encoder.encode(graph, wallet)
        return {f"gnn_{i}": float(embedding[i]) for i in range(len(embedding))}
    except Exception:
        return zero_features


def compute_solana_linked_features(
    wallet: str,
    identity_graph=None,
    solana_risk_cache: dict[str, float] | None = None,
) -> dict:
    """Compute features based on linked Solana addresses via Wormhole bridge.

    Queries the identity graph to find Solana addresses linked to a Stellar
    wallet via Wormhole bridge transactions. If found and a cached risk score
    is available for the Solana address, surfaces that signal.

    Args:
        wallet: Stellar account id to score.
        identity_graph: IdentityGraph instance (from detection.cross_chain.identity_graph).
            If None, returns zero features.
        solana_risk_cache: Optional cache mapping Solana address -> risk_score.
            Risk scores should be in [0, 100]. If None or address not in cache,
            returns 0.0.

    Returns:
        A dictionary with:
        - ``solana_linked_wash_score``: highest risk score among linked Solana
          addresses, or 0.0 if none found.

    Raises:
        Any exceptions from the identity graph are caught and logged.
    """
    if identity_graph is None or solana_risk_cache is None:
        return {"solana_linked_wash_score": 0.0}

    try:
        # Resolve Stellar wallet to linked Solana addresses
        component = identity_graph.get_connected_component(wallet)
        linked_sol_addresses = [node["address"] for node in component.get("sol", [])]

        if not linked_sol_addresses:
            return {"solana_linked_wash_score": 0.0}

        # Find max risk score among linked Solana addresses
        max_risk_score = 0.0
        for sol_addr in linked_sol_addresses:
            if sol_addr in solana_risk_cache:
                risk = solana_risk_cache[sol_addr]
                max_risk_score = max(max_risk_score, float(risk))

        return {"solana_linked_wash_score": max_risk_score}

    except Exception as exc:
        logger.error("Failed to compute Solana linked features for %s: %s", wallet, exc)
        return {"solana_linked_wash_score": 0.0}


def build_feature_vector(
    wallet: str,
    wallet_trades: pd.DataFrame,
    activity: AccountActivity | None = None,
    orderbook_events: pd.DataFrame | None = None,
    funding_graph: nx.DiGraph | None = None,
    all_pairs_df: pd.DataFrame | None = None,
    amm_trades: pd.DataFrame | None = None,
    gnn_encoder=None,
    benford_metrics: dict | None = None,
    pair_benford_sketches: dict | None = None,
    community_map: dict[str, int] | None = None,
    ring_stats: dict[int, dict] | None = None,
) -> dict:
    """Assemble the full feature row for a single wallet.

    `wallet_trades` should already be filtered to trades involving `wallet`
    as base or counter account. `orderbook_events` (optional) is the output
    of `ingestion.orderbook_loader.load_accounts_orderbook_events`, used to
    compute `order_cancellation_rate`. `funding_graph` (optional) is the
    output of `detection.wallet_graph.build_funding_graph`, used for the
    wallet graph features. `all_pairs_df` (optional) enables cross-asset
    coordination features. `amm_trades` (optional) enables cross-venue
    coordination features.
    """
    reference_time = (
        pd.to_datetime(wallet_trades["ledger_close_time"], utc=True).max()
        if not wallet_trades.empty
        else pd.Timestamp.now(tz="UTC")
    )

    features: dict[str, float | str] = {"wallet": wallet}
    features.update(compute_benford_features(wallet_trades, precomputed_metrics=benford_metrics))
    features.update(compute_trade_pattern_features(wallet, wallet_trades, orderbook_events))
    features.update(compute_volume_timing_features(wallet_trades))
    features.update(
        compute_wallet_graph_features(
            wallet, activity, reference_time, funding_graph, community_map, ring_stats
        )
    )
    if all_pairs_df is not None:
        features.update(
            compute_cross_asset_features(
                wallet, all_pairs_df, pair_benford_sketches=pair_benford_sketches
            )
        )
    features.update(compute_hardening_features(wallet_trades))
    if amm_trades is not None:
        features.update(compute_cross_venue_features(wallet, wallet_trades, amm_trades))

    # GNN embedding features — graceful zero-fallback when encoder is absent
    if gnn_encoder is not None and funding_graph is not None:
        features.update(compute_graph_embedding_features(wallet, funding_graph, gnn_encoder))
    else:
        features.update({f"gnn_{i}": 0.0 for i in range(config.GNN_EMBEDDING_DIM)})

    return {k: (0.0 if isinstance(v, float) and pd.isna(v) else v) for k, v in features.items()}


def compute_hardening_features(wallet_trades: pd.DataFrame) -> dict:
    """Hardening features resistant to common adversarial attacks.

    - ``inter_arrival_cv``: coefficient of variation of inter-trade intervals
      (robust to TemporalSpreading — uniform spreading drives CV toward 0).
    - ``entropy_of_amounts``: Shannon entropy of the amount distribution
      (robust to AmountRounding — rounding collapses entropy).
    - ``cross_wallet_volume_corr``: Pearson correlation of per-minute volumes
      across the two most-frequent counterparties (lag-0).
    """
    if wallet_trades.empty:
        return {
            "inter_arrival_cv": 0.0,
            "entropy_of_amounts": 0.0,
            "cross_wallet_volume_corr": 0.0,
        }

    timestamps = pd.to_datetime(wallet_trades["ledger_close_time"]).sort_values()

    # Inter-arrival CV
    if len(timestamps) > 1:
        inter_arrivals = timestamps.diff().dt.total_seconds().dropna()
        mean_ia = inter_arrivals.mean()
        cv = float(inter_arrivals.std() / mean_ia) if mean_ia > 0 else 0.0
    else:
        cv = 0.0

    # Shannon entropy of amounts (binned into up to 50 bins)
    amounts = wallet_trades["amount"].clip(lower=1e-12)
    counts, _ = np.histogram(amounts, bins=min(50, len(amounts)))
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs)))

    # Cross-wallet volume correlation (top-2 counterparties, lag-0)
    corr = 0.0
    top_cps = wallet_trades["counter_account"].value_counts().head(2).index.tolist()
    if len(top_cps) >= 2:
        df_tmp = wallet_trades.copy()
        df_tmp["minute"] = pd.to_datetime(df_tmp["ledger_close_time"]).dt.floor("min")

        vol_a = df_tmp[df_tmp["counter_account"] == top_cps[0]].groupby("minute")["amount"].sum()
        vol_b = df_tmp[df_tmp["counter_account"] == top_cps[1]].groupby("minute")["amount"].sum()
        aligned = pd.concat([vol_a, vol_b], axis=1, keys=["a", "b"]).fillna(0.0)
        if len(aligned) > 1 and aligned["a"].std() > 0 and aligned["b"].std() > 0:
            corr = float(aligned["a"].corr(aligned["b"]))

    return {
        "inter_arrival_cv": cv,
        "entropy_of_amounts": entropy,
        "cross_wallet_volume_corr": float(np.nan_to_num(corr)),
    }


def build_feature_matrix(
    trades_df: pd.DataFrame,
    orderbook_events: pd.DataFrame | None = None,
    funding_graph: nx.DiGraph | None = None,
    all_pairs_df: pd.DataFrame | None = None,
    amm_trades: pd.DataFrame | None = None,
    gnn_embeddings: dict[str, dict] | None = None,
    community_map: dict[str, int] | None = None,
    ring_stats: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """Build a feature matrix with one row per wallet observed in `trades_df`.

    `orderbook_events` and `funding_graph` (both optional) are threaded
    through to `build_feature_vector` for `order_cancellation_rate` and the
    wallet graph features respectively. `all_pairs_df` (optional, should be
    the same as `trades_df` or a superset with a `pair_id` column) enables
    cross-asset coordination features. `amm_trades` (optional) enables
    cross-venue coordination features.

    When `funding_graph` is provided (and `community_map` is not supplied
    explicitly), wash-trading rings are detected once for the whole graph via
    `detect_wash_trading_rings`; the resulting community map and ring statistics
    add the `in_wash_trading_ring`, `ring_size`, and `ring_internal_density`
    columns. Without a funding graph the feature schema is unchanged.
    """
    if trades_df.empty:
        return pd.DataFrame()

    if funding_graph is not None and community_map is None:
        community_map = detect_wash_trading_rings(funding_graph)
        ring_stats = build_ring_statistics(community_map, funding_graph)

    wallets = pd.unique(trades_df[["base_account", "counter_account"]].values.ravel())

    rows = []
    for wallet in wallets:
        mask = (trades_df["base_account"] == wallet) | (trades_df["counter_account"] == wallet)
        row = build_feature_vector(
            wallet,
            trades_df[mask],
            orderbook_events=orderbook_events,
            funding_graph=funding_graph,
            all_pairs_df=all_pairs_df if all_pairs_df is not None else trades_df,
            amm_trades=amm_trades,
            community_map=community_map,
            ring_stats=ring_stats,
        )
        if gnn_embeddings and wallet in gnn_embeddings:
            row.update(gnn_embeddings[wallet])
        rows.append(row)

    return pd.DataFrame(rows)
