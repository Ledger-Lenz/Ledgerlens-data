"""Builds the 30+ feature vector consumed by the ensemble ML models.

Feature groups (see README):
  - Benford features (15): chi-square / Z-score / MAD across 5 windows
  - Trade pattern features
  - Volume and timing features
  - Wallet graph features
  - Cross-asset coordination features (6): synchrony, net flow, counterparty overlap, volume correlation, pair diversity, Benford MAD std

Each `compute_*_features` function operates on the trade DataFrame produced
by `ingestion.historical_loader.trades_to_dataframe` (or the streamer,
buffered into a DataFrame) for a single wallet.
"""

import math

import networkx as nx
import pandas as pd

from config import config
from detection.benford_engine import (
    compute_benford_metrics_for_windows,
    cross_pair_benford_consistency,
)
from detection.wallet_graph import compute_wallet_graph_metrics
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
}


def compute_benford_features(wallet_trades: pd.DataFrame) -> dict:
    """Flatten per-window Benford metrics into a feature row.

    Produces `benford_chi_square_{h}h`, `benford_mad_{h}h`, and
    `benford_z_max_{h}h` for each configured window.
    """
    per_window = compute_benford_metrics_for_windows(wallet_trades)

    features = {}
    for hours, metrics in per_window.items():
        features[f"benford_chi_square_{hours}h"] = metrics["chi_square"]
        features[f"benford_mad_{hours}h"] = metrics["mad"]
        features[f"benford_z_max_{hours}h"] = max(metrics["z_scores"].values(), default=0.0)

    return features


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
    """Counterparty concentration, round-trips, self-matching, cancellations."""
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
    """Volume concentration and timing-based anomaly features."""
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
) -> dict:
    """Funding-source similarity, network centrality, account age.

    `funding_source_similarity` and `network_centrality` are computed from
    `funding_graph` (see `detection.wallet_graph.build_funding_graph`) when
    provided, and default to `0.0` otherwise.
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

    return {
        **graph_metrics,
        "account_age_days": float(account_age_days),
    }


def compute_cross_asset_features(
    wallet: str,
    all_pairs_df: pd.DataFrame,
) -> dict:
    """Cross-asset coordination features computed from multi-pair trade data.

    `all_pairs_df` should contain trades across all pairs for the wallet, with
    a `pair_id` column identifying which pair each trade belongs to.

    Returns a dict with 6 cross-asset features:
    - cross_pair_trade_synchrony: fraction of trades with simultaneous activity on other pairs
    - net_asset_flow_deviation: max absolute net flow normalized by volume (close to 0 = closed cycle)
    - cross_pair_counterparty_overlap: Jaccard similarity of counterparty sets across pairs
    - cross_pair_volume_correlation: Pearson correlation of volumes across pairs (by minute)
    - pair_diversity_score: Shannon entropy of volume distribution across pairs
    - cross_pair_mad_std: standard deviation of Benford MAD scores across pairs
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
    for pair_id in wallet_trades["pair_id"].unique():
        pair_trades = wallet_trades[wallet_trades["pair_id"] == pair_id]
        metrics = compute_benford_metrics_for_windows(pair_trades)
        # Average MAD across all windows for this pair
        per_pair_metrics[pair_id] = {
            "mad": (
                sum(m.get("mad", 0.0) for m in metrics.values()) / len(metrics) if metrics else 0.0
            )
        }

    features["cross_pair_mad_std"] = cross_pair_benford_consistency(per_pair_metrics)

    return features


def build_feature_vector(
    wallet: str,
    wallet_trades: pd.DataFrame,
    activity: AccountActivity | None = None,
    orderbook_events: pd.DataFrame | None = None,
    funding_graph: nx.DiGraph | None = None,
    all_pairs_df: pd.DataFrame | None = None,
) -> dict:
    """Assemble the full feature row for a single wallet.

    `wallet_trades` should already be filtered to trades involving `wallet`
    as base or counter account. `orderbook_events` (optional) is the output
    of `ingestion.orderbook_loader.load_accounts_orderbook_events`, used to
    compute `order_cancellation_rate`. `funding_graph` (optional) is the
    output of `detection.wallet_graph.build_funding_graph`, used for the
    wallet graph features. `all_pairs_df` (optional) enables cross-asset
    coordination features.
    """
    reference_time = (
        pd.to_datetime(wallet_trades["ledger_close_time"], utc=True).max()
        if not wallet_trades.empty
        else pd.Timestamp.now(tz="UTC")
    )

    features = {"wallet": wallet}
    features.update(compute_benford_features(wallet_trades))
    features.update(compute_trade_pattern_features(wallet, wallet_trades, orderbook_events))
    features.update(compute_volume_timing_features(wallet_trades))
    features.update(compute_wallet_graph_features(wallet, activity, reference_time, funding_graph))
    if all_pairs_df is not None:
        features.update(compute_cross_asset_features(wallet, all_pairs_df))

    return features


def build_feature_matrix(
    trades_df: pd.DataFrame,
    orderbook_events: pd.DataFrame | None = None,
    funding_graph: nx.DiGraph | None = None,
    all_pairs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a feature matrix with one row per wallet observed in `trades_df`.

    `orderbook_events` and `funding_graph` (both optional) are threaded
    through to `build_feature_vector` for `order_cancellation_rate` and the
    wallet graph features respectively. `all_pairs_df` (optional, should be
    the same as `trades_df` or a superset with a `pair_id` column) enables
    cross-asset coordination features.
    """
    if trades_df.empty:
        return pd.DataFrame()

    wallets = pd.unique(trades_df[["base_account", "counter_account"]].values.ravel())

    rows = []
    for wallet in wallets:
        mask = (trades_df["base_account"] == wallet) | (trades_df["counter_account"] == wallet)
        rows.append(
            build_feature_vector(
                wallet,
                trades_df[mask],
                orderbook_events=orderbook_events,
                funding_graph=funding_graph,
                all_pairs_df=all_pairs_df if all_pairs_df is not None else trades_df,
            )
        )

    return pd.DataFrame(rows)
