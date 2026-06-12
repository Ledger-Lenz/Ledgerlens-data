"""Builds the 30+ feature vector consumed by the ensemble ML models.

Feature groups (see README):
  - Benford features (15): chi-square / Z-score / MAD across 5 windows
  - Trade pattern features
  - Volume and timing features
  - Wallet graph features

Each `compute_*_features` function operates on the trade DataFrame produced
by `ingestion.historical_loader.trades_to_dataframe` (or the streamer,
buffered into a DataFrame) for a single wallet.
"""

import pandas as pd

from config import config
from detection.benford_engine import compute_benford_metrics_for_windows
from ingestion.data_models import AccountActivity


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


def compute_trade_pattern_features(wallet: str, wallet_trades: pd.DataFrame) -> dict:
    """Counterparty concentration, round-trips, self-matching, cancellations."""
    if wallet_trades.empty:
        return {
            "counterparty_concentration_ratio": 0.0,
            "round_trip_frequency": 0.0,
            "self_matching_rate": 0.0,
            "order_cancellation_rate": 0.0,
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

    # Order cancellation rate requires order book events; placeholder until
    # that ingestion path is wired up.
    order_cancellation_rate = 0.0

    return {
        "counterparty_concentration_ratio": float(concentration),
        "round_trip_frequency": float(round_trip_frequency),
        "self_matching_rate": float(self_matching_rate),
        "order_cancellation_rate": float(order_cancellation_rate),
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
    wallet: str, activity: AccountActivity | None, reference_time: pd.Timestamp
) -> dict:
    """Funding-source similarity, network centrality, account age.

    `funding_source_similarity` and `network_centrality` require a wallet
    graph built across many accounts and are left as placeholders (0.0)
    until the graph-construction job is implemented.
    """
    account_age_days = 0.0
    if activity is not None:
        created_at = pd.to_datetime(activity.account_created_at, utc=True)
        account_age_days = (reference_time - created_at).total_seconds() / 86400

    return {
        "funding_source_similarity": 0.0,
        "network_centrality": 0.0,
        "account_age_days": float(account_age_days),
    }


def build_feature_vector(
    wallet: str,
    wallet_trades: pd.DataFrame,
    activity: AccountActivity | None = None,
) -> dict:
    """Assemble the full feature row for a single wallet.

    `wallet_trades` should already be filtered to trades involving `wallet`
    as base or counter account.
    """
    reference_time = (
        pd.to_datetime(wallet_trades["ledger_close_time"], utc=True).max()
        if not wallet_trades.empty
        else pd.Timestamp.now(tz="UTC")
    )

    features = {"wallet": wallet}
    features.update(compute_benford_features(wallet_trades))
    features.update(compute_trade_pattern_features(wallet, wallet_trades))
    features.update(compute_volume_timing_features(wallet_trades))
    features.update(compute_wallet_graph_features(wallet, activity, reference_time))

    return features


def build_feature_matrix(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix with one row per wallet observed in `trades_df`."""
    if trades_df.empty:
        return pd.DataFrame()

    wallets = pd.unique(trades_df[["base_account", "counter_account"]].values.ravel())

    rows = []
    for wallet in wallets:
        mask = (trades_df["base_account"] == wallet) | (trades_df["counter_account"] == wallet)
        rows.append(build_feature_vector(wallet, trades_df[mask]))

    return pd.DataFrame(rows)
