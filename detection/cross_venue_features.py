"""Cross-pair aggregator and cross-venue feature computation.

After per-pair Benford metrics are computed by independent partition workers,
this module provides:
  1. An aggregator consumer that reads from all partitions
  2. Cross-pair feature functions (e.g., venue concentration, multi-pair volume)

The aggregator consumer is typically run in a dedicated process to build
cross-pair statistics without interfering with per-pair scorers.

Architecture:
    Per-partition workers compute: per-pair Benford, trade patterns (partition-specific)
    Aggregator consumer computes: cross-pair statistics (all partitions)
    → Features fed into ML model for final risk score
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config import config
from utils.logging import get_logger

if TYPE_CHECKING:
    from kafka.structs import TopicPartition

logger = get_logger(__name__)


class CrossVenueAggregator:
    """Aggregates trade data across all partitions for cross-pair analysis."""

    def __init__(
        self,
        topic: str,
        group_id: str = "ledgerlens-aggregator",
        bootstrap_servers: list[str] | str = "localhost:9092",
    ):
        """Initialize aggregator consumer.

        Args:
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            bootstrap_servers: Kafka bootstrap server(s)
        """
        if isinstance(bootstrap_servers, str):
            bootstrap_servers = [bootstrap_servers]

        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers

        self.consumer = KafkaConsumer(
            topic,
            group_id=group_id,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        # Buffer for cross-pair aggregation
        self._trades_by_wallet: dict[str, list[dict]] = {}
        self._trades_by_pair: dict[str, list[dict]] = {}

        logger.info(
            "CrossVenueAggregator initialized: topic=%s, group_id=%s",
            topic,
            group_id,
        )

    def collect_trades(self, max_batches: int = 100) -> None:
        """Consume and buffer trades from all partitions.

        Args:
            max_batches: Number of poll batches before returning
        """
        for batch_idx in range(max_batches):
            messages = self.consumer.poll(timeout_ms=1000, max_records=100)
            if not messages:
                logger.debug("No messages in batch %d", batch_idx)
                continue

            for topic_partition, records in messages.items():
                for record in records:
                    payload = record.value
                    self._buffer_trade(payload)

    def _buffer_trade(self, payload: dict) -> None:
        """Buffer a trade for aggregation.

        Args:
            payload: Trade event dict from Kafka
        """
        wallet_record = {
            "ledger_close_time": payload.get("ledger_close_time"),
            "base_account": payload.get("base_account"),
            "counter_account": payload.get("counter_account"),
            "base_amount": payload.get("base_amount"),
            "counter_amount": payload.get("counter_amount"),
            "pair_id": payload.get("pair_id"),
        }

        pair_id = payload.get("pair_id")

        # Buffer by wallet (for cross-pair analysis)
        for wallet in (payload.get("base_account"), payload.get("counter_account")):
            if wallet:
                self._trades_by_wallet.setdefault(wallet, []).append(wallet_record)

        # Buffer by pair (for cross-venue analysis)
        if pair_id:
            self._trades_by_pair.setdefault(pair_id, []).append(wallet_record)

    def get_cross_pair_features(self, wallet: str) -> dict:
        """Compute cross-pair features for a wallet.

        Cross-pair features include:
          - Number of distinct pairs the wallet traded on
          - Cross-pair volume concentration
          - Venue diversity score

        Args:
            wallet: Wallet address

        Returns:
            dict with cross-pair feature values
        """
        trades = self._trades_by_wallet.get(wallet, [])
        if not trades:
            return {
                "n_distinct_pairs": 0,
                "cross_pair_volume_concentration": 0.0,
                "venue_diversity_score": 0.0,
            }

        trades_df = pd.DataFrame(trades)
        n_pairs = trades_df["pair_id"].nunique()
        total_volume = trades_df["base_amount"].sum()

        # Volume concentration by pair
        if n_pairs > 0 and total_volume > 0:
            volume_by_pair = trades_df.groupby("pair_id")["base_amount"].sum()
            concentration = (volume_by_pair.max() / total_volume) if total_volume > 0 else 0.0
            # Venue diversity: inverse of concentration, normalized by pair count
            diversity = (1.0 - concentration) / max(n_pairs, 1)
        else:
            concentration = 0.0
            diversity = 0.0

        return {
            "n_distinct_pairs": int(n_pairs),
            "cross_pair_volume_concentration": float(concentration),
            "venue_diversity_score": float(diversity),
        }

    def get_pair_cross_venue_features(self, pair_id: str) -> dict:
        """Compute cross-venue features for a specific pair.

        Features include:
          - Number of distinct counterparties
          - Self-trading frequency
          - Venue-specific anomalies

        Args:
            pair_id: Asset pair ID (canonical format)

        Returns:
            dict with pair-specific cross-venue features
        """
        trades = self._trades_by_pair.get(pair_id, [])
        if not trades:
            return {
                "n_distinct_wallets": 0,
                "self_trading_frequency": 0.0,
                "pair_volume": 0.0,
            }

        trades_df = pd.DataFrame(trades)
        n_wallets = pd.unique(
            trades_df[["base_account", "counter_account"]].values.ravel()
        ).size

        # Self-trading: same account as both base and counter
        self_trades = (trades_df["base_account"] == trades_df["counter_account"]).sum()
        self_trading_freq = self_trades / len(trades_df) if len(trades_df) > 0 else 0.0

        total_volume = trades_df["base_amount"].sum()

        return {
            "n_distinct_wallets": int(n_wallets),
            "self_trading_frequency": float(self_trading_freq),
            "pair_volume": float(total_volume),
        }

    def clear_buffers(self) -> None:
        """Clear buffered trades (after aggregation is complete)."""
        self._trades_by_wallet.clear()
        self._trades_by_pair.clear()

    def close(self) -> None:
        """Close consumer."""
        self.consumer.close()


def compute_cross_pair_features(
    wallet: str,
    trades_df: pd.DataFrame,
) -> dict:
    """Compute cross-pair features from a DataFrame of trades.

    This is the batch equivalent of CrossVenueAggregator.get_cross_pair_features()
    and is used by the historical pipeline (run_pipeline.py).

    Args:
        wallet: Wallet address
        trades_df: DataFrame with all trades (across all pairs)

    Returns:
        dict with cross-pair feature values
    """
    if trades_df.empty:
        return {
            "n_distinct_pairs": 0,
            "cross_pair_volume_concentration": 0.0,
            "venue_diversity_score": 0.0,
        }

    # Filter to trades involving this wallet
    mask = (trades_df["base_account"] == wallet) | (trades_df["counter_account"] == wallet)
    wallet_trades = trades_df[mask]

    if wallet_trades.empty:
        return {
            "n_distinct_pairs": 0,
            "cross_pair_volume_concentration": 0.0,
            "venue_diversity_score": 0.0,
        }

    # Compute features
    n_pairs = wallet_trades["base_asset"].combine(
        wallet_trades["counter_asset"],
        lambda x, y: f"{x}/{y}",
    ).nunique()

    total_volume = wallet_trades["amount"].sum()

    # Volume concentration by pair
    if n_pairs > 0 and total_volume > 0:
        wallet_trades_copy = wallet_trades.copy()
        wallet_trades_copy["pair"] = wallet_trades_copy["base_asset"].combine(
            wallet_trades_copy["counter_asset"],
            lambda x, y: f"{x}/{y}",
        )
        volume_by_pair = wallet_trades_copy.groupby("pair")["amount"].sum()
        concentration = (volume_by_pair.max() / total_volume) if total_volume > 0 else 0.0
        diversity = (1.0 - concentration) / max(n_pairs, 1)
    else:
        concentration = 0.0
        diversity = 0.0

    return {
        "n_distinct_pairs": int(n_pairs),
        "cross_pair_volume_concentration": float(concentration),
        "venue_diversity_score": float(diversity),
    }
