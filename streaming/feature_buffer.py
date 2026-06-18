"""Rolling per-wallet trade buffer and on-the-fly streaming scorer.

Phase 1 of the real-time detection pipeline (Issue #012).
FeatureBuffer accumulates trades per wallet; StreamingScorer builds a feature
vector and scores once the wallet has enough history.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pandas as pd

from detection.feature_engineering import build_feature_vector
from ingestion.data_models import Trade
from utils.logging import get_logger

if TYPE_CHECKING:
    from detection.model_inference import RiskScorer

logger = get_logger(__name__)


class FeatureBuffer:
    """Thread-safe rolling trade buffer; builds feature vectors on demand."""

    def __init__(self, max_trades_per_wallet: int = 5000):
        self._buffer: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self.max_trades_per_wallet = max_trades_per_wallet

    def update(self, trade: Trade) -> None:
        """Append trade to both the base_account and counter_account buffers."""
        record = {
            "ledger_close_time": trade.ledger_close_time,
            "base_account": trade.base_account,
            "counter_account": trade.counter_account,
            "amount": trade.amount,
        }
        with self._lock:
            for wallet in (trade.base_account, trade.counter_account):
                buf = self._buffer.setdefault(wallet, [])
                buf.append(record)
                if len(buf) > self.max_trades_per_wallet:
                    self._buffer[wallet] = buf[-self.max_trades_per_wallet :]

    def get_wallet_df(self, wallet: str) -> pd.DataFrame:
        """Return buffered trades for *wallet* as a DataFrame."""
        with self._lock:
            records = list(self._buffer.get(wallet, []))
        if not records:
            return pd.DataFrame(
                columns=["ledger_close_time", "base_account", "counter_account", "amount"]
            )
        return pd.DataFrame(records)

    def wallet_trade_count(self, wallet: str) -> int:
        """Number of trades buffered for *wallet*."""
        with self._lock:
            return len(self._buffer.get(wallet, []))


class StreamingScorer:
    """Scores wallets in real time using buffered trades and a trained RiskScorer."""

    def __init__(
        self,
        risk_scorer: RiskScorer,
        buffer: FeatureBuffer,
        min_trades: int = 20,
    ):
        self._risk_scorer = risk_scorer
        self._buffer = buffer
        self.min_trades = min_trades

    def score_wallet(self, wallet: str) -> dict | None:
        """Return a RiskScore dict, or *None* if the wallet lacks enough history."""
        if self._buffer.wallet_trade_count(wallet) < self.min_trades:
            return None

        wallet_df = self._buffer.get_wallet_df(wallet)
        features = build_feature_vector(wallet, wallet_df)
        feature_row = pd.Series(features)

        try:
            return self._risk_scorer.score(feature_row)
        except Exception as exc:
            logger.warning("Scoring failed for wallet %s: %s", wallet, exc)
            return None
