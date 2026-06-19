"""Thread-safe wrapper around RiskScorer for real-time wallet scoring.

Phase 1 of the real-time detection pipeline (Issue #12).

``RiskScorer.score()`` is stateless (no mutable model state per call), so
``StreamingScorer`` needs no additional locking — concurrent calls from
multiple threads are safe.
"""

from __future__ import annotations

from config import config
from detection.model_inference import RiskScorer
from streaming.feature_buffer import FeatureBuffer
from utils.logging import get_logger

logger = get_logger(__name__)


class StreamingScorer:
    """Scores a wallet on demand using its buffered trades.

    Returns ``None`` when the wallet has fewer than ``min_trades`` buffered
    trades (not enough history for a reliable score).
    """

    def __init__(self, model_dir: str | None = None) -> None:
        self._risk_scorer = RiskScorer(model_dir=model_dir)
        self.min_trades: int = config.MIN_TRADES_FOR_SCORING

    def score_wallet(self, wallet: str, buffer: FeatureBuffer) -> dict | None:
        """Build feature row from *buffer* and score *wallet*.

        Returns a risk-score dict ``{score, benford_flag, ml_flag, confidence}``
        or ``None`` if the wallet has fewer than ``min_trades`` buffered trades.
        """
        if buffer.wallet_trade_count(wallet) < self.min_trades:
            return None

        feature_row = buffer.get_feature_row(wallet)
        if feature_row is None:
            return None

        try:
            return self._risk_scorer.score(feature_row)
        except Exception as exc:
            logger.warning("Scoring failed for wallet %s: %s", wallet, exc)
            return None
