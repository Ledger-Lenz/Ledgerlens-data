"""Metrics collector that feeds scored wallet events into the CUSUM detector (issue #289).

Usage
-----
    collector = MetricsCollector()
    collector.record_score(wallet="G...", score=72.0)
"""

from __future__ import annotations

import logging

from monitoring.cusum_detector import CUSUMDetector

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect per-wallet risk score events and forward them to CUSUM.

    Args:
        cusum: Optional pre-configured ``CUSUMDetector``. A default instance
            (using ``config`` defaults) is created when ``None``.
        redis_client: Forwarded to a default ``CUSUMDetector`` when one is
            created internally.
    """

    def __init__(
        self,
        cusum: CUSUMDetector | None = None,
        redis_client=None,
    ) -> None:
        self._cusum = cusum or CUSUMDetector(
            metric_name="risk_score", redis_client=redis_client
        )

    def record_score(self, wallet: str, score: float) -> bool:
        """Record a scored wallet event; return True if CUSUM alarm fires.

        Args:
            wallet: Stellar account ID (used only for logging).
            score: Risk score in [0, 100].

        Returns:
            True if the CUSUM alarm was just triggered by this observation.
        """
        alarmed = self._cusum.update(score)
        if alarmed:
            logger.warning("CUSUM alarm triggered by wallet %s with score %.1f", wallet, score)
        return alarmed

    @property
    def cusum(self) -> CUSUMDetector:
        return self._cusum
