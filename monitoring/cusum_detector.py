"""CUSUM (Cumulative Sum) control chart for online change-point detection (issue #289).

Detects sustained upward or downward shifts in a streaming metric (e.g. the
LedgerLens risk score stream) in O(1) time and O(1) space per update.

Theory
------
The two-sided Page-CUSUM statistic maintains:

    S_high[n] = max(0, S_high[n-1] + x_n - (μ₀ + k))
    S_low[n]  = max(0, S_low[n-1]  - x_n + (μ₀ - k))

An alarm fires when either statistic exceeds h (decision threshold). After
acknowledgement both statistics are reset to zero.

Parameter guidance (in-control ARL ≈ 500, out-of-control ARL ≈ 10 for a
10-point shift with σ ≈ 15):
    k = 5.0   (half the minimum detectable shift in score units)
    h = 25.0

Public API
----------
CUSUMDetector
    .update(value)   -> bool   (True = alarm just triggered)
    .is_alarm        -> bool
    .acknowledge()
"""

from __future__ import annotations

import logging

from prometheus_client import Gauge

from config import config

logger = logging.getLogger(__name__)

_cusum_alarm_gauge = Gauge(
    "ledgerlens_cusum_alarm",
    "CUSUM change-point alarm (1=alarm, 0=in-control)",
    ["metric"],
)

_REDIS_KEY_PREFIX = "ledgerlens:cusum:"


class CUSUMDetector:
    """Two-sided CUSUM control chart with optional Redis state persistence.

    Args:
        metric_name: Logical name used for Prometheus labels and Redis key.
        target_mean: Expected in-control mean (μ₀).
        allowable_slack: Allowable slack k (typically half the minimum shift).
        decision_threshold: Alarm threshold h.
        redis_client: Optional ``redis.Redis`` instance for alarm persistence
            across worker restarts. When ``None`` alarm state is in-memory only.
    """

    def __init__(
        self,
        metric_name: str = "risk_score",
        target_mean: float | None = None,
        allowable_slack: float | None = None,
        decision_threshold: float | None = None,
        redis_client=None,
    ) -> None:
        self.metric_name = metric_name
        self.mu0: float = target_mean if target_mean is not None else config.CUSUM_TARGET_MEAN
        self.k: float = (
            allowable_slack if allowable_slack is not None else config.CUSUM_ALLOWABLE_SLACK
        )
        self.h: float = (
            decision_threshold
            if decision_threshold is not None
            else config.CUSUM_DECISION_THRESHOLD
        )

        if self.k < 0:
            raise ValueError("allowable_slack (k) must be >= 0")
        if self.h <= 0:
            raise ValueError("decision_threshold (h) must be > 0")

        self._redis = redis_client
        self._s_high: float = 0.0
        self._s_low: float = 0.0
        self._alarm: bool = False

        # Restore alarm state from Redis if available
        if self._redis is not None:
            try:
                val = self._redis.get(f"{_REDIS_KEY_PREFIX}{metric_name}:alarm")
                self._alarm = val == b"1"
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, value: float) -> bool:
        """Ingest one observation; return True if alarm newly triggered."""
        if self._alarm:
            return False  # already alarming — call acknowledge() first

        self._s_high = max(0.0, self._s_high + value - (self.mu0 + self.k))
        self._s_low = max(0.0, self._s_low - value + (self.mu0 - self.k))

        if self._s_high >= self.h or self._s_low >= self.h:
            self._alarm = True
            _cusum_alarm_gauge.labels(metric=self.metric_name).set(1)
            logger.warning(
                "CUSUM alarm: metric=%s s_high=%.2f s_low=%.2f h=%.2f",
                self.metric_name,
                self._s_high,
                self._s_low,
                self.h,
            )
            self._persist_alarm(True)
            return True

        return False

    def acknowledge(self) -> None:
        """Reset CUSUM statistics and clear the alarm."""
        self._s_high = 0.0
        self._s_low = 0.0
        self._alarm = False
        _cusum_alarm_gauge.labels(metric=self.metric_name).set(0)
        self._persist_alarm(False)
        logger.info("CUSUM alarm acknowledged and reset: metric=%s", self.metric_name)

    @property
    def is_alarm(self) -> bool:
        return self._alarm

    @property
    def s_high(self) -> float:
        return self._s_high

    @property
    def s_low(self) -> float:
        return self._s_low

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _persist_alarm(self, state: bool) -> None:
        if self._redis is None:
            return
        try:
            key = f"{_REDIS_KEY_PREFIX}{self.metric_name}:alarm"
            self._redis.set(key, "1" if state else "0")
        except Exception as exc:
            logger.warning("Failed to persist CUSUM alarm state to Redis: %s", exc)
