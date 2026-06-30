"""Sliding-window Benford's Law aggregator (Issue #254).

Maintains running per-digit counts updated incrementally as trades arrive
and old trades expire, so chi-square / MAD / Z-scores can be computed in
O(1) rather than by re-scanning historical records.

Expiry policy
~~~~~~~~~~~~~
Trades expire when ``current_time - timestamp > window_hours * 3600``.
Expiry is *lazy*: the ``_expiry_heap`` is only drained on each
``add_trade`` call, keeping the hot path cheap.

Concurrency
~~~~~~~~~~~
All mutations are guarded by ``asyncio.Lock``.  The class is designed to be
used from a single asyncio event loop; call the async ``add_trade`` /
``expire_trade`` coroutines from your scoring coroutines.

Tolerance guarantee
~~~~~~~~~~~~~~~~~~~
Running chi-square matches the batch-computed value within 1e-6 absolute
tolerance on synthetic data (verified in ``tests/test_sliding_window_benford.py``).

Security
~~~~~~~~
Trade amounts are validated (positive, finite floats) before the leading
digit is extracted.  Invalid amounts are skipped with a WARNING log.
"""

from __future__ import annotations

import asyncio
import heapq
import math
from datetime import datetime, timezone

import numpy as np

from detection.benford_engine import BENFORD_EXPECTED, MAD_NONCONFORMITY_THRESHOLD, BenfordMetrics
from utils.logging import get_logger

logger = get_logger(__name__)


def _leading_digit(amount: float) -> int | None:
    """Return the leading (first significant) digit 1–9, or None for invalid input."""
    if not math.isfinite(amount) or amount <= 0:
        return None
    mag = math.floor(math.log10(amount))
    digit = int(amount / (10.0 ** mag))
    return max(1, min(9, digit))


class SlidingWindowBenfordAggregator:
    """Incremental Benford aggregator over a fixed-size trailing time window.

    Args:
        window_hours: Window width in hours.  Trades older than this are expired.
    """

    def __init__(self, window_hours: float) -> None:
        self.window_seconds: float = window_hours * 3600.0
        # digit_counts[d-1] = count of trades whose leading digit is d (d in 1..9)
        self._digit_counts: list[int] = [0] * 9
        self._total: int = 0
        # Min-heap of (timestamp_float, digit) for lazy expiry
        self._heap: list[tuple[float, int]] = []
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def add_trade(self, amount: float, timestamp: datetime) -> None:
        """Ingest one trade and lazily expire stale trades.

        ``timestamp`` should be timezone-aware; naive timestamps are treated as UTC.
        """
        ts = _to_utc_float(timestamp)
        async with self._lock:
            self._lazy_expire(ts)
            digit = _leading_digit(amount)
            if digit is None:
                logger.warning("Skipping invalid trade amount: %r", amount)
                return
            self._digit_counts[digit - 1] += 1
            self._total += 1
            heapq.heappush(self._heap, (ts, digit))

    async def expire_trade(self, amount: float, timestamp: datetime) -> None:
        """Explicitly remove a specific trade from the running counts.

        Prefer relying on lazy expiry via ``add_trade``; this method is
        provided for tests and manual backfill scenarios.
        """
        digit = _leading_digit(amount)
        if digit is None:
            return
        async with self._lock:
            if self._digit_counts[digit - 1] > 0:
                self._digit_counts[digit - 1] -= 1
                self._total -= 1

    # ------------------------------------------------------------------
    # Metric computation (synchronous — safe to call inside the lock context
    # or from a single-threaded test without holding the lock)
    # ------------------------------------------------------------------

    def chi_square(self) -> float:
        if self._total == 0:
            return 0.0
        chi_sq = 0.0
        for d in range(1, 10):
            expected = BENFORD_EXPECTED[d] * self._total
            observed = self._digit_counts[d - 1]
            if expected > 0:
                chi_sq += (observed - expected) ** 2 / expected
        return float(chi_sq)

    def mad(self) -> float:
        if self._total == 0:
            return 0.0
        deviations = [
            abs(self._digit_counts[d - 1] / self._total - BENFORD_EXPECTED[d])
            for d in range(1, 10)
        ]
        return float(sum(deviations) / 9)

    def z_scores(self) -> dict[int, float]:
        n = self._total
        if n == 0:
            return {d: 0.0 for d in range(1, 10)}
        scores: dict[int, float] = {}
        for d in range(1, 10):
            p = BENFORD_EXPECTED[d]
            observed_p = self._digit_counts[d - 1] / n
            std_err = math.sqrt(p * (1 - p) / n)
            if std_err == 0:
                scores[d] = 0.0
            else:
                z = (abs(observed_p - p) - 1 / (2 * n)) / std_err
                scores[d] = float(max(z, 0.0))
        return scores

    def to_metrics(self) -> BenfordMetrics:
        mad_val = self.mad()
        return BenfordMetrics(
            chi_square=self.chi_square(),
            mad=mad_val,
            mad_nonconforming=mad_val > MAD_NONCONFORMITY_THRESHOLD,
            z_scores=self.z_scores(),
            sample_size=self._total,
        )

    @property
    def sample_size(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lazy_expire(self, current_ts: float) -> None:
        """Remove all heap entries older than the window (called while lock is held)."""
        cutoff = current_ts - self.window_seconds
        while self._heap and self._heap[0][0] <= cutoff:
            _, digit = heapq.heappop(self._heap)
            if self._digit_counts[digit - 1] > 0:
                self._digit_counts[digit - 1] -= 1
                self._total -= 1


def _to_utc_float(ts: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()
