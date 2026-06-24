"""Online, memory-bounded Benford's Law estimator using EWMA decay."""

import math
from datetime import datetime

import numpy as np

from detection.benford_engine import BENFORD_EXPECTED, MAD_NONCONFORMITY_THRESHOLD, BenfordMetrics


class StreamingBenfordSketch:
    """Online, additive sketch over leading-digit frequencies.

    Uses EWMA decay to implement a sliding window without storing individual
    trades. The effective window length is determined by `window_seconds`.
    """

    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self._counts = np.zeros(9, dtype=float)
        self._last_update: datetime | None = None

    def _apply_decay(self, current_time: datetime) -> None:
        """Apply exponential decay based on time elapsed since last update."""
        if self._last_update is None:
            self._last_update = current_time
            return

        delta_t = (current_time - self._last_update).total_seconds()
        if delta_t <= 0:
            return

        # decay = exp(-delta_t / window_seconds)
        decay_factor = math.exp(-delta_t / self.window_seconds)
        self._counts *= decay_factor
        self._last_update = current_time

    def update(self, amount: float, timestamp: datetime) -> None:
        """Ingest one new trade amount."""
        self._apply_decay(timestamp)

        if amount <= 0:
            return

        # Extract leading digit
        mag = math.floor(math.log10(amount))
        digit = int(amount / (10.0**mag))
        if 1 <= digit <= 9:
            self._counts[digit - 1] += 1.0

    @property
    def n(self) -> float:
        """Effective sample size (sum of decayed counts)."""
        return float(np.sum(self._counts))

    def observed_distribution(self) -> dict[int, float]:
        """Estimated frequency of each leading digit 1-9."""
        n = self.n
        if n == 0:
            return {d: 0.0 for d in range(1, 10)}
        return {d: float(self._counts[d - 1] / n) for d in range(1, 10)}

    def chi_square(self) -> float:
        """Estimated chi-square goodness-of-fit statistic."""
        n = self.n
        if n == 0:
            return 0.0

        chi_sq = 0.0
        for d in range(1, 10):
            expected_count = BENFORD_EXPECTED[d] * n
            observed_count = self._counts[d - 1]
            if expected_count > 0:
                chi_sq += (observed_count - expected_count) ** 2 / expected_count
        return float(chi_sq)

    def mad(self) -> float:
        """Estimated Mean Absolute Deviation."""
        n = self.n
        if n == 0:
            return 0.0

        observed = self.observed_distribution()
        deviations = [abs(observed[d] - BENFORD_EXPECTED[d]) for d in range(1, 10)]
        return float(sum(deviations) / len(deviations))

    def z_scores(self) -> dict[int, float]:
        """Estimated per-digit Z-scores."""
        n = self.n
        if n == 0:
            return {d: 0.0 for d in range(1, 10)}

        observed = self.observed_distribution()
        scores = {}
        for d in range(1, 10):
            p = BENFORD_EXPECTED[d]
            # Standard error for a proportion with continuity correction
            std_err = math.sqrt(p * (1 - p) / n)
            if std_err == 0:
                scores[d] = 0.0
                continue
            z = (abs(observed[d] - p) - (1 / (2 * n))) / std_err
            scores[d] = float(max(z, 0.0))

        return scores

    def to_metrics(self) -> BenfordMetrics:
        """Convert current sketch state to BenfordMetrics."""
        mad = self.mad()
        return BenfordMetrics(
            chi_square=self.chi_square(),
            mad=mad,
            mad_nonconforming=mad > MAD_NONCONFORMITY_THRESHOLD,
            z_scores=self.z_scores(),
            sample_size=int(self.n),
        )
