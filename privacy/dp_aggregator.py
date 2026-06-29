"""Differential-privacy noise calibration for training data aggregation.

Wraps standard aggregation functions (mean, count, histogram) with calibrated
DP noise using the Laplace mechanism (mean, count) or Gaussian mechanism
(histogram) as specified in issue #299.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when DP configuration parameters are invalid."""


@dataclass
class PrivacyBudget:
    epsilon_used: float = 0.0
    delta_used: float = 0.0
    queries: int = 0


class DPAggregator:
    """Applies calibrated differential-privacy noise to aggregate statistics.

    Supports:
        - Laplace mechanism for mean and count queries (pure DP, δ=0)
        - Gaussian mechanism for histogram queries (approximate DP, δ>0)

    Args:
        epsilon: Privacy budget ε. Must be > 0.
        delta: Privacy budget δ. Must be in (0, 0.5). Used only for Gaussian.
        random_seed: Optional seed for reproducibility in tests; None = unseeded.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        random_seed: Optional[int] = None,
    ) -> None:
        self._validate_params(epsilon, delta)
        self.epsilon = epsilon
        self.delta = delta
        self._rng = np.random.default_rng(random_seed)
        self._budget = PrivacyBudget()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_params(epsilon: float, delta: float) -> None:
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be > 0, got {epsilon}"
            )
        if not (0 < delta < 0.5):
            raise ConfigurationError(
                f"delta must be in (0, 0.5), got {delta}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def private_mean(
        self,
        values: np.ndarray,
        feature_min: float,
        feature_max: float,
    ) -> float:
        """Return a differentially-private estimate of the mean.

        Uses the Laplace mechanism.  L1 sensitivity = (feature_max - feature_min) / n.

        Args:
            values: 1-D array of feature values.
            feature_min: Known lower bound of the feature range.
            feature_max: Known upper bound of the feature range.

        Returns:
            Noised mean estimate.
        """
        n = len(values)
        if n == 0:
            return 0.0

        true_mean = float(np.mean(values))
        sensitivity = (feature_max - feature_min) / n
        scale = sensitivity / self.epsilon
        noise = self._rng.laplace(0.0, scale)

        self._budget.epsilon_used += self.epsilon
        self._budget.queries += 1

        return true_mean + noise

    def private_count(self, values: np.ndarray) -> float:
        """Return a differentially-private count.

        L1 sensitivity for count = 1.  Uses the Laplace mechanism.

        Args:
            values: Array whose length is the count.

        Returns:
            Noised count (may be non-integer).
        """
        true_count = float(len(values))
        sensitivity = 1.0
        scale = sensitivity / self.epsilon
        noise = self._rng.laplace(0.0, scale)

        self._budget.epsilon_used += self.epsilon
        self._budget.queries += 1

        return max(0.0, true_count + noise)

    def private_histogram(
        self,
        values: np.ndarray,
        bins: int | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a differentially-private histogram using the Gaussian mechanism.

        Sensitivity for a histogram bin count = 1 (one record shifts exactly
        one bin by ±1).  σ is calibrated to (ε, δ) via the standard Gaussian
        mechanism formula.

        Args:
            values: 1-D array of values to bin.
            bins: Number of bins or explicit bin edges (passed to np.histogram).

        Returns:
            (noised_counts, bin_edges) where noised_counts are clipped to ≥ 0.
        """
        counts, edges = np.histogram(values, bins=bins)

        sensitivity = 1.0  # L2 sensitivity per bin
        sigma = self._gaussian_sigma(sensitivity)
        noise = self._rng.normal(0.0, sigma, size=counts.shape)
        noised = np.maximum(0.0, counts.astype(float) + noise)

        self._budget.epsilon_used += self.epsilon
        self._budget.delta_used += self.delta
        self._budget.queries += 1

        return noised, edges

    def budget_consumed(self) -> PrivacyBudget:
        """Return accumulated privacy budget consumed so far."""
        return PrivacyBudget(
            epsilon_used=self._budget.epsilon_used,
            delta_used=self._budget.delta_used,
            queries=self._budget.queries,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gaussian_sigma(self, sensitivity: float) -> float:
        """Compute Gaussian noise σ = Δ · √(2 ln(1.25/δ)) / ε."""
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
