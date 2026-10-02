"""Tests for PerPairScoreNormaliser edge cases (#677)."""

from unittest.mock import MagicMock

import pytest

from detection.score_normaliser import PerPairScoreNormaliser, NormalisedScore


class TestConstantInputNormalisation:
    """Verify the normaliser handles constant-valued score arrays gracefully (#677)."""

    ALLOWED_PAIR = "USDC:GA5ZSEJYBY3RRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"

    def _make_redis_with_constant_scores(self, count: int, value: float = 42.0):
        """Create a mock Redis that returns `count` identical scores."""
        mock_redis = MagicMock()
        mock_redis.zrange.return_value = [(str(i), value) for i in range(count)]
        return mock_redis

    def test_all_identical_scores_return_finite_percentile(self):
        """When every score in the window is identical, the percentile must be finite."""
        mock_redis = self._make_redis_with_constant_scores(100, 42.0)
        normaliser = PerPairScoreNormaliser(mock_redis)
        normaliser.min_samples = 50

        result = normaliser.normalise(self.ALLOWED_PAIR, 42.0)

        assert isinstance(result, NormalisedScore)
        assert result.normalisation_skipped is False
        assert 0.0 < result.normalised_risk_score <= 1.0
        assert result.normalised_risk_score == pytest.approx(0.5 / 100, abs=0.01)

    def test_constant_scores_higher_value_gets_max_percentile(self):
        """A score above all constant values gets a percentile above 0.5/n."""
        mock_redis = self._make_redis_with_constant_scores(100, 42.0)
        normaliser = PerPairScoreNormaliser(mock_redis)
        normaliser.min_samples = 50

        result = normaliser.normalise(self.ALLOWED_PAIR, 99.0)

        assert not result.normalisation_skipped
        assert result.normalised_risk_score > 0.0
        assert result.normalised_risk_score <= 1.5  # well within bounds

    def test_constant_scores_lower_value_gets_min_percentile(self):
        """A score below all constant values gets the minimum percentile."""
        mock_redis = self._make_redis_with_constant_scores(100, 42.0)
        normaliser = PerPairScoreNormaliser(mock_redis)
        normaliser.min_samples = 50

        result = normaliser.normalise(self.ALLOWED_PAIR, 1.0)

        assert not result.normalisation_skipped
        assert result.normalised_risk_score > 0.0
        assert result.normalised_risk_score == pytest.approx(0.5 / 100, abs=0.01)

    def test_single_unique_value_in_window(self):
        """Even a window with a single unique value returns a finite result."""
        mock_redis = self._make_redis_with_constant_scores(60, 7.0)
        normaliser = PerPairScoreNormaliser(mock_redis)
        normaliser.min_samples = 50

        result = normaliser.normalise(self.ALLOWED_PAIR, 7.0)

        assert not result.normalisation_skipped
        import math
        assert math.isfinite(result.normalised_risk_score)
        assert 0.0 < result.normalised_risk_score <= 1.0

    def test_below_min_samples_returns_raw_score(self):
        """When the window has fewer than min_samples, the raw score passes through."""
        mock_redis = self._make_redis_with_constant_scores(10, 42.0)
        normaliser = PerPairScoreNormaliser(mock_redis)
        normaliser.min_samples = 50

        result = normaliser.normalise(self.ALLOWED_PAIR, 42.0)

        assert result.normalisation_skipped is True
        assert result.normalised_risk_score == 42.0
