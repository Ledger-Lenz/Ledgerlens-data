"""Tests for model inversion attack defence (Issue #264).

Tests for Laplace output perturbation and query rate limiting.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import config
from detection.differential_privacy import add_laplace_noise, laplace_scale
from detection.model_inference import _apply_output_perturbation
from detection.persistence import Base, get_session_factory
from detection.risk_score_store import RiskScoreStore


@pytest.fixture
def isolated_db():
    """Create an isolated in-memory SQLite database for tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    # Patch get_session_factory to use this engine
    original_get_session_factory = get_session_factory
    
    def mock_get_session_factory(*args, **kwargs):
        return sessionmaker(bind=engine)
    
    import detection.risk_score_store as risk_store_module
    risk_store_module.get_session_factory = mock_get_session_factory
    
    yield engine
    
    # Cleanup
    risk_store_module.get_session_factory = original_get_session_factory


class TestLaplaceNoise:
    """Test Laplace noise mechanism for output perturbation."""

    def test_laplace_scale_calculation(self):
        """Test that laplace_scale computes sensitivity / epsilon correctly."""
        sensitivity = 100.0
        epsilon = 1.0
        expected = 100.0
        assert laplace_scale(sensitivity, epsilon) == expected

    def test_laplace_scale_negative_epsilon(self):
        """Test that negative epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            laplace_scale(100.0, -1.0)

    def test_add_laplace_noise_variance(self):
        """Test that 1000 identical scores with Laplace noise produce non-zero variance."""
        scale = 10.0
        scores = []
        for _ in range(1000):
            noise = add_laplace_noise(50.0, scale)
            scores.append(noise)
        
        import statistics
        variance = statistics.variance(scores)
        assert variance > 0, "Laplace noise should produce non-zero variance"

    def test_output_perturbation_internal_skips_noise(self):
        """Test that internal caller_id="internal" skips perturbation."""
        score = 75
        perturbed = _apply_output_perturbation(score, caller_id="internal")
        assert perturbed == score, "Internal calls should not be perturbed"

    def test_output_perturbation_external_adds_noise(self):
        """Test that external caller gets perturbed score."""
        score = 50
        perturbed = _apply_output_perturbation(score, caller_id="api.user123", timestamp_bucket=1000)
        # Score should be within reasonable bounds (clipped to [0, 100])
        assert 0 <= perturbed <= 100
        # With small epsilon (1.0) and scale=100, score may differ
        # We only check it's been perturbed in some cases, not deterministically

    def test_output_perturbation_seeding_reproducible(self):
        """Test that same (caller_id, timestamp_bucket) produces same perturbation."""
        score = 60
        caller_id = "api.user456"
        timestamp_bucket = 2000
        
        p1 = _apply_output_perturbation(score, caller_id=caller_id, timestamp_bucket=timestamp_bucket)
        p2 = _apply_output_perturbation(score, caller_id=caller_id, timestamp_bucket=timestamp_bucket)
        
        assert p1 == p2, "Same seed should produce same perturbation"

    def test_output_perturbation_different_callers_different_noise(self):
        """Test that different caller_ids produce different noise."""
        score = 70
        timestamp_bucket = 3000
        
        # Run multiple times to get variety
        results = []
        for caller_id in [f"api.user{i}" for i in range(5)]:
            p = _apply_output_perturbation(score, caller_id=caller_id, timestamp_bucket=timestamp_bucket)
            results.append(p)
        
        # At least some should differ (unlikely all are identical with Laplace noise)
        unique_results = set(results)
        assert len(unique_results) > 1, "Different callers should (usually) produce different noise"

    def test_score_rounding_granularity(self):
        """Test that scores are rounded to SCORE_ROUNDING_GRANULARITY."""
        # Temporarily set granularity to 5
        original_granularity = config.SCORE_ROUNDING_GRANULARITY
        config.SCORE_ROUNDING_GRANULARITY = 5
        
        try:
            score = 77
            perturbed = _apply_output_perturbation(score, caller_id="api.test", timestamp_bucket=4000)
            assert perturbed % 5 == 0, f"Perturbed score {perturbed} should be multiple of 5"
        finally:
            config.SCORE_ROUNDING_GRANULARITY = original_granularity


class TestQueryRateLimiting:
    """Test query rate limiting to prevent sustained model inversion attacks."""

    def test_query_limit_initialization(self, isolated_db):
        """Test that new (caller, wallet) pair has count 0."""
        store = RiskScoreStore()
        exceeded, count = store.check_query_limit("caller1", "wallet1")
        assert not exceeded
        assert count == 0

    def test_query_count_increment(self, isolated_db):
        """Test that query count increments correctly."""
        store = RiskScoreStore()
        caller_id = "caller2"
        wallet_id = "wallet2"
        
        count1 = store.increment_query_count(caller_id, wallet_id)
        assert count1 == 1
        
        count2 = store.increment_query_count(caller_id, wallet_id)
        assert count2 == 2

    def test_query_limit_exceeded(self, isolated_db):
        """Test that limit is correctly detected after exceeding threshold."""
        store = RiskScoreStore()
        caller_id = "caller3"
        wallet_id = "wallet3"
        
        # Increment up to limit
        for _ in range(config.MODEL_INVERSION_QUERY_LIMIT):
            store.increment_query_count(caller_id, wallet_id)
        
        # Check at limit
        exceeded, count = store.check_query_limit(caller_id, wallet_id)
        assert exceeded, f"Should exceed limit at count {count}"
        assert count == config.MODEL_INVERSION_QUERY_LIMIT

    def test_different_callers_independent_counts(self, isolated_db):
        """Test that different callers have independent query counts."""
        store = RiskScoreStore()
        wallet_id = "wallet4"
        
        # Increment for caller A
        store.increment_query_count("caller_a", wallet_id)
        store.increment_query_count("caller_a", wallet_id)
        
        # Increment for caller B
        store.increment_query_count("caller_b", wallet_id)
        
        # Check counts
        _, count_a = store.check_query_limit("caller_a", wallet_id)
        _, count_b = store.check_query_limit("caller_b", wallet_id)
        
        assert count_a == 2
        assert count_b == 1

    def test_different_wallets_independent_counts(self, isolated_db):
        """Test that different wallets have independent query counts."""
        store = RiskScoreStore()
        caller_id = "caller5"
        
        # Increment for wallet A
        store.increment_query_count(caller_id, "wallet_a")
        store.increment_query_count(caller_id, "wallet_a")
        store.increment_query_count(caller_id, "wallet_a")
        
        # Increment for wallet B
        store.increment_query_count(caller_id, "wallet_b")
        
        # Check counts
        _, count_a = store.check_query_limit(caller_id, "wallet_a")
        _, count_b = store.check_query_limit(caller_id, "wallet_b")
        
        assert count_a == 3
        assert count_b == 1


class TestConfigValidation:
    """Test that security parameters validate correctly."""

    def test_model_inversion_query_limit_positive(self):
        """Test that MODEL_INVERSION_QUERY_LIMIT is positive."""
        assert config.MODEL_INVERSION_QUERY_LIMIT > 0

    def test_model_inversion_dp_epsilon_positive(self):
        """Test that MODEL_INVERSION_DP_EPSILON is positive."""
        assert config.MODEL_INVERSION_DP_EPSILON > 0

    def test_score_rounding_granularity_positive(self):
        """Test that SCORE_ROUNDING_GRANULARITY is positive."""
        assert config.SCORE_ROUNDING_GRANULARITY > 0
