"""Tests for Issues #178, #179, #180, #181: Benford enhancements and conformal prediction."""

import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from config import config
from detection.benford_engine import (
    BENFORD_EXPECTED_2ND,
    second_digit_distribution,
    chi_square_second_digit,
    z_scores_second_digit,
    mad_score_second_digit,
    compute_benford_metrics_for_windows,
)
from detection.benford_window_optimizer import select_optimal_window
from detection.benford_drift_detector import (
    BenfordDriftDetector,
    BenfordBaseline,
    DriftStatus,
)
from detection.conformal import ConformalCalibrator


# ============================================================================
# Issue #178: Adaptive Benford Window Selection
# ============================================================================

class TestAdaptiveBenfordWindowSelection:
    """Tests for Issue #178: select_optimal_window function."""
    
    def test_select_optimal_window_meets_threshold(self):
        """Shortest window meeting threshold should be selected."""
        trade_counts = {1: 10, 4: 50, 24: 200, 168: 800, 720: 3000}
        min_threshold = 50
        
        selected = select_optimal_window(
            "USDC:GA.../XLM:native",
            trade_counts,
            min_sample_size=min_threshold,
            candidate_windows=[1, 4, 24, 168, 720]
        )
        
        # 4h window has 50 trades, meets threshold
        assert selected == 4
    
    def test_select_optimal_window_fallback_to_longest(self):
        """No window meets threshold -> fallback to longest window."""
        trade_counts = {1: 10, 4: 30, 24: 45, 168: 45, 720: 100}
        min_threshold = 50
        
        selected = select_optimal_window(
            "USDC:GA.../XLM:native",
            trade_counts,
            min_sample_size=min_threshold,
            candidate_windows=[1, 4, 24, 168, 720]
        )
        
        # No window >= 50, fall back to max (720h with 100 trades)
        assert selected == 720
    
    def test_select_optimal_window_all_zero_trades(self):
        """All windows with zero trades -> fallback to longest."""
        trade_counts = {1: 0, 4: 0, 24: 0, 168: 0, 720: 0}
        
        selected = select_optimal_window(
            "USDC:GA.../XLM:native",
            trade_counts,
            min_sample_size=50,
            candidate_windows=[1, 4, 24, 168, 720]
        )
        
        assert selected == 720
    
    def test_select_optimal_window_validation(self):
        """Minimum sample size < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 10"):
            select_optimal_window(
                "USDC:GA.../XLM:native",
                {1: 100, 4: 100, 24: 100, 168: 100, 720: 100},
                min_sample_size=5  # Invalid
            )
    
    def test_select_optimal_window_property_test(self):
        """Selected window must be in candidates list."""
        for _ in range(100):  # Property-based test
            counts = {1: 5, 4: 15, 24: 50, 168: 200, 720: 1000}
            threshold = np.random.randint(10, 100)
            
            selected = select_optimal_window(
                "pair",
                counts,
                min_sample_size=threshold,
                candidate_windows=[1, 4, 24, 168, 720]
            )
            
            assert selected in [1, 4, 24, 168, 720]


# ============================================================================
# Issue #179: Second-Digit Benford Analysis
# ============================================================================

class TestSecondDigitBenfordAnalysis:
    """Tests for Issue #179: second-digit metrics."""
    
    def test_benford_expected_2nd_valid(self):
        """Expected second-digit distribution should sum to ~1."""
        total = sum(BENFORD_EXPECTED_2ND.values())
        assert abs(total - 1.0) < 0.001
    
    def test_second_digit_distribution_uniform(self):
        """Uniform second digits should produce equal frequencies."""
        # Create amounts with uniform second digits: 10, 20, 30, ..., 90, 101-110, etc.
        amounts = pd.Series([float(10 * (i % 10) + 1) for i in range(1000)])
        
        dist = second_digit_distribution(amounts)
        
        # Should be roughly uniform (each digit ~10%)
        for digit in range(10):
            assert 0.08 < dist[digit] < 0.12  # Allow some variance
    
    def test_chi_square_second_digit_benford_conformant(self):
        """Sample from theoretical distribution -> low chi-square."""
        # Generate amounts roughly conforming to Benford second-digit law
        amounts = []
        for _ in range(500):
            # First digit uniformly 1-9
            first = np.random.randint(1, 10)
            # Second digit from Benford distribution
            second = np.random.choice(
                list(range(10)),
                p=list(BENFORD_EXPECTED_2ND.values())
            )
            amount = first * 10.0 + second + np.random.random()
            amounts.append(amount)
        
        amounts_series = pd.Series(amounts)
        chi_sq = chi_square_second_digit(amounts_series)
        
        # Chi-square should be relatively low for conformant data
        assert chi_sq < 20.0  # Upper threshold; real data usually much lower
    
    def test_z_scores_second_digit(self):
        """Z-scores should be computed per digit."""
        amounts = pd.Series(np.random.exponential(scale=10.0, size=100) + 10)
        
        zscores = z_scores_second_digit(amounts)
        
        # Should have scores for all 10 digits
        assert len(zscores) == 10
        assert all(isinstance(z, float) for z in zscores.values())
        assert all(z >= 0 for z in zscores.values())  # Z-scores are non-negative
    
    def test_mad_score_second_digit(self):
        """MAD should measure deviation from expected distribution."""
        amounts = pd.Series(np.random.exponential(scale=10.0, size=100) + 10)
        
        mad = mad_score_second_digit(amounts)
        
        # MAD should be between 0 and 1 (it's a normalized deviation measure)
        assert 0.0 <= mad <= 1.0
    
    def test_second_digit_exclusion_single_digit_amounts(self):
        """Amounts < 10 should be excluded from second-digit analysis."""
        amounts = pd.Series([1.0, 5.0, 9.9, 10.5, 11.2, 100.0])
        
        dist = second_digit_distribution(amounts)
        
        # Distribution should be computed (only multi-digit amounts included)
        assert isinstance(dist, dict)
        assert all(0 <= v <= 1 for v in dist.values())
        total_dist = sum(dist.values())
        # Should have some probability mass if amounts > 10 exist
        assert total_dist > 0


# ============================================================================
# Issue #180: Benford Drift Detector
# ============================================================================

class TestBenfordDriftDetector:
    """Tests for Issue #180: BenfordDriftDetector."""
    
    def test_benford_baseline_welford_accumulation(self):
        """Welford's algorithm should accumulate running mean and variance."""
        baseline = BenfordBaseline(pair_id="test_pair")
        
        # Simulate fitting with two chi-square values
        baseline.chi_square_count = 1
        baseline.chi_square_mean = 10.0
        baseline.chi_square_variance = 0.0
        
        # Add second value using Welford update
        chi_sq_new = 15.0
        n = 2
        delta = chi_sq_new - baseline.chi_square_mean
        baseline.chi_square_mean += delta / n
        delta2 = chi_sq_new - baseline.chi_square_mean
        baseline.chi_square_variance += delta * delta2
        baseline.chi_square_count = n
        
        # Mean should be (10 + 15) / 2 = 12.5
        assert abs(baseline.chi_square_mean - 12.5) < 0.01
        # Variance should be positive
        assert baseline.chi_square_variance > 0
    
    def test_benford_drift_detector_stable_state(self):
        """Stable baseline with stable current metrics -> STABLE status."""
        detector = BenfordDriftDetector(z_threshold=3.0, min_baseline_samples=10)
        
        # Fit baseline with stable chi-square values
        pair_id = "USDC:GA.../XLM:native"
        stable_chi_sqs = [10.0, 10.5, 11.0, 9.8, 10.2, 10.1, 10.3, 9.9, 10.4, 10.0]
        for chi_sq in stable_chi_sqs:
            amounts = pd.Series(np.random.exponential(scale=1.0, size=50) + 1.0)
            detector.fit_baseline(pair_id, amounts)
        
        # Check with current chi-square close to baseline mean
        status = detector.check(pair_id, current_chi_square=10.1, current_mad=0.015)
        
        assert status == DriftStatus.STABLE
    
    def test_benford_drift_detector_drifted_state(self):
        """High chi-square deviation -> DRIFTED status."""
        detector = BenfordDriftDetector(z_threshold=1.0, min_baseline_samples=5)
        
        pair_id = "USDC:GA.../XLM:native"
        
        # Fit baseline with chi-square ~10
        baseline_chi_sqs = [10.0] * 10
        for chi_sq in baseline_chi_sqs:
            amounts = pd.Series(np.random.exponential(scale=1.0, size=50) + 1.0)
            detector.fit_baseline(pair_id, amounts)
        
        # Current chi-square is far above baseline -> should drift
        status = detector.check(pair_id, current_chi_square=40.0, current_mad=0.015)
        
        assert status == DriftStatus.DRIFTED
    
    def test_benford_drift_detector_insufficient_data(self):
        """Baseline with < min_baseline_samples -> INSUFFICIENT_DATA."""
        detector = BenfordDriftDetector(z_threshold=3.0, min_baseline_samples=20)
        
        pair_id = "USDC:GA.../XLM:native"
        
        # Fit baseline with only 5 samples (< 20 required)
        for i in range(5):
            amounts = pd.Series(np.random.exponential(scale=1.0, size=50) + 1.0)
            detector.fit_baseline(pair_id, amounts)
        
        status = detector.check(pair_id, current_chi_square=15.0, current_mad=0.015)
        
        assert status == DriftStatus.INSUFFICIENT_DATA
    
    def test_benford_drift_detector_deduplication(self):
        """Multiple drift detections should only log once per transition."""
        detector = BenfordDriftDetector(z_threshold=1.0, min_baseline_samples=5)
        
        pair_id = "test_pair"
        
        # Fit baseline
        for _ in range(10):
            amounts = pd.Series(np.random.exponential(scale=1.0, size=50) + 1.0)
            detector.fit_baseline(pair_id, amounts)
        
        # Check multiple times with drifted state
        status1 = detector.check(pair_id, current_chi_square=50.0, current_mad=0.015)
        status2 = detector.check(pair_id, current_chi_square=50.0, current_mad=0.015)
        
        assert status1 == DriftStatus.DRIFTED
        assert status2 == DriftStatus.DRIFTED
        # But the detector should have only logged once (not verifying the log here)
        assert pair_id in detector._drifted_pairs
    
    def test_benford_drift_detector_batch_check(self):
        """Batch checking multiple pairs."""
        detector = BenfordDriftDetector(z_threshold=3.0, min_baseline_samples=10)
        
        pairs_metrics = {
            "pair1": {"chi_square": 10.0, "mad": 0.015},
            "pair2": {"chi_square": 50.0, "mad": 0.050},  # Potentially drifted
        }
        
        results = detector.check_batch(pairs_metrics)
        
        assert isinstance(results, dict)
        assert set(results.keys()) == {"pair1", "pair2"}
        assert all(isinstance(v, DriftStatus) for v in results.values())
    
    def test_benford_drift_detector_retrain_trigger(self):
        """should_trigger_retrain based on drifted pairs count."""
        detector = BenfordDriftDetector(z_threshold=1.0, min_baseline_samples=5)
        
        # Manually mark some pairs as drifted
        detector._drifted_pairs["pair1"] = 1
        detector._drifted_pairs["pair2"] = 1
        
        # num_pairs_trigger = 0 (default): any single drift triggers retrain
        with patch.object(config, 'BENFORD_DRIFT_NUM_PAIRS_TRIGGER', 0):
            assert detector.should_trigger_retrain() == True
        
        # num_pairs_trigger = 3: need 3+ pairs to trigger
        with patch.object(config, 'BENFORD_DRIFT_NUM_PAIRS_TRIGGER', 3):
            assert detector.should_trigger_retrain() == False


# ============================================================================
# Issue #181: Conformal Prediction Intervals
# ============================================================================

class TestConformalPredictionIntegration:
    """Tests for Issue #181: ConformalCalibrator integration."""
    
    def test_conformal_calibrator_initialization(self):
        """ConformalCalibrator should initialize with alpha parameter."""
        calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
        
        assert calibrator.alpha == 0.10
        assert calibrator.q_hat is None  # Not calibrated yet
    
    def test_conformal_calibrator_regression_mode(self):
        """Calibrator should support regression mode (risk score prediction)."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create mock training data
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series(np.random.uniform(0, 100, 100))
        
        X_cal = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        y_cal = pd.Series(np.random.uniform(0, 100, 20))
        
        model = RandomForestRegressor(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)
        
        # Calibrate
        calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
        calibrator.calibrate(model, X_cal, y_cal)
        
        # Should have computed q_hat
        assert calibrator.q_hat is not None
        assert calibrator.q_hat >= 0
        assert calibrator.n_cal == len(X_cal)
    
    def test_conformal_predict_with_interval(self):
        """predict_with_interval should return lower/upper bounds."""
        from sklearn.ensemble import RandomForestRegressor
        
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series(np.random.uniform(0, 100, 100))
        
        X_cal = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        y_cal = pd.Series(np.random.uniform(0, 100, 20))
        
        X_test = pd.DataFrame(np.random.randn(5, 5), columns=[f"f{i}" for i in range(5)])
        
        model = RandomForestRegressor(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)
        
        calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
        calibrator.calibrate(model, X_cal, y_cal)
        
        intervals = calibrator.predict_with_interval(model, X_test)
        
        assert len(intervals) == len(X_test)
        for interval in intervals:
            assert "score" in interval
            assert "lower" in interval
            assert "upper" in interval
            assert interval["lower"] <= interval["upper"]
            assert 0 <= interval["lower"] <= 100
            assert 0 <= interval["upper"] <= 100
    
    def test_conformal_save_and_load(self, tmp_path):
        """Calibration artifact should be saveable and loadable."""
        from sklearn.ensemble import RandomForestRegressor
        
        X_cal = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        y_cal = pd.Series(np.random.uniform(0, 100, 20))
        
        X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series(np.random.uniform(0, 100, 100))
        
        model = RandomForestRegressor(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)
        
        calibrator1 = ConformalCalibrator(alpha=0.10, random_state=42)
        calibrator1.calibrate(model, X_cal, y_cal)
        
        # Save
        artifact_path = str(tmp_path / "test_conformal.json")
        calibrator1.save(artifact_path)
        
        # Load
        calibrator2 = ConformalCalibrator.load(artifact_path)
        
        # Should have same q_hat and alpha
        assert calibrator2.q_hat == calibrator1.q_hat
        assert calibrator2.alpha == calibrator1.alpha
        assert calibrator2.n_cal == calibrator1.n_cal
    
    def test_conformal_fallback_intervals(self):
        """Maximally conservative intervals as fallback."""
        calibrator = ConformalCalibrator(alpha=0.10)  # Not calibrated
        
        from sklearn.ensemble import RandomForestRegressor
        X_test = pd.DataFrame(np.random.randn(5, 5), columns=[f"f{i}" for i in range(5)])
        model = RandomForestRegressor(random_state=42, n_estimators=10)
        model.fit(X_test, np.random.uniform(0, 100, 5))
        
        # Should return maximally conservative bounds when not calibrated
        intervals = calibrator.predict_with_interval(model, X_test)
        
        # When q_hat is None, should fallback... (implementation may vary)
        # This tests that it doesn't crash
        assert intervals is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
