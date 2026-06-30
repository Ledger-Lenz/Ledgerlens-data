"""Tests for monitoring/drift_detector.py — MMD-based covariance shift detection."""

import numpy as np
import pytest

from monitoring.drift_detector import CovarianceShiftDetector


RNG = np.random.default_rng(42)


@pytest.fixture
def detector():
    return CovarianceShiftDetector(threshold=0.05)


def test_identical_windows_mmd_near_zero(detector):
    """Identical reference and test windows must produce MMD near zero."""
    X = RNG.normal(0, 1, (200, 3))
    report = detector.detect(X, X, feature_names=["a", "b", "c"])
    for val in report.mmd_per_feature.values():
        assert abs(val) < 0.05
    assert not report.drift_detected


def test_shifted_gaussian_triggers_drift(detector):
    """Test window drawn from a shifted Gaussian must produce MMD above threshold."""
    ref = RNG.normal(0, 1, (200, 2))
    cur = RNG.normal(5, 1, (200, 2))
    report = detector.detect(ref, cur, feature_names=["x", "y"])
    for val in report.mmd_per_feature.values():
        assert val > detector.threshold
    assert report.drift_detected


def test_near_zero_variance_feature_excluded(detector):
    """Near-zero-variance feature in reference window must be excluded from report."""
    ref = np.hstack([np.ones((100, 1)), RNG.normal(0, 1, (100, 1))])
    cur = np.hstack([np.ones((100, 1)), RNG.normal(0, 1, (100, 1))])
    report = detector.detect(ref, cur, feature_names=["constant", "normal"])
    assert "constant" not in report.mmd_per_feature
    assert "normal" in report.mmd_per_feature
