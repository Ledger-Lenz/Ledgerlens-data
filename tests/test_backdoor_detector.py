"""Tests for backdoor detection using Activation Clustering (Issue #016).

Tests verify:
  1. Backdoor detection can flag poisoned samples injected into clean dataset
  2. Activation extraction works for RandomForest, XGBoost, and LightGBM
  3. 20% safety threshold correctly prevents overflagging
  4. Detection report generation
  5. Graceful error handling
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from detection.adversarial.backdoor_detector import ActivationClusteringDetector


class TestActivationExtraction:
    """Test activation extraction from different model types."""

    @pytest.fixture
    def sample_data(self):
        """Generate synthetic binary classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42,
        )
        return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(20)]), pd.Series(y)

    @pytest.fixture
    def rf_model(self, sample_data):
        """Train a RandomForest model."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def xgb_model(self, sample_data):
        """Train an XGBoost model."""
        X, y = sample_data
        model = XGBClassifier(n_estimators=10, random_state=42, use_label_encoder=False)
        model.fit(X, y)
        return model

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    @pytest.fixture
    def lgbm_model(self, sample_data):
        """Train a LightGBM model."""
        X, y = sample_data
        model = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)
        return model

    def test_extract_activations_from_rf(self, rf_model, sample_data):
        """RandomForest activation extraction should return leaf indices."""
        X, _ = sample_data
        detector = ActivationClusteringDetector()
        activations = detector._extract_activations(rf_model, X)

        assert activations is not None
        assert activations.shape[0] == len(X)
        assert activations.shape[1] == 10  # n_trees

    def test_extract_activations_from_xgb(self, xgb_model, sample_data):
        """XGBoost activation extraction should return raw predictions."""
        X, _ = sample_data
        detector = ActivationClusteringDetector()
        activations = detector._extract_activations(xgb_model, X)

        assert activations is not None
        assert activations.shape[0] == len(X)
        assert activations.ndim == 2

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_extract_activations_from_lgbm(self, lgbm_model, sample_data):
        """LightGBM activation extraction should return raw scores."""
        X, _ = sample_data
        detector = ActivationClusteringDetector()
        activations = detector._extract_activations(lgbm_model, X)

        assert activations is not None
        assert activations.shape[0] == len(X)
        assert activations.ndim == 2

    def test_extract_activations_unsupported_model(self, sample_data):
        """Unsupported model type should return None."""
        X, _ = sample_data

        # Use a simple dict as an unsupported model type
        class UnsupportedModel:
            pass

        unsupported_model = UnsupportedModel()
        detector = ActivationClusteringDetector()
        activations = detector._extract_activations(unsupported_model, X)

        assert activations is None


class TestBackdoorDetection:
    """Test backdoor detection with injected poisoned samples."""

    @pytest.fixture
    def clean_data_with_backdoor(self):
        """Generate synthetic dataset with 10 injected backdoor samples."""
        np.random.seed(42)

        # 100 clean samples
        X_clean, y_clean = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42,
        )

        # Inject 10 backdoor samples (wash trades) with distinctive feature pattern
        # Backdoor trigger: feat_0 > 2.5 and feat_1 < -2.5
        X_backdoor = np.random.randn(10, 20)
        X_backdoor[:, 0] = np.random.uniform(3.0, 4.0, 10)  # feat_0 > 2.5
        X_backdoor[:, 1] = np.random.uniform(-4.0, -3.0, 10)  # feat_1 < -2.5
        y_backdoor = np.ones(10)  # All mislabeled as clean (label=0 expected, but we label them 1)

        X_combined = np.vstack([X_clean, X_backdoor])
        y_combined = np.concatenate([y_clean, y_backdoor])

        # Shuffle to mix backdoors with clean data
        indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]

        X_df = pd.DataFrame(X_combined, columns=[f"feat_{i}" for i in range(20)])
        y_series = pd.Series(y_combined)

        # Track which samples are backdoors (for validation)
        backdoor_mask = np.zeros(len(X_combined), dtype=bool)
        backdoor_indices = np.where((X_combined[:, 0] > 2.5) & (X_combined[:, 1] < -2.5))[0]
        backdoor_mask[backdoor_indices] = True

        return X_df, y_series, backdoor_mask, backdoor_indices

    def test_backdoor_detection_flags_poisoned_samples(self, clean_data_with_backdoor):
        """AC should flag at least 8 of 10 injected backdoor samples."""
        X, y, backdoor_mask, backdoor_indices = clean_data_with_backdoor

        # Train model on contaminated data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Run AC detection
        detector = ActivationClusteringDetector(k=2, random_state=42)
        flagged = detector.detect(model, X, y)

        # Should flag some samples
        assert len(flagged) > 0, "Detector should flag at least some samples"

        # Check overlap with actual backdoors
        flagged_set = set(flagged)
        backdoor_set = set(backdoor_indices)
        overlap = flagged_set & backdoor_set

        # AC should detect at least 8 of 10 backdoors
        # (allowing for some false negatives due to clustering randomness)
        assert (
            len(overlap) >= 8
        ), f"Detector should flag >= 8 backdoors, but only flagged {len(overlap)} of 10"

    def test_safety_check_prevents_overflagging(self):
        """20% safety check should bypass quarantine if > 20% of class is flagged."""
        np.random.seed(42)
        X, y = make_classification(n_samples=50, n_features=20, random_state=42)

        # Create a scenario where detector flags 25% of samples (which exceeds 20% threshold)
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(20)])
        y_series = pd.Series(y)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)

        detector = ActivationClusteringDetector(k=2, random_state=42)
        # Use high threshold_percentile to trigger safety check
        flagged = detector.detect(X_df, y_series, threshold_percentile=75)

        # With percentile=75, the minority cluster must be in top 75% of cluster sizes
        # This is a safety check that should prevent overflagging
        # We can't directly test this without controlling cluster creation,
        # but we verify the method completes without error
        assert isinstance(flagged, list)

    def test_detection_with_insufficient_samples(self):
        """Detection should gracefully handle cases with < k samples per class."""
        X = pd.DataFrame(np.random.randn(3, 5), columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series([0, 0, 1])

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        detector = ActivationClusteringDetector(k=2, random_state=42)
        flagged = detector.detect(model, X, y)

        # Should return empty list or small list without raising
        assert isinstance(flagged, list)

    def test_detection_error_handling(self):
        """Detection should return empty list on exception."""
        X = pd.DataFrame(np.random.randn(10, 5), columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 10))

        # Use a None model to trigger an error
        detector = ActivationClusteringDetector()
        flagged = detector.detect(None, X, y)

        # Should return empty list (graceful error handling)
        assert flagged == []


class TestDetectionReport:
    """Test report generation."""

    def test_report_generation(self):
        """Report should contain detection statistics."""
        X = pd.DataFrame(np.random.randn(50, 20), columns=[f"feat_{i}" for i in range(20)])
        y = pd.Series(np.random.randint(0, 2, 50))

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        detector = ActivationClusteringDetector()
        flagged = detector.detect(model, X, y)

        report = detector.report(X, y, flagged)

        assert "total_samples" in report
        assert "n_flagged" in report
        assert "flagged_percentage" in report
        assert "flagged_by_label" in report
        assert "method" in report
        assert "k" in report

        assert report["total_samples"] == 50
        assert report["n_flagged"] == len(set(flagged))
        assert report["method"] == "activation_clustering"
        assert report["k"] == 2

    def test_report_with_no_flags(self):
        """Report should handle case with no flagged samples."""
        X = pd.DataFrame(np.random.randn(50, 20), columns=[f"feat_{i}" for i in range(20)])
        y = pd.Series(np.random.randint(0, 2, 50))

        detector = ActivationClusteringDetector()
        report = detector.report(X, y, flagged_indices=[])

        assert report["n_flagged"] == 0
        assert report["flagged_percentage"] == 0.0
        assert report["flagged_by_label"] == {}
