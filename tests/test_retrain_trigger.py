"""Tests for scripts/retrain_if_drifted.py.

Validates promotion gate logic, archive creation, exit codes, and the
end-to-end drift → retrain → promote flow via mocked dependencies.
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from scripts.retrain_if_drifted import (
    archive_current_models,
    should_promote,
)


@pytest.fixture
def temp_model_dir(tmp_path):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


@pytest.fixture
def sample_old_metrics():
    return {
        "random_forest": {"auc_roc": 0.90, "pr_auc": 0.85, "f1": 0.82},
        "xgboost": {"auc_roc": 0.92, "pr_auc": 0.88, "f1": 0.84},
        "lightgbm": {"auc_roc": 0.91, "pr_auc": 0.86, "f1": 0.83},
    }


@pytest.fixture
def sample_model_metadata():
    return {
        "trained_at": "2026-06-01T00:00:00Z",
        "feature_columns": ["feat_a", "feat_b"],
        "feature_schema_hash": "sha256:abc123",
        "model_names": ["random_forest", "xgboost", "lightgbm"],
        "feature_distributions": {
            "feat_a": {
                "bin_edges": [0.0, 1.0, 2.0, 3.0],
                "expected_proportions": [0.25, 0.25, 0.25, 0.25],
            },
            "feat_b": {
                "bin_edges": [0.0, 1.0, 2.0, 3.0],
                "expected_proportions": [0.25, 0.25, 0.25, 0.25],
            },
        },
    }


class TestArchiveCurrentModels:
    def test_archive_created_with_correct_permissions(self, temp_model_dir):
        model_files = ["random_forest.joblib", "model_metadata.json", "metrics.json"]
        for fname in model_files:
            with open(os.path.join(temp_model_dir, fname), "w") as f:
                f.write("test")

        archive_path = archive_current_models(temp_model_dir)
        assert os.path.exists(archive_path)
        assert oct(os.stat(archive_path).st_mode & 0o777) == "0o750"

        for fname in model_files:
            assert os.path.exists(os.path.join(archive_path, fname))

    def test_archive_created_before_promotion(self, temp_model_dir):
        model_files = ["random_forest.joblib", "model_metadata.json", "metrics.json"]
        for fname in model_files:
            with open(os.path.join(temp_model_dir, fname), "w") as f:
                f.write("test")

        archive_path = archive_current_models(temp_model_dir)
        assert os.path.isdir(archive_path)

        for fname in model_files:
            assert os.path.exists(os.path.join(temp_model_dir, fname))
            assert os.path.exists(os.path.join(archive_path, fname))


class TestShouldPromote:
    def test_promotion_gate_allows_identical_metrics(self, sample_old_metrics):
        promote, reason = should_promote(sample_old_metrics, sample_old_metrics)
        assert promote is True

    def test_promotion_gate_allows_improvement(self, sample_old_metrics):
        new_metrics = {
            name: {k: v * 1.05 if k in ("auc_roc", "f1") else v for k, v in m.items()}
            for name, m in sample_old_metrics.items()
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is True

    def test_promotion_gate_allows_minor_regression(self, sample_old_metrics):
        new_metrics = {
            name: {k: (v - 0.005 if k in ("auc_roc", "f1") else v) for k, v in m.items()}
            for name, m in sample_old_metrics.items()
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is True

    def test_promotion_gate_blocks_regression(self, sample_old_metrics):
        new_metrics = {
            name: {k: (v - 0.02 if k in ("auc_roc", "f1") else v) for k, v in m.items()}
            for name, m in sample_old_metrics.items()
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is False
        assert "AUC-ROC" in reason or "F1" in reason

    def test_promotion_gate_blocks_single_model_regression(self, sample_old_metrics):
        new_metrics = dict(sample_old_metrics)
        new_metrics["xgboost"] = {
            "auc_roc": 0.80,
            "pr_auc": 0.88,
            "f1": 0.84,
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is False
        assert "xgboost" in reason

    def test_promotion_blocks_missing_model(self, sample_old_metrics):
        new_metrics = {
            "random_forest": sample_old_metrics["random_forest"],
            "xgboost": sample_old_metrics["xgboost"],
        }
        promote, reason = should_promote(sample_old_metrics, new_metrics)
        assert promote is False
        assert "lightgbm" in reason


class TestRetrainScriptExitCodes:
    def _run_retrain_main(self, argv: list[str]) -> int:
        from scripts.retrain_if_drifted import main

        return main(argv)

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    def test_exit_code_0_no_drift(self, mock_load_metadata, mock_get_feature_data, temp_model_dir):
        """No drift → exit code 0."""
        mock_load_metadata.return_value = {
            "feature_distributions": {
                "feat_a": {
                    "bin_edges": [0.0, 0.5, 1.0],
                    "expected_proportions": [0.5, 0.5],
                },
            }
        }

        rng = np.random.default_rng(42)
        mock_get_feature_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 1, 500),
            }
        )

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
            ]
        )
        assert code == 0

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    @patch("scripts.retrain_if_drifted.load_training_data")
    @patch("scripts.retrain_if_drifted.train_models")
    @patch("scripts.retrain_if_drifted.load_metrics")
    def test_exit_code_2_retrained_and_promoted(
        self,
        mock_load_metrics,
        mock_train_models,
        mock_load_training_data,
        mock_load_metadata,
        mock_get_feature_data,
        temp_model_dir,
    ):
        """Drift detected, retrained, promoted → exit code 2."""
        mock_load_metadata.return_value = {
            "feature_distributions": {
                "feat_a": {
                    "bin_edges": [0.0, 0.5, 1.0],
                    "expected_proportions": [0.5, 0.5],
                },
            }
        }

        rng = np.random.default_rng(42)

        # Shifted distribution → triggers drift
        mock_get_feature_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(10, 20, 500),
            }
        )

        mock_load_training_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 1, 100),
                "label": [1, 0] * 50,
            }
        )

        dummy = DummyClassifier(strategy="constant", constant=1)
        dummy.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))

        mock_results = {
            name: {"model": dummy, "metrics": {"auc_roc": 0.95, "pr_auc": 0.90, "f1": 0.88}}
            for name in ["random_forest", "xgboost", "lightgbm"]
        }
        mock_train_models.return_value = {
            "results": mock_results,
            "feature_columns": ["feat_a"],
            "feature_distributions": {},
            "n_train": 80,
            "n_test": 20,
        }

        mock_load_metrics.side_effect = [
            {
                "random_forest": {"auc_roc": 0.90, "pr_auc": 0.85, "f1": 0.82},
                "xgboost": {"auc_roc": 0.92, "pr_auc": 0.88, "f1": 0.84},
                "lightgbm": {"auc_roc": 0.91, "pr_auc": 0.86, "f1": 0.83},
            },
            {
                "random_forest": {"auc_roc": 0.95, "pr_auc": 0.90, "f1": 0.88},
                "xgboost": {"auc_roc": 0.95, "pr_auc": 0.90, "f1": 0.88},
                "lightgbm": {"auc_roc": 0.95, "pr_auc": 0.90, "f1": 0.88},
            },
        ]

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
                "--retrain-data-path",
                "/fake/path.parquet",
            ]
        )
        assert code == 2

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    @patch("scripts.retrain_if_drifted.load_training_data")
    @patch("scripts.retrain_if_drifted.train_models")
    @patch("scripts.retrain_if_drifted.load_metrics")
    def test_exit_code_3_retrained_not_promoted(
        self,
        mock_load_metrics,
        mock_train_models,
        mock_load_training_data,
        mock_load_metadata,
        mock_get_feature_data,
        temp_model_dir,
    ):
        """Drift detected, retrained, NOT promoted (regression) → exit code 3."""
        mock_load_metadata.return_value = {
            "feature_distributions": {
                "feat_a": {
                    "bin_edges": [0.0, 0.5, 1.0],
                    "expected_proportions": [0.5, 0.5],
                },
            }
        }

        rng = np.random.default_rng(42)

        mock_get_feature_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(10, 20, 500),
            }
        )

        mock_load_training_data.return_value = pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 1, 100),
                "label": [1, 0] * 50,
            }
        )

        dummy = DummyClassifier(strategy="constant", constant=1)
        dummy.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))

        mock_results = {
            name: {"model": dummy, "metrics": {"auc_roc": 0.85, "pr_auc": 0.80, "f1": 0.78}}
            for name in ["random_forest", "xgboost", "lightgbm"]
        }
        mock_train_models.return_value = {
            "results": mock_results,
            "feature_columns": ["feat_a"],
            "feature_distributions": {},
            "n_train": 80,
            "n_test": 20,
        }

        mock_load_metrics.side_effect = [
            {
                "random_forest": {"auc_roc": 0.90, "pr_auc": 0.85, "f1": 0.82},
                "xgboost": {"auc_roc": 0.92, "pr_auc": 0.88, "f1": 0.84},
                "lightgbm": {"auc_roc": 0.91, "pr_auc": 0.86, "f1": 0.83},
            },
            {
                "random_forest": {"auc_roc": 0.85, "pr_auc": 0.80, "f1": 0.78},
                "xgboost": {"auc_roc": 0.85, "pr_auc": 0.80, "f1": 0.78},
                "lightgbm": {"auc_roc": 0.85, "pr_auc": 0.80, "f1": 0.78},
            },
        ]

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
                "--retrain-data-path",
                "/fake/path.parquet",
            ]
        )
        assert code == 3

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    def test_exit_code_1_missing_metadata(
        self, mock_load_metadata, mock_get_feature_data, temp_model_dir
    ):
        """Missing model_metadata.json → exit code 1."""
        mock_load_metadata.return_value = None

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
            ]
        )
        assert code == 1

    @patch("scripts.retrain_if_drifted.get_feature_data")
    @patch("scripts.retrain_if_drifted.load_model_metadata")
    def test_exit_code_1_missing_distributions(
        self, mock_load_metadata, mock_get_feature_data, temp_model_dir
    ):
        """model_metadata.json without feature_distributions → exit code 1."""
        mock_load_metadata.return_value = {"trained_at": "2026-01-01"}

        code = self._run_retrain_main(
            [
                "--lookback-days",
                "7",
                "--model-dir",
                temp_model_dir,
            ]
        )
        assert code == 1


class TestRetrainEndToEnd:
    def test_archive_created_during_retrain(
        self, tmp_path, sample_old_metrics, sample_model_metadata
    ):
        """Archive directory is populated during a triggered retrain."""
        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(sample_model_metadata, f)
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(sample_old_metrics, f)
        for name in ["random_forest.joblib", "xgboost.joblib", "lightgbm.joblib"]:
            with open(os.path.join(model_dir, name), "w") as f:
                f.write("test")

        archive_path = archive_current_models(model_dir)
        assert os.path.isdir(archive_path)
        for name in ["random_forest.joblib", "model_metadata.json", "metrics.json"]:
            assert os.path.exists(os.path.join(archive_path, name))
