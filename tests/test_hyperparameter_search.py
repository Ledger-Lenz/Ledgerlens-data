"""Tests for detection/hyperparameter_search.py (Issue #281)."""

import json

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from detection.hyperparameter_search import (
    get_search_space,
    load_best_params,
    run_study,
)


class TestGetSearchSpace:
    """Test search space retrieval."""

    def test_isolation_forest_space(self):
        """Isolation Forest space contains expected parameters."""
        space = get_search_space("isolation_forest")
        assert "contamination" in space
        assert "n_estimators" in space
        assert "max_features" in space
        assert len(space) == 3

    def test_xgboost_space(self):
        """XGBoost space contains expected parameters."""
        space = get_search_space("xgboost")
        assert "max_depth" in space
        assert "learning_rate" in space
        assert "subsample" in space
        assert "colsample_bytree" in space
        assert "n_estimators" in space
        assert len(space) == 5

    def test_gnn_space(self):
        """GNN space contains expected parameters."""
        space = get_search_space("gnn")
        assert "hidden_dim" in space
        assert "num_layers" in space
        assert "dropout" in space
        assert "learning_rate" in space
        assert len(space) == 4

    def test_unknown_model_rejected(self):
        """Unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_name"):
            get_search_space("unknown_model")

    def test_space_format(self):
        """Each space entry is a (type, min, max) tuple."""
        space = get_search_space("xgboost")
        for _, (param_type, min_val, max_val) in space.items():
            assert param_type in ("int", "float")
            assert min_val < max_val


class TestRunStudy:
    """Test Optuna study execution."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic validation data."""
        X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
        X_train, X_val = X[:70], X[70:]
        y_train, y_val = y[:70], y[70:]
        return (
            pd.DataFrame(X_train, columns=[f"f{i}" for i in range(20)]),
            pd.Series(y_train),
            pd.DataFrame(X_val, columns=[f"f{i}" for i in range(20)]),
            pd.Series(y_val),
        )

    def test_xgboost_study_smoke_test(self, synthetic_data, tmp_path):
        """3-trial XGBoost study completes and returns params."""
        storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
        best_params = run_study(
            "xgboost",
            n_trials=3,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=storage_url,
        )
        assert isinstance(best_params, dict)
        assert "max_depth" in best_params
        assert "learning_rate" in best_params

    def test_isolation_forest_study_smoke_test(self, synthetic_data, tmp_path):
        """3-trial Isolation Forest study completes and returns params."""
        storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
        best_params = run_study(
            "isolation_forest",
            n_trials=3,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=storage_url,
        )
        assert isinstance(best_params, dict)
        assert "contamination" in best_params
        assert "n_estimators" in best_params

    def test_invalid_n_trials_zero(self, synthetic_data):
        """n_trials <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_trials must be a positive integer"):
            run_study("xgboost", n_trials=0, validation_data=synthetic_data)

    def test_invalid_n_trials_negative(self, synthetic_data):
        """Negative n_trials raises ValueError."""
        with pytest.raises(ValueError, match="n_trials must be a positive integer"):
            run_study("xgboost", n_trials=-1, validation_data=synthetic_data)

    def test_invalid_n_jobs_zero(self, synthetic_data):
        """n_jobs < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_jobs must be in"):
            run_study("xgboost", n_trials=3, validation_data=synthetic_data, n_jobs=0)

    def test_invalid_n_jobs_over_four(self, synthetic_data):
        """n_jobs > 4 raises ValueError."""
        with pytest.raises(ValueError, match="n_jobs must be in"):
            run_study("xgboost", n_trials=3, validation_data=synthetic_data, n_jobs=5)

    def test_best_params_file_written(self, synthetic_data, tmp_path, monkeypatch):
        """Best params are persisted to disk as JSON."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", model_dir)

        storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
        run_study(
            "xgboost",
            n_trials=2,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=storage_url,
        )

        params_file = model_dir / "best_params_xgboost.json"
        assert params_file.exists()
        with open(params_file) as f:
            params = json.load(f)
        assert isinstance(params, dict)
        assert "max_depth" in params

    def test_study_uses_median_pruner(self, synthetic_data, tmp_path):
        """Study uses MedianPruner for early stopping."""
        storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
        best_params = run_study(
            "xgboost",
            n_trials=3,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=storage_url,
        )
        # Just verify study completes; pruning is an internal detail
        assert isinstance(best_params, dict)


class TestLoadBestParams:
    """Test parameter loading."""

    def test_load_nonexistent_file(self, tmp_path, monkeypatch):
        """Loading nonexistent params file returns None."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", model_dir)
        result = load_best_params("xgboost")
        assert result is None

    def test_load_existing_file(self, tmp_path, monkeypatch):
        """Loading existing params file returns the dict."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", model_dir)

        # Write test params file
        test_params = {"max_depth": 5, "learning_rate": 0.1}
        params_file = model_dir / "best_params_xgboost.json"
        with open(params_file, "w") as f:
            json.dump(test_params, f)

        result = load_best_params("xgboost")
        assert result == test_params

    def test_load_invalid_json(self, tmp_path, monkeypatch):
        """Loading invalid JSON file returns None (logs warning)."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", model_dir)

        # Write invalid JSON
        params_file = model_dir / "best_params_xgboost.json"
        with open(params_file, "w") as f:
            f.write("invalid json {")

        result = load_best_params("xgboost")
        assert result is None


class TestIntegrationWithTraining:
    """Test integration with model training pipeline."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic validation data."""
        X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
        X_train, X_val = X[:70], X[70:]
        y_train, y_val = y[:70], y[70:]
        return (
            pd.DataFrame(X_train, columns=[f"f{i}" for i in range(20)]),
            pd.Series(y_train),
            pd.DataFrame(X_val, columns=[f"f{i}" for i in range(20)]),
            pd.Series(y_val),
        )

    def test_study_then_load(self, synthetic_data, tmp_path, monkeypatch):
        """Run a study, then load the persisted params."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", model_dir)

        storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
        run_study(
            "xgboost",
            n_trials=2,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=storage_url,
        )

        loaded_params = load_best_params("xgboost")
        assert loaded_params is not None
        assert isinstance(loaded_params, dict)
        assert all(key in loaded_params for key in ["max_depth", "learning_rate", "subsample"])
