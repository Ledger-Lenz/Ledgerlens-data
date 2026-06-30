"""Tests for detection/hyperparameter_search.py (Issue #213).

Coverage:
  - Search space correctness for all three ensemble models
  - Bounds validation (security requirement)
  - Single-objective study: completion, params written, non-decreasing AUC
  - Multi-objective study: Pareto front returned, select_pareto_point
  - Unified best_hyperparams.json round-trip
  - load_best_params / load_best_hyperparams edge cases
  - Integration: optimised params produce a trainable model
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from detection.hyperparameter_search import (
    HyperparameterSearchError,
    get_search_space,
    load_best_hyperparams,
    load_best_params,
    run_multiobjective_study,
    run_study,
    save_unified_best_hyperparams,
    select_pareto_point,
    validate_hyperparams,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_data():
    """Small synthetic binary-classification dataset used across all tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return (
        pd.DataFrame(X_train, columns=[f"f{i}" for i in range(20)]),
        pd.Series(y_train, name="label"),
        pd.DataFrame(X_val, columns=[f"f{i}" for i in range(20)]),
        pd.Series(y_val, name="label"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Search space tests
# ──────────────────────────────────────────────────────────────────────────────


class TestGetSearchSpace:
    def test_random_forest_keys(self):
        space = get_search_space("random_forest")
        assert "n_estimators" in space
        assert "max_depth" in space
        assert "min_samples_split" in space
        assert "min_samples_leaf" in space
        assert "max_features" in space

    def test_xgboost_keys(self):
        space = get_search_space("xgboost")
        assert "n_estimators" in space
        assert "max_depth" in space
        assert "learning_rate" in space
        assert "subsample" in space
        assert "colsample_bytree" in space

    def test_lightgbm_keys(self):
        space = get_search_space("lightgbm")
        assert "n_estimators" in space
        assert "max_depth" in space
        assert "learning_rate" in space
        assert "subsample" in space
        assert "colsample_bytree" in space
        assert "num_leaves" in space  # LightGBM-specific

    def test_each_entry_is_triple(self):
        for model in ("random_forest", "xgboost", "lightgbm"):
            for name, spec in get_search_space(model).items():
                assert len(spec) == 3, f"{model}.{name} spec should be (type, min, max)"
                param_type, lo, hi = spec
                assert param_type in ("int", "float", "float_log"), (
                    f"Unknown type {param_type!r} in {model}.{name}"
                )
                assert lo < hi, f"min >= max for {model}.{name}"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model_name"):
            get_search_space("not_a_model")

    # Backwards-compatibility: old models should still be retrievable
    def test_legacy_isolation_forest(self):
        space = get_search_space("isolation_forest")
        assert "contamination" in space

    def test_legacy_gnn(self):
        space = get_search_space("gnn")
        assert "hidden_dim" in space


# ──────────────────────────────────────────────────────────────────────────────
# Bounds validation (security)
# ──────────────────────────────────────────────────────────────────────────────


class TestValidateHyperparams:
    def test_valid_rf_params_pass(self):
        validate_hyperparams(
            "random_forest",
            {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
             "min_samples_leaf": 2, "max_features": 0.8},
        )  # should not raise

    def test_valid_xgboost_params_pass(self):
        validate_hyperparams(
            "xgboost",
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "min_child_weight": 3, "gamma": 1.0},
        )

    def test_valid_lightgbm_params_pass(self):
        validate_hyperparams(
            "lightgbm",
            {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.05,
             "subsample": 0.9, "colsample_bytree": 0.7,
             "num_leaves": 63, "min_child_samples": 20},
        )

    def test_max_depth_zero_rejected(self):
        """max_depth=0 would cause a silent failure in RF — must be rejected."""
        with pytest.raises(ValueError, match="max_depth"):
            validate_hyperparams("random_forest", {"n_estimators": 100, "max_depth": 0,
                                                    "min_samples_split": 2, "min_samples_leaf": 1,
                                                    "max_features": 0.5})

    def test_negative_learning_rate_rejected(self):
        with pytest.raises(ValueError, match="learning_rate"):
            validate_hyperparams("xgboost", {"n_estimators": 100, "max_depth": 3,
                                              "learning_rate": -0.1, "subsample": 0.8,
                                              "colsample_bytree": 0.8,
                                              "min_child_weight": 1, "gamma": 0.0})

    def test_n_estimators_zero_rejected(self):
        with pytest.raises(ValueError, match="n_estimators"):
            validate_hyperparams("xgboost", {"n_estimators": 0, "max_depth": 3,
                                              "learning_rate": 0.1, "subsample": 0.8,
                                              "colsample_bytree": 0.8,
                                              "min_child_weight": 1, "gamma": 0.0})

    def test_subsample_above_one_rejected(self):
        with pytest.raises(ValueError, match="subsample"):
            validate_hyperparams("lightgbm", {"n_estimators": 100, "max_depth": 5,
                                               "learning_rate": 0.1, "subsample": 1.5,
                                               "colsample_bytree": 0.8, "num_leaves": 31,
                                               "min_child_samples": 20})

    def test_num_leaves_one_rejected(self):
        """num_leaves=1 is degenerate for LightGBM (single leaf = no split)."""
        with pytest.raises(ValueError, match="num_leaves"):
            validate_hyperparams("lightgbm", {"n_estimators": 100, "max_depth": 5,
                                               "learning_rate": 0.1, "subsample": 0.8,
                                               "colsample_bytree": 0.8, "num_leaves": 1,
                                               "min_child_samples": 20})

    def test_unknown_param_rejected(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            validate_hyperparams("xgboost", {"totally_fake_param": 42})

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError, match="Unknown model_name"):
            validate_hyperparams("not_a_model", {})


# ──────────────────────────────────────────────────────────────────────────────
# Single-objective study
# ──────────────────────────────────────────────────────────────────────────────


class TestRunStudy:
    def test_random_forest_completes(self, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        params = run_study(
            "random_forest",
            n_trials=5,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
            random_state=42,
        )
        assert isinstance(params, dict)
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "min_samples_split" in params

    def test_xgboost_completes(self, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        params = run_study(
            "xgboost",
            n_trials=5,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
            random_state=42,
        )
        assert "learning_rate" in params
        assert "max_depth" in params

    def test_lightgbm_completes(self, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        params = run_study(
            "lightgbm",
            n_trials=5,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
            random_state=42,
        )
        assert "num_leaves" in params
        assert "learning_rate" in params

    def test_best_params_file_written(self, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        run_study(
            "xgboost",
            n_trials=3,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
        )
        path = tmp_path / "best_params_xgboost.json"
        assert path.exists()
        with open(path) as fh:
            saved = json.load(fh)
        assert isinstance(saved, dict)
        assert "max_depth" in saved

    def test_invalid_n_trials_zero(self, synthetic_data):
        with pytest.raises(ValueError, match="n_trials must be a positive integer"):
            run_study("xgboost", n_trials=0, validation_data=synthetic_data)

    def test_invalid_n_trials_negative(self, synthetic_data):
        with pytest.raises(ValueError, match="n_trials must be a positive integer"):
            run_study("xgboost", n_trials=-5, validation_data=synthetic_data)

    def test_invalid_n_jobs_zero(self, synthetic_data):
        with pytest.raises(ValueError, match="n_jobs must be in"):
            run_study("xgboost", n_trials=3, validation_data=synthetic_data, n_jobs=0)

    def test_invalid_n_jobs_too_high(self, synthetic_data):
        with pytest.raises(ValueError, match="n_jobs must be in"):
            run_study("xgboost", n_trials=3, validation_data=synthetic_data, n_jobs=99)

    def test_unknown_model_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="Unknown model_name"):
            run_study("not_a_model", n_trials=3, validation_data=synthetic_data)

    def test_returned_params_pass_bounds_validation(self, synthetic_data, tmp_path, monkeypatch):
        """Every param returned by run_study must pass validate_hyperparams."""
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        for model_name in ("random_forest", "xgboost", "lightgbm"):
            params = run_study(
                model_name,
                n_trials=3,
                validation_data=synthetic_data,
                n_jobs=1,
                storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
                random_state=42,
            )
            # Should not raise:
            validate_hyperparams(model_name, params)

    def test_non_decreasing_best_auc(self, synthetic_data, tmp_path, monkeypatch):
        """The best AUC must be non-decreasing as the number of trials increases.

        We run 3 trials, then 5 trials on the same study (load_if_exists=True).
        The second run's best value must be >= the first run's best value.
        """
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
        X_train, y_train, X_val, y_val = synthetic_data

        import optuna
        from optuna.samplers import TPESampler

        from detection.hyperparameter_search import _build_model, _suggest_params

        def _run_and_get_best(n_trials: int) -> float:
            study = optuna.create_study(
                study_name="auc_monotone_test",
                storage=storage_url,
                sampler=TPESampler(seed=42),
                direction="maximize",
                load_if_exists=True,
            )

            def objective(trial):
                params = _suggest_params(trial, "xgboost")
                m = _build_model("xgboost", params, random_state=42)
                m.fit(X_train, y_train)
                proba = m.predict_proba(X_val)[:, 1]
                return float(roc_auc_score(y_val, proba))

            study.optimize(objective, n_trials=n_trials)
            return study.best_value

        auc_after_3 = _run_and_get_best(3)
        auc_after_5 = _run_and_get_best(5)  # adds 5 more (load_if_exists=True)

        assert auc_after_5 >= auc_after_3 - 1e-9, (
            f"Best AUC decreased: {auc_after_3:.6f} → {auc_after_5:.6f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Multi-objective study
# ──────────────────────────────────────────────────────────────────────────────


class TestRunMultiobjectiveStudy:
    def test_returns_list(self, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        front = run_multiobjective_study(
            "xgboost",
            n_trials=8,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
            random_state=42,
        )
        assert isinstance(front, list)

    def test_pareto_entries_have_correct_keys(self, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        front = run_multiobjective_study(
            "xgboost",
            n_trials=8,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
            random_state=42,
        )
        for entry in front:
            assert "auc" in entry
            assert "latency_ms" in entry
            assert "params" in entry
            assert isinstance(entry["params"], dict)

    def test_pareto_sorted_by_descending_auc(self, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        front = run_multiobjective_study(
            "xgboost",
            n_trials=10,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
            random_state=42,
        )
        aucs = [e["auc"] for e in front]
        assert aucs == sorted(aucs, reverse=True)

    def test_pareto_front_file_written(self, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        run_multiobjective_study(
            "lightgbm",
            n_trials=8,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
            random_state=42,
        )
        assert (tmp_path / "pareto_front_lightgbm.json").exists()

    def test_invalid_n_trials_rejected(self, synthetic_data):
        with pytest.raises(ValueError, match="n_trials must be a positive integer"):
            run_multiobjective_study("xgboost", n_trials=0, validation_data=synthetic_data)


class TestSelectParetoPoint:
    def _make_front(self, entries):
        return [{"auc": a, "latency_ms": l, "params": {}} for a, l in entries]

    def test_picks_lowest_latency_above_auc_floor(self):
        front = self._make_front([(0.90, 2.0), (0.85, 1.0), (0.80, 0.5), (0.70, 0.1)])
        chosen = select_pareto_point(front, min_auc=0.80, max_latency_ms=5.0)
        assert chosen is not None
        assert chosen["auc"] >= 0.80
        assert chosen["latency_ms"] == 0.5  # fastest among AUC >= 0.80

    def test_returns_none_when_nothing_meets_constraints(self):
        front = self._make_front([(0.60, 0.5), (0.55, 0.1)])
        chosen = select_pareto_point(front, min_auc=0.80, max_latency_ms=5.0)
        assert chosen is None

    def test_empty_front_returns_none(self):
        assert select_pareto_point([], min_auc=0.75, max_latency_ms=5.0) is None

    def test_latency_constraint_applied(self):
        front = self._make_front([(0.90, 10.0), (0.85, 3.0)])
        chosen = select_pareto_point(front, min_auc=0.80, max_latency_ms=5.0)
        assert chosen is not None
        assert chosen["latency_ms"] == 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_load_best_params_nonexistent_returns_none(self, tmp_path):
        result = load_best_params("xgboost", model_dir=tmp_path)
        assert result is None

    def test_load_best_params_roundtrip(self, tmp_path):
        data = {"max_depth": 6, "learning_rate": 0.1}
        (tmp_path / "best_params_xgboost.json").write_text(json.dumps(data))
        loaded = load_best_params("xgboost", model_dir=tmp_path)
        assert loaded == data

    def test_load_best_params_invalid_json_returns_none(self, tmp_path):
        (tmp_path / "best_params_xgboost.json").write_text("not valid json {{{")
        result = load_best_params("xgboost", model_dir=tmp_path)
        assert result is None

    def test_save_and_load_unified_best_hyperparams(self, tmp_path):
        params = {
            "random_forest": {"n_estimators": 100, "max_depth": 10},
            "xgboost": {"n_estimators": 200, "learning_rate": 0.05},
            "lightgbm": {"num_leaves": 63, "n_estimators": 150},
        }
        path = save_unified_best_hyperparams(params, model_dir=tmp_path)
        assert path == tmp_path / "best_hyperparams.json"
        loaded = load_best_hyperparams(model_dir=tmp_path)
        assert loaded == params

    def test_load_best_hyperparams_nonexistent_returns_none(self, tmp_path):
        assert load_best_hyperparams(model_dir=tmp_path) is None

    def test_load_best_hyperparams_invalid_json_returns_none(self, tmp_path):
        (tmp_path / "best_hyperparams.json").write_text("{{bad json")
        assert load_best_hyperparams(model_dir=tmp_path) is None


# ──────────────────────────────────────────────────────────────────────────────
# Integration: optimised params produce a trainable model
# ──────────────────────────────────────────────────────────────────────────────


class TestOptimisedParamsProduceTrainableModel:
    """After running a study and loading the saved params, instantiating the
    model with those params and training it must succeed and produce a
    non-trivial AUC.
    """

    @pytest.mark.parametrize("model_name", ["random_forest", "xgboost", "lightgbm"])
    def test_params_produce_trainable_model(
        self, model_name, synthetic_data, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        X_train, y_train, X_val, y_val = synthetic_data

        params = run_study(
            model_name,
            n_trials=5,
            validation_data=synthetic_data,
            n_jobs=1,
            storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
            random_state=42,
        )

        # Reload from disk to exercise the full round-trip
        loaded = load_best_params(model_name, model_dir=tmp_path)
        assert loaded is not None

        # Build and train with loaded params
        from detection.hyperparameter_search import _build_model
        model = _build_model(model_name, loaded, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)

        # A model trained on 150 rows of 20 informative features should do
        # better than random chance.
        assert auc > 0.5, f"Expected AUC > 0.5, got {auc:.4f} for {model_name}"

    def test_unified_json_round_trip(self, synthetic_data, tmp_path, monkeypatch):
        """save_unified_best_hyperparams → load_best_hyperparams → train all models."""
        monkeypatch.setattr("detection.hyperparameter_search.MODEL_DIR", tmp_path)
        X_train, y_train, X_val, y_val = synthetic_data

        all_params: dict = {}
        storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
        for model_name in ("random_forest", "xgboost", "lightgbm"):
            p = run_study(
                model_name,
                n_trials=3,
                validation_data=synthetic_data,
                n_jobs=1,
                storage_url=storage_url,
                random_state=42,
            )
            all_params[model_name] = p

        save_unified_best_hyperparams(all_params, model_dir=tmp_path)
        loaded = load_best_hyperparams(model_dir=tmp_path)
        assert loaded is not None
        assert set(loaded.keys()) == {"random_forest", "xgboost", "lightgbm"}

        from detection.hyperparameter_search import _build_model
        for model_name, params in loaded.items():
            m = _build_model(model_name, params, random_state=42)
            m.fit(X_train, y_train)
            assert m.predict_proba(X_val).shape == (len(X_val), 2)
