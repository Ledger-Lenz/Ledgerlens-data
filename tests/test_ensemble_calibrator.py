"""Tests for detection/ensemble_calibrator.py — NSGA-II Pareto front search."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from detection.ensemble_calibrator import (
    EnsembleCalibrator,
    ParetoSolution,
    _build_validation_context,
    _normalize_weights,
    objective_precision,
    objective_recall,
    objective_shap_stability,
)

# Small NSGA-II budget so the toy-problem tests stay fast; the issue's
# defaults (population=50, generations=100) are exercised via model_training's
# --calibrate-ensemble CLI flag, which is an opt-in offline step.
_TOY_POPULATION = 12
_TOY_GENERATIONS = 8
_TOY_N_PERTURB = 3


@pytest.fixture(scope="module")
def two_model_setup(tmp_path_factory):
    """A toy 2-model (RF + XGBoost) ensemble with a deliberately ambiguous
    (overlapping, label-noised) classification problem.

    `scripts.generate_synthetic_dataset` produces cleanly-separable classes by
    design (good for exercising training/inference plumbing), which means RF
    and XGBoost agree on every prediction and no weight vector can trade
    precision against recall. Calibration only has something to search for
    when the models actually disagree, so this fixture uses
    `sklearn.datasets.make_classification` with class overlap and label noise
    instead.
    """
    X, y = make_classification(
        n_samples=220,
        n_features=14,
        n_informative=5,
        n_redundant=2,
        class_sep=0.5,
        flip_y=0.15,
        random_state=5,
    )
    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    X_val_df = pd.DataFrame(X_val, columns=feature_cols)
    y_val_s = pd.Series(y_val)

    # Deliberately differently-biased hyperparameters: a shallow, well-averaged
    # RF (high precision, lower recall) vs. a deep, lightly-regularized XGBoost
    # (higher recall, lower precision) — gives weight blending an actual
    # precision/recall frontier to search over, rather than one model
    # dominating the other on every objective.
    models = {
        "random_forest": RandomForestClassifier(random_state=1, n_estimators=60, max_depth=4).fit(
            X_train_df, y_train
        ),
        "xgboost": XGBClassifier(
            random_state=1, max_depth=9, n_estimators=6, subsample=1.0, reg_lambda=0
        ).fit(X_train_df, y_train),
    }
    model_dir = str(tmp_path_factory.mktemp("models"))
    return models, X_val_df, y_val_s, model_dir


# ---------------------------------------------------------------------------
# Weight simplex projection
# ---------------------------------------------------------------------------


def test_normalize_weights_sums_to_one_and_nonnegative():
    raw = np.array([[0.2, 0.8], [5.0, 5.0], [-1.0, 3.0]])
    normalized = _normalize_weights(raw)
    assert np.allclose(normalized.sum(axis=-1), 1.0)
    assert (normalized >= 0).all()


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------


def test_objective_precision_and_recall_are_bounded(two_model_setup):
    models, X_val, y_val, _ = two_model_setup
    val_data = _build_validation_context(
        models, X_val, y_val, n_perturb=_TOY_N_PERTURB, random_state=1
    )

    weights = _normalize_weights(np.array([[0.5, 0.5], [1.0, 0.0]]))
    precision = objective_precision(weights, val_data)
    recall = objective_recall(weights, val_data)

    assert precision.shape == (2,)
    assert recall.shape == (2,)
    assert ((precision >= 0) & (precision <= 1)).all()
    assert ((recall >= 0) & (recall <= 1)).all()


def test_objective_shap_stability_is_bounded(two_model_setup):
    models, X_val, y_val, _ = two_model_setup
    val_data = _build_validation_context(
        models, X_val, y_val, n_perturb=_TOY_N_PERTURB, random_state=1
    )

    weights = _normalize_weights(np.array([[0.5, 0.5], [0.9, 0.1]]))
    stability = objective_shap_stability(weights, val_data)

    assert stability.shape == (2,)
    assert ((stability >= -1 - 1e-9) & (stability <= 1 + 1e-9)).all()


# ---------------------------------------------------------------------------
# NSGA-II search (acceptance criterion: 2-model toy problem)
# ---------------------------------------------------------------------------


def test_nsga2_two_model_toy_problem_pareto_front_has_distinct_points(two_model_setup):
    models, X_val, y_val, model_dir = two_model_setup
    calibrator = EnsembleCalibrator(model_dir=model_dir)

    solutions = calibrator.run_search(
        models,
        X_val,
        y_val,
        population_size=_TOY_POPULATION,
        n_generations=_TOY_GENERATIONS,
        n_perturb=_TOY_N_PERTURB,
        random_state=1,
    )

    assert len(solutions) >= 2
    distinct_weight_points = {tuple(round(w, 3) for w in sol.weights.values()) for sol in solutions}
    assert len(distinct_weight_points) >= 2
    for sol in solutions:
        assert set(sol.weights) == {"random_forest", "xgboost"}
        assert abs(sum(sol.weights.values()) - 1.0) < 1e-6


def test_pareto_front_round_trips_through_disk(two_model_setup):
    models, X_val, y_val, model_dir = two_model_setup
    calibrator = EnsembleCalibrator(model_dir=model_dir)
    calibrator.run_search(
        models,
        X_val,
        y_val,
        population_size=_TOY_POPULATION,
        n_generations=_TOY_GENERATIONS,
        n_perturb=_TOY_N_PERTURB,
        random_state=2,
    )

    loaded = calibrator.load_pareto_front()
    assert loaded
    assert all(isinstance(s, ParetoSolution) for s in loaded)


def test_load_pareto_front_raises_when_missing(tmp_path):
    calibrator = EnsembleCalibrator(model_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        calibrator.load_pareto_front()


# ---------------------------------------------------------------------------
# Operator interface
# ---------------------------------------------------------------------------


def test_select_operating_point_respects_constraints(two_model_setup):
    models, X_val, y_val, model_dir = two_model_setup
    calibrator = EnsembleCalibrator(model_dir=model_dir)
    calibrator.run_search(
        models,
        X_val,
        y_val,
        population_size=_TOY_POPULATION,
        n_generations=_TOY_GENERATIONS,
        n_perturb=_TOY_N_PERTURB,
        random_state=3,
    )

    weights = calibrator.select_operating_point(min_precision=0.0, min_recall=0.0)
    assert set(weights) == set(models)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_select_operating_point_raises_when_infeasible(two_model_setup):
    models, X_val, y_val, model_dir = two_model_setup
    calibrator = EnsembleCalibrator(model_dir=model_dir)
    calibrator.run_search(
        models,
        X_val,
        y_val,
        population_size=_TOY_POPULATION,
        n_generations=_TOY_GENERATIONS,
        n_perturb=_TOY_N_PERTURB,
        random_state=4,
    )

    with pytest.raises(ValueError):
        calibrator.select_operating_point(min_precision=0.999, min_recall=0.999)
