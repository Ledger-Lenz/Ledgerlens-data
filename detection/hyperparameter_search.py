"""Automated hyperparameter search using Optuna for detection model tuning (Issue #281).

API:
  - run_study(model_name, n_trials, validation_data, n_jobs, storage_url, sampler)
    Launch an Optuna study to optimize model hyperparameters.
  - load_best_params(model_name)
    Load persisted best trial parameters from disk.
  - get_search_space(model_name)
    Retrieve the default search space for a given model.
"""

import json
import logging
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from config import config

logger = logging.getLogger(__name__)

MODEL_DIR = Path(config.MODEL_DIR or "./models")
STUDY_DB_URL = f"sqlite:///{MODEL_DIR / 'optuna_studies.db'}"


class HyperparameterSearchError(Exception):
    """Raised when hyperparameter search fails."""

    pass


def get_search_space(model_name: str) -> dict[str, Any]:
    """Retrieve the default search space for a given model.

    Args:
        model_name: One of "isolation_forest", "xgboost", "gnn".

    Returns:
        A dict defining the search space (param_name -> (type, min, max)).

    Raises:
        ValueError: If model_name is unknown.
    """
    spaces = {
        "isolation_forest": {
            "contamination": ("float", 0.01, 0.2),
            "n_estimators": ("int", 50, 500),
            "max_features": ("float", 0.5, 1.0),
        },
        "xgboost": {
            "max_depth": ("int", 2, 10),
            "learning_rate": ("float", 0.01, 0.3),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "n_estimators": ("int", 50, 500),
        },
        "gnn": {
            "hidden_dim": ("int", 16, 128),
            "num_layers": ("int", 1, 4),
            "dropout": ("float", 0.0, 0.5),
            "learning_rate": ("float", 0.0001, 0.01),
        },
    }
    if model_name not in spaces:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Must be one of {list(spaces.keys())}"
        )
    return spaces[model_name]


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest hyperparameters for a given model using an Optuna trial.

    Args:
        trial: The Optuna trial object.
        model_name: Model identifier.

    Returns:
        A dict of suggested parameters.
    """
    space = get_search_space(model_name)
    params = {}

    for param_name, (param_type, min_val, max_val) in space.items():
        if param_type == "int":
            params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
        elif param_type == "float":
            params[param_name] = trial.suggest_float(param_name, min_val, max_val)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def _train_isolation_forest(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]
) -> float:
    """Train Isolation Forest and return F1 score."""
    model = IsolationForest(**params, random_state=42)
    model.fit(X_train)
    y_pred = model.predict(X_train)
    y_pred_binary = (y_pred == -1).astype(int)
    return f1_score(y_train, y_pred_binary, zero_division=0)


def _train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]
) -> float:
    """Train XGBoost and return F1 score."""
    model = XGBClassifier(**params, random_state=42, verbosity=0)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_train)
    return f1_score(y_train, y_pred, zero_division=0)


def _objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Objective function for Optuna trial.

    Args:
        trial: Optuna trial.
        model_name: Model identifier.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Validation F1 score (to maximize).
    """
    params = _suggest_params(trial, model_name)

    try:
        if model_name == "isolation_forest":
            model = IsolationForest(**params, random_state=42)
            model.fit(X_train)
            y_pred = model.predict(X_val)
            y_pred_binary = (y_pred == -1).astype(int)
            score = f1_score(y_val, y_pred_binary, zero_division=0)
        elif model_name == "xgboost":
            model = XGBClassifier(**params, random_state=42, verbosity=0)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return score

    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        raise optuna.TrialPruned() from e


def run_study(
    model_name: str,
    n_trials: int,
    validation_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    n_jobs: int = 1,
    storage_url: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> dict[str, Any]:
    """Run an Optuna study to optimize model hyperparameters.

    Args:
        model_name: Model identifier ("isolation_forest", "xgboost", "gnn").
        n_trials: Number of trials to run. Must be > 0.
        validation_data: Tuple (X_train, y_train, X_val, y_val).
        n_jobs: Number of parallel workers (1-4). Default 1 (serial).
        storage_url: Optuna storage URL. Defaults to SQLite in MODEL_DIR.
        sampler: Optuna sampler. Defaults to TPESampler.

    Returns:
        Best trial parameters as a dict.

    Raises:
        ValueError: If n_trials <= 0 or n_jobs out of range.
        HyperparameterSearchError: If the study fails.
    """
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials}")
    if not isinstance(n_jobs, int) or not (1 <= n_jobs <= 4):
        raise ValueError(f"n_jobs must be in [1, 4], got {n_jobs}")

    X_train, y_train, X_val, y_val = validation_data

    if storage_url is None:
        storage_url = STUDY_DB_URL

    if sampler is None:
        sampler = TPESampler(seed=42)

    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=1)

    study_name = f"study_{model_name}"

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> float:
            return _objective(trial, model_name, X_train, y_train, X_val, y_val)

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        best_params = study.best_params
        best_value = study.best_value

        logger.info(
            f"Study {study_name} completed. Best F1: {best_value:.4f}. "
            f"Best params: {best_params}"
        )

        # Persist best params
        params_file = MODEL_DIR / f"best_params_{model_name}.json"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Persisted best params to {params_file}")

        return best_params

    except Exception as e:
        raise HyperparameterSearchError(f"Study failed for {model_name}: {e}") from e


def load_best_params(model_name: str) -> dict[str, Any] | None:
    """Load persisted best trial parameters from disk.

    Args:
        model_name: Model identifier.

    Returns:
        Best params dict if file exists, else None.
    """
    params_file = MODEL_DIR / f"best_params_{model_name}.json"
    if not params_file.exists():
        return None
    try:
        with open(params_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load best params for {model_name}: {e}")
        return None
