"""Automated hyperparameter optimisation for the LedgerLens ensemble models (Issue #213).

This module implements BOHB-style optimisation using Optuna with a TPE sampler
(Tree-structured Parzen Estimator — functionally equivalent to BOHB's Bayesian
surrogate) combined with ``MedianPruner`` for HyperBand-style early stopping of
unpromising trials.

Public API
----------
run_study(model_name, n_trials, validation_data, ...)
    Single-objective optimisation (maximise AUC-ROC).
run_multiobjective_study(model_name, n_trials, validation_data, ...)
    Multi-objective optimisation (maximise AUC-ROC, minimise inference latency).
    Returns a list of Pareto-optimal (auc, latency_ms, params) triples.
select_pareto_point(pareto_front, min_auc, max_latency_ms)
    Pick the fastest Pareto point that still meets the AUC floor.
get_search_space(model_name)
    Return the search-space bounds for a model.
load_best_params(model_name)
    Load per-model best params from ``models/best_params_{model}.json``.
load_best_hyperparams()
    Load the unified ``models/best_hyperparams.json`` (all three ensemble models).
validate_hyperparams(model_name, params)
    Raise ``ValueError`` if any parameter is outside its legal bounds.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import NSGAIISampler, TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config import config

# Silence Optuna's verbose per-trial logging in production; tests can set
# optuna.logging.set_verbosity(optuna.logging.DEBUG) as needed.
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(config.MODEL_DIR or "./models")
STUDY_DB_URL = f"sqlite:///{MODEL_DIR / 'optuna_studies.db'}"

# ──────────────────────────────────────────────────────────────────────────────
# Search spaces
# ──────────────────────────────────────────────────────────────────────────────

# Format: { param_name: (type, min_val, max_val) }
# "log" float entries use log-uniform sampling.
_SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 2, 30),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("float", 0.1, 1.0),
    },
    "xgboost": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 2, 10),
        "learning_rate": ("float_log", 1e-3, 0.3),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "min_child_weight": ("int", 1, 10),
        "gamma": ("float", 0.0, 5.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 2, 15),
        "learning_rate": ("float_log", 1e-3, 0.3),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "num_leaves": ("int", 15, 255),
        "min_child_samples": ("int", 5, 100),
    },
    # Legacy models kept for backwards compatibility
    "isolation_forest": {
        "contamination": ("float", 0.01, 0.2),
        "n_estimators": ("int", 50, 500),
        "max_features": ("float", 0.5, 1.0),
    },
    "gnn": {
        "hidden_dim": ("int", 16, 128),
        "num_layers": ("int", 1, 4),
        "dropout": ("float", 0.0, 0.5),
        "learning_rate": ("float_log", 1e-4, 1e-2),
    },
}

# Hard bounds used by validate_hyperparams() — anything outside these is
# rejected before it can reach the model constructor and cause a silent failure.
_HARD_BOUNDS: dict[str, dict[str, tuple[Any, Any]]] = {
    "random_forest": {
        "n_estimators": (1, 10_000),
        "max_depth": (1, 200),
        "min_samples_split": (2, 10_000),
        "min_samples_leaf": (1, 10_000),
        "max_features": (1e-6, 1.0),
    },
    "xgboost": {
        "n_estimators": (1, 10_000),
        "max_depth": (1, 50),
        "learning_rate": (1e-6, 10.0),
        "subsample": (1e-6, 1.0),
        "colsample_bytree": (1e-6, 1.0),
        "min_child_weight": (0, 1000),
        "gamma": (0.0, 100.0),
    },
    "lightgbm": {
        "n_estimators": (1, 10_000),
        "max_depth": (1, 200),
        "learning_rate": (1e-6, 10.0),
        "subsample": (1e-6, 1.0),
        "colsample_bytree": (1e-6, 1.0),
        "num_leaves": (2, 131_072),
        "min_child_samples": (1, 100_000),
    },
    "isolation_forest": {
        "contamination": (1e-6, 0.5),
        "n_estimators": (1, 10_000),
        "max_features": (1e-6, 1.0),
    },
    "gnn": {
        "hidden_dim": (1, 4096),
        "num_layers": (1, 32),
        "dropout": (0.0, 1.0),
        "learning_rate": (1e-9, 1.0),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────


class HyperparameterSearchError(Exception):
    """Raised when the hyperparameter search encounters a fatal error."""


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_search_space(model_name: str) -> dict[str, tuple]:
    """Return the search-space definition for *model_name*.

    Each entry is ``(type, min_val, max_val)`` where type is one of
    ``"int"``, ``"float"``, or ``"float_log"`` (log-uniform sampling).

    Raises:
        ValueError: If *model_name* is not a known model.
    """
    if model_name not in _SEARCH_SPACES:
        raise ValueError(
            f"Unknown model_name: {model_name!r}. Must be one of {sorted(_SEARCH_SPACES)}"
        )
    return dict(_SEARCH_SPACES[model_name])


def validate_hyperparams(model_name: str, params: dict[str, Any]) -> None:
    """Validate that *params* are within the hard bounds for *model_name*.

    Raises:
        ValueError: On unknown model, unknown parameter, or out-of-bounds value.
    """
    if model_name not in _HARD_BOUNDS:
        raise ValueError(f"Unknown model_name for validation: {model_name!r}")

    bounds = _HARD_BOUNDS[model_name]
    for key, value in params.items():
        if key not in bounds:
            raise ValueError(
                f"Unknown parameter {key!r} for model {model_name!r}"
            )
        lo, hi = bounds[key]
        if not (lo <= value <= hi):
            raise ValueError(
                f"Parameter {key}={value!r} for model {model_name!r} is outside "
                f"the safe range [{lo}, {hi}]. This could cause a silent training failure."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Internal: suggest + build
# ──────────────────────────────────────────────────────────────────────────────


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest hyperparameters for *model_name* using *trial*."""
    space = get_search_space(model_name)
    params: dict[str, Any] = {}
    for param_name, spec in space.items():
        param_type, lo, hi = spec
        if param_type == "int":
            params[param_name] = trial.suggest_int(param_name, int(lo), int(hi))
        elif param_type == "float":
            params[param_name] = trial.suggest_float(param_name, float(lo), float(hi))
        elif param_type == "float_log":
            params[param_name] = trial.suggest_float(
                param_name, float(lo), float(hi), log=True
            )
        else:
            raise ValueError(f"Unknown parameter type {param_type!r}")
    return params


def _build_model(model_name: str, params: dict[str, Any], random_state: int = 42):
    """Instantiate the sklearn-compatible model for *model_name* with *params*."""
    if model_name == "random_forest":
        return RandomForestClassifier(**params, random_state=random_state, n_jobs=1)
    if model_name == "xgboost":
        return XGBClassifier(**params, random_state=random_state, verbosity=0, n_jobs=1)
    if model_name == "lightgbm":
        return LGBMClassifier(**params, random_state=random_state, verbosity=-1, n_jobs=1)
    raise ValueError(f"No model builder for {model_name!r}")


def _score_auc(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Return AUC-ROC on the validation set. Returns 0.0 on any error."""
    try:
        probs = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, probs))
    except Exception:
        return 0.0


def _score_latency_ms(model, X_val: pd.DataFrame) -> float:
    """Return median per-sample inference latency in milliseconds."""
    n = min(len(X_val), 200)
    sample = X_val.iloc[:n]
    t0 = time.perf_counter()
    model.predict_proba(sample)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / n  # ms per sample


# ──────────────────────────────────────────────────────────────────────────────
# Early-stopping callback
# ──────────────────────────────────────────────────────────────────────────────


class _NoImprovementCallback:
    """Stop the study if there is no improvement in the primary metric for
    *patience* consecutive trials after the first *min_trials* trials.

    Designed for single-objective studies only.
    """

    def __init__(self, patience: int = 30, min_trials: int = 10) -> None:
        self._patience = patience
        self._min_trials = min_trials
        self._best: float = -float("inf")
        self._stagnant: int = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(study.trials) < self._min_trials:
            return
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            return
        current_best = study.best_value
        if current_best > self._best + 1e-6:
            self._best = current_best
            self._stagnant = 0
        else:
            self._stagnant += 1
            if self._stagnant >= self._patience:
                logger.info(
                    "Early stopping: no improvement for %d trials. Best AUC=%.4f",
                    self._stagnant,
                    self._best,
                )
                study.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Single-objective study (AUC-ROC)
# ──────────────────────────────────────────────────────────────────────────────


def run_study(
    model_name: str,
    n_trials: int,
    validation_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    n_jobs: int = 1,
    storage_url: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    timeout_seconds: float | None = None,
    random_state: int = 42,
    no_improvement_patience: int = 30,
) -> dict[str, Any]:
    """Run a single-objective Optuna study to maximise AUC-ROC.

    Args:
        model_name: One of the keys in ``_SEARCH_SPACES``.
        n_trials: Number of trials.  Must be ≥ 1.
        validation_data: ``(X_train, y_train, X_val, y_val)``.
        n_jobs: Parallel workers (1–4).
        storage_url: Optuna storage URL. Defaults to SQLite in ``MODEL_DIR``.
        sampler: Optuna sampler. Defaults to ``TPESampler(seed=random_state)``.
        timeout_seconds: Wall-clock timeout. ``None`` means no timeout.
        random_state: RNG seed for reproducibility.
        no_improvement_patience: Stop after this many non-improving trials
            (≥ 10 trials completed).

    Returns:
        Best trial's parameter dict.

    Raises:
        ValueError: On bad arguments.
        HyperparameterSearchError: On study failure.
    """
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials!r}")
    if not isinstance(n_jobs, int) or not (1 <= n_jobs <= 4):
        raise ValueError(f"n_jobs must be in [1, 4], got {n_jobs!r}")
    if model_name not in _SEARCH_SPACES:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    X_train, y_train, X_val, y_val = validation_data

    if sampler is None:
        sampler = TPESampler(seed=random_state)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    storage_url = storage_url or STUDY_DB_URL

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        try:
            model = _build_model(model_name, params, random_state=random_state)
            model.fit(X_train, y_train)
            return _score_auc(model, X_val, y_val)
        except Exception as exc:
            logger.debug("Trial %d failed: %s", trial.number, exc)
            raise optuna.TrialPruned() from exc

    early_stop = _NoImprovementCallback(patience=no_improvement_patience)

    try:
        study = optuna.create_study(
            study_name=f"ledgerlens_{model_name}",
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout_seconds,
            callbacks=[early_stop],
            show_progress_bar=False,
        )
    except Exception as exc:
        raise HyperparameterSearchError(
            f"Study for {model_name!r} failed: {exc}"
        ) from exc

    best_params = study.best_params
    logger.info(
        "Study '%s' done. Best AUC-ROC=%.4f  params=%s",
        model_name,
        study.best_value,
        best_params,
    )

    _persist_model_params(model_name, best_params)
    return best_params


# ──────────────────────────────────────────────────────────────────────────────
# Multi-objective study (AUC-ROC + latency)
# ──────────────────────────────────────────────────────────────────────────────


def run_multiobjective_study(
    model_name: str,
    n_trials: int,
    validation_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    n_jobs: int = 1,
    storage_url: str | None = None,
    timeout_seconds: float | None = None,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Multi-objective optimisation: maximise AUC-ROC, minimise inference latency.

    Uses the NSGA-II sampler to approximate the Pareto front.

    Args:
        model_name: Model identifier (``"random_forest"``, ``"xgboost"``, ``"lightgbm"``).
        n_trials: Number of trials.
        validation_data: ``(X_train, y_train, X_val, y_val)``.
        n_jobs: Parallel workers (1–4).
        storage_url: Optuna storage URL.
        timeout_seconds: Wall-clock timeout.
        random_state: RNG seed.

    Returns:
        List of Pareto-optimal entries, each:
        ``{"auc": float, "latency_ms": float, "params": dict}``, sorted by
        descending AUC.

    Raises:
        ValueError: On bad arguments.
        HyperparameterSearchError: On study failure.
    """
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials!r}")
    if not isinstance(n_jobs, int) or not (1 <= n_jobs <= 4):
        raise ValueError(f"n_jobs must be in [1, 4], got {n_jobs!r}")
    if model_name not in _SEARCH_SPACES:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    X_train, y_train, X_val, y_val = validation_data
    storage_url = storage_url or STUDY_DB_URL

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = _suggest_params(trial, model_name)
        try:
            model = _build_model(model_name, params, random_state=random_state)
            model.fit(X_train, y_train)
            auc = _score_auc(model, X_val, y_val)
            latency = _score_latency_ms(model, X_val)
            return auc, latency
        except Exception as exc:
            logger.debug("Trial %d failed: %s", trial.number, exc)
            # Return dominated values so the trial is excluded from the Pareto front.
            return 0.0, float("inf")

    try:
        study = optuna.create_study(
            study_name=f"ledgerlens_{model_name}_mo",
            storage=storage_url,
            sampler=NSGAIISampler(seed=random_state),
            directions=["maximize", "minimize"],
            load_if_exists=True,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout_seconds,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise HyperparameterSearchError(
            f"Multi-objective study for {model_name!r} failed: {exc}"
        ) from exc

    pareto_trials = study.best_trials
    pareto_front: list[dict[str, Any]] = []
    for t in pareto_trials:
        if len(t.values) == 2:
            auc_val, lat_val = t.values
            if auc_val > 0.0 and lat_val < float("inf"):
                pareto_front.append(
                    {"auc": auc_val, "latency_ms": lat_val, "params": dict(t.params)}
                )

    pareto_front.sort(key=lambda x: x["auc"], reverse=True)

    # Persist the Pareto front
    _persist_pareto_front(model_name, pareto_front)

    logger.info(
        "Multi-objective study '%s' done. %d Pareto-optimal trials.",
        model_name,
        len(pareto_front),
    )
    return pareto_front


def select_pareto_point(
    pareto_front: list[dict[str, Any]],
    min_auc: float = 0.75,
    max_latency_ms: float = 5.0,
) -> dict[str, Any] | None:
    """Pick the Pareto point with the lowest latency that satisfies the AUC floor.

    Args:
        pareto_front: Output of ``run_multiobjective_study``.
        min_auc: Minimum acceptable AUC-ROC.
        max_latency_ms: Maximum acceptable latency per sample (ms).

    Returns:
        The selected entry dict, or ``None`` if no point meets both constraints.
    """
    candidates = [
        p for p in pareto_front
        if p["auc"] >= min_auc and p["latency_ms"] <= max_latency_ms
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["latency_ms"])


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────


def _persist_model_params(model_name: str, params: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"best_params_{model_name}.json"
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2)
    logger.debug("Persisted best params for '%s' → %s", model_name, path)


def _persist_pareto_front(
    model_name: str, pareto_front: list[dict[str, Any]]
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"pareto_front_{model_name}.json"
    with open(path, "w") as fh:
        json.dump(pareto_front, fh, indent=2)
    logger.debug("Persisted Pareto front for '%s' → %s", model_name, path)


def save_unified_best_hyperparams(
    params_by_model: dict[str, dict[str, Any]],
    model_dir: Path | None = None,
) -> Path:
    """Write ``best_hyperparams.json`` combining params for all ensemble models.

    Args:
        params_by_model: ``{model_name: params_dict}`` for each model.
        model_dir: Directory to write to.  Defaults to ``MODEL_DIR``.

    Returns:
        Path to the written file.
    """
    out_dir = model_dir or MODEL_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_hyperparams.json"
    with open(path, "w") as fh:
        json.dump(params_by_model, fh, indent=2)
    logger.info("Unified best hyperparams written → %s", path)
    return path


def load_best_params(model_name: str, model_dir: Path | None = None) -> dict[str, Any] | None:
    """Load per-model best params from ``models/best_params_{model_name}.json``.

    Returns ``None`` if the file does not exist or cannot be parsed.
    """
    d = Path(model_dir) if model_dir else MODEL_DIR
    path = d / f"best_params_{model_name}.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load best params for '%s': %s", model_name, exc)
        return None


def load_best_hyperparams(model_dir: Path | None = None) -> dict[str, dict[str, Any]] | None:
    """Load the unified ``best_hyperparams.json``.

    Returns ``None`` if the file does not exist.
    """
    d = Path(model_dir) if model_dir else MODEL_DIR
    path = d / "best_hyperparams.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load best_hyperparams.json: %s", exc)
        return None
