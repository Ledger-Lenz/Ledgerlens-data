"""Multi-objective calibration of ensemble weights via NSGA-II Pareto search.

`RiskScorer` (see `detection/model_inference.py`) ordinarily combines the
three ensemble models' predicted probabilities with BFT trimmed-mean voting,
which optimizes for robustness rather than for any particular operating
point on the precision/recall/explainability tradeoff. This module searches
the space of per-model *combination weights* for the set of Pareto-optimal
tradeoffs between three competing objectives:

  - precision (investigators don't want to waste time on false positives)
  - recall (compliance can't afford to miss confirmed wash traders)
  - SHAP stability (auditors need explanations that don't flip under noise)

`EnsembleCalibrator.run_search` runs NSGA-II (via `pymoo`) over the weight
simplex and persists the resulting front to `models/pareto_front.json`.
`EnsembleCalibrator.select_operating_point` then lets an operator pick a
concrete weight vector subject to precision/recall floors.

Performance note: the only expensive part of each objective — per-model
predicted probabilities and per-model SHAP vectors (including the Monte
Carlo input perturbations used for the stability objective) — does not
depend on the candidate weight vector, only on the fixed validation set and
the already-trained models. It is therefore computed exactly once up front
(`_build_validation_context`); every NSGA-II generation then only performs
cheap weighted sums over those precomputed arrays. This is what makes a
population=50 / generations=100 search tractable as an offline training step.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import shap
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from config import config
from detection.model_training import FEATURE_COLUMNS_EXCLUDE
from utils.logging import get_logger

logger = get_logger(__name__)

PARETO_FRONT_FILENAME = "pareto_front.json"

_WEIGHT_FLOOR = 1e-6  # keeps the simplex projection away from all-zero vectors
_SHAP_PERTURBATION_SCALE = 0.01  # x' = x + N(0, 0.01 * std(x)), per the design doc
_COSINE_EPSILON = 1e-12


@dataclass(slots=True)
class ValidationContext:
    """Precomputed, weight-independent quantities shared by every objective."""

    model_names: list[str]
    y_true: np.ndarray  # (n_wallets,)
    probs: np.ndarray  # (n_models, n_wallets)
    shap_base: np.ndarray  # (n_models, n_wallets, n_features)
    shap_perturbed: np.ndarray  # (n_perturb, n_models, n_wallets, n_features)


@dataclass(slots=True)
class ParetoSolution:
    """One non-dominated (weights, objectives) point on the Pareto front."""

    weights: dict[str, float]
    objectives: dict[str, float]

    def to_dict(self) -> dict:
        return {"weights": self.weights, "objectives": self.objectives}

    @classmethod
    def from_dict(cls, data: dict) -> ParetoSolution:
        return cls(weights=dict(data["weights"]), objectives=dict(data["objectives"]))


def _normalize_weights(raw: np.ndarray) -> np.ndarray:
    """Project raw NSGA-II decision vectors onto the weight simplex (sum=1, >=0)."""
    clipped = np.clip(raw, _WEIGHT_FLOOR, None)
    return clipped / clipped.sum(axis=-1, keepdims=True)


def _multi_row_shap_values(model, X: pd.DataFrame, cache: dict) -> np.ndarray:
    """Positive-class SHAP values for every row in `X`. Shape (n_rows, n_features).

    Binary tree models may return a list `[class_0, class_1]`, or a single
    ndarray shaped `(n_rows, n_features, n_classes)` — mirrors the branching
    in `detection.shap_explainer.ShapExplainer`, but for the whole matrix
    rather than a single row.
    """
    key = id(model)
    if key not in cache:
        cache[key] = shap.TreeExplainer(model)
    shap_values = cache[key].shap_values(X)
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def _build_validation_context(
    models: dict[str, object],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_perturb: int = 50,
    random_state: int = 42,
) -> ValidationContext:
    """Compute per-model probabilities and base/perturbed SHAP vectors once."""
    model_names = sorted(models)
    rng = np.random.default_rng(random_state)
    explainer_cache: dict = {}

    feature_cols = [c for c in X_val.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    X = X_val[feature_cols].astype(float)

    probs = np.stack([models[name].predict_proba(X)[:, 1] for name in model_names])
    shap_base = np.stack(
        [_multi_row_shap_values(models[name], X, explainer_cache) for name in model_names]
    )

    feature_std = X.std(axis=0).replace(0.0, 1.0).to_numpy()
    perturbed_batches = []
    for _ in range(n_perturb):
        noise = rng.normal(0.0, _SHAP_PERTURBATION_SCALE, size=X.shape) * feature_std
        X_perturbed = X + noise
        perturbed_batches.append(
            np.stack(
                [
                    _multi_row_shap_values(models[name], X_perturbed, explainer_cache)
                    for name in model_names
                ]
            )
        )
    shap_perturbed = np.stack(perturbed_batches)

    return ValidationContext(
        model_names=model_names,
        y_true=y_val.to_numpy(),
        probs=probs,
        shap_base=shap_base,
        shap_perturbed=shap_perturbed,
    )


def _ensemble_probs(weights: np.ndarray, val_data: ValidationContext) -> np.ndarray:
    """Weighted-average P(wash) per wallet, for each candidate weight vector.

    `weights` is (n_candidates, n_models); returns (n_candidates, n_wallets).
    """
    return weights @ val_data.probs


def objective_precision(weights: np.ndarray, val_data: ValidationContext) -> np.ndarray:
    """Precision at config.RISK_SCORE_FLAG_THRESHOLD threshold, per candidate weight vector."""
    threshold = config.RISK_SCORE_FLAG_THRESHOLD / 100.0
    preds = _ensemble_probs(weights, val_data) >= threshold
    is_wash = val_data.y_true.astype(bool)

    tp = (preds & is_wash).sum(axis=-1)
    fp = (preds & ~is_wash).sum(axis=-1)
    denom = tp + fp
    return np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)


def objective_recall(weights: np.ndarray, val_data: ValidationContext) -> np.ndarray:
    """Recall at config.RISK_SCORE_FLAG_THRESHOLD threshold, per candidate weight vector."""
    threshold = config.RISK_SCORE_FLAG_THRESHOLD / 100.0
    preds = _ensemble_probs(weights, val_data) >= threshold
    is_wash = val_data.y_true.astype(bool)

    tp = (preds & is_wash).sum(axis=-1)
    fn = (~preds & is_wash).sum(axis=-1)
    denom = tp + fn
    return np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)


def objective_shap_stability(weights: np.ndarray, val_data: ValidationContext) -> np.ndarray:
    """Mean cosine similarity between SHAP vectors under small input perturbations.

    High value = stable explanations. Estimated via Monte Carlo over the
    perturbations baked into `val_data.shap_perturbed`.
    """
    base = np.einsum("cm,mwf->cwf", weights, val_data.shap_base)
    perturbed = np.einsum("cm,kmwf->ckwf", weights, val_data.shap_perturbed)

    dot = np.einsum("cwf,ckwf->ckw", base, perturbed)
    base_norm = np.linalg.norm(base, axis=-1)  # (c, w)
    perturbed_norm = np.linalg.norm(perturbed, axis=-1)  # (c, k, w)
    denom = base_norm[:, None, :] * perturbed_norm

    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > _COSINE_EPSILON)
    return cosine.mean(axis=(1, 2))


class _CalibrationProblem(Problem):
    """NSGA-II problem over the ensemble weight simplex.

    pymoo minimizes by convention, so the (precision, recall, stability)
    objectives — all of which we want to maximize — are negated in `_evaluate`.
    """

    def __init__(self, val_data: ValidationContext):
        super().__init__(n_var=len(val_data.model_names), n_obj=3, xl=0.0, xu=1.0)
        self.val_data = val_data

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        weights = _normalize_weights(x)
        precision = objective_precision(weights, self.val_data)
        recall = objective_recall(weights, self.val_data)
        stability = objective_shap_stability(weights, self.val_data)
        out["F"] = -np.column_stack([precision, recall, stability])


def _extract_pareto_solutions(result, model_names: list[str]) -> list[ParetoSolution]:
    if result.X is None:
        return []

    weights = _normalize_weights(np.atleast_2d(result.X))
    objectives = -np.atleast_2d(result.F)

    solutions = []
    for w, obj in zip(weights, objectives, strict=True):
        solutions.append(
            ParetoSolution(
                weights={name: float(v) for name, v in zip(model_names, w, strict=True)},
                objectives={
                    "precision": float(obj[0]),
                    "recall": float(obj[1]),
                    "shap_stability": float(obj[2]),
                },
            )
        )
    return solutions


def summarize_pareto_front(solutions: list[ParetoSolution]) -> str:
    """Render a compact text summary of the front's extremes for the training log."""
    if not solutions:
        return "Pareto front is empty."

    lines = [f"Pareto front summary ({len(solutions)} non-dominated solutions):"]
    for label, key in (
        ("max precision", "precision"),
        ("max recall", "recall"),
        ("max SHAP stability", "shap_stability"),
    ):
        best = max(solutions, key=lambda s: s.objectives[key])
        weights_str = ", ".join(f"{name}={w:.2f}" for name, w in best.weights.items())
        objectives_str = ", ".join(f"{k}={v:.3f}" for k, v in best.objectives.items())
        lines.append(f"  {label}: weights=({weights_str}) objectives=({objectives_str})")
    return "\n".join(lines)


class EnsembleCalibrator:
    """Searches for, persists, and selects from the ensemble's Pareto front."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.pareto_front_path = os.path.join(self.model_dir, PARETO_FRONT_FILENAME)

    def run_search(
        self,
        models: dict[str, object],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        population_size: int = 50,
        n_generations: int = 100,
        n_perturb: int = 50,
        random_state: int = 42,
    ) -> list[ParetoSolution]:
        """Run NSGA-II over `models`' combination weights and persist the front.

        `models` maps model name -> fitted estimator (any non-empty subset of
        `detection.model_training.MODEL_REGISTRY` works, which is what lets a
        2-model toy problem exercise the same code path as the full ensemble).
        """
        if not models:
            raise ValueError("run_search requires at least one model")

        val_data = _build_validation_context(
            models, X_val, y_val, n_perturb=n_perturb, random_state=random_state
        )
        problem = _CalibrationProblem(val_data)
        algorithm = NSGA2(pop_size=population_size)

        logger.info(
            "Running NSGA-II ensemble calibration: %d models, pop=%d, gen=%d, n_perturb=%d",
            len(val_data.model_names),
            population_size,
            n_generations,
            n_perturb,
        )
        result = minimize(
            problem, algorithm, ("n_gen", n_generations), seed=random_state, verbose=False
        )

        solutions = _extract_pareto_solutions(result, val_data.model_names)
        self._save(solutions)
        logger.info("NSGA-II found %d non-dominated solutions", len(solutions))
        return solutions

    def _save(self, solutions: list[ParetoSolution]) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.pareto_front_path, "w") as f:
            json.dump([s.to_dict() for s in solutions], f, indent=2)

    def load_pareto_front(self) -> list[ParetoSolution]:
        """Load the Pareto front previously written by `run_search`."""
        if not os.path.exists(self.pareto_front_path):
            raise FileNotFoundError(
                f"No Pareto front found at {self.pareto_front_path}. Run run_search() first."
            )
        with open(self.pareto_front_path) as f:
            raw = json.load(f)
        return [ParetoSolution.from_dict(item) for item in raw]

    def select_operating_point(
        self,
        min_precision: float = 0.80,
        min_recall: float = 0.70,
        pareto_front: list[ParetoSolution] | None = None,
    ) -> dict[str, float]:
        """Select the Pareto solution with highest SHAP stability subject to
        precision and recall constraints."""
        front = pareto_front if pareto_front is not None else self.load_pareto_front()
        feasible = [
            s
            for s in front
            if s.objectives["precision"] >= min_precision and s.objectives["recall"] >= min_recall
        ]
        if not feasible:
            raise ValueError(
                f"No Pareto-optimal solution satisfies precision >= {min_precision} "
                f"and recall >= {min_recall}"
            )
        return max(feasible, key=lambda s: s.objectives["shap_stability"]).weights


# ---------------------------------------------------------------------------
# Dynamic weight adjustment based on observed per-model false positive rates
# ---------------------------------------------------------------------------

_DYNAMIC_WEIGHT_MIN = 0.05
_DYNAMIC_WEIGHT_MAX = 0.80
_MIN_FP_FEEDBACK = 10
_PREDICTION_FP_THRESHOLD = 0.5  # model probability above which it "voted" for the alert


def _setup_weight_gauges(model_names: list[str]) -> dict[str, Any]:
    """Create Prometheus gauges for per-model dynamic weights (graceful if absent)."""
    gauges: dict[str, Any] = {}
    try:
        from prometheus_client import Gauge

        for name in model_names:
            metric_name = f"ensemble_dynamic_weight_{name.replace('-', '_').replace(' ', '_')}"
            try:
                gauges[name] = Gauge(
                    metric_name,
                    f"Current dynamic ensemble weight for model {name}",
                )
            except Exception:
                # Gauge already registered (common in test suites); retrieve it
                from prometheus_client import REGISTRY

                for collector in list(REGISTRY._names_to_collectors.values()):
                    if hasattr(collector, "_name") and collector._name == metric_name:
                        gauges[name] = collector
                        break
    except Exception:  # pragma: no cover
        pass
    return gauges


@dataclass
class _ModelObservations:
    """Rolling observation buffer for one model."""

    fp_count: int = 0
    total_count: int = 0

    @property
    def fp_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.fp_count / self.total_count

    def record(self, is_fp: bool) -> None:
        self.total_count += 1
        if is_fp:
            self.fp_count += 1


class EnsembleDynamicWeightController:
    """Adjusts per-model ensemble weights inversely proportional to observed FP rates.

    Feedback from operator-verified false positive annotations is used to penalise
    models that produce frequent false alerts.  Weight updates are exponentially
    smoothed (``ENSEMBLE_WEIGHT_SMOOTHING_ALPHA``, default 0.1) to prevent
    overcorrection on small samples.

    A minimum of ``_MIN_FP_FEEDBACK`` (10) confirmed FP annotations are required
    before any weight is changed from the training-time baseline.

    Systemic reset: when **all** models simultaneously exceed
    ``ENSEMBLE_SYSTEMIC_FP_THRESHOLD`` (default 0.5), weights revert to the
    training-time values and a WARNING is logged.

    Weight bounds: [0.05, 0.80] per model.

    Security: every FP observation requires a non-empty ``annotator_id`` and
    ``audit_trail_id`` so that feedback is linked to the operator's audit trail,
    preventing anonymous manipulation of ensemble weights.
    """

    def __init__(
        self,
        training_weights: dict[str, float],
        session_factory=None,
        smoothing_alpha: float | None = None,
        systemic_fp_threshold: float | None = None,
    ) -> None:
        if not training_weights:
            raise ValueError("training_weights must map at least one model name to a weight")

        self.training_weights: dict[str, float] = dict(training_weights)
        self._session_factory = session_factory
        self._alpha: float = (
            smoothing_alpha
            if smoothing_alpha is not None
            else config.ENSEMBLE_WEIGHT_SMOOTHING_ALPHA
        )
        self._systemic_threshold: float = (
            systemic_fp_threshold
            if systemic_fp_threshold is not None
            else config.ENSEMBLE_SYSTEMIC_FP_THRESHOLD
        )

        self._lock = threading.Lock()
        self._current_weights: dict[str, float] = dict(training_weights)
        self._observations: dict[str, _ModelObservations] = {
            name: _ModelObservations() for name in training_weights
        }
        self._systemic_reset_count: int = 0

        self._gauges = _setup_weight_gauges(list(training_weights))
        self._update_gauges()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_false_positive(
        self,
        wallet: str,
        model_predictions: dict[str, float],
        annotator_id: str,
        audit_trail_id: str,
    ) -> None:
        """Record a confirmed false positive from an authenticated operator.

        Parameters
        ----------
        wallet:
            The wallet address that was falsely flagged (used for logging only).
        model_predictions:
            Per-model predicted wash-trading probability (0–1) at the time of the
            alert.  Keys must match the model names in ``training_weights``.
        annotator_id:
            Non-empty operator identifier — required for authentication.
        audit_trail_id:
            Non-empty audit trail entry ID linking this feedback to the verified
            operator action.  Prevents anonymous weight manipulation.
        """
        if not annotator_id:
            raise ValueError("annotator_id must be a non-empty string")
        if not audit_trail_id:
            raise ValueError("audit_trail_id must be a non-empty string (linked to audit trail)")

        with self._lock:
            for name, obs in self._observations.items():
                pred = model_predictions.get(name, 0.0)
                # Model "voted" for the alert if it predicted above the FP threshold
                is_fp = pred >= _PREDICTION_FP_THRESHOLD
                obs.record(is_fp)

            logger.info(
                "FP feedback recorded for wallet=%s by annotator=%s (audit=%s); "
                "total observations: %s",
                wallet,
                annotator_id,
                audit_trail_id,
                {name: obs.total_count for name, obs in self._observations.items()},
            )
            self._maybe_update_weights()

    def current_weights(self) -> dict[str, float]:
        """Return the current (possibly adjusted) per-model weight vector."""
        with self._lock:
            return dict(self._current_weights)

    def fp_rates(self) -> dict[str, float]:
        """Return the current observed FP rate per model."""
        with self._lock:
            return {name: obs.fp_rate for name, obs in self._observations.items()}

    def observation_counts(self) -> dict[str, int]:
        """Return how many FP feedback events have been recorded per model."""
        with self._lock:
            return {name: obs.total_count for name, obs in self._observations.items()}

    def reset_to_training_weights(self, reason: str = "manual") -> None:
        """Revert dynamic weights to the training-time values."""
        with self._lock:
            self._current_weights = dict(self.training_weights)
            self._update_gauges()
            self._persist_weights(is_systemic_reset=True)
            logger.warning("Ensemble weights reset to training-time values: reason=%s", reason)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_update_weights(self) -> None:
        """Recompute and smooth weights if minimum feedback count is met.

        Must be called while holding ``self._lock``.
        """
        total = min(obs.total_count for obs in self._observations.values())
        if total < _MIN_FP_FEEDBACK:
            return

        rates = {name: obs.fp_rate for name, obs in self._observations.items()}

        # Systemic reset: all models simultaneously above threshold
        if all(r >= self._systemic_threshold for r in rates.values()):
            self._systemic_reset_count += 1
            self._current_weights = dict(self.training_weights)
            self._update_gauges()
            self._persist_weights(is_systemic_reset=True)
            logger.warning(
                "Systemic FP rate detected (all models >= %.2f). "
                "Resetting to training-time weights. Reset count: %d",
                self._systemic_threshold,
                self._systemic_reset_count,
            )
            return

        # Compute raw new weights: inversely proportional to FP rate
        _eps = 1e-6
        raw = {name: 1.0 / (rate + _eps) for name, rate in rates.items()}
        total_raw = sum(raw.values())
        normalised = {name: v / total_raw for name, v in raw.items()}

        # Clamp to [_DYNAMIC_WEIGHT_MIN, _DYNAMIC_WEIGHT_MAX] then re-normalise
        clamped = {name: max(_DYNAMIC_WEIGHT_MIN, min(_DYNAMIC_WEIGHT_MAX, w))
                   for name, w in normalised.items()}
        total_clamped = sum(clamped.values())
        target = {name: v / total_clamped for name, v in clamped.items()}

        # Exponential smoothing
        for name in self._current_weights:
            old = self._current_weights[name]
            new = target.get(name, old)
            self._current_weights[name] = self._alpha * new + (1.0 - self._alpha) * old

        # Re-normalise after smoothing so weights always sum to 1
        total_w = sum(self._current_weights.values())
        for name in self._current_weights:
            self._current_weights[name] /= total_w

        self._update_gauges()
        self._persist_weights(is_systemic_reset=False)
        logger.info("Dynamic ensemble weights updated: %s", self._current_weights)

    def _update_gauges(self) -> None:
        for name, weight in self._current_weights.items():
            g = self._gauges.get(name)
            if g is not None:
                try:
                    g.set(weight)
                except Exception:  # pragma: no cover
                    pass

    def _persist_weights(self, *, is_systemic_reset: bool) -> None:
        if self._session_factory is None:
            return
        try:
            from detection.persistence import EnsembleWeightRecord

            with self._session_factory() as session:
                for name, weight in self._current_weights.items():
                    session.add(
                        EnsembleWeightRecord(
                            model_name=name,
                            weight=weight,
                            fp_rate=self._observations[name].fp_rate,
                            observation_count=self._observations[name].total_count,
                            is_systemic_reset=is_systemic_reset,
                        )
                    )
                session.commit()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to persist ensemble weight history: %s", exc)
