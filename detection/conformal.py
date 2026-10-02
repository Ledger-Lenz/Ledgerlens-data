"""Conformal Prediction calibration and inference.

# =============================================================================
# Issue #859 — Formalize conformal prediction coverage guarantees under
# distribution shift
# https://github.com/Ledger-Lenz/Ledgerlens-data/issues/859
#
# ─── PROBLEM ─────────────────────────────────────────────────────────────────
#
# The current ConformalCalibrator assumes exchangeability of calibration and
# test data. When drift_monitor.py detects covariate shift (PSI > threshold),
# the conformal coverage guarantee is silently invalidated — the q_hat computed
# on the original calibration set is no longer valid for the shifted
# distribution, but no user-facing signal is emitted.
#
# ─── PROPOSED IMPLEMENTATION ─────────────────────────────────────────────────
#
# Step 1 — Weighted/adaptive conformal calibration (ACI or Mondrian)
# ------------------------------------------------------------------
# Two complementary approaches; implement both, selecting via config:
#
# APPROACH A: Covariate-shift-weighted conformal (Tibshirani et al. 2019)
# -----------------------------------------------------------------------
# Weight each calibration sample by an importance ratio:
#   w_i = p_test(x_i) / p_cal(x_i)
#
# In practice, estimate weights using a density ratio classifier:
#   fit classifier C on (X_cal labeled 0, X_test labeled 1)
#   w_i = C.predict_proba(x_i)[1] / C.predict_proba(x_i)[0]
#
# Replace the unweighted quantile in _calibrate_*:
#   # Weighted quantile
#   w = np.array(weights) / np.array(weights).sum()
#   sorted_idx = np.argsort(nonconformity)
#   cum_w = np.cumsum(w[sorted_idx])
#   q_hat = nonconformity[sorted_idx][np.searchsorted(cum_w, 1 - alpha)]
#
# APPROACH B: Mondrian conformal by drift regime
# ------------------------------------------------
# Partition calibration samples into drift regimes based on DriftReport:
#   regime 0 = PSI < 0.1   (no drift)
#   regime 1 = 0.1 <= PSI < 0.25  (moderate drift)
#   regime 2 = PSI >= 0.25  (significant drift)
#
# Compute a separate q_hat per regime. At inference time, look up the
# regime of the current request (from LiveDriftMonitor.update()) and use
# the matching q_hat:
#
#   class MondricanConformalCalibrator(ConformalCalibrator):
#       def calibrate_by_regime(self, model, X_cal, y_cal, regimes):
#           for regime in set(regimes):
#               mask = [r == regime for r in regimes]
#               X_r, y_r = X_cal[mask], y_cal[mask]
#               # Compute q_hat_r for this regime
#               self._q_hat_by_regime[regime] = ...
#
#       def predict_set(self, model, X, drift_regimes=None):
#           # Use regime-specific q_hat
#           ...
#
# Step 2 — Wire drift_monitor.py output into calibration weighting
# ----------------------------------------------------------------
# Add a method to ConformalCalibrator:
#
#   def recalibrate_with_drift_weights(
#       self,
#       model,
#       X_cal: pd.DataFrame,
#       y_cal: pd.Series,
#       drift_report: DriftReport,
#       psi_cap: float = 5.0,
#   ) -> None:
#       """Recompute q_hat weighting calibration samples by their drift exposure.
#
#       For each calibration sample, the weight is 1 / (1 + max_psi_of_its_features).
#       Samples from high-drift feature directions are down-weighted so the
#       quantile is driven by the low-drift (more reliable) samples.
#       """
#       drifted_features = {f["feature"]: f["psi"] for f in drift_report.features
#                           if f["drift_flag"]}
#       weights = []
#       for i in range(len(X_cal)):
#           row_psi = max(
#               (drifted_features.get(col, 0.0) for col in X_cal.columns),
#               default=0.0
#           )
#           weights.append(1.0 / (1.0 + min(row_psi, psi_cap)))
#       # Use weighted quantile with these weights
#       self._recalibrate_weighted(model, X_cal, y_cal, np.array(weights))
#
# Step 3 — Coverage-confidence flag on every emitted risk score
# -------------------------------------------------------------
# Add a `coverage_confidence` field to every result returned by
# predict_set() and predict_with_interval():
#
#   {
#     "score": 74.2,
#     "prediction_set": [1],
#     "coverage_guarantee": 0.90,
#     "q_hat": 0.23,
#     "coverage_confidence": "low",  ← NEW: "high" | "moderate" | "low"
#     "drift_context": {             ← NEW: populated when drift detected
#       "drifted_features": ["benford_mad_24h", "round_trip_frequency"],
#       "max_psi": 0.41,
#     }
#   }
#
# coverage_confidence is computed from the current PSI:
#   PSI < 0.1  → "high"     (guarantee is well-founded)
#   PSI < 0.25 → "moderate" (moderate drift, coverage may be slightly off)
#   PSI >= 0.25 → "low"     (significant drift, guarantee is unreliable)
#
# Step 4 — Surface low-confidence flags in forensic_report.py
# -----------------------------------------------------------
# forensic_report.py should include a "conformal_coverage_warning" section
# when any scored transaction's coverage_confidence is "low":
#
#   {
#     "conformal_coverage_warning": {
#       "active": true,
#       "reason": "PSI drift detected on features: benford_mad_24h (PSI=0.41)",
#       "affected_scores": 12,
#       "recommendation": "Recalibrate conformal predictor on recent data."
#     }
#   }
#
# ─── EMPIRICAL COVERAGE TEST ─────────────────────────────────────────────────
#
# Add tests/test_conformal_under_shift.py:
#
#   def test_coverage_under_covariate_shift():
#       """Empirical coverage on a synthetic shifted test set stays within
#       ±2% of the nominal target (90% coverage)."""
#       rng = np.random.default_rng(42)
#       # Calibration: N(0, 1) features
#       X_cal = pd.DataFrame(rng.normal(0, 1, (500, 10)),
#                            columns=[f"f{i}" for i in range(10)])
#       y_cal = pd.Series((X_cal.sum(axis=1) > 0).astype(int))
#       # Shifted test: N(2, 1) features (covariate shift)
#       X_test = pd.DataFrame(rng.normal(2, 1, (200, 10)),
#                             columns=[f"f{i}" for i in range(10)])
#       y_test = pd.Series((X_test.sum(axis=1) > 0).astype(int))
#
#       # Weighted calibration
#       calibrator = ConformalCalibrator(alpha=0.10)
#       calibrator.recalibrate_with_drift_weights(model, X_cal, y_cal, drift_report)
#       results = calibrator.predict_set(model, X_test)
#       empirical_coverage = np.mean([
#           y_test.iloc[i] in r["prediction_set"] for i, r in enumerate(results)
#       ])
#       assert abs(empirical_coverage - 0.90) <= 0.02
#
# ─── ACCEPTANCE CRITERIA MAPPING ─────────────────────────────────────────────
#
#  ✅  Empirical coverage stays within ±2% of nominal target under shift
#      → tests/test_conformal_under_shift.py
#
#  ✅  Flagged low-confidence scores visible in forensic_report.py
#      → coverage_confidence field in every result; warning in forensic report
#
#  ✅  Unit tests for weighting function with adversarially shifted calibration data
#      → tests/test_conformal_under_shift.py::test_weighting_with_adversarial_shift
#
# ─── FILES TO MODIFY ─────────────────────────────────────────────────────────
#
#   detection/conformal.py          ← (THIS FILE) weighted calibration,
#                                      Mondrian by drift regime,
#                                      coverage_confidence field
#   detection/drift_monitor.py      ← export DriftReport.max_psi() helper
#   forensic_report.py              ← add conformal_coverage_warning section
#   tests/test_conformal_under_shift.py  ← new test file
#   config.py                       ← CONFORMAL_DRIFT_RECAL_PSI_THRESHOLD (0.1),
#                                      CONFORMAL_WEIGHTED_MODE ("weighted"|"mondrian")
#
# =============================================================================

Implements split conformal prediction (classification with RAPS extension
and regression framing) producing distribution-free prediction intervals
at a user-specified coverage level (default 90%).

References
----------
Angelopoulos, A.N. & Bates, S. (2023) "Conformal prediction: A gentle
introduction." Foundations and Trends in Machine Learning, 16(4), 494–591.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

RAPS_LAMBDA: float = 0.1
RAPS_K0: int = 5


class CalibrationIntegrityError(Exception):
    """Raised when a calibration artifact's SHA-256 does not match on load."""


class ConformalCalibrator:
    """Calibrate and apply conformal prediction for a trained classifier.

    Two modes:
      - **classification** (default): uses RAPS nonconformity scores.
        ``predict_set`` returns a set of class labels guaranteed to contain
        the true label with probability >= ``1 - alpha``.
      - **regression**: uses absolute residual nonconformity.
        ``predict_with_interval`` returns ``[score - q_hat, score + q_hat]``.

    Parameters
    ----------
    alpha:
        Desired miscoverage level (default 0.10 → 90% coverage).
    random_state:
        Seed for reproducible RAPS penalty tie-breaking.
    """

    def __init__(self, alpha: float = 0.10, random_state: int = 42) -> None:
        self.alpha: float = alpha
        self.random_state: int = random_state
        self.q_hat: float | None = None
        self.n_cal: int | None = None
        self.feature_columns: list[str] | None = None
        self.classes_: list[int] | None = None
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model: Any,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        alpha: float | None = None,
    ) -> None:
        """Compute the conformal threshold ``q_hat`` from a calibration split.

        For **classification** mode (model has ``predict_proba``):
        nonconformity score = 1 - softmax score of the true class.

        For **regression** mode (model has ``predict``, returning a scalar
        risk score 0-100): nonconformity score = absolute residual.

        Parameters
        ----------
        model:
            A fitted classifier with ``predict_proba`` or a regressor with
            ``predict``.
        X_cal:
            Calibration feature matrix.
        y_cal:
            Calibration labels (int 0/1 for classification, float for
            regression).
        alpha:
            Override the instance's ``alpha`` for this calibration.
        """
        if alpha is not None:
            self.alpha = alpha

        n = len(X_cal)
        if n == 0:
            raise ValueError("Calibration split is empty")

        self.feature_columns = list(X_cal.columns)

        # Determine mode and compute nonconformity scores
        if hasattr(model, "predict_proba"):
            self._mode = "classification"
            self._calibrate_classification(model, X_cal, y_cal)
        elif hasattr(model, "predict"):
            self._mode = "regression"
            self._calibrate_regression(model, X_cal, y_cal)
        else:
            raise TypeError(
                "model must have predict_proba (classification) or predict (regression)"
            )

    def _calibrate_classification(self, model: Any, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        probs = model.predict_proba(X_cal)
        n_classes = probs.shape[1]
        self.classes_ = list(range(n_classes))

        nonconformity = np.array([1.0 - probs[i, int(y_cal.iloc[i])] for i in range(len(X_cal))])
        self._nonconformity_scores = nonconformity
        self.n_cal = len(X_cal)
        self.q_hat = float(np.quantile(nonconformity, self._coverage_guarantee()))
        logger.info(
            "Conformal calibration (classification) done: n_cal=%d, q_hat=%.6f, alpha=%.2f",
            self.n_cal,
            self.q_hat,
            self.alpha,
        )

    def _calibrate_regression(self, model: Any, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        y_pred = model.predict(X_cal)
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.flatten()
        residuals = np.abs(np.array(y_cal) - np.array(y_pred))
        self._nonconformity_scores = residuals
        self.n_cal = len(X_cal)
        self.q_hat = float(np.quantile(residuals, self._coverage_guarantee()))
        logger.info(
            "Conformal calibration (regression) done: n_cal=%d, q_hat=%.6f, alpha=%.2f",
            self.n_cal,
            self.q_hat,
            self.alpha,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _coverage_guarantee(self) -> float:
        """Return the coverage guarantee (``1 - alpha``) clamped to ``[0, 1]``.

        ``coverage_guarantee`` is surfaced as a probability in every
        ``RiskScore`` record and consumed by the API/dashboard and
        ``ledgerlens-core``'s shared type. A configured ``alpha`` outside
        ``[0, 1]`` would yield a nonsensical coverage value, so it is clamped
        here with a logged warning rather than propagated downstream.
        """
        coverage = 1.0 - self.alpha
        clamped = min(1.0, max(0.0, coverage))
        if clamped != coverage:
            logger.warning(
                "coverage_guarantee %.4f is outside [0, 1] (alpha=%.4f); clamping to %.4f",
                coverage,
                self.alpha,
                clamped,
            )
        return clamped

    def predict_set(self, model: Any, X: pd.DataFrame) -> list[dict]:
        """Return a prediction set for each row using RAPS.

        Only available in classification mode.

        Returns a list of dicts, one per row:
            ``{"score": float, "prediction_set": list[int],
              "coverage_guarantee": float, "q_hat": float}``
        """
        if self.q_hat is None:
            raise RuntimeError("ConformalCalibrator has not been calibrated yet")

        if self._mode != "classification":
            raise RuntimeError("predict_set is only available in classification mode")

        if not hasattr(model, "predict_proba"):
            raise TypeError("model must have predict_proba for classification mode")

        probs = model.predict_proba(X)
        n_classes = probs.shape[1]
        coverage_guarantee = self._coverage_guarantee()

        results = []
        for row_probs in probs:
            sorted_idx = np.argsort(row_probs)[::-1]
            cumulative = 0.0
            prediction_set: list[int] = []
            penalty = 0.0
            for k, idx in enumerate(sorted_idx):
                softmax_k = float(row_probs[idx])
                cumulative += softmax_k
                regularized_score = cumulative - penalty

                if regularized_score > 1.0 - self.q_hat or k < 1:
                    prediction_set.append(int(idx))
                else:
                    break

                if k >= RAPS_K0:
                    penalty += RAPS_LAMBDA

            results.append(
                {
                    "score": float(row_probs[1]) * 100 if n_classes == 2 else 50.0,
                    "prediction_set": sorted(prediction_set),
                    "coverage_guarantee": coverage_guarantee,
                    "q_hat": self.q_hat,
                }
            )

        return results

    def predict_with_interval(self, model: Any, X: pd.DataFrame) -> list[dict]:
        """Return a prediction interval for each row.

        Available in both modes:
          - **classification**: derives interval from softmax scores.
          - **regression**: ``[predict - q_hat, predict + q_hat]``.

        Returns a list of dicts, one per row:
            ``{"score": float, "lower": float, "upper": float}``
        """
        if self.q_hat is None:
            raise RuntimeError("ConformalCalibrator has not been calibrated yet")

        if hasattr(model, "predict_proba") and self._mode == "classification":
            return self._interval_classification(model, X)
        elif hasattr(model, "predict"):
            return self._interval_regression(model, X)
        else:
            raise TypeError("model must have predict or predict_proba")

    def _interval_classification(self, model: Any, X: pd.DataFrame) -> list[dict]:
        probs = model.predict_proba(X)
        results = []
        for row_probs in probs:
            score = (
                float(row_probs[1]) * 100
                if probs.shape[1] == 2
                else float(row_probs.argmax()) / (probs.shape[1] - 1) * 100
            )
            margin = self.q_hat * 100
            results.append(
                {
                    "score": score,
                    "lower": max(0.0, score - margin),
                    "upper": min(100.0, score + margin),
                }
            )
        return results

    def _interval_regression(self, model: Any, X: pd.DataFrame) -> list[dict]:
        y_pred = model.predict(X)
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.flatten()
        results = []
        for pred in y_pred:
            pred_f = float(pred)
            margin = self.q_hat
            results.append(
                {
                    "score": pred_f,
                    "lower": max(0.0, pred_f - margin),
                    "upper": min(100.0, pred_f + margin),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence (auditable JSON + SHA-256 integrity check)
    # ------------------------------------------------------------------

    def _compute_sha256(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def save(self, path: str) -> None:
        """Persist the calibration artifact as a human-readable JSON file.

        The payload includes a ``sha256`` field computed over the sorted JSON
        representation of all other fields, providing tamper evidence.
        """
        if self.q_hat is None:
            raise RuntimeError("Cannot save — calibrator has not been calibrated")

        payload: dict[str, Any] = {
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "n_cal": self.n_cal,
            "random_state": self.random_state,
            "mode": getattr(self, "_mode", "classification"),
            "feature_columns": self.feature_columns,
            "classes": self.classes_,
        }

        content = {k: v for k, v in payload.items() if k != "sha256"}
        content["sha256"] = self._compute_sha256(content)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(content, f, indent=2)

        logger.info("Saved calibration artifact to %s (sha256=%s)", path, content["sha256"])

    @classmethod
    def load(cls, path: str) -> "ConformalCalibrator":
        """Load a calibration artifact from a JSON file.

        Verifies the embedded SHA-256 before returning the calibrator.
        Raises ``CalibrationIntegrityError`` on mismatch.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration artifact not found: {path}")

        with open(path) as f:
            content = json.load(f)

        stored_sha = content.pop("sha256", None)
        if stored_sha is None:
            raise CalibrationIntegrityError(
                "Calibration artifact is missing sha256 field — cannot verify integrity"
            )

        computed = cls._compute_sha256_static(content)
        if computed != stored_sha:
            raise CalibrationIntegrityError(
                f"Calibration artifact SHA-256 mismatch: stored={stored_sha}, computed={computed}"
            )

        calibrator = cls(alpha=content["alpha"], random_state=content.get("random_state", 42))
        calibrator.q_hat = content["q_hat"]
        calibrator.n_cal = content["n_cal"]
        calibrator.feature_columns = content.get("feature_columns")
        calibrator.classes_ = content.get("classes")
        calibrator._mode = content.get("mode", "classification")

        logger.info(
            "Loaded calibration artifact from %s (q_hat=%.6f, n_cal=%d, alpha=%.2f)",
            path,
            calibrator.q_hat,
            calibrator.n_cal or 0,
            calibrator.alpha,
        )
        return calibrator

    @staticmethod
    def _compute_sha256_static(content: dict) -> str:
        raw = json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()
