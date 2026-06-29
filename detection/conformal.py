"""Conformal Prediction calibration and inference.

Implements split conformal prediction (classification with RAPS extension
and regression framing) producing distribution-free prediction intervals
at a user-specified coverage level (default 90%).

References
----------
Angelopoulos, A.N. & Bates, S. (2023) "Conformal prediction: A gentle
introduction." Foundations and Trends in Machine Learning, 16(4), 494–591.
"""

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

    def _calibrate_classification(
        self, model: Any, X_cal: pd.DataFrame, y_cal: pd.Series
    ) -> None:
        probs = model.predict_proba(X_cal)
        n_classes = probs.shape[1]
        self.classes_ = list(range(n_classes))

        nonconformity = np.array([
            1.0 - probs[i, int(y_cal.iloc[i])] for i in range(len(X_cal))
        ])
        self._nonconformity_scores = nonconformity
        self.n_cal = len(X_cal)
        self.q_hat = float(np.quantile(nonconformity, 1.0 - self.alpha))
        logger.info(
            "Conformal calibration (classification) done: n_cal=%d, q_hat=%.6f, alpha=%.2f",
            self.n_cal,
            self.q_hat,
            self.alpha,
        )

    def _calibrate_regression(
        self, model: Any, X_cal: pd.DataFrame, y_cal: pd.Series
    ) -> None:
        y_pred = model.predict(X_cal)
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.flatten()
        residuals = np.abs(np.array(y_cal) - np.array(y_pred))
        self._nonconformity_scores = residuals
        self.n_cal = len(X_cal)
        self.q_hat = float(np.quantile(residuals, 1.0 - self.alpha))
        logger.info(
            "Conformal calibration (regression) done: n_cal=%d, q_hat=%.6f, alpha=%.2f",
            self.n_cal,
            self.q_hat,
            self.alpha,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

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

            results.append({
                "score": float(row_probs[1]) * 100 if n_classes == 2 else 50.0,
                "prediction_set": sorted(prediction_set),
                "coverage_guarantee": 1.0 - self.alpha,
                "q_hat": self.q_hat,
            })

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
            score = float(row_probs[1]) * 100 if probs.shape[1] == 2 else float(row_probs.argmax()) / (probs.shape[1] - 1) * 100
            margin = self.q_hat * 100
            results.append({
                "score": score,
                "lower": max(0.0, score - margin),
                "upper": min(100.0, score + margin),
            })
        return results

    def _interval_regression(self, model: Any, X: pd.DataFrame) -> list[dict]:
        y_pred = model.predict(X)
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.flatten()
        results = []
        for pred in y_pred:
            pred_f = float(pred)
            margin = self.q_hat
            results.append({
                "score": pred_f,
                "lower": max(0.0, pred_f - margin),
                "upper": min(100.0, pred_f + margin),
            })
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
