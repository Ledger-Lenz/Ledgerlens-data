"""Platt scaling calibration for raw detector scores (issue #288).

Platt scaling fits a logistic regression on held-out (score, label) pairs to
map raw anomaly detector outputs to calibrated probabilities in [0, 1].

Theory
------
Given a raw score s, Platt scaling learns:
    P(y=1 | s) = 1 / (1 + exp(A·s + B))

where A and B are fitted by MLE on a held-out calibration set (Platt, 1999).
This preserves rank order (monotonically increasing with raw score) while
producing well-calibrated probability estimates.

When to retrain the calibrator
-------------------------------
Retrain whenever the underlying detector is retrained or when the Expected
Calibration Error (ECE) on recent production labels exceeds 0.05. The
calibrator is lightweight (two parameters) so retraining is cheap.

ECE interpretation
------------------
ECE measures the average gap between predicted confidence and actual accuracy
across equal-width probability bins. ECE = 0 is perfect calibration; ECE < 0.05
is considered acceptable for production use (Guo et al., 2017).

Public API
----------
PlattCalibrator
    .fit(scores, labels)
    .calibrate(scores) -> np.ndarray
    .save(path)
    .load(path)         (classmethod)
    .ece               -> float (after fit)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from config import config

logger = logging.getLogger(__name__)

_ECE_N_BINS = 10


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = _ECE_N_BINS) -> float:
    """Compute Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


class PlattCalibrator:
    """Platt scaling wrapper around scikit-learn LogisticRegression.

    Platt scaling fits a two-parameter sigmoid on held-out detector scores and
    binary labels. The resulting model is monotonically increasing with raw
    score (rank order preserved) and outputs calibrated probabilities in [0, 1].

    When to retrain: whenever the base detector is retrained or ECE on recent
    production data exceeds 0.05.

    ECE interpretation: 0 = perfect calibration; < 0.05 = acceptable for
    production alerting and forensic reporting.
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=config.CALIBRATION_RANDOM_SEED,
        )
        self._ece: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        """Fit the calibrator on held-out (score, label) pairs.

        Args:
            scores: 1-D array of raw detector scores.
            labels: 1-D binary array (0 = benign, 1 = wash trade).

        Returns:
            self (for chaining).
        """
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels, dtype=int)
        self._lr.fit(scores, labels)
        probs = self._lr.predict_proba(scores)[:, 1]
        self._ece = _compute_ece(probs, labels)
        logger.info("Platt calibration ECE = %.4f", self._ece)
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in [0, 1].

        Args:
            scores: 1-D array of raw scores.

        Returns:
            1-D array of calibrated probabilities, same length as ``scores``.
        """
        check_is_fitted(self._lr)
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(scores)[:, 1]

    @property
    def ece(self) -> float | None:
        """Expected Calibration Error on the training split, or None before fit."""
        return self._ece

    def save(self, path: str | os.PathLike) -> None:
        """Persist calibrator to ``path`` using pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved PlattCalibrator to %s", path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "PlattCalibrator":
        """Load a persisted calibrator from ``path``."""
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301 — trusted internal model artifact
        if not isinstance(obj, cls):
            raise TypeError(f"Expected PlattCalibrator, got {type(obj)}")
        return obj

    def plot_calibration_curve(self, scores: np.ndarray, labels: np.ndarray, path: str) -> None:
        """Emit a reliability diagram (calibration curve) to ``path``.

        Args:
            scores: Raw detector scores used for evaluation.
            labels: True binary labels.
            path: Output file path (PNG).
        """
        probs = self.calibrate(scores)
        bins = np.linspace(0.0, 1.0, _ECE_N_BINS + 1)
        mean_probs, mean_labels = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.any():
                mean_probs.append(float(probs[mask].mean()))
                mean_labels.append(float(labels[mask].mean()))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_probs, mean_labels, "o-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Reliability diagram (Platt scaling)")
        ax.legend()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved calibration curve to %s", path)
