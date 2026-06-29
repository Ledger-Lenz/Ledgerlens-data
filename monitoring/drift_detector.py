"""Sliding window covariance shift detection using Maximum Mean Discrepancy (MMD)."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Gauge
    _drift_gauge = Gauge("ledgerlens_feature_drift_detected", "1=drift detected, 0=stable")
except Exception:  # pragma: no cover
    _drift_gauge = None


@dataclass
class DriftReport:
    mmd_per_feature: dict[str, float]
    drift_detected: bool

    def to_dict(self) -> dict:
        return {"drift_detected": self.drift_detected, "mmd_per_feature": self.mmd_per_feature}


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * bandwidth ** 2))


def _mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute unbiased MMD² with RBF kernel; bandwidth via median heuristic."""
    all_points = np.vstack([X, Y])
    dists = np.linalg.norm(all_points[:, None] - all_points[None, :], axis=-1)
    bandwidth = float(np.median(dists[dists > 0])) or 1.0

    kxx = _rbf_kernel(X, X, bandwidth)
    kyy = _rbf_kernel(Y, Y, bandwidth)
    kxy = _rbf_kernel(X, Y, bandwidth)

    n, m = len(X), len(Y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()


class CovarianceShiftDetector:
    """Detects feature distribution drift between a reference and current window using MMD."""

    def __init__(self, threshold: float = 0.05) -> None:
        try:
            from config import Config
            self._ref_hours = Config.DRIFT_REFERENCE_WINDOW_HOURS
            self._test_hours = Config.DRIFT_TEST_WINDOW_HOURS
            self._interval = Config.DRIFT_CHECK_INTERVAL_MINUTES
        except Exception:
            self._ref_hours = 168
            self._test_hours = 1
            self._interval = 30
        self.threshold = threshold

    def detect(self, reference: np.ndarray, current: np.ndarray, feature_names: list[str] | None = None) -> DriftReport:
        """Compare reference and current windows per feature; return DriftReport.

        Args:
            reference: 2-D array of shape (n_ref, n_features).
            current:   2-D array of shape (n_cur, n_features).
            feature_names: Optional list of feature name strings.
        """
        n_features = reference.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        mmd_scores: dict[str, float] = {}
        for i, name in enumerate(names):
            ref_col = reference[:, i]
            cur_col = current[:, i]
            if ref_col.std() < 1e-8:  # skip near-zero-variance features
                continue
            mmd_scores[name] = _mmd(ref_col.reshape(-1, 1), cur_col.reshape(-1, 1))

        drift_detected = any(v > self.threshold for v in mmd_scores.values())

        if _drift_gauge is not None:
            _drift_gauge.set(1 if drift_detected else 0)

        if drift_detected:
            top5 = sorted(mmd_scores, key=mmd_scores.get, reverse=True)[:5]
            logger.warning("Feature drift detected. Top drifted features: %s", top5)

        return DriftReport(mmd_per_feature=mmd_scores, drift_detected=drift_detected)
