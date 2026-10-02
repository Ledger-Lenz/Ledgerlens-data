"""Feature drift detection using Population Stability Index (PSI).

PSI is the industry-standard metric for detecting feature drift in tabular ML
systems (see https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf for a
reference implementation). It measures how much the current feature distribution
has shifted from the reference (training-time) distribution.

PSI formula:
    PSI = Σ_i (observed_i - expected_i) * ln(observed_i / expected_i)
where i iterates over bins of each feature's value distribution.

This module also provides a unified triage view that correlates upstream
feature drift with model-output drift, so an operator investigating an alert
can immediately tell whether output drift is explained by feature drift or is
unexplained. See ``build_triage_view`` and ``TRIAGE_DECISION_TREE``.
"""

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

PSI_MODERATE_DRIFT_THRESHOLD = 0.25
PSI_EPSILON = 1e-4

REPORTS_DIR = "reports"

# Documented triage decision tree based on the combination of signals.
# Keys are the correlation states produced by ``correlate_drift_signals``.
TRIAGE_DECISION_TREE: dict[str, str] = {
    "both_drifted": (
        "Both input feature drift and output drift are flagged in the same "
        "window. Output drift is likely EXPLAINED by upstream feature drift. "
        "Action: inspect the drifted features, confirm the upstream data "
        "source change, and plan a retrain on fresh data. No model bug "
        "investigation required unless feature drift is expected/benign."
    ),
    "only_output_drifted": (
        "Output drift is flagged but no tracked input feature drifted. Output "
        "drift is UNEXPLAINED by feature drift. Action: treat as a potential "
        "model or pipeline defect — check scoring code, label leakage, "
        "post-processing, and untracked inputs before retraining."
    ),
    "only_input_drifted": (
        "Input feature drift is flagged but output drift is not. The model is "
        "currently robust to the upstream shift. Action: monitor; no immediate "
        "retrain required, but watch for output drift developing in later "
        "windows."
    ),
    "no_drift": (
        "Neither input feature drift nor output drift is flagged. Action: no "
        "triage required."
    ),
}


@dataclass
class DriftReport:
    features: list[dict] = field(default_factory=list)
    any_drift_detected: bool = False

    def to_dict(self) -> dict:
        return {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "any_drift_detected": self.any_drift_detected,
            "n_features_checked": len(self.features),
            "n_features_drifted": sum(1 for f in self.features if f["drift_flag"]),
            "features": self.features,
        }


def compute_psi(
    expected_proportions: np.ndarray,
    observed_proportions: np.ndarray,
    epsilon: float = PSI_EPSILON,
) -> float:
    """Compute PSI between two sets of bin proportions.

    Arrays are clipped to >= epsilon to prevent log(0) and division-by-zero
    errors, then re-normalised to sum to 1.0.
    """

    expected = np.maximum(expected_proportions, epsilon)
    observed = np.maximum(observed_proportions, epsilon)

    expected = expected / expected.sum()
    observed = observed / observed.sum()

    psi = np.sum((observed - expected) * np.log(observed / expected))
    return float(psi)


def correlate_drift_signals(
    feature_drift: bool,
    output_drift: bool,
    drifted_features: list[str] | None = None,
) -> dict:
    """Correlate feature-drift and model-output-drift signals for one window.

    When output drift is flagged, this checks whether any tracked input
    feature also drifted in the same window, producing one of three
    drift-combination states:

    * ``both_drifted`` — output drift is explained by upstream feature drift.
    * ``only_output_drifted`` — output drift is unexplained by feature drift.
    * ``only_input_drifted`` — feature drift without output drift.

    Returns a dict with the ``state``, the raw signals, the drifted feature
    names, and the matching ``triage`` guidance from ``TRIAGE_DECISION_TREE``.
    """

    drifted_features = list(drifted_features or [])

    if feature_drift and output_drift:
        state = "both_drifted"
    elif output_drift:
        state = "only_output_drifted"
    elif feature_drift:
        state = "only_input_drifted"
    else:
        state = "no_drift"

    return {
        "state": state,
        "feature_drift": bool(feature_drift),
        "output_drift": bool(output_drift),
        "drifted_features": drifted_features,
        "output_drift_explained": state == "both_drifted",
        "triage": TRIAGE_DECISION_TREE[state],
    }


def build_triage_view(
    feature_report: DriftReport | dict | None,
    output_drift: bool,
    output_metric: str | None = None,
) -> dict:
    """Build a single unified triage view combining both drift signals.

    Accepts either a ``DriftReport`` or its ``to_dict()`` form for the
    feature-drift side, plus the model-output-drift flag, and returns a
    dashboard/report-ready payload with the correlation annotation.
    """

    if isinstance(feature_report, DriftReport):
        feature_report = feature_report.to_dict()
    feature_report = feature_report or {}

    features = feature_report.get("features", [])
    drifted_features = [f["feature"] for f in features if f.get("drift_flag")]
    feature_drift = bool(feature_report.get("any_drift_detected", False)) or bool(
        drifted_features
    )

    correlation = correlate_drift_signals(
        feature_drift=feature_drift,
        output_drift=output_drift,
        drifted_features=drifted_features,
    )

    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "feature_drift": {
            "any_drift_detected": feature_drift,
            "n_features_drifted": len(drifted_features),
            "drifted_features": drifted_features,
            "features": features,
        },
        "output_drift": {
            "drift_detected": bool(output_drift),
            "metric": output_metric,
        },
        "correlation": correlation,
    }


class DriftMonitor:
    """Offline PSI computation over a feature matrix."""

    def __init__(self, reference_distribution: dict[str, dict]):
        self.reference = reference_distribution

    def compute(self, current_data: pd.DataFrame) -> DriftReport:
        features = []
        any_drift = False

        for col in current_data.columns:
            if col not in self.reference:
                logger.warning(
                    "Skipping drift computation for feature %r: no reference "
                    "distribution in model metadata (feature may be new since "
                    "the last training run)",
                    col,
                )
                continue

            ref = self.reference[col]
            bin_edges = np.array(ref["bin_edges"])
            expected = np.array(ref["expected_proportions"])

            col_data = current_data[col].dropna().values
            if len(col_data) == 0:
                continue

            n_bins = len(bin_edges) - 1
            bin_indices = np.digitize(col_data, bins=bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            counts = np.bincount(bin_indices, minlength=n_bins)
            total = counts.sum()
            observed = counts / total if total > 0 else np.zeros(n_bins, dtype=float)

            psi = compute_psi(expected, observed)

            drift_flag = psi >= PSI_MODERATE_DRIFT_THRESHOLD
            if drift_flag:
                any_drift = True
                # Only include features that breached the threshold
                features.append({"feature": col, "psi": float(psi), "drift_flag": drift_flag})

        # Sort by PSI descending (worst-drifted features first)
        features.sort(key=lambda f: f["psi"], reverse=True)

        report = DriftReport(features=features, any_drift_detected=any_drift)
        self._write_report(report)
        return report

    def _write_report(self, report: DriftReport) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        return path


class LiveDriftMonitor:
    """Live PSI monitoring for streaming scored feature rows.

    Computes per-feature PSI against a reference distribution using a rolling
    window of the last *window_size* observed values.

    Returns:
        list[str]: feature names where PSI > threshold.
    """

    def __init__(
        self,
        reference_path: str,
        threshold: float | None = None,
        window_size: int | None = None,
    ) -> None:
        # Import config lazily so this module can be unit-tested even in
        # minimal environments where optional deps (python-dotenv) are absent.
        try:
            from config import config  # type: ignore

            self.threshold = config.DRIFT_PSI_THRESHOLD if threshold is None else float(threshold)
            self.window_size = config.DRIFT_WINDOW_SIZE if window_size is None else int(window_size)
        except Exception:  # pragma: no cover
            self.threshold = 0.2 if threshold is None else float(threshold)
            self.window_size = 1000 if window_size is None else int(window_size)

        with open(reference_path, encoding="utf-8") as f:
            raw = json.load(f)

        # Support either the raw dict or wrapper under "feature_distributions".
        self.reference: dict[str, dict] = raw.get("feature_distributions", raw)

        self._buffers: dict[str, deque[float]] = {}

        # Prometheus gauge (optional).
        try:
            from prometheus_client import Gauge

            self._psi_gauge = Gauge(
                "feature_psi",
                "Population Stability Index (PSI) per feature (live monitor)",
                ["feature"],
            )
        except Exception:  # pragma: no cover
            self._psi_gauge = None

    def update(self, feature_row: pd.Series) -> list[str]:
        drifted: list[str] = []

        for feature, ref in self.reference.items():
            if feature not in feature_row.index:
                continue

            val = feature_row.get(feature)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            if feature not in self._buffers:
                self._buffers[feature] = deque(maxlen=self.window_size)
            self._buffers[feature].append(float(val))

            # Avoid noisy PSI with tiny buffers.
            if len(self._buffers[feature]) < min(10, self.window_size):
                continue

            bin_edges = np.array(ref["bin_edges"], dtype=float)
            expected = np.array(ref["expected_proportions"], dtype=float)
            n_bins = len(bin_edges) - 1

            current_values = np.array(self._buffers[feature], dtype=float)
            bin_indices = np.digitize(current_values, bins=bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            counts = np.bincount(bin_indices, minlength=n_bins)
            total = counts.sum()
            observed = counts / total if total > 0 else np.zeros(n_bins, dtype=float)

            psi = compute_psi(expected, observed)

            if self._psi_gauge is not None:
                self._psi_gauge.labels(feature=feature).set(psi)

            if psi > self.threshold:
                drifted.append(feature)

        return drifted
