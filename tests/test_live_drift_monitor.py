import json
from pathlib import Path

import numpy as np
import pandas as pd

from detection.drift_monitor import LiveDriftMonitor, compute_psi


def _build_reference(values: np.ndarray, n_bins: int = 10) -> dict:
    bin_edges = np.histogram_bin_edges(values, bins=n_bins)
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    expected = np.maximum(counts / total, 1e-4)
    expected = expected / expected.sum()
    return {
        "feature_distributions": {
            "feat_x": {
                "bin_edges": bin_edges.tolist(),
                "expected_proportions": expected.tolist(),
            }
        }
    }


def test_live_drift_monitor_stable_psi_below_0_1(tmp_path: Path):
    rng = np.random.default_rng(42)
    reference_vals = rng.normal(0.0, 1.0, size=2000)
    current_vals = rng.normal(0.0, 1.0, size=1000)

    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(_build_reference(reference_vals)), encoding="utf-8")

    monitor = LiveDriftMonitor(
        reference_path=str(ref_path),
        threshold=0.2,
        window_size=1000,
    )

    drifted = []
    for v in current_vals:
        drifted = monitor.update(pd.Series({"feat_x": float(v)}))

    # With stable distribution we should not exceed threshold.
    assert "feat_x" not in drifted


def test_live_drift_monitor_shifted_psi_above_0_2(tmp_path: Path):
    rng = np.random.default_rng(42)
    reference_vals = rng.normal(0.0, 1.0, size=2000)
    shifted_vals = rng.normal(5.0, 0.5, size=1000)

    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(_build_reference(reference_vals)), encoding="utf-8")

    monitor = LiveDriftMonitor(
        reference_path=str(ref_path),
        threshold=0.2,
        window_size=1000,
    )

    drifted = []
    for v in shifted_vals:
        drifted = monitor.update(pd.Series({"feat_x": float(v)}))

    assert "feat_x" in drifted


