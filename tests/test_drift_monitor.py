"""Tests for detection/drift_monitor.py.

Validates PSI computation, threshold behaviour, zero-frequency bin handling,
and report serialisation.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from detection.drift_monitor import (
    REPORTS_DIR,
    DriftMonitor,
    DriftReport,
    compute_psi,
)


def _build_reference(columns: list[str], n_bins: int = 10) -> dict[str, dict]:
    """Build a synthetic reference distribution dict for testing."""
    ref = {}
    for col in columns:
        col_data = np.random.default_rng(42).normal(loc=0.0, scale=1.0, size=1000)
        bin_edges = np.histogram_bin_edges(col_data, bins=n_bins)
        counts, _ = np.histogram(col_data, bins=bin_edges)
        total = counts.sum()
        expected = np.maximum(counts / total, 1e-4)
        expected = expected / expected.sum()
        ref[col] = {
            "bin_edges": bin_edges.tolist(),
            "expected_proportions": expected.tolist(),
        }
    return ref


class TestComputePsi:
    def test_identical_distributions_psi_zero(self):
        props = np.array([0.2, 0.3, 0.5])
        psi = compute_psi(props, props)
        assert psi == pytest.approx(0.0, abs=1e-10)

    def test_different_distributions_psi_positive(self):
        expected = np.array([0.9, 0.05, 0.05])
        observed = np.array([0.05, 0.05, 0.9])
        psi = compute_psi(expected, observed)
        assert psi > 0.0

    def test_zero_frequencies_clipped(self):
        expected = np.array([0.0, 1.0])
        observed = np.array([1.0, 0.0])
        psi = compute_psi(expected, observed)
        assert np.isfinite(psi)
        assert psi > 0.0

    def test_all_zero_frequencies(self):
        expected = np.array([0.0, 0.0])
        observed = np.array([0.0, 0.0])
        psi = compute_psi(expected, observed)
        assert np.isfinite(psi)

    def test_known_psi_value(self):
        """Validate against a manually computed PSI value.

        Reference: expected=[0.5, 0.5], observed=[0.3, 0.7]
        PSI = (0.3-0.5)*ln(0.3/0.5) + (0.7-0.5)*ln(0.7/0.5)
            = (-0.2)*ln(0.6) + (0.2)*ln(1.4)
            = (-0.2)*(-0.5108) + 0.2*0.3365
            = 0.1022 + 0.0673
            = 0.1695
        """
        expected = np.array([0.5, 0.5])
        observed = np.array([0.3, 0.7])
        psi = compute_psi(expected, observed)
        assert psi == pytest.approx(0.1695, abs=1e-3)


class TestDriftMonitor:
    def test_psi_below_threshold_no_drift(self):
        """Two identical distributions → psi < 0.1 and drift_flag = False."""
        rng = np.random.default_rng(42)
        n = 500
        columns = ["feat_a", "feat_b", "feat_c"]

        ref_data = pd.DataFrame({c: rng.normal(0, 1, n) for c in columns})
        ref = _build_reference(columns)

        monitor = DriftMonitor(ref)
        report = monitor.compute(ref_data)

        assert not report.any_drift_detected
        for feat in report.features:
            assert feat["psi"] < 0.1
            assert not feat["drift_flag"]

    def test_psi_above_threshold_triggers_drift(self):
        """Shifted distribution → psi >= 0.25 and drift_flag = True."""
        rng = np.random.default_rng(42)
        n = 500

        reference_col_data = rng.normal(0, 1, n)
        bin_edges = np.histogram_bin_edges(reference_col_data, bins=5)
        counts, _ = np.histogram(reference_col_data, bins=bin_edges)
        total = counts.sum()
        expected = np.maximum(counts / total, 1e-4)
        expected = expected / expected.sum()

        ref = {
            "shifted_feat": {
                "bin_edges": bin_edges.tolist(),
                "expected_proportions": expected.tolist(),
            }
        }

        current_data = pd.DataFrame(
            {
                "shifted_feat": rng.normal(5.0, 0.5, n),
            }
        )

        monitor = DriftMonitor(ref)
        report = monitor.compute(current_data)

        assert report.any_drift_detected
        shifted = [f for f in report.features if f["feature"] == "shifted_feat"][0]
        assert shifted["psi"] >= 0.25
        assert shifted["drift_flag"]

    def test_psi_handles_zero_frequency_bins(self):
        """Zero-frequency bins in reference do not cause ZeroDivisionError."""
        ref = {
            "sparse_feat": {
                "bin_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
                "expected_proportions": [0.0, 0.0, 0.0, 1.0],
            }
        }
        current_data = pd.DataFrame(
            {
                "sparse_feat": np.random.default_rng(42).uniform(0.8, 1.0, 200),
            }
        )

        monitor = DriftMonitor(ref)
        report = monitor.compute(current_data)

        assert np.all(np.isfinite([f["psi"] for f in report.features]))
        assert report.any_drift_detected is not None

    def test_drift_report_written_to_json(self):
        """DriftMonitor.compute() writes a JSON report with correct schema."""
        rng = np.random.default_rng(42)
        n = 200
        columns = ["feat_x", "feat_y"]

        ref = _build_reference(columns)
        current = pd.DataFrame({c: rng.normal(0, 1, n) for c in columns})

        monitor = DriftMonitor(ref)
        monitor.compute(current)

        report_files = [
            f for f in os.listdir(REPORTS_DIR) if re.match(r"drift_report_\d{8}_\d{6}\.json$", f)
        ]
        assert len(report_files) >= 1

        latest = sorted(report_files)[-1]
        with open(os.path.join(REPORTS_DIR, latest)) as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "any_drift_detected" in data
        assert "n_features_checked" in data
        assert "n_features_drifted" in data
        assert "features" in data

        for feat in data["features"]:
            assert "feature" in feat
            assert "psi" in feat
            assert "drift_flag" in feat
            assert isinstance(feat["feature"], str)
            assert isinstance(feat["psi"], float)
            assert isinstance(feat["drift_flag"], bool)

    def test_only_known_features_checked(self):
        """Features in current_data but not in reference are skipped."""
        ref = _build_reference(["known_feat"])
        current = pd.DataFrame(
            {
                "known_feat": np.random.default_rng(42).normal(0, 1, 100),
                "unknown_feat": np.random.default_rng(42).normal(0, 1, 100),
            }
        )

        monitor = DriftMonitor(ref)
        report = monitor.compute(current)

        names = [f["feature"] for f in report.features]
        assert "known_feat" in names
        assert "unknown_feat" not in names

    def test_empty_current_data_no_features(self):
        """No feature columns matching reference results in empty report."""
        ref = _build_reference(["feat_a"])
        current = pd.DataFrame({"wallet": ["G1", "G2"]})

        monitor = DriftMonitor(ref)
        report = monitor.compute(current)

        assert len(report.features) == 0
        assert not report.any_drift_detected


class TestDriftReport:
    def test_to_dict(self):
        report = DriftReport(
            features=[
                {"feature": "f1", "psi": 0.05, "drift_flag": False},
                {"feature": "f2", "psi": 0.35, "drift_flag": True},
            ],
            any_drift_detected=True,
        )
        d = report.to_dict()
        assert d["n_features_checked"] == 2
        assert d["n_features_drifted"] == 1
        assert d["any_drift_detected"] is True
        assert d["features"][0]["feature"] == "f1"
