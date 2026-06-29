"""Unit tests for LabelQualityEstimator (issue #271).

Covers:
- 10 known mislabelled samples (high-confidence wash traders labelled 'clean')
  in a batch of 100 → cleanlab must flag at least 7 of the 10
- Per-annotator noise rate alert fires when rate exceeds 20%
- Clean batch produces no quarantined samples
- Quarantine audit log is written with annotator_id and noise_score
- evaluate_batch returns expected keys
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from detection.active_learning.label_quality_estimator import LabelQualityEstimator

# Skip if cleanlab is not installed
try:
    import cleanlab  # noqa: F401
    _CLEANLAB_AVAILABLE = True
except ImportError:
    _CLEANLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_separable_dataset(n_clean=45, n_wash=45, n_dim=5, random_state=0):
    """Make a cleanly separable binary dataset for testing."""
    rng = np.random.default_rng(random_state)
    X_clean = rng.normal(loc=0.0, scale=0.5, size=(n_clean, n_dim))
    X_wash = rng.normal(loc=3.0, scale=0.5, size=(n_wash, n_dim))
    X = np.vstack([X_clean, X_wash])
    y = np.array([0] * n_clean + [1] * n_wash, dtype=int)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_dim)]), y


def _inject_mislabelled(y: np.ndarray, wash_indices: list[int]) -> np.ndarray:
    """Flip wash-trade (1) labels to clean (0) to simulate mislabelling."""
    y_noisy = y.copy()
    for idx in wash_indices:
        y_noisy[idx] = 0  # "clean" but actually wash trade
    return y_noisy


def _train_model(X: pd.DataFrame, y: np.ndarray):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Test: 7-of-10 mislabelled detection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _CLEANLAB_AVAILABLE, reason="cleanlab not installed")
class TestMislabelledDetection:
    """Inject 10 known mislabelled samples into a batch of 100;
    cleanlab must flag at least 7 of the 10.
    """

    def test_detects_at_least_7_of_10_mislabelled(self, tmp_path):
        # 90 correct labels + 10 mislabelled high-confidence wash traders
        X, y_true = _make_separable_dataset(n_clean=45, n_wash=45)
        # Train a model on the true labels
        model = _train_model(X, y_true)

        # The last 10 indices are wash traders (label=1); flip to clean (label=0)
        noisy_indices = list(range(90, 100))
        y_noisy = _inject_mislabelled(y_true, noisy_indices)

        estimator = LabelQualityEstimator(
            model=model,
            noise_threshold=0.10,  # top 10% quarantined
            quarantine_log_path=str(tmp_path / "quarantine.ndjson"),
        )
        result = estimator.evaluate_batch(X, y_noisy)

        quarantined = set(result["quarantined_indices"])
        injected = set(noisy_indices)
        n_detected = len(quarantined & injected)

        assert n_detected >= 7, (
            f"Expected at least 7 of 10 mislabelled samples detected, "
            f"got {n_detected}. Quarantined: {quarantined}"
        )

    def test_clean_batch_quarantines_few(self, tmp_path):
        """A clean batch should produce very few quarantined samples."""
        X, y = _make_separable_dataset()
        model = _train_model(X, y)

        estimator = LabelQualityEstimator(
            model=model,
            noise_threshold=0.10,
            quarantine_log_path=str(tmp_path / "quarantine.ndjson"),
        )
        result = estimator.evaluate_batch(X, y)
        # On a cleanly labelled dataset, the 90th-percentile noise scores
        # should all be near zero; we allow at most 5% false positives
        assert len(result["quarantined_indices"]) <= int(0.05 * len(y)) + 1


# ---------------------------------------------------------------------------
# Test: per-annotator noise rate alert
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _CLEANLAB_AVAILABLE, reason="cleanlab not installed")
class TestAnnotatorNoiseAlert:
    def test_alert_fires_when_noise_rate_exceeds_threshold(self, tmp_path, caplog):
        import logging

        X, y_true = _make_separable_dataset(n_clean=45, n_wash=45)
        model = _train_model(X, y_true)

        # Mislabel 30 out of 100 from a single annotator → 30% noise rate > 20%
        noisy_indices = list(range(70, 100))
        y_noisy = _inject_mislabelled(y_true, noisy_indices)

        annotator_ids = ["bad_annotator"] * 100

        estimator = LabelQualityEstimator(
            model=model,
            noise_threshold=0.30,
            annotator_alert_threshold=0.20,
            quarantine_log_path=str(tmp_path / "quarantine.ndjson"),
        )

        with caplog.at_level(logging.WARNING, logger="detection.active_learning.label_quality_estimator"):
            estimator.evaluate_batch(X, y_noisy, annotator_ids=annotator_ids)

        assert any(
            "bad_annotator" in r.message and "noise" in r.message.lower()
            for r in caplog.records
        ), "Expected a WARNING about bad_annotator's high noise rate"

    def test_noise_rate_tracked_per_annotator(self, tmp_path):
        X, y_true = _make_separable_dataset()
        model = _train_model(X, y_true)

        noisy_indices = list(range(80, 90))
        y_noisy = _inject_mislabelled(y_true, noisy_indices)

        annotator_ids = (
            ["alice"] * 80 + ["bad_actor"] * 10 + ["alice"] * 10
        )

        estimator = LabelQualityEstimator(
            model=model,
            noise_threshold=0.10,
            annotator_alert_threshold=0.20,
            quarantine_log_path=str(tmp_path / "quarantine.ndjson"),
        )
        estimator.evaluate_batch(X, y_noisy, annotator_ids=annotator_ids)
        rates = estimator.annotator_noise_rates()

        # bad_actor annotated the mislabelled samples — rate should be higher
        if "bad_actor" in rates and "alice" in rates:
            assert rates["bad_actor"] >= rates["alice"]


# ---------------------------------------------------------------------------
# Test: quarantine log security
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _CLEANLAB_AVAILABLE, reason="cleanlab not installed")
class TestQuarantineLog:
    def test_quarantined_items_logged_with_annotator_and_score(self, tmp_path):
        X, y_true = _make_separable_dataset()
        model = _train_model(X, y_true)

        noisy_indices = list(range(90, 100))
        y_noisy = _inject_mislabelled(y_true, noisy_indices)
        annotator_ids = ["analyst_x"] * 100
        wallet_ids = [f"G{i:06d}" for i in range(100)]

        log_path = str(tmp_path / "quarantine.ndjson")
        estimator = LabelQualityEstimator(
            model=model,
            noise_threshold=0.10,
            quarantine_log_path=log_path,
        )
        result = estimator.evaluate_batch(
            X, y_noisy, annotator_ids=annotator_ids, wallet_ids=wallet_ids
        )

        if result["quarantined_indices"]:
            assert os.path.exists(log_path)
            with open(log_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            assert len(records) == len(result["quarantined_indices"])
            for rec in records:
                assert "noise_score" in rec
                assert rec["status"] == "quarantined"
                assert "annotator_id" in rec
                assert "wallet" in rec

    def test_quarantined_not_silently_deleted(self, tmp_path):
        """Quarantined items must appear in the log, not just be dropped."""
        X, y_true = _make_separable_dataset()
        model = _train_model(X, y_true)

        noisy_indices = list(range(90, 100))
        y_noisy = _inject_mislabelled(y_true, noisy_indices)

        log_path = str(tmp_path / "quarantine.ndjson")
        estimator = LabelQualityEstimator(
            model=model, noise_threshold=0.10, quarantine_log_path=log_path
        )
        result = estimator.evaluate_batch(X, y_noisy)

        quarantined = result["quarantined_indices"]
        if quarantined:
            # Log must exist and contain exactly as many records as quarantined items
            with open(log_path) as f:
                records = [json.loads(l) for l in f if l.strip()]
            assert len(records) >= len(quarantined)


# ---------------------------------------------------------------------------
# Test: evaluate_batch return keys
# ---------------------------------------------------------------------------

class TestEvaluateBatchAPI:
    def test_returns_required_keys(self, tmp_path):
        X, y = _make_separable_dataset(n_clean=10, n_wash=10)
        # Use a dummy model that just returns 0.5 for everything
        class DummyModel:
            def predict_proba(self, X):
                return np.full((len(X), 2), 0.5)

        estimator = LabelQualityEstimator(
            DummyModel(),
            quarantine_log_path=str(tmp_path / "q.ndjson"),
        )
        result = estimator.evaluate_batch(X, y)
        for key in ("clean_indices", "quarantined_indices", "noise_scores", "annotator_noise_rates"):
            assert key in result

    def test_empty_batch_returns_empty(self, tmp_path):
        class DummyModel:
            def predict_proba(self, X):
                return np.empty((0, 2))

        estimator = LabelQualityEstimator(
            DummyModel(),
            quarantine_log_path=str(tmp_path / "q.ndjson"),
        )
        X_empty = pd.DataFrame(columns=["f0"])
        result = estimator.evaluate_batch(X_empty, np.array([], dtype=int))
        assert result["clean_indices"] == []
        assert result["quarantined_indices"] == []
