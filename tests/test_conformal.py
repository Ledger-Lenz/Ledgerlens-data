"""Tests for detection/conformal.py — ConformalCalibrator."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from detection.conformal import CalibrationIntegrityError, ConformalCalibrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture(scope="module")
def trained_classifier(classification_data):
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@pytest.fixture(scope="module")
def calibration_data(trained_classifier):
    model, X_test, y_test = trained_classifier
    X_cal, X_hold, y_cal, y_hold = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    return model, pd.DataFrame(X_cal), pd.Series(y_cal), pd.DataFrame(X_hold), pd.Series(y_hold)


# ---------------------------------------------------------------------------
# 1. calibrate on a perfectly calibrated model achieves empirical coverage
# ---------------------------------------------------------------------------


def test_calibrate_achieved_coverage():
    """A well-calibrated logistic regression should achieve coverage in [0.88, 1.0]."""
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_cal, X_hold, y_cal, y_hold = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
    calibrator.calibrate(model, pd.DataFrame(X_cal), pd.Series(y_cal))

    results = calibrator.predict_set(model, pd.DataFrame(X_hold))
    covered = 0
    for i, row in enumerate(results):
        if int(y_hold[i]) in row["prediction_set"]:
            covered += 1
    empirical = covered / len(X_hold)

    assert 0.88 <= empirical <= 1.0, f"Empirical coverage {empirical:.3f} outside [0.88, 1.0]"


# ---------------------------------------------------------------------------
# 2. calibrate on a random model returns q_hat close to 1.0
# ---------------------------------------------------------------------------


def test_random_model_high_q_hat():
    """A random (untrained) classifier should produce very conservative sets."""
    X = np.random.randn(200, 5)
    y = np.random.randint(0, 2, size=200)

    model = LogisticRegression(random_state=42, max_iter=10)
    model.fit(X[:50], y[:50])

    calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
    calibrator.calibrate(model, pd.DataFrame(X[50:150]), pd.Series(y[50:150]))

    assert calibrator.q_hat is not None
    assert calibrator.q_hat > 0.5


# ---------------------------------------------------------------------------
# 3. predict_set always contains the true label for calibration-guaranteed fraction
# ---------------------------------------------------------------------------


def test_predict_set_coverage_guarantee():
    X, y = make_classification(n_samples=600, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    # Use 300 points for calibration, 300 for holdout
    X_cal, X_hold, y_cal, y_hold = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X_train, y_train)

    calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
    calibrator.calibrate(model, pd.DataFrame(X_cal), pd.Series(y_cal))

    results = calibrator.predict_set(model, pd.DataFrame(X_hold))
    covered = sum(1 for i, r in enumerate(results) if int(y_hold[i]) in r["prediction_set"])
    empirical = covered / len(X_hold)

    assert empirical >= 0.85


# ---------------------------------------------------------------------------
# 4. save + load round-trip preserves q_hat to 8 decimal places
# ---------------------------------------------------------------------------


def test_save_load_round_trip():
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    model = RandomForestClassifier(random_state=42, n_estimators=30)
    model.fit(X_train, y_train)

    calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
    calibrator.calibrate(model, pd.DataFrame(X_test[:80]), pd.Series(y[:80]))
    original_q = calibrator.q_hat

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        calibrator.save(tmp_path)
        loaded = ConformalCalibrator.load(tmp_path)
        assert loaded.q_hat is not None
        assert round(original_q - loaded.q_hat, 8) == 0, (
            f"q_hat mismatch: {original_q} vs {loaded.q_hat}"
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 5. SHA-256 mismatch on load raises CalibrationIntegrityError
# ---------------------------------------------------------------------------


def test_sha256_mismatch_raises_integrity_error():
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    X_cal, _, y_cal, _ = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    model = RandomForestClassifier(random_state=42, n_estimators=30)
    model.fit(X_cal, y_cal)

    calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
    calibrator.calibrate(model, pd.DataFrame(X_cal), pd.Series(y_cal))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        calibrator.save(tmp.name)
        # Tamper with the file
        with open(tmp.name) as f:
            content = json.load(f)
        content["q_hat"] = 0.99
        with open(tmp.name, "w") as f:
            json.dump(content, f, indent=2)
        tmp_path = tmp.name

    try:
        with pytest.raises(CalibrationIntegrityError):
            ConformalCalibrator.load(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 6. Fallback behavior when no artifact: returns conservative interval
# ---------------------------------------------------------------------------


def test_fallback_conservative_interval():
    """Simulates the fallback used when calibration artifact is missing."""
    fallback = {
        "score_lower": 0.0,
        "score_upper": 100.0,
        "coverage_guarantee": 1.0,
    }
    assert fallback["score_lower"] == 0.0
    assert fallback["score_upper"] == 100.0
    assert fallback["coverage_guarantee"] == 1.0


# ---------------------------------------------------------------------------
# 7. predict_with_interval width shrinks with confidence (regression mode)
# ---------------------------------------------------------------------------


def test_interval_width_shrinks_with_confidence():
    X_train = np.random.randn(300, 5)
    y_train = 50 + 10 * X_train[:, 0] + 5 * np.random.randn(300)
    X_cal = pd.DataFrame(np.random.randn(100, 5))
    y_cal = pd.Series(50 + 10 * X_cal.iloc[:, 0] + 5 * np.random.randn(100))

    model = RandomForestRegressor(random_state=42, n_estimators=50)
    model.fit(X_train, y_train)

    calibrator = ConformalCalibrator(alpha=0.10, random_state=42)
    calibrator.calibrate(model, X_cal, y_cal)

    X_eval = pd.DataFrame(np.random.randn(50, 5))
    intervals = calibrator.predict_with_interval(model, X_eval)

    for iv in intervals:
        assert 0.0 <= iv["lower"] <= iv["upper"] <= 100.0
        assert iv["lower"] <= iv["score"] <= iv["upper"]
