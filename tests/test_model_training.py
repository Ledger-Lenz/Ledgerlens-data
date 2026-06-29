"""Tests for detection/model_training.py — provenance, poisoning detection."""

import json
import os

import pandas as pd
import pytest

from detection.model_training import (
    MODEL_REGISTRY,
    compute_feature_schema_hash,
    detect_label_poisoning,
    save_models,
    save_training_artifacts,
    sha256_dataframe,
    split_features_labels,
    train_models,
)
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


@pytest.fixture(scope="module")
def trained_output():
    df = generate_synthetic_dataset(n_wallets=60, seed=1)
    return train_models(df, test_size=0.3, random_state=1), df


def test_split_features_labels_excludes_wallet_and_label():
    df = generate_synthetic_dataset(n_wallets=10, seed=1)
    X, y = split_features_labels(df)
    assert "wallet" not in X.columns
    assert "label" not in X.columns
    assert len(X) == len(y)


def test_train_models_returns_metrics_for_each_model(trained_output):
    output, _ = trained_output
    results = output["results"]
    assert set(results) == set(MODEL_REGISTRY)
    for result in results.values():
        base_keys = {"auc_roc", "pr_auc", "f1"}
        conformal_keys = {"conformal_empirical_coverage", "conformal_q_hat", "calibration_split_size"}
        assert base_keys.issubset(set(result["metrics"]))
        assert conformal_keys.issubset(set(result["metrics"]))
        assert 0.0 <= result["metrics"]["auc_roc"] <= 1.0
        assert 0.0 <= result["metrics"]["conformal_empirical_coverage"] <= 1.0


def test_train_models_returns_held_out_split(trained_output):
    output, _ = trained_output
    assert len(output["X_test"]) == output["n_test"]
    assert len(output["y_test"]) == output["n_test"]
    assert "label" not in output["X_test"].columns


def test_save_models_and_training_artifacts(tmp_path, trained_output):
    output, _ = trained_output
    results = output["results"]
    model_dir = str(tmp_path)

    save_models(results, model_dir)
    for name in MODEL_REGISTRY:
        assert os.path.exists(os.path.join(model_dir, f"{name}.joblib"))

    save_training_artifacts(output, "data/synthetic.parquet", model_dir)
    assert os.path.exists(os.path.join(model_dir, "metrics.json"))
    assert os.path.exists(os.path.join(model_dir, "model_metadata.json"))

    with open(os.path.join(model_dir, "metrics.json")) as f:
        metrics = json.load(f)
    assert set(MODEL_REGISTRY).issubset(set(metrics))

    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        meta = json.load(f)
    expected_hash = compute_feature_schema_hash(output["feature_columns"])
    assert meta["feature_schema_hash"] == expected_hash


# ---------------------------------------------------------------------------
# Provenance: SHA-256 of training data
# ---------------------------------------------------------------------------


def test_training_data_sha256_changes_when_row_added():
    df = generate_synthetic_dataset(n_wallets=20, seed=5)
    sha1 = sha256_dataframe(df)

    extra = df.iloc[[0]].copy()
    extra["wallet"] = "GNEW"
    df2 = pd.concat([df, extra], ignore_index=True)
    sha2 = sha256_dataframe(df2)

    assert sha1 != sha2


# ---------------------------------------------------------------------------
# Label poisoning detection
# ---------------------------------------------------------------------------


def test_detect_label_poisoning_returns_true_when_ratio_shifts(tmp_path):
    baseline_path = str(tmp_path / "baseline.json")
    # Write a baseline with ~10% wash-trade ratio
    baseline_ratio = 0.10
    with open(baseline_path, "w") as f:
        json.dump({"wash_trade_ratio": baseline_ratio}, f)

    # Current distribution: ~30% wash trades → shift = 0.20 > 0.15 threshold
    distribution = {0: 70, 1: 30}
    assert detect_label_poisoning(distribution, baseline_path=baseline_path, threshold=0.15)


def test_detect_label_poisoning_returns_false_when_ratio_ok(tmp_path):
    baseline_path = str(tmp_path / "baseline.json")
    with open(baseline_path, "w") as f:
        json.dump({"wash_trade_ratio": 0.20}, f)

    distribution = {0: 82, 1: 18}  # 18% — shift < 15%
    assert not detect_label_poisoning(distribution, baseline_path=baseline_path, threshold=0.15)


def test_detect_label_poisoning_creates_baseline_when_missing(tmp_path):
    baseline_path = str(tmp_path / "new_baseline.json")
    assert not os.path.exists(baseline_path)

    distribution = {0: 90, 1: 10}
    result = detect_label_poisoning(distribution, baseline_path=baseline_path)
    assert result is False
    assert os.path.exists(baseline_path)


def test_detect_label_poisoning_aborts_training(tmp_path, monkeypatch):
    """When poisoning is detected, no .pkl / .joblib files should be written."""
    import detection.model_training as mt

    baseline_path = str(tmp_path / "baseline.json")
    with open(baseline_path, "w") as f:
        json.dump({"wash_trade_ratio": 0.05}, f)

    # Patch baseline path and threshold so poisoning is always detected
    monkeypatch.setattr(mt, "LABEL_DISTRIBUTION_BASELINE_PATH", baseline_path)
    monkeypatch.setattr(mt.config, "POISON_LABEL_RATIO_THRESHOLD", 0.05)
    monkeypatch.setattr(mt.config, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(mt.config, "MODEL_SIGNING_PRIVATE_KEY_PATH", "")

    # Build a minimal dataset with a high wash-trade ratio (60%)
    df = generate_synthetic_dataset(n_wallets=40, seed=7)
    # Override labels so ratio = 60%
    df["label"] = [1 if i % 5 != 0 else 0 for i in range(len(df))]

    # Simulate main() with a temp data file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        df.to_parquet(tmp_file.name)
        tmp_file_path = tmp_file.name

    monkeypatch.setattr(
        "sys.argv",
        ["model_training", "--data-path", tmp_file_path, "--model-dir", str(tmp_path / "models")],
    )

    mt.main()

    # No model artifacts should have been written
    model_dir = str(tmp_path / "models")
    for name in MODEL_REGISTRY:
        assert not os.path.exists(os.path.join(model_dir, f"{name}.joblib"))

    os.unlink(tmp_file_path)
