import json
import os

import pytest

from detection.model_training import (
    MODEL_REGISTRY,
    compute_feature_schema_hash,
    save_models,
    save_training_artifacts,
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
        assert set(result["metrics"]) == {"auc_roc", "pr_auc", "f1"}
        assert 0.0 <= result["metrics"]["auc_roc"] <= 1.0


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
    assert set(metrics) == set(MODEL_REGISTRY)


def test_save_training_artifacts_writes_metadata(tmp_path, trained_output):
    output, _ = trained_output
    model_dir = str(tmp_path)
    data_path = "data/synthetic_dataset.parquet"

    save_training_artifacts(output, data_path, model_dir)
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    assert os.path.exists(metadata_path)

    with open(metadata_path) as f:
        meta = json.load(f)

    assert "trained_at" in meta
    assert meta["data_path"] == data_path
    assert meta["n_training_rows"] == output["n_train"]
    assert meta["n_test_rows"] == output["n_test"]
    assert meta["feature_columns"] == output["feature_columns"]
    assert "feature_schema_hash" in meta
    assert meta["model_names"] == list(MODEL_REGISTRY.keys())
    assert "python_version" in meta
    assert meta["ledgerlens_version"] == "0.2.0"


def test_metadata_feature_hash_matches_training_columns(tmp_path, trained_output):
    output, _ = trained_output
    model_dir = str(tmp_path)
    save_training_artifacts(output, "data/test.parquet", model_dir)

    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        meta = json.load(f)

    expected_hash = compute_feature_schema_hash(output["feature_columns"])
    assert meta["feature_schema_hash"] == expected_hash
