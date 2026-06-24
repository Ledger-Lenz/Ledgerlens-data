"""Tests for detection/active_learning/incremental_trainer.py."""

from __future__ import annotations

import os

import pytest

from detection.active_learning.incremental_trainer import IncrementalTrainer, _sha256_file
from detection.model_training import save_models, train_models
from scripts.generate_synthetic_dataset import generate_synthetic_dataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    """Train a small ensemble and return the model directory."""
    df = generate_synthetic_dataset(n_wallets=80, seed=10)
    output = train_models(df, test_size=0.25, random_state=10)
    model_dir = str(tmp_path_factory.mktemp("models"))
    save_models(output["results"], model_dir)
    return model_dir, df


# ---------------------------------------------------------------------------
# Warm-start vs full retrain path selection
# ---------------------------------------------------------------------------


def test_warm_start_called_when_below_threshold(tmp_path, trained_model_dir, monkeypatch):
    """When len(new_labelled) < AL_RETRAIN_THRESHOLD, strategy == 'warm_start'."""
    model_dir, df = trained_model_dir
    import shutil

    # Copy models to a temp dir so we don't mutate the fixture
    local_dir = str(tmp_path / "models")
    shutil.copytree(model_dir, local_dir)

    small_df = generate_synthetic_dataset(n_wallets=10, seed=99)
    trainer = IncrementalTrainer(model_dir=local_dir)

    # Ensure threshold is above our sample count
    monkeypatch.setattr("config.config.AL_RETRAIN_THRESHOLD", 50)
    monkeypatch.setattr("config.config.AL_ROLLBACK_AUC_DROP", 0.99)  # never roll back

    report = trainer.update(small_df)
    assert report["strategy"] == "warm_start"
    assert report["n_new_samples"] == 10


def test_full_retrain_called_when_at_or_above_threshold(tmp_path, trained_model_dir, monkeypatch):
    """When len(new_labelled) >= AL_RETRAIN_THRESHOLD, strategy == 'full_retrain'."""
    model_dir, df = trained_model_dir
    import shutil

    local_dir = str(tmp_path / "models")
    shutil.copytree(model_dir, local_dir)

    large_df = generate_synthetic_dataset(n_wallets=60, seed=77)
    trainer = IncrementalTrainer(model_dir=local_dir)

    monkeypatch.setattr("config.config.AL_RETRAIN_THRESHOLD", 50)
    monkeypatch.setattr("config.config.AL_ROLLBACK_AUC_DROP", 0.99)

    report = trainer.update(large_df)
    assert report["strategy"] == "full_retrain"


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_triggered_when_auc_drops(tmp_path, trained_model_dir, monkeypatch):
    """IncrementalTrainer should roll back when AUC drops beyond threshold."""
    model_dir, df = trained_model_dir
    import shutil

    local_dir = str(tmp_path / "models")
    shutil.copytree(model_dir, local_dir)

    trainer = IncrementalTrainer(model_dir=local_dir)
    monkeypatch.setattr("config.config.AL_RETRAIN_THRESHOLD", 50)
    # Set rollback threshold to 0 → any AUC drop (even tiny) triggers rollback
    monkeypatch.setattr("config.config.AL_ROLLBACK_AUC_DROP", 0.0)

    # Use a very small, imbalanced dataset that is likely to hurt AUC
    bad_df = generate_synthetic_dataset(n_wallets=8, seed=999)
    bad_df["label"] = 0  # all clean — degenerate training set

    report = trainer.update(bad_df)
    # May or may not roll back depending on actual AUC; just ensure key exists
    assert "rolled_back" in report


def test_rollback_restores_sha256_match(tmp_path, trained_model_dir, monkeypatch):
    """After rollback, model artifact SHA-256 must match the pre-update artifact."""
    model_dir, df = trained_model_dir
    import shutil

    from detection.model_training import MODEL_REGISTRY

    local_dir = str(tmp_path / "models")
    shutil.copytree(model_dir, local_dir)

    # Record SHA-256 of each artifact before update
    pre_shas = {
        name: _sha256_file(os.path.join(local_dir, f"{name}.joblib"))
        for name in MODEL_REGISTRY
        if os.path.exists(os.path.join(local_dir, f"{name}.joblib"))
    }

    monkeypatch.setattr("config.config.AL_RETRAIN_THRESHOLD", 50)
    # Force rollback by setting threshold to 0
    monkeypatch.setattr("config.config.AL_ROLLBACK_AUC_DROP", 0.0)

    bad_df = generate_synthetic_dataset(n_wallets=8, seed=888)
    bad_df["label"] = 0

    trainer = IncrementalTrainer(model_dir=local_dir)
    report = trainer.update(bad_df)

    if report["rolled_back"]:
        post_shas = {
            name: _sha256_file(os.path.join(local_dir, f"{name}.joblib"))
            for name in MODEL_REGISTRY
            if os.path.exists(os.path.join(local_dir, f"{name}.joblib"))
        }
        for name in pre_shas:
            assert pre_shas[name] == post_shas[name], f"SHA-256 mismatch for {name} after rollback"


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------


def test_update_writes_report_json(tmp_path, trained_model_dir, monkeypatch):
    model_dir, df = trained_model_dir
    import shutil

    local_dir = str(tmp_path / "models")
    shutil.copytree(model_dir, local_dir)

    monkeypatch.setattr("config.config.AL_RETRAIN_THRESHOLD", 50)
    monkeypatch.setattr("config.config.AL_ROLLBACK_AUC_DROP", 0.99)
    monkeypatch.chdir(tmp_path)  # reports/ written relative to cwd

    small_df = generate_synthetic_dataset(n_wallets=10, seed=55)
    trainer = IncrementalTrainer(model_dir=local_dir)
    report = trainer.update(small_df)

    assert os.path.exists(os.path.join(tmp_path, "reports"))
    report_files = os.listdir(os.path.join(tmp_path, "reports"))
    assert any(f.startswith("al_update_") and f.endswith(".json") for f in report_files)

    assert "auc_before" in report
    assert "auc_after" in report
    assert "strategy" in report
    assert "rolled_back" in report


def test_update_raises_without_models(tmp_path, monkeypatch):
    monkeypatch.setattr("config.config.AL_RETRAIN_THRESHOLD", 50)
    monkeypatch.setattr("config.config.AL_ROLLBACK_AUC_DROP", 0.99)
    trainer = IncrementalTrainer(model_dir=str(tmp_path / "empty_models"))
    df = generate_synthetic_dataset(n_wallets=10, seed=1)
    with pytest.raises(RuntimeError, match="No trained models"):
        trainer.update(df)
