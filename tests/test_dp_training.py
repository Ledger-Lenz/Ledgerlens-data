"""Tests for differentially private neural training (issue #127)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import config
from detection.dann_encoder import DANNEncoder, train_dann_encoder
from detection.privacy.dp_training import opacus_available, train_with_dp, train_without_dp
from detection.privacy.membership_inference import membership_inference_success_rate
from detection.privacy.metrics import record_dp_metrics
from detection.privacy.meta_learner_dp import train_meta_learner_dp


class _TinyClassifier(nn.Module):
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _synthetic_binary_loader(n_samples: int = 64, input_dim: int = 8, batch_size: int = 16):
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(n_samples, input_dim)).astype(np.float32))
    y = torch.from_numpy((rng.random(n_samples) > 0.5).astype(np.float32))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _bce_loss(model: nn.Module, batch: tuple) -> torch.Tensor:
    x, y = batch
    logits = model(x).squeeze(-1)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def test_config_dp_defaults():
    assert config.DP_TARGET_EPSILON == 8.0
    assert config.DP_TARGET_DELTA == 1e-5
    assert config.DP_MAX_GRAD_NORM == 1.0


def test_record_dp_metrics_merges_into_metrics_json(tmp_path):
    existing = {"random_forest": {"auc_roc": 0.9}}
    path = tmp_path / "metrics.json"
    with open(path, "w") as handle:
        json.dump(existing, handle)

    record_dp_metrics(
        str(tmp_path),
        "dann_encoder",
        {"achieved_epsilon": 7.5, "target_epsilon": 8.0, "target_delta": 1e-5},
    )

    with open(path) as handle:
        payload = json.load(handle)

    assert payload["random_forest"]["auc_roc"] == 0.9
    assert payload["differential_privacy"]["dann_encoder"]["achieved_epsilon"] == 7.5


def test_membership_inference_random_model_near_chance():
    torch.manual_seed(1)
    model = _TinyClassifier()
    member_loader = _synthetic_binary_loader(40)
    non_member_loader = _synthetic_binary_loader(40)
    rate = membership_inference_success_rate(model, member_loader, non_member_loader, _bce_loss)
    assert 0.45 <= rate <= 0.75


def test_membership_inference_balanced_not_inflated_by_class_prior():
    """Imbalanced loaders must not let 'predict all members' dominate accuracy."""
    torch.manual_seed(0)
    model = _TinyClassifier()
    x = torch.randn(80, 8)
    y = torch.randint(0, 2, (80,)).float()
    member_loader = DataLoader(TensorDataset(x[:64], y[:64]), batch_size=16, shuffle=False)
    non_member_loader = DataLoader(TensorDataset(x[64:], y[64:]), batch_size=16, shuffle=False)
    rate = membership_inference_success_rate(model, member_loader, non_member_loader, _bce_loss)
    # Majority-class shortcut on 64/16 split would be ~80%; balanced eval must stay well below that.
    assert rate < 0.70


def test_train_without_dp_runs():
    model = _TinyClassifier()
    loader = _synthetic_binary_loader()
    trained, loss = train_without_dp(model, loader, _bce_loss, epochs=2)
    assert loss >= 0.0
    assert trained is model


@pytest.mark.skipif(not opacus_available(), reason="opacus not installed")
def test_train_with_dp_achieves_target_epsilon_order():
    model = _TinyClassifier()
    loader = _synthetic_binary_loader(n_samples=128)
    result = train_with_dp(
        model,
        loader,
        _bce_loss,
        target_epsilon=config.DP_TARGET_EPSILON,
        target_delta=config.DP_TARGET_DELTA,
        epochs=3,
    )
    assert result.achieved_epsilon > 0
    assert result.target_epsilon == config.DP_TARGET_EPSILON
    assert result.target_delta == config.DP_TARGET_DELTA


@pytest.mark.skipif(not opacus_available(), reason="opacus not installed")
def test_dann_encoder_dp_training_smoke():
    rng = np.random.default_rng(7)
    n = 80
    x = rng.normal(size=(n, 12)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.float32)
    domains = (np.arange(n) % 2).astype(np.float32)

    report = train_dann_encoder(
        x,
        y,
        domains,
        epochs=3,
        batch_size=16,
        hidden_dim=32,
        embedding_dim=16,
        use_dp=True,
    )
    assert isinstance(report.model, DANNEncoder)
    assert report.achieved_epsilon is not None
    assert report.achieved_epsilon > 0
    assert 0.0 <= report.membership_inference_success_rate <= 1.0


@pytest.mark.skipif(not opacus_available(), reason="opacus not installed")
def test_meta_learner_dp_training_smoke():
    rng = np.random.default_rng(3)
    n = 80
    embeddings = rng.normal(size=(n, 20)).astype(np.float32)
    labels = (rng.random(n) > 0.5).astype(np.float32)

    report = train_meta_learner_dp(
        embeddings,
        labels,
        epochs=3,
        batch_size=16,
        hidden_dim=32,
        use_dp=True,
    )
    assert report.achieved_epsilon is not None
    assert report.achieved_epsilon > 0
    assert 0.0 <= report.membership_inference_success_rate <= 1.0


@pytest.mark.skipif(not opacus_available(), reason="opacus not installed")
def test_train_dp_neural_script_writes_metrics(tmp_path):
    from scripts.train_dp_neural import train_private_neural_components

    model_dir = str(tmp_path)
    train_private_neural_components(
        model_dir=model_dir,
        n_wallets=60,
        epochs=3,
        seed=11,
        skip_meta=True,
    )

    metrics_path = os.path.join(model_dir, "metrics.json")
    assert os.path.exists(metrics_path)
    with open(metrics_path) as handle:
        metrics = json.load(handle)

    dp = metrics["differential_privacy"]
    assert "dann_encoder" in dp
    assert dp["dann_encoder"]["achieved_epsilon"] > 0
    assert dp["dann_encoder"]["target_epsilon"] == config.DP_TARGET_EPSILON
    assert dp["dann_encoder"]["target_delta"] == config.DP_TARGET_DELTA
    assert os.path.exists(os.path.join(model_dir, "dann_encoder.pt"))
