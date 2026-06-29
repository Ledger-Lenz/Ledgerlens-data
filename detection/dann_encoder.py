"""Domain-adversarial neural encoder for cross-venue wallet representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import config
from detection.privacy.dp_training import DPTrainingResult, train_with_dp, train_without_dp
from detection.privacy.membership_inference import membership_inference_success_rate
from utils.logging import get_logger

logger = get_logger(__name__)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return cast(torch.Tensor, _GradientReversalFn.apply(x, lambda_))


class DANNEncoder(nn.Module):
    """Wallet feature encoder with label and domain-adversarial heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.label_classifier = nn.Linear(embedding_dim, 1)
        self.domain_classifier = nn.Linear(embedding_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.feature_extractor(x))

    def forward(
        self,
        x: torch.Tensor,
        *,
        domain_lambda: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encode(x)
        label_logits = self.label_classifier(embeddings)
        reversed_embeddings = gradient_reversal(embeddings, domain_lambda)
        domain_logits = self.domain_classifier(reversed_embeddings)
        return label_logits, domain_logits, embeddings

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            label_logits, _, _ = self.forward(x)
            return torch.sigmoid(label_logits.squeeze(-1)).cpu().numpy()


def _dann_loss(model: nn.Module, batch: tuple) -> torch.Tensor:
    x, y, domain = batch
    label_logits, domain_logits, _ = model(x, domain_lambda=1.0)
    label_loss = F.binary_cross_entropy_with_logits(label_logits.squeeze(-1), y)
    domain_loss = F.binary_cross_entropy_with_logits(domain_logits.squeeze(-1), domain)
    return label_loss + 0.5 * domain_loss


def _dann_binary_loss(model: nn.Module, batch: tuple) -> torch.Tensor:
    x, y = batch
    label_logits, _, _ = model(x)
    return F.binary_cross_entropy_with_logits(label_logits.squeeze(-1), y)


def _build_dann_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    batch_size: int,
    train_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    n_train = max(2, int(round(len(indices) * train_fraction)))
    if n_train >= len(indices):
        n_train = len(indices) - 1

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def _loader(idx: np.ndarray, include_domain: bool) -> DataLoader:
        x = torch.from_numpy(features[idx]).float()
        y = torch.from_numpy(labels[idx]).float()
        if include_domain:
            d = torch.from_numpy(domains[idx]).float()
            dataset = TensorDataset(x, y, d)
        else:
            dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = _loader(train_idx, include_domain=True)
    member_loader = _loader(train_idx, include_domain=False)
    test_loader = _loader(test_idx, include_domain=False)
    non_member_loader = test_loader
    return train_loader, member_loader, test_loader, non_member_loader


def _auc_roc(model: DANNEncoder, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].cpu().numpy()
            probs = model.predict_proba(x)
            labels.extend(y.tolist())
            scores.extend(probs.tolist())
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


@dataclass
class DANNTrainingReport:
    model: DANNEncoder
    auc_roc: float
    baseline_auc_roc: float
    auc_roc_degradation: float
    membership_inference_success_rate: float
    achieved_epsilon: float | None = None
    target_epsilon: float | None = None
    target_delta: float | None = None


def train_dann_encoder(
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    domains: pd.Series | np.ndarray | None = None,
    *,
    hidden_dim: int = 128,
    embedding_dim: int = 64,
    epochs: int | None = None,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    train_fraction: float = 0.8,
    seed: int = 42,
    use_dp: bool = False,
    target_epsilon: float | None = None,
    target_delta: float | None = None,
    max_grad_norm: float | None = None,
    device: torch.device | str | None = None,
) -> DANNTrainingReport:
    """Train (or DP-train) a DANN encoder and evaluate privacy metrics."""
    epochs = epochs if epochs is not None else config.DP_EPOCHS
    target_epsilon = target_epsilon if target_epsilon is not None else config.DP_TARGET_EPSILON
    target_delta = target_delta if target_delta is not None else config.DP_TARGET_DELTA
    device = torch.device(device or "cpu")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    if domains is None:
        domains = (np.arange(len(y)) % 2).astype(np.float32)
    else:
        domains = np.asarray(domains, dtype=np.float32)

    train_loader, member_loader, test_loader, non_member_loader = _build_dann_loaders(
        x, y, domains, batch_size=batch_size, train_fraction=train_fraction, seed=seed
    )

    baseline_model = DANNEncoder(x.shape[1], hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    baseline_model = cast(
        DANNEncoder,
        train_without_dp(
            baseline_model,
            train_loader,
            _dann_loss,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )[0],
    )
    baseline_auc = _auc_roc(baseline_model, test_loader, device)

    dp_model = DANNEncoder(x.shape[1], hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    achieved_epsilon: float | None = None
    if use_dp:
        dp_result: DPTrainingResult = train_with_dp(
            dp_model,
            train_loader,
            _dann_loss,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
        dp_model = cast(DANNEncoder, dp_result.model)
        achieved_epsilon = dp_result.achieved_epsilon
        trained_model = dp_model
    else:
        trained_model = cast(
            DANNEncoder,
            train_without_dp(
                dp_model,
                train_loader,
                _dann_loss,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
            )[0],
        )

    auc = _auc_roc(trained_model, test_loader, device)
    degradation = max(0.0, baseline_auc - auc) if not np.isnan(baseline_auc) else float("nan")
    mia_rate = membership_inference_success_rate(
        trained_model,
        member_loader,
        non_member_loader,
        _dann_binary_loss,
        device=device,
    )

    logger.info(
        "DANN encoder %s: auc_roc=%.4f baseline=%.4f degradation=%.4f mia=%.2f%% ε=%s",
        "DP" if use_dp else "baseline",
        auc,
        baseline_auc,
        degradation,
        mia_rate * 100,
        f"{achieved_epsilon:.4f}" if achieved_epsilon is not None else "n/a",
    )

    return DANNTrainingReport(
        model=trained_model,
        auc_roc=auc,
        baseline_auc_roc=baseline_auc,
        auc_roc_degradation=degradation,
        membership_inference_success_rate=mia_rate,
        achieved_epsilon=achieved_epsilon,
        target_epsilon=target_epsilon if use_dp else None,
        target_delta=target_delta if use_dp else None,
    )
