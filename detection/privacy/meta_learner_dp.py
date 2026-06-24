"""Supervised DP training for the MAML meta-learner adapter head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import config
from detection.meta_learner import MAMLAdapter
from detection.privacy.dp_training import DPTrainingResult, train_with_dp, train_without_dp
from detection.privacy.membership_inference import membership_inference_success_rate
from utils.logging import get_logger

logger = get_logger(__name__)


def _maml_loss(model: torch.nn.Module, batch: tuple) -> torch.Tensor:
    x, y = batch
    logits = model(x).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, y)


def _build_loaders(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    train_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    n_train = max(2, int(round(len(indices) * train_fraction)))
    if n_train >= len(indices):
        n_train = len(indices) - 1

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def _loader(idx: np.ndarray) -> DataLoader:
        x = torch.from_numpy(embeddings[idx]).float()
        y = torch.from_numpy(labels[idx]).float()
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    return _loader(train_idx), _loader(train_idx), _loader(test_idx)


def _auc_roc(model: MAMLAdapter, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = model.predict_proba(x)
            labels.extend(y.numpy().tolist())
            scores.extend(probs.tolist())
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


@dataclass
class MetaLearnerTrainingReport:
    model: MAMLAdapter
    auc_roc: float
    baseline_auc_roc: float
    auc_roc_degradation: float
    membership_inference_success_rate: float
    achieved_epsilon: float | None = None
    target_epsilon: float | None = None
    target_delta: float | None = None


def train_meta_learner_dp(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_dim: int = 128,
    epochs: int | None = None,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    train_fraction: float = 0.8,
    seed: int = 42,
    use_dp: bool = True,
    target_epsilon: float | None = None,
    target_delta: float | None = None,
    max_grad_norm: float | None = None,
    device: torch.device | str | None = None,
) -> MetaLearnerTrainingReport:
    """Train the MAML adapter head with optional DP-SGD."""
    epochs = epochs if epochs is not None else config.DP_EPOCHS
    target_epsilon = target_epsilon if target_epsilon is not None else config.DP_TARGET_EPSILON
    target_delta = target_delta if target_delta is not None else config.DP_TARGET_DELTA
    device = torch.device(device or "cpu")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    train_loader, member_loader, test_loader = _build_loaders(
        embeddings,
        labels,
        batch_size=batch_size,
        train_fraction=train_fraction,
        seed=seed,
    )

    baseline = MAMLAdapter(embeddings.shape[1], hidden_dim=hidden_dim)
    baseline = cast(
        MAMLAdapter,
        train_without_dp(
            baseline,
            train_loader,
            _maml_loss,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )[0],
    )
    baseline_auc = _auc_roc(baseline, test_loader, device)

    model = MAMLAdapter(embeddings.shape[1], hidden_dim=hidden_dim)
    achieved_epsilon: float | None = None
    if use_dp:
        dp_result: DPTrainingResult = train_with_dp(
            model,
            train_loader,
            _maml_loss,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
        model = cast(MAMLAdapter, dp_result.model)
        achieved_epsilon = dp_result.achieved_epsilon
    else:
        model = cast(
            MAMLAdapter,
            train_without_dp(
                model,
                train_loader,
                _maml_loss,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
            )[0],
        )

    auc = _auc_roc(model, test_loader, device)
    degradation = max(0.0, baseline_auc - auc) if not np.isnan(baseline_auc) else float("nan")
    mia_rate = membership_inference_success_rate(
        model,
        member_loader,
        test_loader,
        _maml_loss,
        device=device,
    )

    logger.info(
        "Meta-learner %s: auc_roc=%.4f baseline=%.4f degradation=%.4f mia=%.2f%% ε=%s",
        "DP" if use_dp else "baseline",
        auc,
        baseline_auc,
        degradation,
        mia_rate * 100,
        f"{achieved_epsilon:.4f}" if achieved_epsilon is not None else "n/a",
    )

    return MetaLearnerTrainingReport(
        model=model,
        auc_roc=auc,
        baseline_auc_roc=baseline_auc,
        auc_roc_degradation=degradation,
        membership_inference_success_rate=mia_rate,
        achieved_epsilon=achieved_epsilon,
        target_epsilon=target_epsilon if use_dp else None,
        target_delta=target_delta if use_dp else None,
    )
