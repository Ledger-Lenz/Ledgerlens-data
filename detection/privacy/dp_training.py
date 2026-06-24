"""Differentially private SGD training via Opacus PrivacyEngine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from opacus import PrivacyEngine

    _OPACUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PrivacyEngine = None  # type: ignore[assignment,misc]
    _OPACUS_AVAILABLE = False


@dataclass
class DPTrainingResult:
    """Outcome of a DP-SGD training run."""

    model: nn.Module
    achieved_epsilon: float
    target_epsilon: float
    target_delta: float
    max_grad_norm: float
    epochs: int
    final_loss: float
    privacy_engine: object | None = None


def opacus_available() -> bool:
    return _OPACUS_AVAILABLE


def unwrap_private_model(model: nn.Module) -> nn.Module:
    """Return the inner module when Opacus wraps it for DP-SGD."""
    inner = getattr(model, "_module", None)
    if inner is not None:
        return inner
    return model


def train_with_dp(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[nn.Module, tuple], torch.Tensor],
    *,
    target_epsilon: float | None = None,
    target_delta: float | None = None,
    max_grad_norm: float | None = None,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: torch.device | str | None = None,
) -> DPTrainingResult:
    """Wrap *model* with Opacus DP-SGD and run a standard training loop.

    Parameters
    ----------
    model:
        PyTorch module to train (e.g. DANNEncoder, MAMLAdapter).
    dataloader:
        Training DataLoader; Opacus replaces its sampler for per-sample
        gradient clipping.
    loss_fn:
        ``loss_fn(model, batch) -> scalar loss`` where *batch* is a tuple of
        tensors moved to *device*.
    """
    if not _OPACUS_AVAILABLE:
        raise ImportError(
            "opacus is required for differentially private training. "
            "Install with: pip install opacus"
        )

    target_epsilon = target_epsilon if target_epsilon is not None else config.DP_TARGET_EPSILON
    target_delta = target_delta if target_delta is not None else config.DP_TARGET_DELTA
    max_grad_norm = max_grad_norm if max_grad_norm is not None else config.DP_MAX_GRAD_NORM
    device = torch.device(device or "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            batch = tuple(tensor.to(device) for tensor in batch)
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)
        logger.debug("DP epoch %d/%d loss=%.4f", epoch + 1, epochs, final_loss)

    achieved_epsilon = privacy_engine.get_epsilon(target_delta)
    logger.info(
        "DP training complete: achieved ε=%.4f (target=%.4f, δ=%.1e)",
        achieved_epsilon,
        target_epsilon,
        target_delta,
    )

    inference_model = unwrap_private_model(model)

    return DPTrainingResult(
        model=inference_model,
        achieved_epsilon=float(achieved_epsilon),
        target_epsilon=float(target_epsilon),
        target_delta=float(target_delta),
        max_grad_norm=float(max_grad_norm),
        epochs=epochs,
        final_loss=float(final_loss),
        privacy_engine=privacy_engine,
    )


def train_without_dp(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[nn.Module, tuple], torch.Tensor],
    *,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, float]:
    """Non-private baseline training loop matching :func:`train_with_dp`."""
    device = torch.device(device or "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_loss = 0.0
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            batch = tuple(tensor.to(device) for tensor in batch)
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)

    return model, float(final_loss)
