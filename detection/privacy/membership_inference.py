"""Membership inference attack evaluation for trained neural models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _per_sample_losses(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[nn.Module, tuple], torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Compute mean loss per sample (handles Opacus batch layouts)."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(tensor.to(device) for tensor in batch)
            # Per-sample BCE for binary classifiers when batch is (x, y).
            if len(batch) == 2:
                x, y = batch
                logits = _forward_logits(model, x).squeeze(-1)
                sample_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y.float(), reduction="none"
                )
                losses.extend(sample_losses.cpu().numpy().tolist())
            else:
                losses.append(loss_fn(model, batch).item())
    return np.asarray(losses, dtype=np.float64)


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    if isinstance(output, tuple):
        return cast(torch.Tensor, output[0])
    return cast(torch.Tensor, output)


def membership_inference_success_rate(
    model: nn.Module,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    loss_fn: Callable[[nn.Module, tuple], torch.Tensor],
    *,
    device: torch.device | str | None = None,
) -> float:
    """Loss-threshold membership inference attack success rate.

    Members (training set) typically achieve lower loss than non-members.
    The attacker picks the threshold that maximises classification accuracy
    on a **balanced** member/non-member evaluation set so the random baseline
    is 50%, not the training-set prevalence.
    """
    device = torch.device(device or "cpu")
    model = model.to(device)

    member_losses = _per_sample_losses(model, member_loader, loss_fn, device)
    non_member_losses = _per_sample_losses(model, non_member_loader, loss_fn, device)
    if len(member_losses) == 0 or len(non_member_losses) == 0:
        return 0.5

    n_eval = min(len(member_losses), len(non_member_losses))
    rng = np.random.default_rng(0)
    member_eval = rng.choice(member_losses, n_eval, replace=False)
    non_member_eval = rng.choice(non_member_losses, n_eval, replace=False)

    all_losses = np.concatenate([member_eval, non_member_eval])
    labels = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

    best_accuracy = 0.0
    for threshold in np.unique(all_losses):
        predictions = (all_losses < threshold).astype(np.float64)
        accuracy = float(np.mean(predictions == labels))
        best_accuracy = max(best_accuracy, accuracy, 1.0 - accuracy)

    return best_accuracy
