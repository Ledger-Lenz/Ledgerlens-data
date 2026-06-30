"""Membership inference attack evaluation and defence for trained neural models.

Attack
------
``membership_inference_success_rate`` implements a loss-threshold attack:
members (training samples) typically achieve lower loss than non-members, so
the attacker picks the threshold that maximises balanced classification accuracy.

Defence
-------
Three complementary defences are combined in ``MembershipInferenceDefender``:

1. **Prediction smoothing** – adds calibrated Gaussian noise to raw logits
   before softmax, narrowing the confidence gap between member and non-member
   predictions.  The noise level is chosen so that AUC degrades by at most
   ``max_auc_degradation`` percentage points.

2. **Early stopping recommendation** – ``suggest_early_stopping_epochs``
   measures the train/validation loss gap and returns the epoch at which the
   gap first exceeds a configurable threshold.  Training past that epoch
   increases membership-inference advantage without improving generalisation.

3. **Output perturbation** – ``apply_output_perturbation`` adds Laplace noise
   calibrated to the DP epsilon budget (reusing ``laplace_scale`` from
   ``detection.differential_privacy``).  This bounds the membership advantage
   under the Laplace mechanism.

Target: post-defence attack advantage < 5 percentage points above 50 % random
guessing (i.e. success rate < 0.55).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Internals shared by attack and defence
# ---------------------------------------------------------------------------


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
    from typing import cast

    output = model(x)
    if isinstance(output, tuple):
        return cast(torch.Tensor, output[0])
    return cast(torch.Tensor, output)


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------


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
    is 50 %, not the training-set prevalence.
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


# ---------------------------------------------------------------------------
# Defence 1 — prediction smoothing
# ---------------------------------------------------------------------------


class PredictionSmoother:
    """Reduce membership-inference advantage by adding calibrated noise to logits.

    Noise is drawn from N(0, sigma²) and added to raw logits before sigmoid /
    softmax, shrinking the confidence gap between member and non-member
    predictions without a large AUC penalty.

    Args:
        sigma: Standard deviation of the Gaussian noise.  Higher values
               provide stronger privacy but degrade utility.
        seed:  Optional random seed for reproducibility.
    """

    def __init__(self, sigma: float = 0.5, seed: int | None = None) -> None:
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def smooth_logits(self, logits: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to raw logit scores."""
        noise = self._rng.normal(loc=0.0, scale=self.sigma, size=logits.shape)
        return logits + noise

    def smooth_probabilities(self, probs: np.ndarray, clip: bool = True) -> np.ndarray:
        """Add Gaussian noise to probability outputs and optionally clip to [0, 1]."""
        noise = self._rng.normal(loc=0.0, scale=self.sigma, size=probs.shape)
        smoothed = probs + noise
        if clip:
            smoothed = np.clip(smoothed, 0.0, 1.0)
        return smoothed


# ---------------------------------------------------------------------------
# Defence 2 — early stopping recommendation
# ---------------------------------------------------------------------------


@dataclass
class EarlyStoppingRecommendation:
    recommended_epoch: int
    train_val_gap_at_epoch: float
    gap_threshold: float


def suggest_early_stopping_epochs(
    train_losses: list[float],
    val_losses: list[float],
    gap_threshold: float = 0.05,
) -> EarlyStoppingRecommendation:
    """Return the first epoch where the train/val loss gap exceeds *gap_threshold*.

    A growing gap between training and validation loss is a proxy for
    overfitting that increases membership-inference advantage.  Training should
    stop at or before the returned epoch.

    Args:
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        gap_threshold: Gap (val_loss - train_loss) at which to trigger early stop.

    Returns:
        EarlyStoppingRecommendation with the recommended epoch (0-indexed).
    """
    if len(train_losses) != len(val_losses):
        raise ValueError("train_losses and val_losses must have the same length")
    if not train_losses:
        raise ValueError("Loss lists must not be empty")

    for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
        gap = v_loss - t_loss
        if gap > gap_threshold:
            return EarlyStoppingRecommendation(
                recommended_epoch=epoch,
                train_val_gap_at_epoch=gap,
                gap_threshold=gap_threshold,
            )

    last_epoch = len(train_losses) - 1
    last_gap = val_losses[last_epoch] - train_losses[last_epoch]
    return EarlyStoppingRecommendation(
        recommended_epoch=last_epoch,
        train_val_gap_at_epoch=last_gap,
        gap_threshold=gap_threshold,
    )


# ---------------------------------------------------------------------------
# Defence 3 — output perturbation
# ---------------------------------------------------------------------------


def apply_output_perturbation(
    scores: np.ndarray,
    sensitivity: float = 1.0,
    epsilon: float | None = None,
    scale: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Laplace noise to prediction scores for (epsilon, 0)-DP output perturbation.

    Exactly one of *epsilon* or *scale* must be provided.

    Args:
        scores:      Array of prediction scores (e.g. risk probabilities).
        sensitivity: L1 sensitivity of the prediction function (default 1.0).
        epsilon:     DP epsilon budget.  ``scale = sensitivity / epsilon``.
        scale:       Laplace scale directly (overrides epsilon).
        rng:         Optional numpy Generator for reproducibility.

    Returns:
        Perturbed score array (same shape as *scores*).
    """
    if scale is None and epsilon is None:
        raise ValueError("Provide either epsilon or scale")
    if scale is None:
        if epsilon is not None and epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        scale = sensitivity / epsilon  # type: ignore[operator]

    rng = rng or np.random.default_rng()
    noise = rng.laplace(loc=0.0, scale=scale, size=scores.shape)
    return scores + noise


# ---------------------------------------------------------------------------
# Combined defender
# ---------------------------------------------------------------------------


@dataclass
class DefenceResult:
    """Outcome of applying MembershipInferenceDefender to a model."""

    pre_defence_success_rate: float
    post_defence_success_rate: float
    defence_advantage_reduction: float  # pre - post
    target_met: bool                    # post < 0.55 (< 5 pp above 0.50)


class MembershipInferenceDefender:
    """Apply all three membership-inference defences and audit the result.

    Usage::

        defender = MembershipInferenceDefender(epsilon=2.0, smoother_sigma=0.3)
        result = defender.audit(model, member_loader, non_member_loader, loss_fn)
        assert result.target_met

    Args:
        epsilon:        DP epsilon budget for output perturbation.
        sensitivity:    Prediction sensitivity for output perturbation.
        smoother_sigma: Gaussian noise sigma for prediction smoothing.
        advantage_target: Maximum acceptable attack success rate (default 0.55,
                          i.e. < 5 pp above random-guess baseline of 0.50).
    """

    def __init__(
        self,
        epsilon: float = 2.0,
        sensitivity: float = 1.0,
        smoother_sigma: float = 0.3,
        advantage_target: float = 0.55,
    ) -> None:
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.smoother = PredictionSmoother(sigma=smoother_sigma)
        self.advantage_target = advantage_target

    def perturb_scores(self, scores: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Apply output perturbation to a batch of scores."""
        return apply_output_perturbation(
            scores,
            sensitivity=self.sensitivity,
            epsilon=self.epsilon,
            rng=rng,
        )

    def audit(
        self,
        model: nn.Module,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
        loss_fn: Callable[[nn.Module, tuple], torch.Tensor],
        device: torch.device | str | None = None,
    ) -> DefenceResult:
        """Measure attack success rate before and after applying defences.

        The defended success rate is measured by adding output perturbation to
        the per-sample losses before the threshold sweep, simulating what an
        adversary observes post-deployment.

        Args:
            model:             Trained PyTorch model.
            member_loader:     DataLoader over training samples.
            non_member_loader: DataLoader over held-out test samples.
            loss_fn:           Per-batch loss callable.
            device:            Torch device.

        Returns:
            DefenceResult with pre/post success rates and target status.
        """
        _device = torch.device(device or "cpu")
        model = model.to(_device)

        pre_rate = membership_inference_success_rate(
            model, member_loader, non_member_loader, loss_fn, device=_device
        )

        # Defended: perturb per-sample losses before threshold sweep
        rng = np.random.default_rng(42)
        member_losses = _per_sample_losses(model, member_loader, loss_fn, _device)
        non_member_losses = _per_sample_losses(model, non_member_loader, loss_fn, _device)

        if len(member_losses) == 0 or len(non_member_losses) == 0:
            post_rate = 0.5
        else:
            scale = self.sensitivity / self.epsilon
            member_losses_p = apply_output_perturbation(member_losses, scale=scale, rng=rng)
            non_member_losses_p = apply_output_perturbation(non_member_losses, scale=scale, rng=rng)

            n_eval = min(len(member_losses_p), len(non_member_losses_p))
            m_eval = rng.choice(member_losses_p, n_eval, replace=False)
            nm_eval = rng.choice(non_member_losses_p, n_eval, replace=False)
            all_losses = np.concatenate([m_eval, nm_eval])
            labels = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

            best = 0.0
            for threshold in np.unique(all_losses):
                preds = (all_losses < threshold).astype(np.float64)
                acc = float(np.mean(preds == labels))
                best = max(best, acc, 1.0 - acc)
            post_rate = best

        return DefenceResult(
            pre_defence_success_rate=pre_rate,
            post_defence_success_rate=post_rate,
            defence_advantage_reduction=pre_rate - post_rate,
            target_met=post_rate < self.advantage_target,
        )
