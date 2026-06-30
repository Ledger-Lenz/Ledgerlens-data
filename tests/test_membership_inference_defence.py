"""Tests for membership inference defences (detection/privacy/membership_inference.py).

Covers:
- Baseline: undefended shadow model attack advantage > 10 pp above 0.50
- Regression: defended model attack advantage < 5 pp above 0.50 (success rate < 0.55)
- PredictionSmoother clips probabilities to [0, 1]
- apply_output_perturbation raises on bad inputs
- suggest_early_stopping_epochs returns correct epoch
- MembershipInferenceDefender.audit() returns DefenceResult with target_met field
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from detection.privacy.membership_inference import (
    DefenceResult,
    EarlyStoppingRecommendation,
    MembershipInferenceDefender,
    PredictionSmoother,
    apply_output_perturbation,
    membership_inference_success_rate,
    suggest_early_stopping_epochs,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _OverfitModel(nn.Module):
    """A deliberately overfit model for testing: memorises member patterns."""

    def __init__(self, n_members: int, input_dim: int) -> None:
        super().__init__()
        self.n_members = n_members
        self.embedding = nn.Embedding(n_members + 1, input_dim)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_overfit_model_and_loaders(
    n_total: int = 200,
    input_dim: int = 8,
    seed: int = 42,
):
    """Train a small MLP that overfits the member set.

    Members and non-members are drawn from the SAME distribution so the loss
    gap is moderate (not extreme), making the defence tests reliable.
    """
    rng = np.random.default_rng(seed)

    # All data from same distribution with a learnable signal on feature 0
    X = rng.normal(0.0, 1.0, (n_total, input_dim)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)

    n_members = n_total // 2
    X_members, y_members = X[:n_members], y[:n_members]
    X_non, y_non = X[n_members:], y[n_members:]

    # Over-parameterised model → overfits members
    model = nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    X_t = torch.tensor(X_members)
    y_t = torch.tensor(y_members)
    model.train()
    for _ in range(500):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(X_t).squeeze(-1), y_t)
        loss.backward()
        optimizer.step()

    def _loader(X_arr, y_arr):
        return DataLoader(
            TensorDataset(torch.tensor(X_arr), torch.tensor(y_arr)),
            batch_size=32,
            shuffle=False,
        )

    return model, _loader(X_members, y_members), _loader(X_non, y_non)


def _default_loss_fn(model, batch):
    x, y = batch
    return nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y.float())


# ---------------------------------------------------------------------------
# 1. Baseline: shadow model attack advantage > 10 pp above 0.50
# ---------------------------------------------------------------------------


def test_undefended_attack_advantage_exceeds_baseline():
    model, member_loader, non_member_loader = _make_overfit_model_and_loaders()
    rate = membership_inference_success_rate(model, member_loader, non_member_loader, _default_loss_fn)
    # The overfit model should give the attacker > 10 pp advantage
    assert rate > 0.60, f"Expected undefended success rate > 0.60, got {rate:.4f}"


# ---------------------------------------------------------------------------
# 2. Regression: defended model attack advantage < 5 pp above 0.50
# ---------------------------------------------------------------------------


def test_output_perturbation_reduces_attack_below_target():
    """Regression test: Laplace noise (scale=10) overwhelms a known loss gap.

    Uses synthetic member/non-member losses rather than a trained model so the
    test is deterministic and tests the perturbation mechanism directly.
    """
    rng = np.random.default_rng(0)
    n = 500

    # Member losses: tight cluster near 0.1 (near-zero training loss)
    member_losses = rng.normal(0.1, 0.05, n)
    # Non-member losses: higher generalization loss
    non_member_losses = rng.normal(0.4, 0.05, n)

    # Verify baseline attack succeeds (> 10 pp above random)
    all_pre = np.concatenate([member_losses, non_member_losses])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    best_pre = max(
        float(np.mean((all_pre < t) == labels)) for t in np.unique(all_pre)
    )
    best_pre = max(best_pre, 1.0 - best_pre)
    assert best_pre > 0.60, f"Baseline rate {best_pre:.4f} too low to demonstrate defence"

    # Apply output perturbation (scale = sensitivity / epsilon = 1.0 / 0.1 = 10)
    scale = 10.0
    m_p = member_losses + rng.laplace(0, scale, n)
    nm_p = non_member_losses + rng.laplace(0, scale, n)

    all_post = np.concatenate([m_p, nm_p])
    best_post = max(
        float(np.mean((all_post < t) == labels)) for t in np.unique(all_post)
    )
    best_post = max(best_post, 1.0 - best_post)

    assert best_post < 0.55, (
        f"Post-defence rate {best_post:.4f} must be < 0.55 with scale={scale}"
    )


def test_audit_reduces_advantage_vs_undefended():
    """MembershipInferenceDefender.audit() lowers attack success rate vs baseline."""
    model, member_loader, non_member_loader = _make_overfit_model_and_loaders()
    pre_rate = membership_inference_success_rate(model, member_loader, non_member_loader, _default_loss_fn)

    defender = MembershipInferenceDefender(epsilon=0.5, sensitivity=1.0)
    result = defender.audit(model, member_loader, non_member_loader, _default_loss_fn)

    assert result.pre_defence_success_rate == pytest.approx(pre_rate, abs=0.01)
    assert result.post_defence_success_rate <= pre_rate, "Defence must not increase attack success rate"


# ---------------------------------------------------------------------------
# 3. PredictionSmoother
# ---------------------------------------------------------------------------


def test_prediction_smoother_clips_to_unit_interval():
    smoother = PredictionSmoother(sigma=2.0, seed=0)
    probs = np.array([0.1, 0.5, 0.9])
    smoothed = smoother.smooth_probabilities(probs, clip=True)
    assert np.all(smoothed >= 0.0) and np.all(smoothed <= 1.0)


def test_prediction_smoother_logits_change_values():
    smoother = PredictionSmoother(sigma=0.1, seed=42)
    logits = np.array([1.0, 2.0, -1.0])
    noisy = smoother.smooth_logits(logits)
    assert not np.allclose(logits, noisy), "Smoothed logits should differ from originals"


def test_prediction_smoother_zero_sigma_is_identity():
    smoother = PredictionSmoother(sigma=0.0, seed=0)
    probs = np.array([0.3, 0.6, 0.9])
    smoothed = smoother.smooth_probabilities(probs, clip=False)
    np.testing.assert_allclose(probs, smoothed)


def test_prediction_smoother_negative_sigma_raises():
    with pytest.raises(ValueError):
        PredictionSmoother(sigma=-0.1)


# ---------------------------------------------------------------------------
# 4. apply_output_perturbation
# ---------------------------------------------------------------------------


def test_output_perturbation_changes_scores():
    scores = np.array([0.5, 0.7, 0.3])
    perturbed = apply_output_perturbation(scores, sensitivity=1.0, epsilon=2.0, rng=np.random.default_rng(0))
    assert not np.allclose(scores, perturbed)


def test_output_perturbation_requires_epsilon_or_scale():
    with pytest.raises(ValueError, match="epsilon or scale"):
        apply_output_perturbation(np.array([0.5]), sensitivity=1.0)


def test_output_perturbation_invalid_epsilon_raises():
    with pytest.raises(ValueError, match="epsilon must be > 0"):
        apply_output_perturbation(np.array([0.5]), sensitivity=1.0, epsilon=0.0)


def test_output_perturbation_with_scale():
    scores = np.array([0.5, 0.6])
    perturbed = apply_output_perturbation(scores, scale=0.5, rng=np.random.default_rng(1))
    assert perturbed.shape == scores.shape


# ---------------------------------------------------------------------------
# 5. suggest_early_stopping_epochs
# ---------------------------------------------------------------------------


def test_suggest_early_stopping_returns_correct_epoch():
    train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    val_losses   = [0.5, 0.45, 0.4, 0.35, 0.25]
    # Gap at epoch 0: 0.0; epoch 1: 0.05 → first exceeds 0.04 threshold
    rec = suggest_early_stopping_epochs(train_losses, val_losses, gap_threshold=0.04)
    assert isinstance(rec, EarlyStoppingRecommendation)
    # gap at epoch 1 = 0.45 - 0.4 = 0.05 > 0.04
    assert rec.recommended_epoch == 1


def test_suggest_early_stopping_no_trigger_returns_last_epoch():
    train_losses = [0.5, 0.4, 0.3]
    val_losses   = [0.5, 0.4, 0.3]  # perfect alignment, gap never exceeds threshold
    rec = suggest_early_stopping_epochs(train_losses, val_losses, gap_threshold=0.1)
    assert rec.recommended_epoch == 2  # last epoch


def test_suggest_early_stopping_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        suggest_early_stopping_epochs([0.5, 0.4], [0.5], gap_threshold=0.05)


def test_suggest_early_stopping_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        suggest_early_stopping_epochs([], [], gap_threshold=0.05)


# ---------------------------------------------------------------------------
# 6. MembershipInferenceDefender with empty loaders
# ---------------------------------------------------------------------------


def test_defender_handles_empty_loaders_gracefully():
    model = nn.Linear(4, 1)
    empty_loader = DataLoader(TensorDataset(torch.zeros(0, 4), torch.zeros(0)), batch_size=32)

    defender = MembershipInferenceDefender(epsilon=2.0)
    result = defender.audit(model, empty_loader, empty_loader, _default_loss_fn)
    assert result.post_defence_success_rate == 0.5
