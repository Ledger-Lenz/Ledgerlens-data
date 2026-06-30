"""Membership inference auditor — measure attack success before and after defence.

Implements a shadow-model-style audit:
  1. Train a shadow model on a subset of the same distribution.
  2. Split shadow training/test data into member/non-member sets.
  3. Measure loss-threshold attack success on the target model (undefended).
  4. Apply ``MembershipInferenceDefender`` and re-measure (defended).
  5. Report aggregate success rates only — no individual wallet identifiers.

Usage::

    python -m scripts.audit_membership_inference \\
        --model-path models/risk_classifier.pt \\
        --data-path data/synthetic_dataset.parquet \\
        --epsilon 2.0 \\
        --output audit_report.json

Exit code 0 if the post-defence advantage is < 5 pp above 0.50; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from detection.privacy.membership_inference import (
    MembershipInferenceDefender,
    membership_inference_success_rate,
)
from utils.logging import get_logger

logger = get_logger(__name__)

ADVANTAGE_TARGET = 0.55  # < 5 pp above random-guess baseline of 0.50


# ---------------------------------------------------------------------------
# Shadow model (minimal 2-layer MLP)
# ---------------------------------------------------------------------------


class _ShadowModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _default_loss_fn(model: nn.Module, batch: tuple) -> torch.Tensor:
    x, y = batch
    logits = model(x).squeeze(-1)
    return nn.functional.binary_cross_entropy_with_logits(logits, y.float())


def _build_loaders(
    X: np.ndarray,
    y: np.ndarray,
    member_fraction: float = 0.5,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split data into member/non-member DataLoaders."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * member_fraction)
    member_idx = idx[:split]
    non_member_idx = idx[split:]

    def _loader(indices: np.ndarray) -> DataLoader:
        X_t = torch.tensor(X[indices], dtype=torch.float32)
        y_t = torch.tensor(y[indices], dtype=torch.float32)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)

    return _loader(member_idx), _loader(non_member_idx)


def _train_shadow_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 10,
    lr: float = 1e-2,
) -> _ShadowModel:
    model = _ShadowModel(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            x_b, y_b = batch
            optimizer.zero_grad()
            logits = model(x_b).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_b)
            loss.backward()
            optimizer.step()
    return model


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------


def run_audit(
    X: np.ndarray,
    y: np.ndarray,
    target_model: nn.Module | None = None,
    epsilon: float = 2.0,
    sensitivity: float = 1.0,
    smoother_sigma: float = 0.3,
    shadow_epochs: int = 10,
) -> dict:
    """Run the full membership inference audit.

    Returns a dict with aggregate attack success rates.  Individual wallet
    membership is NOT reported.

    Args:
        X:             Feature matrix (n_samples, n_features).
        y:             Binary labels.
        target_model:  Model to audit.  A shadow model is trained if None.
        epsilon:       DP epsilon for output-perturbation defence.
        sensitivity:   Prediction sensitivity.
        smoother_sigma: Gaussian sigma for prediction smoothing.
        shadow_epochs: Training epochs for the shadow model.

    Returns:
        Dict with keys: pre_defence_success_rate, post_defence_success_rate,
        target_met, epsilon, advantage_target.
    """
    rng_split = np.random.default_rng(1)
    n = len(X)
    idx = rng_split.permutation(n)
    # 70 % for shadow training, 30 % for audit (member / non-member split)
    train_idx = idx[: int(0.7 * n)]
    audit_idx = idx[int(0.7 * n) :]

    if target_model is None:
        logger.info("Training shadow model on %d samples for %d epochs …", len(train_idx), shadow_epochs)
        model = _train_shadow_model(X[train_idx], y[train_idx], epochs=shadow_epochs)
    else:
        model = target_model

    member_loader, non_member_loader = _build_loaders(X[audit_idx], y[audit_idx])

    logger.info("Measuring undefended attack success rate …")
    pre_rate = membership_inference_success_rate(model, member_loader, non_member_loader, _default_loss_fn)
    logger.info("Pre-defence success rate: %.4f", pre_rate)

    logger.info("Applying defences and re-measuring …")
    defender = MembershipInferenceDefender(
        epsilon=epsilon,
        sensitivity=sensitivity,
        smoother_sigma=smoother_sigma,
        advantage_target=ADVANTAGE_TARGET,
    )
    result = defender.audit(model, member_loader, non_member_loader, _default_loss_fn)
    logger.info("Post-defence success rate: %.4f (target met: %s)", result.post_defence_success_rate, result.target_met)

    return {
        "pre_defence_success_rate": round(result.pre_defence_success_rate, 4),
        "post_defence_success_rate": round(result.post_defence_success_rate, 4),
        "advantage_reduction": round(result.defence_advantage_reduction, 4),
        "target_met": result.target_met,
        "advantage_target": ADVANTAGE_TARGET,
        "epsilon": epsilon,
        "n_samples_audited": len(audit_idx),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Membership inference auditor for LedgerLens models")
    parser.add_argument("--model-path", default=None, help="Path to a saved PyTorch model (.pt).  A shadow model is trained if omitted.")
    parser.add_argument("--data-path", default="data/synthetic_dataset.parquet", help="Parquet feature dataset")
    parser.add_argument("--epsilon", type=float, default=2.0, help="DP epsilon budget for output perturbation (default: 2.0)")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Prediction sensitivity (default: 1.0)")
    parser.add_argument("--smoother-sigma", type=float, default=0.3, help="Prediction smoother Gaussian sigma (default: 0.3)")
    parser.add_argument("--shadow-epochs", type=int, default=10, help="Shadow model training epochs (default: 10)")
    parser.add_argument("--output", default=None, help="Write audit report to this JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import pandas as pd

        df = pd.read_parquet(args.data_path)
        label_col = "label" if "label" in df.columns else df.columns[-1]
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].select_dtypes(include="number").fillna(0.0).values.astype(np.float32)
        y = (df[label_col].values > 0).astype(np.float32)
    except Exception as exc:
        logger.error("Failed to load dataset from %s: %s", args.data_path, exc)
        sys.exit(1)

    target_model: nn.Module | None = None
    if args.model_path:
        try:
            target_model = torch.load(args.model_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", args.model_path, exc)
            sys.exit(1)

    report = run_audit(
        X,
        y,
        target_model=target_model,
        epsilon=args.epsilon,
        sensitivity=args.sensitivity,
        smoother_sigma=args.smoother_sigma,
        shadow_epochs=args.shadow_epochs,
    )

    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Audit report written to %s", args.output)

    sys.exit(0 if report["target_met"] else 1)


if __name__ == "__main__":
    main()
