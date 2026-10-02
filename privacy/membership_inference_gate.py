"""Membership-inference audit gate for model release pipeline.

Integrates automated membership-inference attack testing into the model promotion
workflow. This gate blocks release if post-defence attack success rate exceeds
the configured risk threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from detection.model_governance import PromotionError
from scripts.audit_membership_inference import ADVANTAGE_TARGET, run_audit
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MembershipInferenceAuditResult:
    """Result of membership inference audit."""

    pre_defence_success_rate: float
    post_defence_success_rate: float
    advantage_reduction: float
    target_met: bool
    auc_threshold: float
    passed: bool
    audit_timestamp: str


class MembershipInferenceGateError(PromotionError):
    """Raised when membership-inference audit gate fails."""

    pass


def run_membership_inference_gate(
    target_model,
    audit_data_path: str | Path,
    epsilon: float = 2.0,
    sensitivity: float = 1.0,
    smoother_sigma: float = 0.3,
) -> MembershipInferenceAuditResult:
    """Run membership inference audit as a release gate.

    Args:
        target_model: PyTorch model to audit.
        audit_data_path: Path to audit dataset (parquet).
        epsilon: DP epsilon for defence.
        sensitivity: Prediction sensitivity.
        smoother_sigma: Gaussian smoothing sigma.

    Returns:
        MembershipInferenceAuditResult with audit details.

    Raises:
        MembershipInferenceGateError: If post-defence success rate exceeds threshold.
    """
    import numpy as np
    import pandas as pd
    from datetime import UTC, datetime

    logger.info("Starting membership-inference audit gate")

    try:
        df = pd.read_parquet(audit_data_path)
        label_col = "label" if "label" in df.columns else df.columns[-1]
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].select_dtypes(include="number").fillna(0.0).values.astype(np.float32)
        y = (df[label_col].values > 0).astype(np.float32)
    except Exception as exc:
        raise MembershipInferenceGateError(f"Failed to load audit dataset: {exc}") from exc

    try:
        report = run_audit(
            X,
            y,
            target_model=target_model,
            epsilon=epsilon,
            sensitivity=sensitivity,
            smoother_sigma=smoother_sigma,
        )
    except Exception as exc:
        raise MembershipInferenceGateError(f"Audit execution failed: {exc}") from exc

    result = MembershipInferenceAuditResult(
        pre_defence_success_rate=report["pre_defence_success_rate"],
        post_defence_success_rate=report["post_defence_success_rate"],
        advantage_reduction=report["advantage_reduction"],
        target_met=report["target_met"],
        auc_threshold=ADVANTAGE_TARGET,
        passed=report["target_met"],
        audit_timestamp=datetime.now(UTC).isoformat(),
    )

    if not result.passed:
        logger.error(
            "Membership-inference gate FAILED: post-defence success rate %.4f >= target %.4f",
            result.post_defence_success_rate,
            result.auc_threshold,
        )
        raise MembershipInferenceGateError(
            f"Post-defence attack success rate {result.post_defence_success_rate:.4f} "
            f"exceeds threshold {result.auc_threshold:.4f}"
        )

    logger.info(
        "Membership-inference gate PASSED: post-defence success rate %.4f < target %.4f",
        result.post_defence_success_rate,
        result.auc_threshold,
    )

    return result
