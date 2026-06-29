"""Historical backtest framework.

Replays a labelled, held-out Parquet dataset through the detection
pipeline in batch mode and produces a standardized performance report, so
comparing model versions or catching a regression before deployment
doesn't require manually running scripts and comparing numbers by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 0.5


def _config_hash(model_config: dict[str, Any]) -> str:
    """Stable hash of `model_config` (excluding the non-serializable
    `predict_fn` override, if any) so a report is traceable to an exact
    configuration."""
    serializable = {k: v for k, v in model_config.items() if k != "predict_fn"}
    blob = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _resolve_predict_fn(model_config: dict[str, Any]) -> Callable[[pd.Series], float]:
    """Resolve a row -> risk-probability ([0, 1]) scoring function.

    `model_config["predict_fn"]` is used directly if provided -- the seam
    used by tests and offline replay against a fixed scoring function.
    Otherwise falls back to the live ensemble `RiskScorer`, scoring each
    row's feature columns through the production pipeline.
    """
    predict_fn = model_config.get("predict_fn")
    if predict_fn is not None:
        return predict_fn

    from detection.model_inference import RiskScorer

    scorer = RiskScorer()

    def _score_row(row: pd.Series) -> float:
        return float(scorer.score(row).get("score", 0)) / 100.0

    return _score_row


def _label_to_binary(labels: pd.Series) -> np.ndarray:
    """Binarize either hard labels (`"wash"`/`"clean"`, `0`/`1`, bool) or
    soft labels (0.0-1.0 confidence, thresholded at 0.5)."""
    if labels.dtype == object or labels.dtype == bool:
        positive = {"wash", "1", "true", "positive"}
        return labels.astype(str).str.lower().isin(positive).to_numpy(dtype=int)
    return (labels.to_numpy(dtype=float) >= 0.5).astype(int)


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _per_asset_pair_breakdown(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict[str, Any]]:
    pairs = df["asset_pair"] if "asset_pair" in df.columns else pd.Series(["unknown"] * len(df))
    breakdown: dict[str, dict[str, Any]] = {}
    for pair in pairs.unique():
        mask = (pairs == pair).to_numpy()
        breakdown[str(pair)] = {
            "row_count": int(mask.sum()),
            "precision": float(precision_score(y_true[mask], y_pred[mask], zero_division=0)),
            "recall": float(recall_score(y_true[mask], y_pred[mask], zero_division=0)),
            "f1": float(f1_score(y_true[mask], y_pred[mask], zero_division=0)),
        }
    return breakdown


def _write_pr_curve(y_true: np.ndarray, y_score: np.ndarray, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(np.unique(y_true)) < 2:
        precisions, recalls = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    else:
        precisions, recalls, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots()
    ax.plot(recalls, precisions)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(path)
    plt.close(fig)


def run_backtest(
    dataset_path: str,
    model_config: dict[str, Any],
    output_dir: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Replay a labelled Parquet dataset through the detection pipeline and
    write a standardized performance report.

    Args:
        dataset_path: Parquet file with a `label` column (hard
            wash/clean strings, or soft 0-1 confidence floats) and an
            `asset_pair` column, plus whatever feature columns the scoring
            function needs. Run in batch mode (the whole file is loaded
            up front) for reproducibility.
        model_config: arbitrary dict describing the model under test;
            hashed (excluding any injected `predict_fn`) into the report
            so results stay traceable to an exact configuration. May
            include a `predict_fn` callable to override the live
            `RiskScorer` (used by tests / offline replay against fixed
            predictions).
        output_dir: directory for `backtest_report.json` and
            `pr_curve.png` (created if missing).
        threshold: risk-probability cutoff in [0, 1] used to binarize
            predictions for the confusion matrix / precision / recall /
            F1. Defaults to 0.5.

    Returns:
        The report dict (identical to what's written to
        `backtest_report.json`). Never includes wallet addresses --
        per-asset-pair breakdowns are keyed by asset pair, not by wallet,
        and no raw per-row wallet identifiers are persisted.
    """
    os.makedirs(output_dir, exist_ok=True)
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold

    df = pd.read_parquet(dataset_path).reset_index(drop=True)
    predict_fn = _resolve_predict_fn(model_config)

    y_score = np.array([predict_fn(row) for _, row in df.iterrows()], dtype=float)
    y_true = _label_to_binary(df["label"])
    y_pred = (y_score >= threshold).astype(int)

    report = {
        "model_config_hash": _config_hash(model_config),
        "threshold": threshold,
        "row_count": int(len(df)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": _safe_average_precision(y_true, y_score),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "per_asset_pair": _per_asset_pair_breakdown(df, y_true, y_pred),
    }

    with open(os.path.join(output_dir, "backtest_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    _write_pr_curve(y_true, y_score, os.path.join(output_dir, "pr_curve.png"))

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a historical detection backtest")
    parser.add_argument("dataset_path", help="Path to a labelled Parquet dataset")
    parser.add_argument("output_dir", help="Directory to write the report and PR curve into")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Risk-probability cutoff in [0, 1]"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_backtest(args.dataset_path, {}, args.output_dir, threshold=args.threshold)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
