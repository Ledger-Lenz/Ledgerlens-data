"""CLI script that detects feature drift and triggers automated retraining.

Exit codes:
    0  — No drift detected, no retraining needed.
    2  — Drift detected, retrained and promoted.
    3  — Drift detected, retrained but NOT promoted (metric regression).
    1  — Fatal error.

Usage:
    python -m scripts.retrain_if_drifted --lookback-days 30
    python -m scripts.retrain_if_drifted --lookback-days 30 --retrain-data-path data/synthetic_dataset.parquet
"""

import argparse
from typing import Any, cast

import json
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd

from config import config
from detection.drift_monitor import DriftMonitor
from detection.feature_engineering import build_feature_matrix
from detection.model_training import (
    MODEL_REGISTRY,
    load_training_data,
    save_models,
    save_training_artifacts,
    train_models,
)
from utils.logging import get_logger

logger = get_logger(__name__)

ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "archive"
)
REPORTS_DIR = "reports"
PROMOTION_TOLERANCE = 0.01


def get_feature_data(lookback_days: int) -> pd.DataFrame:
    """Load recent trades from Horizon and build a feature matrix.

    This is the "current distribution" used for drift detection.
    In production this hits the Horizon API; in tests it is mocked.
    """
    from ingestion.historical_loader import load_watched_pairs_to_dataframe

    since = datetime.now(UTC) - timedelta(days=lookback_days)
    logger.info("Loading trades since %s", since.isoformat())
    trades_df = load_watched_pairs_to_dataframe(start_time=since)
    logger.info("Loaded %d trades", len(trades_df))

    if trades_df.empty:
        logger.warning("No trades loaded; returning empty feature matrix")
        return pd.DataFrame()

    logger.info("Building feature matrix for drift detection")
    feature_matrix = build_feature_matrix(trades_df)
    logger.info("Built features for %d wallets", len(feature_matrix))
    return feature_matrix


def load_model_metadata(model_dir: str) -> dict | None:
    path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.exists(path):
        logger.error("model_metadata.json not found in %s", model_dir)
        return None
    with open(path) as f:
        return cast(dict[Any, Any], json.load(f))


def load_metrics(model_dir: str) -> dict | None:
    path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(path):
        logger.error("metrics.json not found in %s", model_dir)
        return None
    with open(path) as f:
        return cast(dict[Any, Any], json.load(f))


def archive_current_models(model_dir: str) -> str:
    """Archive the current production models to models/archive/{timestamp}/.

    Returns the archive path.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(ARCHIVE_DIR, timestamp)
    os.makedirs(archive_path, exist_ok=True)

    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isfile(item_path):
            shutil.copy2(item_path, os.path.join(archive_path, item))

    os.chmod(archive_path, 0o750)
    logger.info("Archived models to %s", archive_path)
    return archive_path


def evaluate_new_model(model_dir: str) -> dict | None:
    """Evaluate the newly trained model's metrics.

    Loads the metrics.json that was written by save_training_artifacts.
    """
    return load_metrics(model_dir)


def should_promote(
    old_metrics: dict[str, dict],
    new_metrics: dict[str, dict],
    tolerance: float = PROMOTION_TOLERANCE,
) -> tuple[bool, str]:
    """Check if the new model should be promoted.

    Requires AUC-ROC >= old_auc - tolerance AND F1 >= old_f1 - tolerance
    for every model in the ensemble.

    Returns (promote: bool, reason: str).
    """
    reasons = []
    for model_name in MODEL_REGISTRY:
        if model_name not in old_metrics:
            reasons.append(f"{model_name}: missing in old metrics")
            continue
        if model_name not in new_metrics:
            reasons.append(f"{model_name}: missing in new metrics")
            continue

        old = old_metrics[model_name]
        new = new_metrics[model_name]

        old_auc = old["auc_roc"]
        new_auc = new["auc_roc"]
        old_f1 = old["f1"]
        new_f1 = new["f1"]

        auc_ok = new_auc >= old_auc - tolerance
        f1_ok = new_f1 >= old_f1 - tolerance

        if not auc_ok:
            reasons.append(
                f"{model_name}: AUC-ROC {new_auc:.4f} < {old_auc:.4f} - {tolerance} "
                f"(delta {new_auc - old_auc:+.4f})"
            )
        if not f1_ok:
            reasons.append(
                f"{model_name}: F1 {new_f1:.4f} < {old_f1:.4f} - {tolerance} "
                f"(delta {new_f1 - old_f1:+.4f})"
            )

    if reasons:
        return False, "; ".join(reasons)
    return True, "All model metrics within tolerance — promoting."


def write_retrain_report(
    drift_report: dict,
    old_metrics: dict | None,
    new_metrics: dict | None,
    promotion_decision: bool,
    reason: str,
    archive_path: str | None,
) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"retrain_report_{timestamp}.json")

    report = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "drift_report": drift_report,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "promotion_decision": promotion_decision,
        "reason": reason,
        "archive_path": archive_path,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote retrain report to %s", path)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect feature drift and trigger retraining")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Number of days to look back for current feature distribution (default: 30)",
    )
    parser.add_argument(
        "--retrain-data-path",
        default=None,
        help="Path to labelled parquet dataset for retraining (required if drift detected)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model directory (default: config.MODEL_DIR)",
    )
    parser.add_argument(
        "--feature-data-path",
        default=None,
        help="Path to pre-computed feature matrix parquet for drift detection (bypasses Horizon API)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split ratio for retraining"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for train/test split"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = args.model_dir or config.MODEL_DIR

    logger.info("Loading model metadata from %s", model_dir)
    metadata = load_model_metadata(model_dir)
    if metadata is None:
        logger.error("Cannot proceed without model_metadata.json")
        return 1

    feature_distributions = metadata.get("feature_distributions")
    if feature_distributions is None:
        logger.error(
            "model_metadata.json has no feature_distributions (re-train with updated model_training.py)"
        )
        return 1

    logger.info("Loading current feature distribution")
    if args.feature_data_path:
        current_data = pd.read_parquet(args.feature_data_path)
    else:
        current_data = get_feature_data(args.lookback_days)

    if current_data.empty:
        logger.warning("Current feature matrix is empty — cannot compute drift")
        return 0

    logger.info("Computing feature drift")
    monitor = DriftMonitor(feature_distributions)
    feature_cols = [c for c in current_data.columns if c not in {"wallet", "label"}]
    drift_report = monitor.compute(current_data[feature_cols])

    drift_dict = drift_report.to_dict()
    logger.info(
        "Drift check complete: %d/%d features drifted",
        drift_dict["n_features_drifted"],
        drift_dict["n_features_checked"],
    )

    if not drift_report.any_drift_detected:
        logger.info("No significant drift detected — no retraining needed")
        write_retrain_report(drift_dict, None, None, False, "No drift detected", None)
        return 0

    logger.info("Drift detected — starting retraining pipeline")

    if not args.retrain_data_path:
        logger.error("Drift detected but --retrain-data-path not provided")
        return 1

    logger.info("Archiving current models")
    archive_path = archive_current_models(model_dir)

    old_metrics = load_metrics(model_dir)

    logger.info("Loading training data from %s", args.retrain_data_path)
    df = load_training_data(args.retrain_data_path)
    logger.info("Loaded %d labelled rows", len(df))

    logger.info("Training new ensemble models")
    training_output = train_models(df, test_size=args.test_size, random_state=args.random_state)
    results = training_output["results"]

    temp_model_dir = model_dir + "_new"
    os.makedirs(temp_model_dir, exist_ok=True)
    save_models(results, temp_model_dir)
    save_training_artifacts(training_output, args.retrain_data_path, temp_model_dir)

    new_metrics = evaluate_new_model(temp_model_dir)
    if new_metrics is None:
        logger.error("Failed to evaluate new model metrics")
        shutil.rmtree(temp_model_dir, ignore_errors=True)
        return 1

    promote, reason = should_promote(old_metrics or {}, new_metrics)
    logger.info("Promotion decision: %s — %s", promote, reason)

    if promote:
        for fname in os.listdir(temp_model_dir):
            src = os.path.join(temp_model_dir, fname)
            dst = os.path.join(model_dir, fname)
            shutil.copy2(src, dst)
        logger.info("New models promoted to %s", model_dir)
        write_retrain_report(
            drift_dict,
            old_metrics,
            new_metrics,
            promote,
            reason,
            archive_path,
        )
        shutil.rmtree(temp_model_dir, ignore_errors=True)
        return 2
    else:
        logger.warning("New models did not meet promotion criteria — archived but not promoted")
        write_retrain_report(
            drift_dict,
            old_metrics,
            new_metrics,
            promote,
            reason,
            archive_path,
        )
        for fname in os.listdir(temp_model_dir):
            src = os.path.join(temp_model_dir, fname)
            dst = os.path.join(archive_path, fname)
            shutil.copy2(src, dst)
        logger.info("New models also archived to %s", archive_path)
        shutil.rmtree(temp_model_dir, ignore_errors=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
