"""Training entry point that trains ensemble models and then fits Platt calibrators (issue #288).

After standard model training (delegated to ``detection.model_training``), this
module:
1. Splits a 20% calibration holdout using ``config.CALIBRATION_RANDOM_SEED``.
2. Fits a ``PlattCalibrator`` per model using held-out scores.
3. Persists each calibrator to ``models/calibrator_{model_name}.pkl``.
4. Emits a reliability diagram PNG to ``models/calibration_curve_{model_name}.png``.

Usage
-----
    python -m training.train --data-path data/synthetic_dataset.parquet
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import config
from training.calibration import PlattCalibrator
from utils.logging import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS_EXCLUDE = {"wallet", "label", "profile"}


def _load_feature_matrix(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(data_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    feature_cols = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    return df[feature_cols], df["label"]


def train_and_calibrate(data_path: str, model_dir: str) -> None:
    """Train ensemble models (via detection.model_training) then calibrate."""
    # --- 1. Run standard model training ---
    # Import and invoke the existing training pipeline
    import subprocess

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "detection.model_training", "--data-path", data_path],
        check=False,
    )
    if result.returncode != 0:
        logger.error("detection.model_training exited with code %d", result.returncode)
        sys.exit(result.returncode)

    # --- 2. Load feature matrix for calibration ---
    X, y = _load_feature_matrix(data_path)
    _, X_cal, _, y_cal = train_test_split(
        X,
        y,
        test_size=config.CALIBRATION_SPLIT,
        random_state=config.CALIBRATION_RANDOM_SEED,
        stratify=y,
    )

    # --- 3. Calibrate each trained model ---
    model_names = ["random_forest", "xgboost", "lightgbm"]
    for model_name in model_names:
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            logger.warning("Model artifact not found, skipping calibration: %s", model_path)
            continue

        model = joblib.load(model_path)
        if hasattr(model, "predict_proba"):
            raw_scores = model.predict_proba(X_cal)[:, 1]
        else:
            raw_scores = model.predict(X_cal).astype(float)

        calibrator = PlattCalibrator()
        calibrator.fit(raw_scores, np.asarray(y_cal))

        cal_path = os.path.join(model_dir, f"calibrator_{model_name}.pkl")
        calibrator.save(cal_path)

        curve_path = os.path.join(model_dir, f"calibration_curve_{model_name}.png")
        calibrator.plot_calibration_curve(raw_scores, np.asarray(y_cal), curve_path)

        logger.info(
            "Calibrated %s | ECE=%.4f | saved to %s",
            model_name,
            calibrator.ece,
            cal_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ensemble + Platt calibration")
    parser.add_argument("--data-path", required=True, help="Path to labelled parquet dataset")
    parser.add_argument(
        "--model-dir", default=config.MODEL_DIR, help="Directory to write model artifacts"
    )
    args = parser.parse_args()
    train_and_calibrate(args.data_path, args.model_dir)


if __name__ == "__main__":
    main()
