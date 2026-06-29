"""Estimate per-feature SHAP sensitivity for the differential-privacy layer.

For each feature the *sensitivity* Δ is the maximum change in that feature's
SHAP value when one trade is added/removed (Issue #59). Because the script
operates on the feature matrix rather than raw trades, the leave-one-trade-out
delta is approximated by the per-trade share of the feature's attribution: a
single trade can move a feature's SHAP value by at most its current attribution
divided across the wallet's trades. Δ for a feature is therefore the maximum,
over all training samples, of ``|SHAP_i| / n_trades`` (or ``|SHAP_i| *
rel_epsilon`` when no ``n_trades`` column is available). This is always
non-negative and is positive whenever the feature ever receives attribution.

The result is written to ``models/shap_sensitivity.json`` as
``{model_name: {feature: sensitivity}}`` and consumed by
``ShapExplainer.explain_private`` to calibrate the Gaussian noise.

Usage:
    python -m scripts.estimate_shap_sensitivity \
        --model-dir ./models --data data/synthetic_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import shap

from config import config
from detection.model_training import FEATURE_COLUMNS_EXCLUDE
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REL_EPSILON = 0.05


def _shap_matrix(model, X: pd.DataFrame) -> np.ndarray:
    """Return the positive-class SHAP values as an (n_samples, n_features) array."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # [class_0, class_1]
        values = values[1]
    else:
        values = np.asarray(values)
        if values.ndim == 3:  # (n_samples, n_features, n_classes)
            values = values[:, :, 1]
    return np.asarray(values)


def estimate_sensitivity(
    model,
    X: pd.DataFrame,
    n_trades: np.ndarray | None = None,
    rel_epsilon: float = DEFAULT_REL_EPSILON,
) -> dict[str, float]:
    """Estimate per-feature SHAP sensitivity for `model` on feature matrix `X`."""
    feature_cols = [c for c in X.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    features = X[feature_cols].astype(float).reset_index(drop=True)

    abs_shap = np.abs(_shap_matrix(model, features))

    if n_trades is not None:
        n_trades = np.asarray(n_trades, dtype=float)
        n_trades = np.where(n_trades > 0, n_trades, 1.0)
        per_trade = abs_shap / n_trades[:, None]
    else:
        per_trade = abs_shap * rel_epsilon

    return {col: float(per_trade[:, idx].max()) for idx, col in enumerate(feature_cols)}


def estimate_for_models(
    models: dict, X: pd.DataFrame, n_trades: np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    """Run `estimate_sensitivity` for every model in `models`."""
    result = {}
    for name, model in models.items():
        logger.info("Estimating SHAP sensitivity for model %s", name)
        result[name] = estimate_sensitivity(model, X, n_trades=n_trades)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate SHAP sensitivity for DP calibration")
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--data", default="data/synthetic_dataset.parquet")
    parser.add_argument("--output", default=config.SHAP_SENSITIVITY_PATH)
    return parser.parse_args()


def main() -> None:
    # Imported lazily so the reusable estimator helpers above don't pull in the
    # full inference stack at module import time.
    from detection.model_inference import RiskScorer

    args = _parse_args()

    df = pd.read_parquet(args.data)
    n_trades = df["n_trades"].to_numpy() if "n_trades" in df.columns else None
    feature_df = df.drop(columns=[c for c in ("label",) if c in df.columns])

    scorer = RiskScorer(model_dir=args.model_dir)
    if not scorer.models:
        raise SystemExit(f"No trained models found in {args.model_dir}")

    sensitivities = estimate_for_models(scorer.models, feature_df, n_trades=n_trades)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(sensitivities, fh, indent=2)
    logger.info("Wrote SHAP sensitivities for %d model(s) to %s", len(sensitivities), args.output)


if __name__ == "__main__":
    main()
