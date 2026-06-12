"""Real-time risk scoring using the trained ensemble.

Loads model artifacts from `config.MODEL_DIR` and combines per-model
probabilities into the single LedgerLens Risk Score (0-100) consumed by
the API and the `ledgerlens-score` Soroban contract.

TODO (not yet implemented):
  - Calibrate ensemble weighting (currently a simple average)
  - Combine ML probability with Benford `mad_nonconforming` flag into the
    final `benford_flag` / `ml_flag` / `confidence` fields described in the
    contract's `RiskScore` struct
  - Batch scoring entry point for the scheduled re-scoring job
"""

import os

import joblib
import pandas as pd

from config import config
from detection.model_training import MODEL_REGISTRY, FEATURE_COLUMNS_EXCLUDE


class RiskScorer:
    """Loads trained ensemble models and produces risk scores."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.models = self._load_models()

    def _load_models(self) -> dict:
        models = {}
        for name in MODEL_REGISTRY:
            path = os.path.join(self.model_dir, f"{name}.joblib")
            if os.path.exists(path):
                models[name] = joblib.load(path)
        return models

    def score(self, feature_row: pd.Series) -> dict:
        """Score a single wallet's feature row.

        Returns a dict matching the on-chain `RiskScore` shape:
            {score, benford_flag, ml_flag, confidence}
        """
        if not self.models:
            raise RuntimeError(
                f"No trained models found in {self.model_dir}. "
                "Run model_training.py first."
            )

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
        X = feature_row[feature_cols].to_frame().T

        probs = [model.predict_proba(X)[0, 1] for model in self.models.values()]
        avg_prob = sum(probs) / len(probs)

        benford_mad_cols = [c for c in feature_row.index if c.startswith("benford_mad_")]
        benford_flag = bool(
            benford_mad_cols and (feature_row[benford_mad_cols] > 0.015).any()
        )

        return {
            "score": int(round(avg_prob * 100)),
            "benford_flag": benford_flag,
            "ml_flag": avg_prob >= 0.5,
            "confidence": int(round(avg_prob * 100)),
        }

    def score_matrix(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Score every row in a feature matrix, returning the matrix with
        `score`, `benford_flag`, `ml_flag`, `confidence` columns appended."""
        scores = feature_matrix.apply(self.score, axis=1, result_type="expand")
        return pd.concat([feature_matrix[["wallet"]], scores], axis=1)
