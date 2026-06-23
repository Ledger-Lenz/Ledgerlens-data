"""Real-time risk scoring using the trained ensemble.

Loads model artifacts from `config.MODEL_DIR`, verifies artifact integrity,
and combines per-model probabilities using Byzantine-Fault-Tolerant (BFT)
trimmed-mean voting into the single LedgerLens Risk Score (0–100).

BFT voting:
- Sort the 3 model scores.
- If |max - min| > BFT_SCORE_DIVERGENCE_THRESHOLD, drop the outliers and
  use the median (for 3 models) — equivalent to a trimmed mean.
- If fewer than BFT_MIN_CONSENSUS models agree (within 10 points), return
  a ``consensus_failure`` score with maximum uncertainty.

Calibrated weighted mode:
- ``RiskScorer(weights=...)`` accepts a ``{model_name: weight}`` mapping
  (e.g. one selected from a Pareto front via
  ``detection.ensemble_calibrator.EnsembleCalibrator.select_operating_point``)
  and combines model probabilities as a weighted average instead of BFT
  voting. ``weights=None`` (the default) preserves the BFT behaviour above.
"""

import json
import math
import os
import statistics

import joblib
import pandas as pd

from config import config
from detection.model_training import (
    FEATURE_COLUMNS_EXCLUDE,
    MODEL_REGISTRY,
    compute_feature_schema_hash,
)
from utils.logging import get_logger

logger = get_logger(__name__)

BENFORD_MAD_FLAG_THRESHOLD = 0.015
ML_FLAG_THRESHOLD = 0.5
_CONSENSUS_WINDOW = 10  # two models must be within this many points of each other

# ---------------------------------------------------------------------------
# Prometheus counter (optional — gracefully absent if prometheus_client not
# installed or not yet wired to an exporter)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter

    bft_divergence_detected_total = Counter(
        "bft_divergence_detected_total",
        "Number of times BFT divergence was detected during ensemble scoring",
    )
except Exception:  # pragma: no cover
    bft_divergence_detected_total = None  # type: ignore[assignment]


def _increment_bft_counter() -> None:
    if bft_divergence_detected_total is not None:
        bft_divergence_detected_total.inc()


# ---------------------------------------------------------------------------
# BFT voting helpers
# ---------------------------------------------------------------------------


def bft_trimmed_mean(scores: list[float]) -> tuple[float, bool]:
    """Return ``(consensus_score, divergence_flag)`` using BFT trimmed mean.

    For exactly 3 models the trimmed mean degenerates to the median.
    Divergence is flagged when ``|max - min| > BFT_SCORE_DIVERGENCE_THRESHOLD``.
    """
    if len(scores) == 1:
        return scores[0], False

    span = max(scores) - min(scores)
    diverged = span > config.BFT_SCORE_DIVERGENCE_THRESHOLD

    if len(scores) == 3:
        return statistics.median(scores), diverged

    if diverged:
        trimmed = sorted(scores)[1:-1]
        return statistics.mean(trimmed), True

    return statistics.mean(scores), False


def _has_consensus(scores: list[float]) -> bool:
    """Return True if at least BFT_MIN_CONSENSUS models agree within the
    consensus window."""
    n = config.BFT_MIN_CONSENSUS
    for a in scores:
        count = sum(1 for b in scores if abs(a - b) <= _CONSENSUS_WINDOW)
        if count >= n:
            return True
    return False


def _confidence_from_probs(probs: list[float], avg_prob: float) -> int:
    certainty = abs(avg_prob - 0.5) * 2
    if len(probs) > 1:
        agreement = 1.0 - (max(probs) - min(probs))
        certainty *= max(agreement, 0.0)
    return int(round(certainty * 100))


def _benford_flag(feature_row: pd.Series) -> bool:
    benford_mad_cols = [c for c in feature_row.index if c.startswith("benford_mad_")]
    return bool(
        benford_mad_cols and (feature_row[benford_mad_cols] > BENFORD_MAD_FLAG_THRESHOLD).any()
    )


class RiskScorer:
    """Loads trained ensemble models and produces BFT-hardened risk scores."""

    def __init__(self, model_dir: str | None = None, weights: dict[str, float] | None = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.weights = self._validate_weights(weights)
        self.metadata = self._load_metadata()
        self.models = self._load_models()

    @staticmethod
    def _validate_weights(weights: dict[str, float] | None) -> dict[str, float] | None:
        if weights is None:
            return None
        total = sum(weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"RiskScorer weights must sum to 1.0, got {total}")
        return weights

    def _load_metadata(self) -> dict | None:
        path = os.path.join(self.model_dir, "model_metadata.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _load_models(self) -> dict:
        from detection.persistence import ModelArtifact, ModelIntegrityError

        artifact = ModelArtifact(self.model_dir)
        models = {}
        for name in MODEL_REGISTRY:
            path = os.path.join(self.model_dir, f"{name}.joblib")
            if os.path.exists(path):
                model = joblib.load(path)
                try:
                    artifact.verify_chain(name)
                except ModelIntegrityError as exc:
                    logger.warning(
                        "Artifact integrity check skipped or failed for %s: %s", name, exc
                    )
                models[name] = model
        return models

    def score(self, feature_row: pd.Series) -> dict:
        """Score a single wallet's feature row with BFT voting.

        Returns a dict matching the on-chain `RiskScore` shape:
            {score, benford_flag, ml_flag, confidence}

        When BFT divergence is detected the dict also contains:
            {"bft_divergence": True}

        When consensus cannot be reached:
            {"score": 100, "consensus_failure": True, ...}
        """
        if not self.models:
            raise RuntimeError(
                f"No trained models found in {self.model_dir}. Run model_training.py first."
            )

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]

        if self.metadata:
            current_hash = compute_feature_schema_hash(feature_cols)
            expected_hash = self.metadata["feature_schema_hash"]

            if current_hash != expected_hash:
                model_cols = set(self.metadata["feature_columns"])
                row_cols = set(feature_cols)
                missing_in_row = model_cols - row_cols
                missing_in_model = row_cols - model_cols

                msg = (
                    f"Feature schema mismatch! Model expected hash {expected_hash}, "
                    f"got {current_hash}."
                )
                if missing_in_row:
                    msg += f" Columns missing from input: {sorted(missing_in_row)}."
                if missing_in_model:
                    msg += f" Columns missing from model: {sorted(missing_in_model)}."
                raise RuntimeError(msg)

        X = feature_row[feature_cols].to_frame().T.astype(float)

        probs = [model.predict_proba(X)[0][1] for model in self.models.values()]
        scores_100 = [p * 100 for p in probs]

        if self.weights is not None:
            missing = set(self.weights) - set(self.models)
            if missing:
                raise ValueError(f"weights reference unknown models: {sorted(missing)}")

            avg_prob = sum(
                self.weights.get(name, 0.0) * prob
                for name, prob in zip(self.models, probs, strict=True)
            )
            return {
                "score": int(round(avg_prob * 100)),
                "benford_flag": _benford_flag(feature_row),
                "ml_flag": bool(avg_prob >= ML_FLAG_THRESHOLD),
                "confidence": _confidence_from_probs(probs, avg_prob),
                "calibrated": True,
            }

        result: dict = {}

        if not _has_consensus(scores_100):
            logger.warning(
                "BFT consensus failure — raw scores: %s",
                [round(s, 1) for s in scores_100],
            )
            _increment_bft_counter()
            avg_prob = sum(probs) / len(probs)
            result = {
                "score": 100,
                "benford_flag": False,
                "ml_flag": True,
                "confidence": 0,
                "consensus_failure": True,
                "bft_divergence": True,
            }
        else:
            final_score, diverged = bft_trimmed_mean(scores_100)

            if diverged:
                logger.warning(
                    "BFT divergence detected — raw model scores: %s",
                    [round(s, 1) for s in scores_100],
                )
                _increment_bft_counter()

            avg_prob = final_score / 100.0

            result = {
                "score": int(round(final_score)),
                "benford_flag": _benford_flag(feature_row),
                "ml_flag": bool(avg_prob >= ML_FLAG_THRESHOLD),
                "confidence": _confidence_from_probs(probs, avg_prob),
            }
            if diverged:
                result["bft_divergence"] = True

        return result

    def score_matrix(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Score every row in a feature matrix."""
        scores = feature_matrix.apply(self.score, axis=1, result_type="expand")
        return pd.concat([feature_matrix[["wallet"]], scores], axis=1)
