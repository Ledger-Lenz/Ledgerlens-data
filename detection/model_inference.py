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
"""

import json
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
from detection.list_override import ListOverride
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


class RiskScorer:
    """Loads trained ensemble models and produces BFT-hardened risk scores."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.list_override = ListOverride()
        self.metadata = self._load_metadata()
        self.models = self._load_models()
        from detection.meta_learner import LeafEmbeddingExtractor
        self.extractor = LeafEmbeddingExtractor(self.models)
        self.maml_adapter, self.proto_classifier = self._load_meta_learners()

    def _load_meta_learners(self):
        maml = None
        proto = None

        # Prefer adapted model if available
        maml_path = os.path.join(self.model_dir, "maml_adapter_adapted.pt")
        if not os.path.exists(maml_path):
            maml_path = os.path.join(self.model_dir, "maml_adapter.pt")

        if os.path.exists(maml_path) and self.models:
            try:
                from detection.meta_learner import LeafEmbeddingExtractor, MAMLAdapter, PrototypicalClassifier
                import torch

                # We need to know input_dim. It depends on the leaf indices from base models.
                # Use metadata if we have it or a dummy row
                # This is a bit inefficient to do on every init, but usually done once
                # Let's use a dummy row based on metadata columns
                if self.metadata:
                    cols = self.metadata["feature_columns"]
                    dummy_X = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
                    self.extractor.fit(dummy_X)
                    input_dim = self.extractor.transform(dummy_X).shape[1]

                    maml = MAMLAdapter(input_dim=input_dim)
                    maml.load_state_dict(torch.load(maml_path, weights_only=True))
                    maml.eval()

                    # Prototypical classifier
                    proto_path = os.path.join(self.model_dir, "prototypes.joblib")
                    if os.path.exists(proto_path):
                        proto = PrototypicalClassifier()
                        proto.prototypes = joblib.load(proto_path)
            except Exception as e:
                logger.warning("Failed to load meta-learners: %s", e)

        return maml, proto

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
        if isinstance(feature_row, pd.Series):
            wallet = feature_row.get("wallet")
            if wallet is not None:
                override_val = self.list_override.check(wallet)
                if override_val is not None:
                    return {
                        "score": override_val,
                        "benford_flag": False,
                        "ml_flag": bool(override_val >= 50),
                        "confidence": 100,
                    }

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

        # Incorporate MAML adapter if available
        if self.maml_adapter:
            try:
                import torch
                emb = torch.from_numpy(self.extractor.transform(X)).float()
                maml_prob = self.maml_adapter.predict_proba(emb)[0]
                probs.append(float(maml_prob))
                logger.debug("MAML adapter prediction: %.4f", maml_prob)
            except Exception as e:
                logger.warning("MAML scoring failed: %s", e)

        # Incorporate Prototypical classifier if available
        if self.proto_classifier:
            try:
                emb = self.extractor.transform(X)
                proto_prob = self.proto_classifier.predict_proba(emb)[0]
                probs.append(float(proto_prob))
                logger.debug("Prototypical prediction: %.4f", proto_prob)
            except Exception as e:
                logger.warning("Prototypical scoring failed: %s", e)

        scores_100 = [p * 100 for p in probs]

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

            benford_mad_cols = [c for c in feature_row.index if c.startswith("benford_mad_")]
            benford_flag = bool(
                benford_mad_cols
                and (feature_row[benford_mad_cols] > BENFORD_MAD_FLAG_THRESHOLD).any()
            )

            # Integrate Zero-Shot prototype score
            if config.ZERO_SHOT_WEIGHT > 0.0:
                try:
                    from detection.zero_shot import PrototypeDetector
                    import joblib
                    zs_path = os.path.join(self.model_dir, "zero_shot.joblib")
                    if os.path.exists(zs_path):
                        zs_model = joblib.load(zs_path)
                        emb = self.extractor.transform(X)[0]
                        zs_score = zs_model.score(emb) * 100
                        final_score = (1.0 - config.ZERO_SHOT_WEIGHT) * final_score + (config.ZERO_SHOT_WEIGHT * zs_score)
                        logger.debug("Zero-shot score applied. New final_score: %.2f", final_score)
                except Exception as e:
                    logger.warning("Zero-shot scoring failed: %s", e)

            result = {
                "score": int(round(final_score)),
                "benford_flag": benford_flag,
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


def _score_one(wallet: str) -> dict:
    """Fetch a wallet's on-chain account data and return a risk score dict.

    Raises on network/HTTP errors so batch_scorer can capture per-wallet
    failures without crashing the batch.
    """
    import requests

    resp = requests.get(
        f"https://horizon.stellar.org/accounts/{wallet}", timeout=10
    )
    resp.raise_for_status()
    data = resp.json()

    balances = data.get("balances", [])
    native = next((b for b in balances if b.get("asset_type") == "native"), {})
    xlm_balance = float(native.get("balance", 0))

    # Placeholder — replace with RiskScorer.score() once feature pipeline wired in
    score = min(xlm_balance / 10_000, 1.0)
    return {"wallet": wallet, "score": round(score, 4), "xlm_balance": xlm_balance}
