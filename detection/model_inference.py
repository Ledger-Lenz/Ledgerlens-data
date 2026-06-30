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

Zero-shot routing (Issue #274):
- When a wallet's asset pair has fewer than ZERO_SHOT_MIN_LABELLED_EXAMPLES
  labelled training examples, scoring is routed through ZeroShotPatternDetector
  instead of the ensemble. The zero-shot confidence is blended with the
  ensemble score via config.ZERO_SHOT_WEIGHT when both are available.
"""

import hashlib
import json
import math
import os
import statistics
import time
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from config import config
from detection.conformal import ConformalCalibrator
from detection.list_override import ListOverride
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
    from prometheus_client import Counter, Summary

    bft_divergence_detected_total: Counter | None = Counter(
        "bft_divergence_detected_total",
        "Number of times BFT divergence was detected during ensemble scoring",
    )
    ledgerlens_cluster_scored_total: Counter | None = Counter(
        "ledgerlens_cluster_scored_total",
        "Total number of wallet clusters scored by score_cluster()",
    )
    ledgerlens_canary_score_delta: Summary | None = Summary(
        "ledgerlens_canary_score_delta_seconds",
        "Absolute score difference between canary and champion model per wallet",
    )
except Exception:  # pragma: no cover
    bft_divergence_detected_total = None
    ledgerlens_cluster_scored_total = None
    ledgerlens_canary_score_delta = None


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


def _zero_shot_score(feature_row: pd.Series) -> dict | None:
    """Score via ZeroShotPatternDetector if patterns file exists.

    Returns a partial score dict on success, None if unavailable.
    """
    try:
        from detection.zero_shot import ZeroShotPatternDetector

        feature_names = [c for c in feature_row.index if c not in ("wallet",)]
        detector = ZeroShotPatternDetector.load(feature_names)
        return detector.score(feature_row.to_dict())
    except Exception as exc:
        logger.debug("Zero-shot scoring unavailable: %s", exc)
        return None


def _benford_flag(feature_row: pd.Series) -> bool:
    benford_mad_cols = [c for c in feature_row.index if c.startswith("benford_mad_")]
    return bool(
        benford_mad_cols and (feature_row[benford_mad_cols] > BENFORD_MAD_FLAG_THRESHOLD).any()
    )


def _apply_output_perturbation(
    score: int, caller_id: str = "internal", timestamp_bucket: int | None = None
) -> int:
    """Apply Laplace output perturbation to defend against model inversion (Issue #264).

    The random seed is derived from (caller_id, timestamp_bucket) to ensure:
    - Different results for identical queries (prevents averaging attacks)
    - Reproducibility within the same time bucket (for testing)

    Args:
        score: The clean 0-100 risk score
        caller_id: Caller identifier (default "internal" skips perturbation)
        timestamp_bucket: Timestamp divided by a bucketing interval (e.g. 60s)

    Returns:
        Perturbed score, rounded to SCORE_ROUNDING_GRANULARITY
    """
    # Internal pipeline calls use caller_id="internal" and skip perturbation
    if caller_id == "internal":
        return score

    # Derive seed from (caller_id, timestamp_bucket) for seeded randomness
    if timestamp_bucket is None:
        timestamp_bucket = int(time.time())

    seed_str = f"{caller_id}:{timestamp_bucket}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31 - 1)
    rng = np.random.Generator(np.random.PCG64(seed))

    # Compute Laplace scale: sensitivity / epsilon
    # Sensitivity is the max score change from one trade, bounded by 100
    sensitivity = 100.0
    scale = laplace_scale(sensitivity, config.MODEL_INVERSION_DP_EPSILON)

    # Add noise and round
    perturbed = add_laplace_noise(float(score), scale, rng)
    perturbed = np.clip(perturbed, 0.0, 100.0)
    rounded = int(round(perturbed / config.SCORE_ROUNDING_GRANULARITY)) * config.SCORE_ROUNDING_GRANULARITY
    return rounded


class RiskScorer:
    """Loads trained ensemble models and produces BFT-hardened risk scores."""

    def __init__(self, model_dir: str | None = None, weights: dict[str, float] | None = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.weights = self._validate_weights(weights)
        self.list_override = ListOverride()
        self.metadata = self._load_metadata()
        self.models = self._load_models()
        self.calibrators: dict[str, ConformalCalibrator] = {}
        self._load_calibrators()
        from detection.meta_learner import LeafEmbeddingExtractor

        self.extractor = LeafEmbeddingExtractor(self.models)
        self.maml_adapter, self.proto_classifier = self._load_meta_learners()

    def _load_calibrators(self) -> None:
        """Load conformal calibration artifacts for each model.

        Missing artifacts are logged as warnings but do not crash — a
        maximally conservative interval is used as fallback.
        """
        from detection.conformal import CalibrationIntegrityError

        for name in MODEL_REGISTRY:
            path = os.path.join(self.model_dir, f"{name}_conformal.json")
            try:
                calibrator = ConformalCalibrator.load(path)
                self.calibrators[name] = calibrator
                logger.info("Loaded conformal calibration for %s", name)
            except FileNotFoundError:
                logger.warning(
                    "No conformal calibration artifact for %s at %s — "
                    "uncertainty scoring will use maximally conservative fallback",
                    name,
                    path,
                )
            except CalibrationIntegrityError:
                logger.warning(
                    "Conformal calibration artifact for %s failed integrity check — "
                    "uncertainty scoring will use maximally conservative fallback",
                    name,
                )
            except Exception:
                logger.warning(
                    "Failed to load conformal calibration for %s — "
                    "uncertainty scoring will use maximally conservative fallback",
                    name,
                    exc_info=True,
                )

    def _load_meta_learners(self):
        maml = None
        proto = None

        # Prefer adapted model if available
        maml_path = os.path.join(self.model_dir, "maml_adapter_adapted.pt")
        if not os.path.exists(maml_path):
            maml_path = os.path.join(self.model_dir, "maml_adapter.pt")

        if os.path.exists(maml_path) and self.models:
            try:
                import torch

                from detection.meta_learner import (
                    MAMLAdapter,
                    PrototypicalClassifier,
                )

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
                        from detection.persistence import ModelArtifact, ModelIntegrityError

                        proto = PrototypicalClassifier()
                        proto.prototypes = joblib.load(proto_path)
                        try:
                            ModelArtifact(self.model_dir).verify_chain("prototypes")
                        except ModelIntegrityError as exc:
                            logger.warning(
                                "Artifact integrity check skipped or failed for prototypes: %s",
                                exc,
                            )
            except Exception as e:
                logger.warning("Failed to load meta-learners: %s", e)

        return maml, proto

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
                return cast(dict[Any, Any], json.load(f))
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

    def _load_selected_features(self) -> list[str] | None:
        if not config.FEATURE_SELECTION_ENABLED:
            return None
        path = config.FEATURE_SELECTION_PATH
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data.get("selected_features")
        return None

    def _ensemble_probabilities(self, feature_row: pd.Series) -> list[float]:
        """Per-model wash-trade probabilities for a single feature row.

        Raises if no models are loaded so callers (`score`,
        `score_continuous`) surface the same error.
        """
        if not self.models:
            raise RuntimeError(
                f"No trained models found in {self.model_dir}. Run model_training.py first."
            )

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]

        # Apply feature selection filter when enabled
        if self.selected_features is not None:
            feature_cols = [c for c in feature_cols if c in self.selected_features]

        if self.metadata and self.selected_features is None:
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
        return [model.predict_proba(X)[0, 1] for model in self.models.values()]

    def _check_override(self, feature_row: pd.Series) -> dict | None:
        wallet = feature_row.get("wallet") if isinstance(feature_row, pd.Series) else None
        if wallet is None:
            return None
        override_val = self.list_override.check(wallet)
        if override_val in (0, 100):
            return {
                "score": override_val,
                "benford_flag": False,
                "ml_flag": bool(override_val >= 50),
                "confidence": 100,
            }
        return None

    def score(
        self,
        feature_row: pd.Series,
        labelled_count: int | None = None,
    ) -> dict:
        """Compute the LedgerLens Risk Score for a single feature row.

        When ``labelled_count`` is provided and is less than
        ``config.ZERO_SHOT_MIN_LABELLED_EXAMPLES``, scoring is routed through
        the zero-shot pattern detector instead of the ensemble.

        Returns a dict matching the on-chain `RiskScore` shape:
            {score, benford_flag, ml_flag, confidence}

        Args:
            feature_row: Feature matrix row for the wallet/pair
            caller_id: Caller identifier (e.g. "api.user123", "internal"). 
                      "internal" skips output perturbation; external callers get Laplace noise.
        """
        override = self._check_override(feature_row)
        if override is not None:
            return override

        # Zero-shot routing for asset pairs with insufficient labelled data
        if (
            labelled_count is not None
            and labelled_count < config.ZERO_SHOT_MIN_LABELLED_EXAMPLES
        ):
            zs = _zero_shot_score(feature_row)
            if zs is not None:
                zs_score = int(round(zs["confidence"] * 100))
                return {
                    "score": zs_score,
                    "benford_flag": _benford_flag(feature_row),
                    "ml_flag": bool(zs["prediction"] == 1),
                    "confidence": int(round(zs["confidence"] * 100)),
                    "zero_shot": True,
                    "matched_pattern": zs.get("matched_pattern"),
                }

        probs = self._ensemble_probabilities(feature_row)

        if self.weights is not None:
            missing = set(self.weights) - set(self.models)
            if missing:
                raise ValueError(f"weights reference unknown models: {sorted(missing)}")

            avg_prob = sum(
                self.weights.get(name, 0.0) * prob
                for name, prob in zip(self.models, probs, strict=True)
            )
            clean_score = int(round(avg_prob * 100))
            perturbed_score = _apply_output_perturbation(clean_score, caller_id)
            return {
                "score": perturbed_score,
                "benford_flag": _benford_flag(feature_row),
                "ml_flag": bool(avg_prob >= ML_FLAG_THRESHOLD),
                "confidence": _confidence_from_probs(probs, avg_prob),
                "calibrated": True,
            }

        scores_100 = [p * 100 for p in probs]
        final_score, diverged = bft_trimmed_mean(scores_100)
        if diverged:
            logger.warning(
                "BFT divergence detected — raw model scores: %s",
                [round(s, 1) for s in scores_100],
            )
            _increment_bft_counter()

        if not _has_consensus(scores_100):
            result: dict = {
                "score": 100,
                "benford_flag": _benford_flag(feature_row),
                "ml_flag": True,
                "confidence": 0,
                "consensus_failure": True,
            }
            if diverged:
                result["bft_divergence"] = True
            return result

        avg_prob = final_score / 100.0
        perturbed_score = _apply_output_perturbation(int(round(final_score)), caller_id)
        result = {
            "score": perturbed_score,
            "benford_flag": _benford_flag(feature_row),
            "ml_flag": bool(avg_prob >= ML_FLAG_THRESHOLD),
            "confidence": _confidence_from_probs(probs, avg_prob),
        }
        if diverged:
            result["bft_divergence"] = True
        return result

    def score_with_uncertainty(self, feature_row: pd.Series) -> dict:
        """Compute risk score with conformal prediction uncertainty bounds.

        Returns the existing score dict plus:
        ``score_lower``, ``score_upper``, ``prediction_set``, ``coverage_guarantee``.

        Falls back to maximally conservative bounds when calibration artifacts
        are not available.
        """
        base_score = self.score(feature_row)

        if not self.calibrators or not self.models:
            return {
                **base_score,
                "score_lower": 0.0,
                "score_upper": 100.0,
                "prediction_set": [],
                "coverage_guarantee": 1.0,
            }

        feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
        X = feature_row[feature_cols].to_frame().T.astype(float)

        lowers: list[float] = []
        uppers: list[float] = []
        for name, model in self.models.items():
            calibrator = self.calibrators.get(name)
            if calibrator is None:
                lowers.append(0.0)
                uppers.append(100.0)
                continue
            try:
                intervals = calibrator.predict_with_interval(model, X)
                lowers.append(intervals[0]["lower"])
                uppers.append(intervals[0]["upper"])
            except Exception:
                lowers.append(0.0)
                uppers.append(100.0)

        score_lower = max(0.0, min(lowers))
        score_upper = min(100.0, max(uppers))

        coverage_guarantee = 1.0
        if self.calibrators:
            coverage_guarantee = 1.0 - next(iter(self.calibrators.values())).alpha

        prediction_set: list[int] = []
        for name, model in self.models.items():
            calibrator = self.calibrators.get(name)
            if calibrator is None:
                continue
            try:
                sets = calibrator.predict_set(model, X)
                if sets:
                    prediction_set = sets[0].get("prediction_set", [])
                    break
            except Exception:
                continue

        return {
            **base_score,
            "score_lower": score_lower,
            "score_upper": score_upper,
            "prediction_set": prediction_set,
            "coverage_guarantee": coverage_guarantee,
        }

    def score_continuous(self, feature_row: pd.Series) -> float:
        """Continuous ensemble risk score in `[0, 100]` (unrounded).

        Mirrors `score()`'s combination logic (list-override short-circuit,
        weighted average, or BFT trimmed mean) so that, absent a consensus
        failure, `round(score_continuous(row)) == score(row)["score"]` holds.
        Deliberately does *not* apply `score()`'s consensus-failure override
        (snapping to 100): the adversarial-robustness tooling perturbs this
        value via gradient/finite-difference search and needs a smooth
        scalar, not a discontinuous step that can rise under attack when a
        perturbation pushes models out of agreement.
        """
        override = self._check_override(feature_row)
        if override is not None:
            return float(override["score"])

        probs = self._ensemble_probabilities(feature_row)

        if self.weights is not None:
            return 100.0 * sum(
                self.weights.get(name, 0.0) * prob
                for name, prob in zip(self.models, probs, strict=True)
            )

        scores_100 = [p * 100 for p in probs]
        final_score, _ = bft_trimmed_mean(scores_100)
        return final_score

    def score_continuous_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Continuous ensemble scores for a batch of feature rows.

        `X` must contain (at least) the model feature columns; non-feature
        columns (`FEATURE_COLUMNS_EXCLUDE`) are dropped. Vectorised over the
        batch so the adversarial tooling can evaluate every finite-difference
        probe in one `predict_proba` call per model instead of one per row.
        """
        if not self.models:
            raise RuntimeError(
                f"No trained models found in {self.model_dir}. " "Run model_training.py first."
            )
        feature_cols = [c for c in X.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        Xf = X[feature_cols].astype(float)
        per_model = np.column_stack([m.predict_proba(Xf)[:, 1] for m in self.models.values()])
        return per_model.mean(axis=1) * 100

    def score_matrix(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Score every row in a feature matrix."""
        scores = feature_matrix.apply(self.score, axis=1, result_type="expand")
        return pd.concat([feature_matrix[["wallet"]], scores], axis=1)


# ---------------------------------------------------------------------------
# Canary deployment monitoring (issue #240)
# ---------------------------------------------------------------------------

_CANARY_P95_DELTA_LIMIT = 15.0  # promotion readiness threshold


class ModelCanaryMonitor:
    """Log per-wallet champion/canary score pairs and check promotion readiness.

    During the shadow period both the champion (production) model and the
    canary (candidate) model score every wallet.  This class records those
    pairs in-memory (or to a supplied DB table via ``log_score_pair``) so
    operators can inspect the delta distribution before promoting.

    Parameters
    ----------
    champion_version:
        Human-readable identifier for the current production model.
    canary_version:
        Human-readable identifier for the candidate model under evaluation.
    """

    def __init__(self, champion_version: str, canary_version: str) -> None:
        self.champion_version = champion_version
        self.canary_version = canary_version
        self._pairs: list[dict] = []

    def log_score_pair(
        self,
        wallet_id_hash: str,
        champion_score: float,
        canary_score: float,
    ) -> None:
        """Record a (champion, canary) score pair for a hashed wallet ID.

        Wallet addresses are never stored; callers must pass a pre-hashed ID
        (e.g. ``hashlib.sha256(wallet.encode()).hexdigest()``).

        Also updates the Prometheus summary metric with the absolute delta.
        """
        delta = abs(canary_score - champion_score)
        self._pairs.append(
            {
                "wallet_id_hash": wallet_id_hash,
                "champion_score": champion_score,
                "canary_score": canary_score,
                "delta": delta,
                "champion_version": self.champion_version,
                "canary_version": self.canary_version,
            }
        )
        if ledgerlens_canary_score_delta is not None:
            try:
                ledgerlens_canary_score_delta.observe(delta)
            except Exception:
                pass

    def score_pairs(self) -> list[dict]:
        """Return all logged score pairs for the current canary version."""
        return list(self._pairs)

    def disagreement_rate(self, band_threshold: float = 15.0) -> float:
        """Fraction of wallets where |canary - champion| > band_threshold."""
        if not self._pairs:
            return 0.0
        disagreements = sum(1 for p in self._pairs if p["delta"] > band_threshold)
        return disagreements / len(self._pairs)

    def top_divergent_wallets(self, n: int = 10) -> list[dict]:
        """Return the n wallets with the largest score divergence."""
        return sorted(self._pairs, key=lambda p: p["delta"], reverse=True)[:n]

    def promotion_readiness(self) -> dict:
        """Compute a promotion readiness summary.

        Returns a dict with ``ready`` (bool) and diagnostic fields.
        The check fails (``ready=False``) if the p95 absolute score delta
        exceeds ``_CANARY_P95_DELTA_LIMIT`` (15 points).
        """
        if not self._pairs:
            return {
                "ready": False,
                "reason": "no score pairs logged",
                "pair_count": 0,
            }

        deltas = sorted(p["delta"] for p in self._pairs)
        n = len(deltas)
        p95_index = min(int(n * 0.95), n - 1)
        p95_delta = deltas[p95_index]
        mean_delta = sum(deltas) / n
        disagreement = self.disagreement_rate()

        ready = p95_delta <= _CANARY_P95_DELTA_LIMIT
        return {
            "ready": ready,
            "pair_count": n,
            "p95_delta": round(p95_delta, 2),
            "mean_delta": round(mean_delta, 2),
            "disagreement_rate": round(disagreement, 4),
            "champion_version": self.champion_version,
            "canary_version": self.canary_version,
            "reason": (
                None
                if ready
                else f"p95 score delta {p95_delta:.1f} exceeds limit {_CANARY_P95_DELTA_LIMIT}"
            ),
        }


def _score_one(wallet: str) -> dict:
    """Fetch a wallet's on-chain account data and return a risk score dict.

    Raises on network/HTTP errors so batch_scorer can capture per-wallet
    failures without crashing the batch.
    """
    import requests

    resp = requests.get(f"https://horizon.stellar.org/accounts/{wallet}", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    balances = data.get("balances", [])
    native: dict[str, Any] = next((b for b in balances if b.get("asset_type") == "native"), {})
    xlm_balance = float(native.get("balance", 0))

    # Placeholder — replace with RiskScorer.score() once feature pipeline wired in
    score = min(xlm_balance / 10_000, 1.0)
    return {"wallet": wallet, "score": round(score, 4), "xlm_balance": xlm_balance}


# ---------------------------------------------------------------------------
# Cluster-level risk scoring via DiffPool graph pooling (issue #269)
# ---------------------------------------------------------------------------


def _cluster_id(wallet_ids: list[str]) -> str:
    """Stable cluster identifier: SHA-256 of the sorted wallet address set.

    Prevents duplicate cluster scoring and lets results be deduplicated.
    """
    key = "|".join(sorted(wallet_ids))
    return hashlib.sha256(key.encode()).hexdigest()


def score_cluster(
    wallet_ids: list[str],
    graph,
    scorer: RiskScorer,
    feature_matrix: pd.DataFrame | None = None,
    pooler=None,
    encoder=None,
    wallet_metadata: dict[str, dict] | None = None,
) -> dict:
    """Score an entire suspected wash-trade ring as a unit.

    Extracts the subgraph for ``wallet_ids``, optionally runs DiffPool
    graph pooling (``pooler``) to capture ring-level topology, and returns a
    cluster-level risk score (0–100) alongside individual wallet scores.

    The cluster score is permutation-invariant: the result is the same
    regardless of the order of ``wallet_ids``.

    A Prometheus counter ``ledgerlens_cluster_scored_total`` is incremented
    on each call.

    Parameters
    ----------
    wallet_ids:
        Wallet addresses forming the suspected ring.
    graph:
        ``networkx.DiGraph`` of the full wallet interaction graph.
    scorer:
        Loaded ``RiskScorer`` instance.
    feature_matrix:
        Optional ``pd.DataFrame`` keyed by wallet, used to obtain individual
        per-wallet risk scores via ``scorer.score()``.
    pooler:
        Optional ``GraphLevelPooling`` instance.  When supplied together with
        ``encoder``, the DiffPool architecture contributes to the cluster score.
    encoder:
        Optional ``GNNEncoder`` instance.
    wallet_metadata:
        Optional per-node metadata forwarded to the encoder.

    Returns
    -------
    dict with keys:
        cluster_id, cluster_score (0–100), individual_scores, wallet_count.
    """
    if not wallet_ids:
        raise ValueError("wallet_ids must not be empty")

    cid = _cluster_id(wallet_ids)

    # Increment Prometheus counter
    if ledgerlens_cluster_scored_total is not None:
        try:
            ledgerlens_cluster_scored_total.inc()
        except Exception:  # pragma: no cover
            pass

    # --- Individual wallet scores (when feature matrix is available) ---
    individual_scores: dict[str, int] = {}
    if feature_matrix is not None:
        for wallet in sorted(wallet_ids):
            if wallet in feature_matrix.index:
                try:
                    row = feature_matrix.loc[wallet]
                    individual_scores[wallet] = scorer.score(row)["score"]
                except Exception as exc:
                    logger.warning("score_cluster: failed to score wallet %s: %s", wallet, exc)

    # --- Graph pooling contribution (when pooler + encoder are available) ---
    pooling_score: float | None = None
    if pooler is not None and encoder is not None:
        try:
            pooling_score = pooler.compute_cluster_score(
                graph, wallet_ids, encoder, wallet_metadata=wallet_metadata
            )
        except Exception as exc:
            logger.warning("score_cluster: DiffPool pooling failed: %s", exc)

    # --- Final cluster score aggregation ---
    if pooling_score is not None and individual_scores:
        # Blend: 50% pooling score + 50% mean individual score
        mean_ind = float(np.mean(list(individual_scores.values())))
        cluster_score = int(round(0.5 * pooling_score + 0.5 * mean_ind))
    elif pooling_score is not None:
        cluster_score = int(round(pooling_score))
    elif individual_scores:
        # No pooler: use the 90th-percentile of individual scores to reflect
        # that rings tend to have uniformly high-scoring members
        cluster_score = int(round(float(np.percentile(list(individual_scores.values()), 90))))
    else:
        # No features and no encoder: cannot score
        cluster_score = 0

    cluster_score = max(0, min(100, cluster_score))

    result = {
        "cluster_id": cid,
        "cluster_score": cluster_score,
        "individual_scores": individual_scores,
        "wallet_count": len(wallet_ids),
    }
    logger.info(
        "Cluster scored: cluster_id=%s wallet_count=%d cluster_score=%d",
        cid,
        len(wallet_ids),
        cluster_score,
    )
    return result
