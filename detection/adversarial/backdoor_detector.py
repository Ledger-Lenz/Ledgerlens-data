"""Backdoor detection using Activation Clustering (AC) defence.

Detects potential backdoor-poisoned training samples by clustering penultimate-layer
activations. Backdoor samples typically form anomalous (minority) clusters with
feature patterns distinct from the majority class activation pattern.

Issue #871 extends plain detection (which cluster is anomalous) with
**trigger localization** (which *input features* define that anomaly) via a
difference-of-means effect size in the original feature space — a
simplified spectral-signature decomposition (Tran, Li & Madry, 2018,
"Spectral Signatures in Backdoor Attacks", https://arxiv.org/abs/1811.00636):
the difference-of-means vector between the flagged and clean samples
approximates the dominant direction of covariance a consistent trigger
pattern introduces, without needing the full SVD of the activation
covariance matrix. It also wires a flagged model into
`detection.model_governance.quarantine_candidate` (`scan_and_quarantine`)
so a suspected backdoor blocks promotion automatically, pending review.

References:
    Wang et al. (2019) "Activation Clustering: An Approach to Detecting Backdoor Attacks"
    https://arxiv.org/abs/1811.03728
    Tran, Li & Madry (2018) "Spectral Signatures in Backdoor Attacks"
    https://arxiv.org/abs/1811.00636

Assumptions:
    - Backdoor samples form a cohesive minority cluster
    - Clean samples have consistent activation patterns within class
    - Known limitations: does NOT detect clean-label attacks (where backdoor
      samples have correct labels but are crafted to trigger specific model behavior)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Non-feature columns to exclude from trigger localization / ring reporting.
_NON_FEATURE_COLUMNS = {"wallet", "label", "profile"}

# Default fraction of samples flagged by AC above which a candidate is
# considered likely-backdoored and recommended for auto-quarantine.
#
# Per-class k=2 clustering on RandomForest/XGBoost/LightGBM activations has a
# high, noisy *baseline* flagged fraction on entirely clean data — empirically
# 25-47% across 15 clean synthetic seeds (see
# tests/test_backdoor_detector.py::TestScanAndQuarantine
# ::test_false_positive_rate_on_clean_models_is_documented), independent of
# dataset size (measured up to n=2000) and largely independent of how extreme
# an injected trigger is (an obvious, cohesive, 10-sigma-shifted trigger
# cluster still did not reliably separate from clean noise in this
# per-class k=2 formulation). This threshold is therefore set *above* the
# observed clean ceiling to keep false positives low, at the cost of only
# catching unusually large-scale poisoning events -- not subtle,
# small-fraction attacks. See docs/adversarial_robustness.md for the full
# caveat and the recommendation to also review `candidate_trigger_features`
# manually even when auto-quarantine does not fire.
DEFAULT_QUARANTINE_FLAGGED_FRACTION_THRESHOLD = 0.5


class ActivationClusteringDetector:
    """Detects backdoor samples using k-means clustering on penultimate-layer activations."""

    def __init__(self, k: int = 2, random_state: int = 42):
        """Initialize detector.

        Args:
            k: Number of clusters (default 2: one majority, one potential backdoor)
            random_state: Random seed for k-means
        """
        self.k = k
        self.random_state = random_state
        self._scaler = StandardScaler()

    def detect(
        self,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        threshold_percentile: int = 25,
    ) -> list[int]:
        """Detect backdoor samples in the dataset.

        Uses k-means clustering on penultimate-layer activations to identify
        anomalous clusters. Flags samples in the smallest cluster (potential backdoor).

        Args:
            model: Trained scikit-learn model (RandomForest, XGBoost, or LightGBM)
            X: Feature matrix
            y: Labels (0=clean, 1=wash trade)
            threshold_percentile: Percentile for size-based outlier detection

        Returns:
            List of row indices flagged as potential backdoor samples

        Raises:
            ValueError: If model type is not supported or activation extraction fails
        """
        try:
            # Extract penultimate-layer activations
            activations = self._extract_activations(model, X)

            if activations is None or len(activations) == 0:
                logger.warning("Failed to extract activations; returning empty flagged list")
                return []

            # Separate by label for per-class clustering
            flagged_indices = []

            for label in sorted(y.unique()):
                mask = y == label
                if mask.sum() < self.k:
                    logger.debug(
                        "Skipping AC for label=%d: insufficient samples (%d < k=%d)",
                        label,
                        mask.sum(),
                        self.k,
                    )
                    continue

                label_activations = activations[mask]
                label_indices = np.where(mask)[0]

                # Cluster activations for this label
                flagged_for_label = self._cluster_and_flag(
                    label_activations,
                    label_indices,
                    label,
                    threshold_percentile,
                )
                flagged_indices.extend(flagged_for_label)

            return sorted(flagged_indices)

        except Exception as exc:
            logger.error("Activation clustering detection failed: %s", exc)
            return []

    def _extract_activations(self, model: object, X: pd.DataFrame) -> np.ndarray | None:
        """Extract penultimate-layer (pre-output) activations from model.

        Supports RandomForest, XGBoost, and LightGBM by extracting leaf indices
        or pre-output layer activations.

        Args:
            model: Trained model
            X: Feature matrix

        Returns:
            Activation matrix of shape (n_samples, n_activations), or None if unsupported
        """
        try:
            model_class_name = model.__class__.__name__

            if "RandomForest" in model_class_name:
                # Extract leaf indices as activations
                leaf_indices = model.apply(X)  # (n_samples, n_trees)
                return leaf_indices.astype(np.float32)

            elif "XGBClassifier" in model_class_name or "XGBRegressor" in model_class_name:
                # Extract leaf predictions (raw model output before final transformation)
                # Get raw predictions (pre-sigmoid for binary classification)
                raw_preds = model.predict(X, output_margin=True)
                # Shape: (n_samples,) for binary, or (n_samples, n_classes) for multiclass
                if raw_preds.ndim == 1:
                    raw_preds = raw_preds.reshape(-1, 1)
                return raw_preds.astype(np.float32)

            elif "LGBMClassifier" in model_class_name or "LGBMRegressor" in model_class_name:
                # Extract leaf predictions
                raw_preds = model.predict(X, raw_score=True)
                if raw_preds.ndim == 1:
                    raw_preds = raw_preds.reshape(-1, 1)
                return raw_preds.astype(np.float32)

            else:
                logger.warning("Unsupported model type: %s", model_class_name)
                return None

        except Exception as exc:
            logger.error("Failed to extract activations: %s", exc)
            return None

    def _cluster_and_flag(
        self,
        activations: np.ndarray,
        indices: np.ndarray,
        label: int,
        threshold_percentile: int,
    ) -> list[int]:
        """Cluster activations and flag minority cluster members.

        Args:
            activations: Activation matrix for this class
            indices: Original row indices corresponding to activations
            label: Class label
            threshold_percentile: Percentile for minimum cluster size

        Returns:
            List of flagged indices from the minority cluster
        """
        if len(activations) < self.k:
            return []

        try:
            # Standardize activations
            activations_scaled = self._scaler.fit_transform(activations)

            # Cluster
            kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(activations_scaled)

            # Find minority cluster
            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            minority_cluster = unique_clusters[np.argmin(counts)]
            minority_size = np.min(counts)

            # Safety check: if minority cluster is too large (> threshold_percentile),
            # something is wrong — don't flag
            min_threshold = np.percentile(counts, threshold_percentile)
            if minority_size > min_threshold:
                logger.debug(
                    "Label=%d: minority cluster size (%d) exceeds percentile threshold (%.1f)",
                    label,
                    minority_size,
                    min_threshold,
                )
                return []

            # Flag samples in minority cluster
            flagged_mask = cluster_labels == minority_cluster
            flagged_indices_for_label = indices[flagged_mask].tolist()

            logger.info(
                "Label=%d: flagged %d samples (cluster size %d / %d total)",
                label,
                len(flagged_indices_for_label),
                minority_size,
                len(activations),
            )

            return flagged_indices_for_label

        except Exception as exc:
            logger.error("Clustering for label=%d failed: %s", label, exc)
            return []

    def report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        flagged_indices: list[int],
    ) -> dict:
        """Generate a detection report.

        Args:
            X: Feature matrix
            y: Labels
            flagged_indices: Flagged sample indices

        Returns:
            Report dict with detection statistics
        """
        total = len(X)
        n_flagged = len(set(flagged_indices))
        flagged_labels = y.iloc[flagged_indices].value_counts().to_dict() if flagged_indices else {}

        return {
            "total_samples": total,
            "n_flagged": n_flagged,
            "flagged_percentage": 100.0 * n_flagged / total if total > 0 else 0.0,
            "flagged_by_label": flagged_labels,
            "method": "activation_clustering",
            "k": self.k,
        }

    def structured_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        flagged_indices: list[int],
        top_k_features: int = 5,
    ) -> dict:
        """Full incident-response report: detection stats, localized
        candidate trigger feature(s), affected sample/wallet identities, and
        (when available) affected wash-trading-ring concentration.

        This is the structured report the issue requires — combining
        `report()`'s aggregate stats with `localize_trigger_features()`'s
        feature-space localization, so an analyst can act on a flag instead
        of just seeing a yes/no.
        """
        base = self.report(X, y, flagged_indices)
        base["candidate_trigger_features"] = localize_trigger_features(
            X, flagged_indices, top_k=top_k_features
        )
        base["affected_samples"] = sorted(set(flagged_indices))
        if "wallet" in X.columns and flagged_indices:
            base["affected_wallets"] = X["wallet"].iloc[flagged_indices].tolist()
        ring_summary = _ring_concentration(X, flagged_indices)
        if ring_summary is not None:
            base["affected_ring_concentration"] = ring_summary
        return base


def localize_trigger_features(
    X: pd.DataFrame,
    flagged_indices: list[int],
    top_k: int = 5,
) -> list[dict]:
    """Rank feature columns by how strongly they separate the flagged
    (suspected-backdoor) samples from the rest of `X`.

    For each feature, computes a Cohen's-d-style standardized effect size —
    ``(mean(flagged) - mean(clean)) / pooled_std`` — so features are
    comparable across wildly different raw units (a Benford MAD of ~0.01 vs.
    an account age of ~1000 days). This approximates the dominant direction
    of the flagged/clean covariance shift a consistent trigger pattern
    introduces (a simplified spectral-signature decomposition; see module
    docstring), without requiring the full SVD Tran et al. (2018) use.

    Returns up to `top_k` entries `{"feature", "score", "direction"}` sorted
    by `|score|` descending (`score` is the signed effect size; `direction`
    is `"higher"` when the flagged group's mean is larger). Returns `[]` if
    there are no flagged indices, no feature columns, or every row is
    flagged (nothing to contrast against).
    """
    feature_cols = [c for c in X.columns if c not in _NON_FEATURE_COLUMNS]
    if not flagged_indices or not feature_cols or len(flagged_indices) >= len(X):
        return []

    flagged_mask = np.zeros(len(X), dtype=bool)
    flagged_mask[np.array(sorted(set(flagged_indices)), dtype=int)] = True

    flagged_X = X.iloc[flagged_mask][feature_cols].astype(float)
    clean_X = X.iloc[~flagged_mask][feature_cols].astype(float)

    scored: list[dict[str, Any]] = []
    for col in feature_cols:
        flagged_vals = flagged_X[col].to_numpy()
        clean_vals = clean_X[col].to_numpy()
        pooled_std = float(np.sqrt((flagged_vals.var(ddof=0) + clean_vals.var(ddof=0)) / 2.0))
        if pooled_std <= 1e-12:
            continue
        effect_size = float(flagged_vals.mean() - clean_vals.mean()) / pooled_std
        scored.append(
            {
                "feature": col,
                "score": effect_size,
                "direction": "higher" if effect_size > 0 else "lower",
            }
        )

    scored.sort(key=lambda entry: -abs(entry["score"]))
    return scored[:top_k]


def _ring_concentration(X: pd.DataFrame, flagged_indices: list[int]) -> dict | None:
    """Fraction of flagged vs. non-flagged samples inside a detected
    wash-trading ring (`in_wash_trading_ring`, see `detection.wallet_graph`),
    a cheap graph-substructure signal alongside the feature-space
    localization above. Returns `None` when that column isn't present."""
    if "in_wash_trading_ring" not in X.columns or not flagged_indices:
        return None

    flagged_mask = np.zeros(len(X), dtype=bool)
    flagged_mask[np.array(sorted(set(flagged_indices)), dtype=int)] = True
    in_ring = X["in_wash_trading_ring"].astype(bool).to_numpy()

    flagged_in_ring = int(np.sum(in_ring[flagged_mask]))
    clean_in_ring = int(np.sum(in_ring[~flagged_mask]))
    n_flagged = int(flagged_mask.sum())
    n_clean = int((~flagged_mask).sum())

    return {
        "flagged_in_ring_fraction": (flagged_in_ring / n_flagged) if n_flagged else 0.0,
        "clean_in_ring_fraction": (clean_in_ring / n_clean) if n_clean else 0.0,
    }


def scan_and_quarantine(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    candidate_dir: str,
    *,
    k: int = 2,
    threshold_percentile: int = 25,
    top_k_features: int = 5,
    flagged_fraction_threshold: float = DEFAULT_QUARANTINE_FLAGGED_FRACTION_THRESHOLD,
    actor: str = "system",
    session_factory: Any = None,
) -> dict:
    """Run activation-clustering backdoor detection + trigger localization
    against `model`/`X`/`y`, and auto-quarantine `candidate_dir` in
    `detection.model_governance` when the flagged fraction of samples
    exceeds `flagged_fraction_threshold`.

    This is the Issue #871 end-to-end entry point: called once after
    training (see `detection.model_training.main`, gated by
    `config.BACKDOOR_SCAN_ENABLED`), it returns the structured report and,
    when quarantine is recommended, the `ModelVersionRecord.version_id` of
    the quarantine entry it just created.
    """
    detector = ActivationClusteringDetector(k=k)
    flagged_indices = detector.detect(model, X, y, threshold_percentile=threshold_percentile)
    report = detector.structured_report(X, y, flagged_indices, top_k_features=top_k_features)

    flagged_fraction = (
        report["n_flagged"] / report["total_samples"] if report["total_samples"] else 0.0
    )
    report["flagged_fraction"] = flagged_fraction
    report["quarantine_recommended"] = flagged_fraction >= flagged_fraction_threshold

    if report["quarantine_recommended"]:
        top_features = ", ".join(
            f"{entry['feature']} ({entry['direction']}, d={entry['score']:.2f})"
            for entry in report["candidate_trigger_features"][:3]
        )
        reason = (
            f"Activation clustering flagged {report['n_flagged']}/{report['total_samples']} "
            f"({100 * flagged_fraction:.1f}%) training samples as a suspected backdoor cluster"
            + (f"; top candidate trigger features: {top_features}" if top_features else "")
        )
        report["reason"] = reason

        from detection.model_governance import quarantine_candidate

        record = quarantine_candidate(
            candidate_dir=candidate_dir,
            reason=reason,
            report=report,
            actor=actor,
            session_factory=session_factory,
        )
        report["quarantine_version_id"] = record.version_id

    return report
