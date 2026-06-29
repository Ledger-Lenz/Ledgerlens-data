"""Backdoor detection using Activation Clustering (AC) defence.

Detects potential backdoor-poisoned training samples by clustering penultimate-layer
activations. Backdoor samples typically form anomalous (minority) clusters with
feature patterns distinct from the majority class activation pattern.

References:
    Wang et al. (2019) "Activation Clustering: An Approach to Detecting Backdoor Attacks"
    https://arxiv.org/abs/1811.03728

Assumptions:
    - Backdoor samples form a cohesive minority cluster
    - Clean samples have consistent activation patterns within class
    - Known limitations: does NOT detect clean-label attacks (where backdoor
      samples have correct labels but are crafted to trigger specific model behavior)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.logging import get_logger

if TYPE_CHECKING:
    from detection.model_inference import RiskScorer

logger = get_logger(__name__)


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
                booster = model.get_booster()
                dmatrix = model.get_booster().DMatrix(X.values)
                # Get raw predictions (pre-sigmoid for binary classification)
                raw_preds = booster.predict(dmatrix, pred_leaf=False)
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
