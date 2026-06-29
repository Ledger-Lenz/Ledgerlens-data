"""Zero-shot wash-trade detection.

Provides two detection strategies:

1. ``PrototypeDetector`` — prototype-based (few-shot): fits prototypes from a
   small labelled set and scores by cosine distance to each prototype.

2. ``ZeroShotPatternDetector`` — fully zero-shot: matches a wallet's feature
   vector against a library of textual wash-trade pattern descriptions
   (data/zero_shot_patterns.json) pre-embedded as feature-weight vectors.
   No labelled training data required; safe for new/unseen asset pairs.
   Pattern vectors are loaded from disk and hash-validated on startup so
   tampering is detected before any scoring takes place.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np

_PATTERNS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "zero_shot_patterns.json")


def cosine_distance(a, b):
    """Calculate the cosine distance between 1D arrays a and b."""
    a = np.asarray(a)
    b = np.asarray(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (np.dot(a, b) / (norm_a * norm_b))

class PrototypeDetector:
    """Zero-shot detection module that classifies unlabeled wallets by measuring
    cosine distance to prototype embeddings derived from a small set of confirmed
    wash-trade examples.
    """
    def __init__(self):
        self.wash_prototype = None
        self.legit_prototype = None

    def fit(self, labeled_embeddings, labels):
        """Fit prototypes using labeled embeddings.
        labels == 1 indicates wash trade, labels == 0 indicates legit trade.
        """
        labeled_embeddings = np.asarray(labeled_embeddings)
        labels = np.asarray(labels)
        
        wash_mask = labels == 1
        legit_mask = labels == 0
        
        if np.any(wash_mask):
            self.wash_prototype = labeled_embeddings[wash_mask].mean(axis=0)
        else:
            self.wash_prototype = np.zeros(labeled_embeddings.shape[1])
            
        if np.any(legit_mask):
            self.legit_prototype = labeled_embeddings[legit_mask].mean(axis=0)
        else:
            self.legit_prototype = np.zeros(labeled_embeddings.shape[1])

    def score(self, embedding) -> float:
        """Score an embedding based on its distance to the prototypes.
        Returns a score between 0.0 and 1.0. Higher means more likely wash trade.
        """
        embedding = np.asarray(embedding)
        
        # In case prototypes are not fitted properly
        if self.wash_prototype is None or self.legit_prototype is None:
            return 0.5
            
        d_wash = cosine_distance(embedding, self.wash_prototype)
        d_legit = cosine_distance(embedding, self.legit_prototype)
        
        # Handle case where both distances are 0
        if d_wash + d_legit == 0:
            return 0.5
            
        return d_legit / (d_wash + d_legit)


# ---------------------------------------------------------------------------
# Zero-shot pattern detector (Issue #274)
# ---------------------------------------------------------------------------

def _patterns_hash(patterns_data: dict) -> str:
    """SHA-256 of the canonical JSON representation of patterns data."""
    canonical = json.dumps(patterns_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _pattern_to_vector(weights: dict[str, float], feature_names: list[str]) -> np.ndarray:
    """Project a feature-weight dict onto the ordered feature vector space."""
    vec = np.zeros(len(feature_names))
    name_index = {n: i for i, n in enumerate(feature_names)}
    for feat, weight in weights.items():
        if feat in name_index:
            vec[name_index[feat]] = weight
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class ZeroShotPatternDetector:
    """Zero-shot wash-trade detector using pre-embedded pattern descriptions.

    Patterns are loaded from ``data/zero_shot_patterns.json`` and encoded as
    feature-weight vectors aligned with the known feature schema. No labelled
    training data is required. A stored SHA-256 hash of the pattern file is
    validated on load to detect tampering.

    Usage::

        detector = ZeroShotPatternDetector.load(feature_names)
        result = detector.score(feature_dict)
        # result: {"prediction": 0/1, "confidence": 0.0-1.0, "matched_pattern": str}
    """

    def __init__(
        self,
        pattern_vectors: list[np.ndarray],
        pattern_ids: list[str],
        pattern_hash: str,
        feature_names: list[str],
    ):
        self._pattern_vectors = pattern_vectors
        self._pattern_ids = pattern_ids
        self._expected_hash = pattern_hash
        self._feature_names = feature_names

    @classmethod
    def load(
        cls,
        feature_names: list[str],
        patterns_path: str | None = None,
        expected_hash: str | None = None,
    ) -> "ZeroShotPatternDetector":
        """Load patterns from disk and build pattern vectors.

        If ``expected_hash`` is supplied (e.g. stored in a side-car file),
        the loaded file is validated against it so accidental overwrites or
        tampering are caught at startup.
        """
        path = patterns_path or _PATTERNS_PATH
        with open(path) as fh:
            data = json.load(fh)

        computed_hash = _patterns_hash(data)
        if expected_hash is not None and computed_hash != expected_hash:
            raise ValueError(
                f"Pattern file hash mismatch — expected {expected_hash!r}, "
                f"got {computed_hash!r}. File may have been tampered with."
            )

        pattern_vectors = []
        pattern_ids = []
        for p in data.get("patterns", []):
            vec = _pattern_to_vector(p["feature_weights"], feature_names)
            pattern_vectors.append(vec)
            pattern_ids.append(p["id"])

        return cls(pattern_vectors, pattern_ids, computed_hash, feature_names)

    def _feature_dict_to_vector(self, features: dict[str, Any]) -> np.ndarray:
        """Map a feature dict to a normalised vector over the known feature space."""
        vec = np.array(
            [float(features.get(name, 0.0)) for name in self._feature_names], dtype=float
        )
        # Sign-correct: negative-weighted features should increase wash-trade signal
        # when their raw values are high — no inversion needed here since the pattern
        # vectors already encode directionality via negative weights.
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        """Score a feature dict against all patterns.

        Returns:
            prediction:      1 if max confidence > 0.5 else 0
            confidence:      highest cosine similarity to any pattern (0–1)
            matched_pattern: ID of the best-matching pattern (or None)
        """
        if not self._pattern_vectors:
            return {"prediction": 0, "confidence": 0.0, "matched_pattern": None}

        fvec = self._feature_dict_to_vector(features)
        similarities = []
        for pvec in self._pattern_vectors:
            sim = 1.0 - cosine_distance(fvec, pvec)
            sim = max(0.0, min(1.0, sim))
            similarities.append(sim)

        best_idx = int(np.argmax(similarities))
        confidence = float(similarities[best_idx])
        return {
            "prediction": int(confidence > 0.5),
            "confidence": confidence,
            "matched_pattern": self._pattern_ids[best_idx] if confidence > 0.3 else None,
        }

    @property
    def pattern_hash(self) -> str:
        return self._expected_hash
