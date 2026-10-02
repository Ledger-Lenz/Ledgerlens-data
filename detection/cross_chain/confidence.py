"""Confidence scoring for cross-chain identity edges (Issue #879).

Cross-chain identity resolution is probabilistic: a memo-tagged bridge
transfer is near-conclusive, whereas a timing correlation or a matching trade
amount can arise by coincidence.  Each piece of evidence linking two addresses
is therefore turned into an independent probability

    p_i = strength_i * reliability(evidence_type_i)

where ``strength_i`` is the detector's own score in [0, 1] (e.g. the Pearson
``r`` from ``BehavioralMatcher.match_timing_correlation``) and ``reliability``
is a per-source prior reflecting how often that evidence type is correct on
its own.  Evidence is combined with a noisy-OR:

    confidence = 1 - prod(1 - p_i)

so independent corroborating signals raise confidence while a single weak
signal stays weak.  Edges whose combined confidence falls below
``config.CROSS_CHAIN_MIN_CONFIDENCE`` are excluded from risk propagation.
See docs/cross_chain_identity.md for the threshold rationale.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Per-source reliability priors.  Bridge links come from explicit memo-encoded
# destination addresses and are treated as conclusive; behavioral signals are
# discounted according to their observed false-positive rate on the labelled
# benchmark (benchmarks/cross_chain.py).
EVIDENCE_RELIABILITY: dict[str, float] = {
    "bridge": 1.0,
    "wormhole_bridge": 1.0,
    "shared_deposit": 0.9,
    "amount_fingerprint": 0.55,
    "behavioral": 0.8,
    "timing_correlation": 0.7,
}
DEFAULT_RELIABILITY = 0.8


@dataclass(frozen=True)
class Evidence:
    """A single piece of evidence linking two addresses across chains."""

    evidence_type: str
    strength: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reliability(self) -> float:
        return EVIDENCE_RELIABILITY.get(self.evidence_type, DEFAULT_RELIABILITY)

    @property
    def probability(self) -> float:
        """Probability this evidence alone implies a shared identity."""
        strength = min(max(float(self.strength), 0.0), 1.0)
        return strength * self.reliability

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "strength": float(self.strength),
            "reliability": self.reliability,
            "probability": self.probability,
            "metadata": dict(self.metadata),
        }


def combine_confidence(evidence: Iterable[Evidence]) -> float:
    """Combine independent evidence into one confidence score via noisy-OR."""
    miss = 1.0
    for item in evidence:
        miss *= 1.0 - item.probability
    return 1.0 - miss


def precision_recall(
    scored_pairs: dict[tuple[str, str], float],
    true_pairs: set[tuple[str, str]],
    threshold: float,
) -> dict[str, float]:
    """Precision/recall/F1 of ``scored_pairs`` at ``threshold``.

    Pairs are treated as unordered.  ``true_pairs`` is the labelled set of
    known cross-chain identity pairs; every pair in ``scored_pairs`` not in it
    is a negative.
    """
    truth = {frozenset(p) for p in true_pairs}
    predicted = {frozenset(p) for p, c in scored_pairs.items() if c >= threshold}
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
