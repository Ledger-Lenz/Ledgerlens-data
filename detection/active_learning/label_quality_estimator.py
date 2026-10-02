"""Label quality estimation using confident learning (cleanlab).

Runs ``cleanlab.filter.find_label_issues`` on each new annotation batch before
adding samples to the training set.  Samples flagged as potentially mislabelled
(top ``LABEL_QUALITY_NOISE_THRESHOLD`` percent, default 10 %) are quarantined
for re-annotation rather than silently included.

Per-annotator noise rates are tracked; when an annotator's estimated noise rate
exceeds ``ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD`` (default 20 %) the operator is
alerted via a structured log WARNING.

Cleanlab requires out-of-sample predicted probabilities.  The current production
model (a ``RiskScorer`` instance) is used for this purpose.

Class-conditional noise rates are used (not overall) to handle the severe class
imbalance typical of wash-trade datasets.

Security
--------
Quarantined labels are logged with their estimated noise score and annotator ID.
They are not silently deleted; operators must manually review and re-annotate.

References
----------
Northcutt, C., Jiang, L., & Chuang, I. (2021). Confident Learning: Estimating
Uncertainty in Dataset Labels. *JAIR*, 70, 1373–1411.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

_QUARANTINE_LOG_PATH = "data/label_quality_quarantine.ndjson"


def _get_predicted_probs(
    labels: np.ndarray,
    features: pd.DataFrame,
    model,
) -> np.ndarray:
    """Return out-of-sample P(class=1) for each sample using the production model.

    If the model exposes ``predict_proba``, it is used directly.  Otherwise the
    model's ``predict`` output is cast to float as a fallback.
    """
    if hasattr(model, "predict_proba"):
        probs_pos = model.predict_proba(features)[:, 1]
    else:
        probs_pos = model.predict(features).astype(float)
    # Clip to avoid log(0) inside cleanlab
    probs_pos = np.clip(probs_pos, 1e-6, 1.0 - 1e-6)
    probs_neg = 1.0 - probs_pos
    return np.column_stack([probs_neg, probs_pos])


class LabelQualityEstimator:
    """Identifies potentially mislabelled annotation samples using cleanlab.

    Parameters
    ----------
    model:
        Production model used to generate out-of-sample predicted probabilities.
        Must expose ``predict_proba(X)`` or ``predict(X)``.
    noise_threshold:
        Fraction (0–1) of the batch to quarantine as potentially noisy
        (``LABEL_QUALITY_NOISE_THRESHOLD``, default 0.10).
    annotator_alert_threshold:
        Alert when an annotator's estimated noise rate exceeds this fraction
        (``ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD``, default 0.20).
    quarantine_log_path:
        NDJSON file where quarantined items are appended for operator review.
    """

    def __init__(
        self,
        model,
        noise_threshold: float | None = None,
        annotator_alert_threshold: float | None = None,
        quarantine_log_path: str = _QUARANTINE_LOG_PATH,
    ) -> None:
        self.model = model
        self.noise_threshold = (
            noise_threshold if noise_threshold is not None else config.LABEL_QUALITY_NOISE_THRESHOLD
        )
        self.annotator_alert_threshold = (
            annotator_alert_threshold
            if annotator_alert_threshold is not None
            else config.ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD
        )
        self.quarantine_log_path = quarantine_log_path

        # Per-annotator noise tracking: annotator_id → {noise_count, total_count}
        self._annotator_stats: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        features: pd.DataFrame,
        labels: np.ndarray | list[int],
        annotator_ids: list[str] | None = None,
        wallet_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run label quality estimation on an annotation batch.

        Parameters
        ----------
        features:
            Feature matrix for the batch (rows match ``labels``).
        labels:
            Integer labels (0 = clean, 1 = wash trade).
        annotator_ids:
            Per-sample annotator identifiers (optional; used for per-annotator
            noise rate tracking).
        wallet_ids:
            Per-sample wallet addresses (optional; used in quarantine log).

        Returns
        -------
        dict with keys:
            ``clean_indices`` — indices NOT flagged as noisy,
            ``quarantined_indices`` — indices flagged and quarantined,
            ``noise_scores`` — per-sample estimated noise score (higher = noisier),
            ``annotator_noise_rates`` — per-annotator estimated noise rate.
        """
        labels_arr = np.asarray(labels, dtype=int)
        n = len(labels_arr)

        if n == 0:
            return {
                "clean_indices": [],
                "quarantined_indices": [],
                "noise_scores": [],
                "annotator_noise_rates": {},
            }

        pred_probs = _get_predicted_probs(labels_arr, features, self.model)
        issue_indices = self._find_issues(labels_arr, pred_probs)

        # Noise score = P(predicted class) for the *given* label
        # Higher score ↔ model is more confident the label is wrong
        noise_scores = np.zeros(n, dtype=float)
        for i in range(n):
            given_class = labels_arr[i]
            noise_scores[i] = pred_probs[i, 1 - given_class]

        # Quarantine the top ``noise_threshold`` fraction by noise score, but
        # only among the samples flagged by cleanlab
        n_quarantine = max(1, int(round(self.noise_threshold * n)))
        if len(issue_indices) > 0:
            ranked = sorted(issue_indices, key=lambda idx: noise_scores[idx], reverse=True)
            quarantined = ranked[:n_quarantine]
        else:
            quarantined = []

        quarantined_set = set(quarantined)
        clean_indices = [i for i in range(n) if i not in quarantined_set]

        # Per-annotator noise tracking
        annotator_noise_rates: dict[str, float] = {}
        if annotator_ids is not None:
            for idx in range(n):
                ann = annotator_ids[idx] if idx < len(annotator_ids) else "unknown"
                if ann not in self._annotator_stats:
                    self._annotator_stats[ann] = {"noise_count": 0, "total_count": 0}
                self._annotator_stats[ann]["total_count"] += 1
                if idx in quarantined_set:
                    self._annotator_stats[ann]["noise_count"] += 1

            for ann, stats in self._annotator_stats.items():
                if stats["total_count"] > 0:
                    rate = stats["noise_count"] / stats["total_count"]
                    annotator_noise_rates[ann] = rate
                    if rate > self.annotator_alert_threshold:
                        logger.warning(
                            "High label noise rate detected for annotator=%s: "
                            "noise_rate=%.2f (threshold=%.2f) "
                            "noise_count=%d total_count=%d",
                            ann,
                            rate,
                            self.annotator_alert_threshold,
                            stats["noise_count"],
                            stats["total_count"],
                        )

        # Append quarantined items to the audit log
        self._log_quarantined(
            quarantined_indices=quarantined,
            labels=labels_arr,
            noise_scores=noise_scores,
            annotator_ids=annotator_ids,
            wallet_ids=wallet_ids,
        )

        logger.info(
            "Label quality check: batch_size=%d flagged=%d quarantined=%d",
            n,
            len(issue_indices),
            len(quarantined),
        )
        return {
            "clean_indices": clean_indices,
            "quarantined_indices": quarantined,
            "noise_scores": noise_scores.tolist(),
            "annotator_noise_rates": annotator_noise_rates,
        }

    def annotator_noise_rates(self) -> dict[str, float]:
        """Return the cumulative estimated noise rate per annotator."""
        rates = {}
        for ann, stats in self._annotator_stats.items():
            if stats["total_count"] > 0:
                rates[ann] = stats["noise_count"] / stats["total_count"]
        return rates

    def reset_annotator_stats(self) -> None:
        """Clear accumulated per-annotator noise statistics."""
        self._annotator_stats.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_issues(self, labels: np.ndarray, pred_probs: np.ndarray) -> list[int]:
        """Delegate to cleanlab's ``find_label_issues``.

        Falls back to an empty list if cleanlab is not installed.
        """
        try:
            from cleanlab.filter import find_label_issues

            issue_mask = find_label_issues(
                labels=labels,
                pred_probs=pred_probs,
                return_indices_ranked_by="normalized_margin",
            )
            if isinstance(issue_mask, np.ndarray) and issue_mask.dtype == bool:
                return list(np.where(issue_mask)[0])
            return list(issue_mask)
        except ImportError:  # pragma: no cover
            logger.warning(
                "cleanlab is not installed; label quality estimation is disabled. "
                "Install it with: pip install cleanlab"
            )
            return []
        except Exception as exc:
            logger.warning("cleanlab.filter.find_label_issues failed: %s", exc)
            return []

    def _log_quarantined(
        self,
        quarantined_indices: list[int],
        labels: np.ndarray,
        noise_scores: np.ndarray,
        annotator_ids: list[str] | None,
        wallet_ids: list[str] | None,
    ) -> None:
        """Append quarantined items to the NDJSON audit log."""
        if not quarantined_indices:
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.quarantine_log_path)), exist_ok=True)
        now = datetime.now(UTC).isoformat()
        with open(self.quarantine_log_path, "a") as f:
            for idx in quarantined_indices:
                record: dict[str, Any] = {
                    "quarantined_at": now,
                    "batch_index": int(idx),
                    "label": int(labels[idx]),
                    "noise_score": float(noise_scores[idx]),
                    "annotator_id": (
                        annotator_ids[idx] if annotator_ids and idx < len(annotator_ids) else None
                    ),
                    "wallet": (wallet_ids[idx] if wallet_ids and idx < len(wallet_ids) else None),
                    "status": "quarantined",
                }
                f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Issue #887: inter-annotator agreement tracking
# ---------------------------------------------------------------------------


def assign_overlapping_items(
    item_ids: list[str],
    annotators: list[str],
    overlap_fraction: float | None = None,
    n_per_overlap: int = 2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Assign queue items to annotators, sending *overlap_fraction* of items to 2+ annotators.

    Returns ``item_id -> [annotator, ...]``. Non-overlap items go round-robin to one annotator.
    """
    if overlap_fraction is None:
        overlap_fraction = config.AL_OVERLAP_FRACTION
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError(f"overlap_fraction must be in [0, 1], got {overlap_fraction}")
    if not annotators:
        raise ValueError("at least one annotator is required")
    rng = np.random.default_rng(seed)
    k = min(max(2, n_per_overlap), len(annotators))
    n_overlap = int(round(len(item_ids) * overlap_fraction)) if len(annotators) > 1 else 0
    overlap = set(rng.choice(len(item_ids), size=n_overlap, replace=False).tolist()) if n_overlap else set()
    assignments: dict[str, list[str]] = {}
    for i, item in enumerate(item_ids):
        if i in overlap:
            assignments[item] = [str(a) for a in rng.choice(annotators, size=k, replace=False)]
        else:
            assignments[item] = [annotators[i % len(annotators)]]
    return assignments


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa between two annotators over the same items."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must be equal length")
    n = len(labels_a)
    if n == 0:
        return float("nan")
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    p_o = float(np.mean(a == b))
    cats = np.union1d(a, b)
    p_e = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(ratings: dict[str, dict[str, Any]]) -> float:
    """Fleiss' kappa over ``item_id -> {annotator: label}`` (items with 2+ ratings only)."""
    items = [list(r.values()) for r in ratings.values() if len(r) >= 2]
    if not items:
        return float("nan")
    cats = sorted({lab for labs in items for lab in labs}, key=str)
    counts = np.array([[labs.count(c) for c in cats] for labs in items], dtype=float)
    n_i = counts.sum(axis=1)
    p_i = ((counts * (counts - 1)).sum(axis=1)) / (n_i * (n_i - 1))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / n_i.sum()
    p_e = float((p_j**2).sum())
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


class AgreementTracker:
    """Track per-annotator agreement on overlapping items and route disagreements.

    ``ratings`` is ``item_id -> {annotator_id: label}``. An annotator's score is the
    mean Cohen's kappa against every peer who shared at least ``min_shared`` items.
    Annotators below ``min_kappa`` are flagged; their votes are down-weighted to
    ``flagged_weight`` in :meth:`resolve`. Items with no weighted majority of at
    least ``consensus`` go to adjudication, so no label is ever picked silently.
    """

    def __init__(
        self,
        min_kappa: float | None = None,
        min_shared: int = 5,
        consensus: float = 0.75,
        flagged_weight: float = 0.0,
    ):
        self.min_kappa = config.AL_MIN_ANNOTATOR_KAPPA if min_kappa is None else min_kappa
        self.min_shared = min_shared
        self.consensus = consensus
        self.flagged_weight = flagged_weight
        self.ratings: dict[str, dict[str, Any]] = {}

    def record(self, item_id: str, annotator_id: str, label: Any) -> None:
        self.ratings.setdefault(item_id, {})[annotator_id] = label

    def annotator_kappas(self) -> dict[str, float]:
        annotators = sorted({a for r in self.ratings.values() for a in r})
        scores: dict[str, list[float]] = {a: [] for a in annotators}
        for i, a in enumerate(annotators):
            for b in annotators[i + 1 :]:
                shared = [r for r in self.ratings.values() if a in r and b in r]
                if len(shared) < self.min_shared:
                    continue
                k = cohens_kappa([r[a] for r in shared], [r[b] for r in shared])
                scores[a].append(k)
                scores[b].append(k)
        return {a: float(np.mean(v)) for a, v in scores.items() if v}

    def flagged_annotators(self) -> list[str]:
        return sorted(a for a, k in self.annotator_kappas().items() if k < self.min_kappa)

    def resolve(self) -> tuple[dict[str, Any], list[str]]:
        """Return ``(resolved_labels, items_needing_adjudication)``."""
        flagged = set(self.flagged_annotators())
        resolved: dict[str, Any] = {}
        adjudicate: list[str] = []
        for item, votes in self.ratings.items():
            tally: dict[Any, float] = {}
            for ann, lab in votes.items():
                tally[lab] = tally.get(lab, 0.0) + (self.flagged_weight if ann in flagged else 1.0)
            total = sum(tally.values())
            if total == 0:
                adjudicate.append(item)
                continue
            label, weight = max(tally.items(), key=lambda kv: kv[1])
            if len(votes) == 1 or weight / total >= self.consensus:
                resolved[item] = label
            else:
                adjudicate.append(item)
        if flagged:
            logger.warning("Low-agreement annotators flagged: %s", sorted(flagged))
        return resolved, adjudicate

    def report(self) -> dict[str, Any]:
        return {
            "fleiss_kappa": fleiss_kappa(self.ratings),
            "annotator_kappas": self.annotator_kappas(),
            "flagged_annotators": self.flagged_annotators(),
            "min_kappa": self.min_kappa,
        }
