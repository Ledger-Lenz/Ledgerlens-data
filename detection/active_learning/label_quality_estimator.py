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
            noise_threshold
            if noise_threshold is not None
            else config.LABEL_QUALITY_NOISE_THRESHOLD
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
                        annotator_ids[idx]
                        if annotator_ids and idx < len(annotator_ids)
                        else None
                    ),
                    "wallet": (
                        wallet_ids[idx]
                        if wallet_ids and idx < len(wallet_ids)
                        else None
                    ),
                    "status": "quarantined",
                }
                f.write(json.dumps(record) + "\n")
