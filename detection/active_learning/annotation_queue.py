"""Annotation queue with HMAC-SHA256 integrity protection.

Each annotation carries an ``annotation_hmac`` field computed as
HMAC-SHA256 of ``wallet|label|annotator_id|annotated_at`` keyed by
``config.ANNOTATION_HMAC_SECRET``.  ``export_labelled`` verifies every
HMAC before including the annotation in the exported dataset; any
annotation with an invalid HMAC is logged as a WARNING and excluded.

The ``AnnotationQueue`` class provides the high-level interface used by
the active learning pipeline:

    queue = AnnotationQueue()
    queue.push(["GABCD...", "GXYZ..."], strategy_name="entropy")
    batch = queue.pop_batch(5)
    queue.annotate("GABCD...", label=1, annotator_id="alice", notes="obvious wash")
    queue.export_labelled("data/annotated.parquet")

The legacy ``add_annotation`` / ``export_labelled`` functions are retained
for backward compatibility with existing tests and callers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import tempfile
from collections import deque
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_QUEUE_PATH = "data/annotation_queue.json"


# ---------------------------------------------------------------------------
# HMAC helpers (shared by class and legacy functions)
# ---------------------------------------------------------------------------


def _compute_hmac(wallet: str, label: int, annotator_id: str, annotated_at: str) -> str:
    secret = config.ANNOTATION_HMAC_SECRET.encode()
    message = f"{wallet}|{label}|{annotator_id}|{annotated_at}".encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _atomic_write(path: str, data: list) -> None:
    """Write *data* as JSON to *path* atomically (write temp, rename)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        os.rename(tmp_path, path)
    except Exception:
        # Clean up the temp file if rename fails
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_queue(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# AnnotationQueue class
# ---------------------------------------------------------------------------


class AnnotationQueue:
    """Persistent annotation queue backed by a JSON file.

    Args:
        queue_path: Path to the JSON queue file (default: data/annotation_queue.json).
    """

    def __init__(self, queue_path: str = DEFAULT_QUEUE_PATH):
        self.queue_path = queue_path

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def push(self, wallets: list[str], strategy_name: str, asset_pair: str = "") -> None:
        """Add *wallets* to the queue with status ``pending``.

        Wallets already present (any status) are skipped.
        """
        queue = _load_queue(self.queue_path)
        existing = {item["wallet"] for item in queue}
        now = datetime.now(UTC).isoformat()
        for wallet in wallets:
            if wallet in existing:
                continue
            queue.append(
                {
                    "wallet": wallet,
                    "asset_pair": asset_pair,
                    "score": None,
                    "query_strategy": strategy_name,
                    "selected_at": now,
                    "status": "pending",
                }
            )
        _atomic_write(self.queue_path, queue)

    def annotate(
        self,
        wallet: str,
        label: int,
        annotator_id: str,
        notes: str = "",
    ) -> None:
        """Record an analyst verdict for *wallet*.

        Raises ``ValueError`` if *annotator_id* is empty (accountability
        requirement) or *label* is not 0 or 1.

        Idempotent: calling again with the same wallet updates the record.
        """
        if not annotator_id:
            raise ValueError("annotator_id must be a non-empty string")
        if label not in (0, 1):
            raise ValueError("label must be 0 (clean) or 1 (wash trading)")

        queue = _load_queue(self.queue_path)
        annotated_at = datetime.now(UTC).isoformat()
        mac = _compute_hmac(wallet, label, annotator_id, annotated_at)

        for item in queue:
            if item["wallet"] == wallet:
                item.update(
                    {
                        "label": label,
                        "annotator_id": annotator_id,
                        "notes": notes,
                        "annotated_at": annotated_at,
                        "status": "annotated",
                        "annotation_hmac": mac,
                    }
                )
                _atomic_write(self.queue_path, queue)
                return

        # Wallet not yet in queue — add it inline
        queue.append(
            {
                "wallet": wallet,
                "asset_pair": "",
                "score": None,
                "query_strategy": "manual",
                "selected_at": annotated_at,
                "status": "annotated",
                "label": label,
                "annotator_id": annotator_id,
                "notes": notes,
                "annotated_at": annotated_at,
                "annotation_hmac": mac,
            }
        )
        _atomic_write(self.queue_path, queue)

    def skip(self, wallet: str) -> None:
        """Mark *wallet* as skipped."""
        queue = _load_queue(self.queue_path)
        for item in queue:
            if item["wallet"] == wallet:
                item["status"] = "skipped"
                break
        _atomic_write(self.queue_path, queue)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def pop_batch(self, n: int) -> list[dict]:
        """Return the next *n* pending wallets (does not change status)."""
        queue = _load_queue(self.queue_path)
        pending = [item for item in queue if item.get("status") == "pending"]
        return pending[:n]

    def pending_wallets(self) -> list[str]:
        return [item["wallet"] for item in self.pop_batch(10**9)]

    def skipped_wallets(self) -> list[str]:
        queue = _load_queue(self.queue_path)
        return [item["wallet"] for item in queue if item.get("status") == "skipped"]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_labelled(self, output_path: str) -> pd.DataFrame:
        """Export verified annotated rows to *output_path* as parquet.

        Only rows with ``status == "annotated"`` and valid HMAC are included.
        Returns the exported DataFrame.
        """
        queue = _load_queue(self.queue_path)
        verified = []
        for item in queue:
            if item.get("status") != "annotated":
                continue
            expected = _compute_hmac(
                item.get("wallet", ""),
                item.get("label", -1),
                item.get("annotator_id", ""),
                item.get("annotated_at", ""),
            )
            if not hmac.compare_digest(expected, item.get("annotation_hmac", "")):
                logger.warning(
                    "Invalid HMAC for annotation wallet=%s — excluded from export",
                    item.get("wallet"),
                )
                continue
            verified.append(item)

        df = pd.DataFrame(verified)
        if not df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            df.to_parquet(output_path, index=False)
        return df


# ---------------------------------------------------------------------------
# Stopping criterion (Issue #256)
# ---------------------------------------------------------------------------


class StoppingCriterion:
    """Active learning stopping criterion based on Expected Error Reduction (EER)
    and rolling AUC improvement.

    Convergence is declared when **either**:
    - The EER of the highest-uncertainty unlabelled sample falls below
      ``eer_threshold`` (default ``ACTIVE_LEARNING_EER_THRESHOLD``), OR
    - The mean AUC improvement over the last ``convergence_window`` rounds
      is below ``auc_improvement_threshold`` (default 0.005).

    The check is designed to run at the end of each annotation batch (not
    after each individual annotation).

    Security: convergence reports log annotator IDs and counts only —
    never raw label values.

    Args:
        eer_threshold: Stop when EER < this value (default 0.001).
        convergence_window: Number of rounds to average for AUC trend (default 5).
        auc_improvement_threshold: Min mean AUC improvement per round (default 0.005).
    """

    def __init__(
        self,
        eer_threshold: float | None = None,
        convergence_window: int | None = None,
        auc_improvement_threshold: float = 0.005,
    ) -> None:
        self.eer_threshold: float = eer_threshold if eer_threshold is not None else float(
            getattr(config, "ACTIVE_LEARNING_EER_THRESHOLD", 0.001)
        )
        self.convergence_window: int = convergence_window if convergence_window is not None else int(
            getattr(config, "ACTIVE_LEARNING_CONVERGENCE_WINDOW", 5)
        )
        self.auc_improvement_threshold = auc_improvement_threshold
        # Rolling AUC history: deque of per-round AUC values
        self._auc_history: deque[float] = deque(maxlen=self.convergence_window + 1)
        self._round: int = 0

    def record_round_auc(self, auc: float) -> None:
        """Record the AUC after one annotation batch completes."""
        self._auc_history.append(auc)
        self._round += 1

    def eer(self, model, unlabelled_pool: pd.DataFrame) -> float:
        """Compute EER: expected error reduction for the highest-uncertainty sample.

        Uses the current production model (no special model trained).  EER is
        approximated as ``1 - max_class_probability`` for the most uncertain sample.

        Args:
            model: Fitted scikit-learn compatible model with ``predict_proba``.
            unlabelled_pool: DataFrame of unlabelled candidates.

        Returns:
            EER estimate (float).  0.0 if pool is empty.
        """
        if unlabelled_pool.empty or model is None:
            return 0.0

        from detection.model_training import FEATURE_COLUMNS_EXCLUDE

        feat_cols = [c for c in unlabelled_pool.columns if c not in FEATURE_COLUMNS_EXCLUDE]
        X = unlabelled_pool[feat_cols].astype(float).fillna(0.0)
        probs = model.predict_proba(X)  # (N, 2)
        # EER ≈ 1 − max_prob for the most uncertain sample
        max_probs = probs.max(axis=1)
        return float(1.0 - max_probs.max())

    def should_stop(
        self,
        model=None,
        unlabelled_pool: "pd.DataFrame | None" = None,
    ) -> bool:
        """Return True if the stopping criterion has fired.

        Checks both EER and rolling AUC trend.  Intended to be called at the
        end of each annotation batch.
        """
        # EER check
        if model is not None and unlabelled_pool is not None and not unlabelled_pool.empty:
            eer_val = self.eer(model, unlabelled_pool)
            if eer_val < self.eer_threshold:
                logger.info(
                    "StoppingCriterion: EER=%.6f < threshold=%.6f — convergence declared",
                    eer_val,
                    self.eer_threshold,
                )
                return True

        # Rolling AUC improvement check
        if len(self._auc_history) >= self.convergence_window + 1:
            # compute round-over-round improvements for the last window rounds
            history = list(self._auc_history)
            improvements = [history[i] - history[i - 1] for i in range(1, len(history))]
            mean_improvement = sum(improvements[-self.convergence_window:]) / self.convergence_window
            if mean_improvement < self.auc_improvement_threshold:
                logger.info(
                    "StoppingCriterion: mean AUC improvement=%.6f < threshold=%.6f "
                    "over last %d rounds — convergence declared",
                    mean_improvement,
                    self.auc_improvement_threshold,
                    self.convergence_window,
                )
                return True

        return False

    def emit_convergence_report(
        self,
        queue_path: str,
        db_path: str | None = None,
    ) -> dict:
        """Write a convergence report without including raw label values.

        Logs annotator IDs and annotation counts only.

        Returns the report dict (also written to ``reports/`` if *db_path* is set).
        """
        queue = _load_queue(queue_path)
        annotated = [item for item in queue if item.get("status") == "annotated"]

        # Count annotations per annotator (no label values)
        annotator_counts: dict[str, int] = {}
        for item in annotated:
            aid = item.get("annotator_id", "unknown")
            annotator_counts[aid] = annotator_counts.get(aid, 0) + 1

        report: dict[str, Any] = {
            "converged_at": datetime.now(UTC).isoformat(),
            "rounds_completed": self._round,
            "total_annotations": len(annotated),
            "annotator_counts": annotator_counts,
            "auc_history": list(self._auc_history),
        }

        # Dispatch alert via streaming alert dispatcher
        try:
            from streaming.alert_dispatcher import AlertDispatcher

            dispatcher = AlertDispatcher(channel="stdout")
            dispatcher.dispatch(
                wallet="__convergence__",
                pair_id="active_learning",
                score=0,
                benford_flag=False,
                ml_flag=False,
                confidence=0,
            )
        except Exception as exc:
            logger.warning("Could not dispatch convergence alert: %s", exc)

        # Persist report
        import os

        os.makedirs("reports", exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        report_path = os.path.join("reports", f"al_convergence_{ts}.json")
        with open(report_path, "w") as f:
            import json as _json

            _json.dump(report, f, indent=2)
        logger.info("Convergence report written to %s", report_path)
        return report


# ---------------------------------------------------------------------------
# Legacy functional API (backward compat)
# ---------------------------------------------------------------------------


def add_annotation(
    queue_path: str,
    wallet: str,
    label: int,
    annotator_id: str,
    annotated_at: str,
) -> dict[str, Any]:
    """Append a new annotation to *queue_path* (JSON list) with an HMAC."""
    annotation: dict[str, Any] = {
        "wallet": wallet,
        "label": label,
        "annotator_id": annotator_id,
        "annotated_at": annotated_at,
        "annotation_hmac": _compute_hmac(wallet, label, annotator_id, annotated_at),
    }

    queue = _load_queue(queue_path)
    queue.append(annotation)
    _atomic_write(queue_path, queue)
    return annotation


def export_labelled(queue_path: str) -> list[dict]:
    """Return verified annotations from *queue_path*.

    Annotations whose HMAC fails verification are logged as WARNING and
    excluded from the returned list.
    """
    queue = _load_queue(queue_path)
    verified = []
    for ann in queue:
        expected = _compute_hmac(
            ann.get("wallet", ""),
            ann.get("label", -1),
            ann.get("annotator_id", ""),
            ann.get("annotated_at", ""),
        )
        if not hmac.compare_digest(expected, ann.get("annotation_hmac", "")):
            logger.warning(
                "Invalid HMAC for annotation wallet=%s annotator=%s — excluded",
                ann.get("wallet"),
                ann.get("annotator_id"),
            )
        else:
            verified.append(ann)
    return verified
