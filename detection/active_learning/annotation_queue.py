"""Annotation queue with HMAC-SHA256 integrity protection and multi-annotator
inter-rater agreement scoring.

Each annotation carries an ``annotation_hmac`` field computed as
HMAC-SHA256 of ``wallet|label|annotator_id|annotated_at`` keyed by
``config.ANNOTATION_HMAC_SECRET``.  ``export_labelled`` verifies every
HMAC before including the annotation in the exported dataset; any
annotation with an invalid HMAC is logged as a WARNING and excluded.

Multi-annotator support
-----------------------
A wallet can receive labels from multiple annotators (blind double-annotation
or senior review).  When at least 2 verified annotations exist for a wallet,
``compute_inter_annotator_agreement(wallet_id)`` returns both Cohen's Kappa
(binary labels) and Krippendorff's Alpha (multi-class / ordinal / continuous).

Wallets with Kappa < ``DISPUTE_KAPPA_THRESHOLD`` (default 0.6) are flagged as
``"disputed"`` and routed to ``get_senior_review_queue()``.

Annotator IDs are pseudonymous opaque strings (e.g. ``"anon-7f3a"``) — never
email addresses — to protect annotator privacy in the DB.

Usage::

    queue = AnnotationQueue()
    queue.push(["GABCD...", "GXYZ..."], strategy_name="entropy")
    batch = queue.pop_batch(5)
    queue.annotate("GABCD...", label=1, annotator_id="anon-alice", notes="obvious wash")
    queue.annotate("GABCD...", label=0, annotator_id="anon-bob")
    result = queue.compute_inter_annotator_agreement("GABCD...")
    # {"kappa": -1.0, "alpha": ..., "n_annotators": 2, "disputed": True}
    disputed = queue.get_senior_review_queue()

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
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    import krippendorff
    _KRIPPENDORFF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _KRIPPENDORFF_AVAILABLE = False

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_QUEUE_PATH = "data/annotation_queue.json"
DISPUTE_KAPPA_THRESHOLD = 0.6  # wallets below this are routed to senior review


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

    def push(
        self,
        wallets: list[str],
        strategy_name: str,
        asset_pair: str = "",
        epistemic_uncertainties: dict[str, float] | None = None,
        aleatoric_uncertainties: dict[str, float] | None = None,
    ) -> None:
        """Add *wallets* to the queue with status ``pending``.

        Wallets already present (any status) are skipped.

        Parameters
        ----------
        wallets:
            Wallet IDs to enqueue.
        strategy_name:
            Name of the query strategy that selected these wallets.
        asset_pair:
            Optional asset pair context.
        epistemic_uncertainties:
            Optional mapping of wallet_id -> epistemic uncertainty [0, 1].
            Used to rank samples for annotation (higher = prioritised).
        aleatoric_uncertainties:
            Optional mapping of wallet_id -> aleatoric uncertainty [0, 1].
            Wallets above ``config.ACTIVE_LEARNING_ALEATORIC_THRESHOLD`` are
            filtered out before enqueuing (inherently noisy, annotation won't help).
        """
        aleatoric_threshold = config.ACTIVE_LEARNING_ALEATORIC_THRESHOLD
        queue = _load_queue(self.queue_path)
        existing = {item["wallet"] for item in queue}
        now = datetime.now(UTC).isoformat()
        skipped_aleatoric = 0
        for wallet in wallets:
            if wallet in existing:
                continue
            aleatoric = (aleatoric_uncertainties or {}).get(wallet)
            if aleatoric is not None and aleatoric > aleatoric_threshold:
                logger.debug(
                    "Skipping wallet %s: aleatoric uncertainty %.4f exceeds threshold %.4f",
                    wallet,
                    aleatoric,
                    aleatoric_threshold,
                )
                skipped_aleatoric += 1
                continue
            queue.append(
                {
                    "wallet": wallet,
                    "asset_pair": asset_pair,
                    "score": None,
                    "query_strategy": strategy_name,
                    "selected_at": now,
                    "status": "pending",
                    "epistemic_uncertainty": (epistemic_uncertainties or {}).get(wallet),
                    "aleatoric_uncertainty": aleatoric,
                }
            )
        if skipped_aleatoric:
            logger.info(
                "push: filtered %d wallet(s) with aleatoric uncertainty above threshold %.4f",
                skipped_aleatoric,
                aleatoric_threshold,
            )
        _atomic_write(self.queue_path, queue)

    def annotate(
        self,
        wallet: str,
        label: int,
        annotator_id: str,
        notes: str = "",
        quarantine: bool = False,
        quarantine_reason: str = "",
    ) -> None:
        """Record an analyst verdict for *wallet*.

        Raises ``ValueError`` if *annotator_id* is empty (accountability
        requirement) or *label* is not 0 or 1.

        Idempotent: calling again with the same wallet updates the record.

        Args:
            wallet: Wallet ID
            label: 0 (clean) or 1 (wash trading)
            annotator_id: Analyst ID (non-empty)
            notes: Optional annotation notes
            quarantine: If True, mark as quarantined (default False)
            quarantine_reason: Reason for quarantine (e.g., "backdoor_ac_detected")
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
                        "quarantine": quarantine,
                        "quarantine_reason": quarantine_reason if quarantine else "",
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
                "quarantine": quarantine,
                "quarantine_reason": quarantine_reason if quarantine else "",
            }
        )
        _atomic_write(self.queue_path, queue)

    def multi_annotate(
        self,
        wallet: str,
        label: int,
        annotator_id: str,
        notes: str = "",
    ) -> None:
        """Record an additional annotation for *wallet* from *annotator_id*.

        Unlike ``annotate``, this does **not** overwrite an existing record.
        Each (wallet, annotator_id) pair is stored as a separate entry under
        the ``"annotations"`` list of the wallet's queue item.

        After recording, if at least 2 verified annotations exist, the wallet's
        status is updated to ``"multi_annotated"`` and agreement is computed;
        if Kappa < ``DISPUTE_KAPPA_THRESHOLD`` the status becomes ``"disputed"``.

        Annotator IDs must be pseudonymous (not email addresses).
        """
        if not annotator_id:
            raise ValueError("annotator_id must be a non-empty string")
        if label not in (0, 1):
            raise ValueError("label must be 0 (clean) or 1 (wash trading)")

        queue = _load_queue(self.queue_path)
        annotated_at = datetime.now(UTC).isoformat()
        mac = _compute_hmac(wallet, label, annotator_id, annotated_at)
        new_entry = {
            "label": label,
            "annotator_id": annotator_id,
            "notes": notes,
            "annotated_at": annotated_at,
            "annotation_hmac": mac,
        }

        for item in queue:
            if item["wallet"] == wallet:
                item.setdefault("annotations", [])
                # prevent duplicate from same annotator
                if any(a["annotator_id"] == annotator_id for a in item["annotations"]):
                    logger.warning("Annotator %s already labelled wallet %s — skipping duplicate", annotator_id, wallet)
                    return
                item["annotations"].append(new_entry)
                self._refresh_agreement_status(item)
                _atomic_write(self.queue_path, queue)
                return

        # Wallet not yet in queue — create it
        queue.append(
            {
                "wallet": wallet,
                "asset_pair": "",
                "score": None,
                "query_strategy": "manual",
                "selected_at": annotated_at,
                "status": "multi_annotated",
                "annotations": [new_entry],
            }
        )
        _atomic_write(self.queue_path, queue)

    # ------------------------------------------------------------------
    # Agreement computation
    # ------------------------------------------------------------------

    def compute_inter_annotator_agreement(self, wallet_id: str) -> dict[str, Any]:
        """Compute Cohen's Kappa and Krippendorff's Alpha for *wallet_id*.

        Requires at least 2 verified annotations.  Returns a dict::

            {
                "kappa": float,
                "alpha": float | None,
                "n_annotators": int,
                "disputed": bool,
            }

        Raises ``ValueError`` if fewer than 2 verified annotations exist.
        """
        labels = self._verified_labels(wallet_id)
        if len(labels) < 2:
            raise ValueError(
                f"wallet {wallet_id!r} has {len(labels)} verified annotation(s); "
                "need at least 2 to compute agreement"
            )

        # Cohen's Kappa: compare every consecutive pair and average.
        # For exactly 2 annotators this is a single pairwise kappa.
        # For N>2, compute all pairwise combinations and return the mean.
        # We implement the formula directly so it handles the single-item
        # case correctly (sklearn's cohen_kappa_score requires ≥1 sample
        # per class across both raters, which fails for single-item disagreement).
        from itertools import combinations

        def _kappa_pair(a: int, b: int) -> float:
            """Cohen's Kappa for two raters each providing one binary label."""
            if a == b:
                return 1.0
            # Observed agreement P_o = 0 (total disagreement)
            # Expected agreement P_e for binary: p_0 * q_0 + p_1 * q_1
            # With one rater each: p_class = (a==c + b==c) / 2 for class c
            p0 = ((a == 0) + (b == 0)) / 2.0
            p1 = ((a == 1) + (b == 1)) / 2.0
            p_e = p0 ** 2 + p1 ** 2
            if p_e == 1.0:
                return 1.0  # both raters always choose same class
            return (0.0 - p_e) / (1.0 - p_e)

        kappas = [_kappa_pair(labels[i], labels[j]) for i, j in combinations(range(len(labels)), 2)]
        kappa = float(np.mean(kappas))

        # Krippendorff's Alpha: expects a reliability matrix (annotators × items)
        alpha: float | None = None
        if _KRIPPENDORFF_AVAILABLE:
            # Build a 2-D array: rows = annotators, cols = items (1 item here)
            reliability = np.array([[float(lbl)] for lbl in labels]).T  # shape (1, n)
            try:
                alpha = float(krippendorff.alpha(reliability_data=reliability, level_of_measurement="nominal"))
            except Exception as exc:  # pragma: no cover
                logger.warning("Krippendorff alpha failed: %s", exc)

        disputed = kappa < DISPUTE_KAPPA_THRESHOLD
        return {
            "kappa": kappa,
            "alpha": alpha,
            "n_annotators": len(labels),
            "disputed": disputed,
        }

    def get_senior_review_queue(self) -> list[str]:
        """Return wallet IDs whose inter-annotator Kappa < ``DISPUTE_KAPPA_THRESHOLD``.

        Only wallets with at least 2 verified annotations are evaluated.
        """
        queue = _load_queue(self.queue_path)
        disputed = []
        for item in queue:
            if item.get("status") not in ("disputed", "multi_annotated"):
                continue
            wallet = item["wallet"]
            try:
                result = self.compute_inter_annotator_agreement(wallet)
                if result["disputed"]:
                    disputed.append(wallet)
            except ValueError:
                pass
        return disputed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verified_labels(self, wallet_id: str) -> list[int]:
        """Return HMAC-verified labels for *wallet_id* from multi-annotation list."""
        queue = _load_queue(self.queue_path)
        for item in queue:
            if item["wallet"] != wallet_id:
                continue
            verified = []
            for ann in item.get("annotations", []):
                expected = _compute_hmac(
                    wallet_id,
                    ann.get("label", -1),
                    ann.get("annotator_id", ""),
                    ann.get("annotated_at", ""),
                )
                if hmac.compare_digest(expected, ann.get("annotation_hmac", "")):
                    verified.append(ann["label"])
                else:
                    logger.warning("Invalid HMAC for multi-annotation wallet=%s annotator=%s", wallet_id, ann.get("annotator_id"))
            return verified
        return []

    def _refresh_agreement_status(self, item: dict) -> None:
        """Update item status based on current annotation count and kappa."""
        verified = []
        wallet = item["wallet"]
        for ann in item.get("annotations", []):
            expected = _compute_hmac(wallet, ann.get("label", -1), ann.get("annotator_id", ""), ann.get("annotated_at", ""))
            if hmac.compare_digest(expected, ann.get("annotation_hmac", "")):
                verified.append(ann["label"])
        if len(verified) < 2:
            item["status"] = "multi_annotated"
            return
        # Quick kappa check (reuse logic without full method call)
        from itertools import combinations

        def _kappa_pair(a: int, b: int) -> float:
            if a == b:
                return 1.0
            p0 = ((a == 0) + (b == 0)) / 2.0
            p1 = ((a == 1) + (b == 1)) / 2.0
            p_e = p0 ** 2 + p1 ** 2
            if p_e == 1.0:
                return 1.0
            return (0.0 - p_e) / (1.0 - p_e)

        kappas = [_kappa_pair(verified[i], verified[j]) for i, j in combinations(range(len(verified)), 2)]
        kappa = float(np.mean(kappas))
        item["status"] = "disputed" if kappa < DISPUTE_KAPPA_THRESHOLD else "multi_annotated"
        item["agreement_kappa"] = kappa

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
        """Return the next *n* pending wallets ranked by epistemic uncertainty.

        Wallets with ``epistemic_uncertainty`` set are sorted in descending
        order (highest epistemic uncertainty first — most informative for
        annotation).  Wallets without uncertainty scores retain their
        insertion order, after the ranked wallets.
        """
        queue = _load_queue(self.queue_path)
        pending = [item for item in queue if item.get("status") == "pending"]

        # Separate ranked vs unranked; sort ranked by descending epistemic uncertainty
        ranked = [item for item in pending if item.get("epistemic_uncertainty") is not None]
        unranked = [item for item in pending if item.get("epistemic_uncertainty") is None]
        ranked.sort(key=lambda x: x["epistemic_uncertainty"], reverse=True)

        return (ranked + unranked)[:n]

    def pending_wallets(self) -> list[str]:
        return [item["wallet"] for item in self.pop_batch(10**9)]

    def skipped_wallets(self) -> list[str]:
        queue = _load_queue(self.queue_path)
        return [item["wallet"] for item in queue if item.get("status") == "skipped"]

    def quarantined_samples(self) -> list[dict]:
        """Return all quarantined annotation records."""
        queue = _load_queue(self.queue_path)
        return [
            item
            for item in queue
            if item.get("quarantine") and item.get("status") == "annotated"
        ]

    def dismiss_quarantine(self, wallet: str) -> None:
        """Remove quarantine flag from a wallet (operator override)."""
        queue = _load_queue(self.queue_path)
        for item in queue:
            if item["wallet"] == wallet:
                item["quarantine"] = False
                item["quarantine_reason"] = ""
                break
        _atomic_write(self.queue_path, queue)

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
