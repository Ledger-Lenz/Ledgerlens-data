"""Partial pipeline execution recovery (Issue #578).

Overview
--------
When a long-running pipeline job is interrupted mid-way — by an OOM kill,
a network time-out, or a KeyboardInterrupt — the next invocation should
*resume* from the last successfully completed stage rather than re-running
everything from scratch.

This module builds on the ``CheckpointStore`` from ``pipeline.idempotency``
(which handles the durable state) and adds:

1. ``StageTracker`` — thin wrapper around a ``CheckpointStore`` that adds
   wall-clock timing, per-stage row counts, and a human-readable run summary.

2. ``RecoveryManager`` — orchestrates a resumable pipeline run.  It exposes
   the ``stage(name)`` context manager that pipelines use to wrap each step,
   and ``resume_info(run_id, pair_id)`` to report what will be skipped on
   the next invocation.

3. ``rollback_partial_writes`` — best-effort DB cleanup for stages that
   touched the risk-score store but did not reach the ``persist`` completion
   checkpoint.  Deletes rows written during a failed run so re-runs start
   from a clean slate.

4. ``verify_recovery`` / ``RecoveryManager.complete_recovery`` (Issue #920)
   — post-recovery consistency verification.  After a recovery completes,
   the expected state (record counts / checksums per stage, captured from
   the source of truth) is compared against the actual recovered state.
   The result is a ``RecoveryReport`` that renders as a human-readable
   summary for on-call responders (see ``docs/recovery_verification.md``).
   A failed check blocks auto-resumption: ``RecoveryManager.stage()``
   raises ``RecoveryBlockedError`` for that ``(run_id, pair_id)`` until an
   operator calls ``approve_resume()`` after manual review.

Design decisions
----------------
* **No saga-style compensating transactions** — the pipeline writes to a
  single SQLite/Postgres DB.  A simple "delete rows with matching run_id"
  rollback is sufficient; there is nothing to compensate in external systems
  because on-chain submission (``onchain`` stage) only runs after ``persist``
  is complete, and a failed ``onchain`` stage is logged and retried rather
  than rolled back.

* **Atomic stage transitions** — the underlying ``CheckpointStore`` uses
  ``UniqueConstraint`` + ``SQLAlchemy`` transactions so concurrent writes to
  the same checkpoint key are safe.

* **Idempotent resume** — calling ``RecoveryManager.stage()`` for an already-
  completed stage returns immediately (the inner block is not executed).

Usage
-----
::

    from pipeline.recovery import RecoveryManager
    from pipeline.idempotency import CheckpointStore

    store = CheckpointStore()
    rm = RecoveryManager(store)
    run_id = CheckpointStore.make_run_id(pair_id, since_iso or "all")

    with rm.stage(run_id, pair_id, "ingest") as ctx:
        if not ctx.skip:
            trades_df = load_pair_to_dataframe(...)
            ctx.set_result({"row_count": len(trades_df)})

    with rm.stage(run_id, pair_id, "features") as ctx:
        if not ctx.skip:
            feature_matrix = build_feature_matrix(trades_df)
            ctx.set_result({"wallet_count": len(feature_matrix)})

    summary = rm.run_summary(run_id, pair_id)
    print(summary)
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pipeline.idempotency import (
    CheckpointStore,
    PipelineCheckpoint,
    _CheckpointState,
)
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# StageResult — rich per-stage execution record
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Execution record for a single pipeline stage."""

    stage: str
    status: str  # "skipped" | "completed" | "failed"
    wall_seconds: float = 0.0
    result_payload: Any = None
    error: str = ""


# ---------------------------------------------------------------------------
# StageTracker
# ---------------------------------------------------------------------------


class StageTracker:
    """Accumulates per-stage timing and result data for a pipeline run.

    Not persisted — lives in memory for the duration of the process.
    Complements the durable ``CheckpointStore`` by adding rich reporting.
    """

    def __init__(self) -> None:
        self._stages: list[StageResult] = []

    def record(self, result: StageResult) -> None:
        self._stages.append(result)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the run."""
        total_wall = sum(r.wall_seconds for r in self._stages)
        return {
            "stages": [
                {
                    "stage": r.stage,
                    "status": r.status,
                    "wall_seconds": round(r.wall_seconds, 3),
                    "result_payload": r.result_payload,
                    "error": r.error or None,
                }
                for r in self._stages
            ],
            "total_wall_seconds": round(total_wall, 3),
            "completed_count": sum(1 for r in self._stages if r.status == "completed"),
            "skipped_count": sum(1 for r in self._stages if r.status == "skipped"),
            "failed_count": sum(1 for r in self._stages if r.status == "failed"),
        }

    def failed_stages(self) -> list[str]:
        return [r.stage for r in self._stages if r.status == "failed"]

    def has_failures(self) -> bool:
        return any(r.status == "failed" for r in self._stages)


# ---------------------------------------------------------------------------
# Post-recovery consistency verification (Issue #920)
# ---------------------------------------------------------------------------


class RecoveryBlockedError(RuntimeError):
    """Raised when a stage is entered for a run whose post-recovery
    consistency check failed and has not yet been approved by an operator."""

    def __init__(self, run_id: str, pair_id: str, report: RecoveryReport) -> None:
        self.run_id = run_id
        self.pair_id = pair_id
        self.report = report
        super().__init__(
            f"auto-resumption blocked for run={run_id} pair={pair_id}: post-recovery "
            f"consistency check failed for stage(s) {report.failed_stages()}; "
            "manual review required (RecoveryManager.approve_resume)"
        )


def compute_checksum(records: Iterable[Any]) -> str:
    """Order-independent SHA-256 checksum over a collection of records.

    Each record is serialised to canonical JSON (sorted keys, ``str`` fallback
    for non-JSON types); the per-record digests are sorted before hashing so
    the checksum does not depend on row order, which recovery does not
    guarantee to preserve.
    """
    digests = sorted(
        hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        for r in records
    )
    return hashlib.sha256("".join(digests).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StageSnapshot:
    """Record count and optional checksum of one stage's output."""

    record_count: int
    checksum: str | None = None

    @classmethod
    def from_records(cls, records: Iterable[Any]) -> StageSnapshot:
        materialised = list(records)
        return cls(record_count=len(materialised), checksum=compute_checksum(materialised))


@dataclass
class ConsistencyCheck:
    """Expected-vs-actual comparison for a single recovered stage."""

    stage: str
    expected: StageSnapshot | None
    actual: StageSnapshot | None

    @property
    def issues(self) -> list[str]:
        if self.expected is None:
            return ["no expected state recorded"]
        if self.actual is None:
            return ["stage output missing after recovery"]
        problems = []
        if self.expected.record_count != self.actual.record_count:
            problems.append(
                f"record count mismatch: expected {self.expected.record_count}, "
                f"actual {self.actual.record_count}"
            )
        if (
            self.expected.checksum is not None
            and self.actual.checksum is not None
            and self.expected.checksum != self.actual.checksum
        ):
            problems.append(
                f"checksum mismatch: expected {self.expected.checksum[:12]}…, "
                f"actual {self.actual.checksum[:12]}…"
            )
        return problems

    @property
    def passed(self) -> bool:
        return not self.issues


@dataclass
class RecoveryReport:
    """Outcome of a post-recovery consistency verification.

    ``render()`` produces the on-call report documented in
    ``docs/recovery_verification.md``; ``to_dict()`` is JSON-serialisable
    for structured logs / ticket attachments.
    """

    run_id: str
    pair_id: str
    recovered_range: tuple[str, str] | None
    recovered_stages: list[str]
    checks: list[ConsistencyCheck]
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    approved_by: str | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def resume_allowed(self) -> bool:
        return self.passed or self.approved_by is not None

    def failed_stages(self) -> list[str]:
        return [c.stage for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pair_id": self.pair_id,
            "generated_at": self.generated_at,
            "recovered_range": list(self.recovered_range) if self.recovered_range else None,
            "recovered_stages": list(self.recovered_stages),
            "result": "PASS" if self.passed else "FAIL",
            "resume_allowed": self.resume_allowed,
            "approved_by": self.approved_by,
            "checks": [
                {
                    "stage": c.stage,
                    "status": "PASS" if c.passed else "FAIL",
                    "expected_count": c.expected.record_count if c.expected else None,
                    "actual_count": c.actual.record_count if c.actual else None,
                    "expected_checksum": c.expected.checksum if c.expected else None,
                    "actual_checksum": c.actual.checksum if c.actual else None,
                    "issues": c.issues,
                }
                for c in self.checks
            ],
        }

    def render(self) -> str:
        """Human-readable report for on-call responders."""
        if self.recovered_range:
            range_text = f"{self.recovered_range[0]} → {self.recovered_range[1]}"
        else:
            range_text = "(not specified)"
        if self.passed:
            action = "Automatic resumption of normal processing: ALLOWED"
        elif self.approved_by:
            action = f"Automatic resumption: ALLOWED after manual review by {self.approved_by}"
        else:
            action = (
                "Automatic resumption: BLOCKED — manual review required. Investigate the "
                "failed stages, then call RecoveryManager.approve_resume(run_id, pair_id, "
                "reviewer) or re-run recovery."
            )
        lines = [
            "=== LedgerLens post-recovery consistency report ===",
            f"Run ID:           {self.run_id}",
            f"Pair:             {self.pair_id}",
            f"Generated at:     {self.generated_at}",
            f"Recovered range:  {range_text}",
            f"Recovered stages: {', '.join(self.recovered_stages) or '(none)'}",
            f"Result:           {'PASS' if self.passed else 'FAIL'}",
            "",
            "Stage checks:",
        ]
        for c in self.checks:
            exp = c.expected.record_count if c.expected else "-"
            act = c.actual.record_count if c.actual else "-"
            lines.append(
                f"  [{'PASS' if c.passed else 'FAIL'}] {c.stage}: expected={exp} actual={act}"
            )
            lines.extend(f"         - {issue}" for issue in c.issues)
        lines += ["", action]
        return "\n".join(lines)


def verify_recovery(
    run_id: str,
    pair_id: str,
    expected: dict[str, StageSnapshot],
    actual: dict[str, StageSnapshot],
    recovered_range: tuple[str, str] | None = None,
    recovered_stages: list[str] | None = None,
) -> RecoveryReport:
    """Compare expected vs actual per-stage state across a recovered range.

    Every stage present in either ``expected`` or ``actual`` is checked, so
    a stage that unexpectedly appears (or disappears) after recovery is
    reported as a failure rather than silently ignored.
    """
    stages = list(dict.fromkeys([*expected, *actual]))
    checks = [ConsistencyCheck(s, expected.get(s), actual.get(s)) for s in stages]
    return RecoveryReport(
        run_id=run_id,
        pair_id=pair_id,
        recovered_range=recovered_range,
        recovered_stages=list(recovered_stages) if recovered_stages is not None else stages,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# RecoveryManager — main API
# ---------------------------------------------------------------------------


class RecoveryManager:
    """Orchestrates resumable, idempotent pipeline execution.

    Parameters
    ----------
    store:
        The ``CheckpointStore`` that durably records stage completion.
    """

    def __init__(self, store: CheckpointStore) -> None:
        self._store = store
        # Keyed by (run_id, pair_id) to support multiple concurrent pairs.
        self._trackers: dict[tuple[str, str], StageTracker] = {}
        # Latest post-recovery verification report per (run_id, pair_id).
        self._reports: dict[tuple[str, str], RecoveryReport] = {}

    def _tracker(self, run_id: str, pair_id: str) -> StageTracker:
        key = (run_id, pair_id)
        if key not in self._trackers:
            self._trackers[key] = StageTracker()
        return self._trackers[key]

    @contextmanager
    def stage(
        self,
        run_id: str,
        pair_id: str,
        stage_name: str,
        force: bool = False,
    ) -> Generator[_CheckpointState, None, None]:
        """Context manager wrapping a single pipeline stage.

        * If the stage already completed (within TTL), yields a
          ``_CheckpointState`` with ``skip=True`` so the caller can short-
          circuit its work.
        * If the stage has not completed, runs the block, records timing, and
          marks the stage ``done`` or ``failed`` on exit.

        Parameters
        ----------
        run_id:
            Unique pipeline run identifier.
        pair_id:
            Asset-pair being processed.
        stage_name:
            Name of the stage (one of ``PIPELINE_STAGES`` or custom).
        force:
            Re-run even if already completed.

        Raises
        ------
        RecoveryBlockedError
            If a post-recovery consistency check for this ``(run_id,
            pair_id)`` failed and has not been approved via
            ``approve_resume``.
        """
        report = self._reports.get((run_id, pair_id))
        if report is not None and not report.resume_allowed:
            raise RecoveryBlockedError(run_id, pair_id, report)

        tracker = self._tracker(run_id, pair_id)
        cp = PipelineCheckpoint(self._store, run_id, pair_id, stage_name, force=force)

        t0 = time.monotonic()
        state: _CheckpointState | None = None
        exc_info: tuple[Any, Any, Any] = (None, None, None)
        cp_entered = False

        try:
            state = cp.__enter__()
            cp_entered = True
            yield state
            exc_info = (None, None, None)
        except Exception as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            elapsed = time.monotonic() - t0
            if cp_entered:
                cp.__exit__(*exc_info)

            if state is not None:
                if state.skip:
                    status = "skipped"
                    error_msg = ""
                elif exc_info[0] is None:
                    status = "completed"
                    error_msg = ""
                else:
                    status = "failed"
                    error_msg = f"{exc_info[0].__name__}: {exc_info[1]}"

                tracker.record(
                    StageResult(
                        stage=stage_name,
                        status=status,
                        wall_seconds=elapsed,
                        result_payload=state._pending_result if state else None,
                        error=error_msg,
                    )
                )

    def resume_info(self, run_id: str, pair_id: str) -> dict[str, Any]:
        """Return a human-readable dict describing what will happen on resume.

        Useful for logging at the start of a pipeline run so operators can
        see what was already completed.
        """
        first_incomplete = self._store.first_incomplete_stage(run_id, pair_id)
        completed_stages = [
            r.stage for r in self._store.list_stages(run_id, pair_id) if r.status == "done"
        ]
        return {
            "run_id": run_id,
            "pair_id": pair_id,
            "completed_stages": completed_stages,
            "first_incomplete_stage": first_incomplete,
            "will_resume": first_incomplete is not None and len(completed_stages) > 0,
        }

    def run_summary(self, run_id: str, pair_id: str) -> dict[str, Any]:
        """Return the in-memory stage tracker summary for (run_id, pair_id)."""
        return self._tracker(run_id, pair_id).summary()

    def has_failures(self, run_id: str, pair_id: str) -> bool:
        """True if any stage for this (run_id, pair_id) failed in this process."""
        return self._tracker(run_id, pair_id).has_failures()

    def complete_recovery(
        self,
        run_id: str,
        pair_id: str,
        expected: dict[str, StageSnapshot],
        actual: dict[str, StageSnapshot],
        recovered_range: tuple[str, str] | None = None,
    ) -> RecoveryReport:
        """Run the post-recovery consistency check and gate resumption.

        Must be called once recovery has finished and before the pipeline is
        declared healthy.  On success normal processing may resume
        automatically; on failure every subsequent ``stage()`` call for this
        ``(run_id, pair_id)`` raises ``RecoveryBlockedError`` until
        ``approve_resume`` is called.
        """
        recovered_stages = [
            r.stage for r in self._tracker(run_id, pair_id)._stages if r.status == "completed"
        ]
        report = verify_recovery(
            run_id,
            pair_id,
            expected,
            actual,
            recovered_range=recovered_range,
            recovered_stages=recovered_stages or None,
        )
        self._reports[(run_id, pair_id)] = report
        if report.passed:
            logger.info(
                "complete_recovery: consistency check PASSED run=%s pair=%s; resuming",
                run_id,
                pair_id,
            )
        else:
            logger.error(
                "complete_recovery: consistency check FAILED run=%s pair=%s stages=%s; "
                "auto-resumption blocked pending manual review\n%s",
                run_id,
                pair_id,
                report.failed_stages(),
                report.render(),
            )
        return report

    def can_auto_resume(self, run_id: str, pair_id: str) -> bool:
        """False only while a failed, unapproved recovery report is on file."""
        report = self._reports.get((run_id, pair_id))
        return report is None or report.resume_allowed

    def recovery_report(self, run_id: str, pair_id: str) -> RecoveryReport | None:
        """Return the latest post-recovery report for (run_id, pair_id), if any."""
        return self._reports.get((run_id, pair_id))

    def approve_resume(self, run_id: str, pair_id: str, reviewer: str) -> None:
        """Record a manual-review sign-off, unblocking a failed recovery."""
        if not reviewer:
            raise ValueError("reviewer must be a non-empty identifier")
        report = self._reports.get((run_id, pair_id))
        if report is None:
            raise KeyError(f"no recovery report for run={run_id} pair={pair_id}")
        report.approved_by = reviewer
        logger.warning(
            "approve_resume: run=%s pair=%s resumption approved by %s despite failed stages %s",
            run_id,
            pair_id,
            reviewer,
            report.failed_stages(),
        )


# ---------------------------------------------------------------------------
# rollback_partial_writes
# ---------------------------------------------------------------------------


def rollback_partial_writes(
    score_store: Any,
    wallets: list[str],
    pair_id: str,
) -> int:
    """Remove risk-score rows written during a failed/incomplete pipeline run.

    Called when a run is abandoned (e.g. the ``persist`` checkpoint is not
    marked ``done``) to ensure the next invocation starts from a clean slate.
    Only deletes rows for ``pair_id`` — other pairs are unaffected.

    Parameters
    ----------
    score_store:
        A ``RiskScoreStore`` (or duck-typed equivalent) with a ``delete``
        method.  If the store does not expose ``delete``, this function logs
        a warning and returns 0.
    wallets:
        List of wallet addresses that may have been written.
    pair_id:
        Asset-pair identifier.

    Returns
    -------
    int
        Number of rows deleted.
    """
    if not hasattr(score_store, "delete"):
        logger.warning(
            "rollback_partial_writes: score_store does not expose .delete(); "
            "no rollback performed for pair=%s",
            pair_id,
        )
        return 0

    deleted = 0
    for wallet in wallets:
        try:
            removed = score_store.delete(wallet, pair_id)
            if removed:
                deleted += 1
                logger.debug("rollback_partial_writes: deleted wallet=%s pair=%s", wallet, pair_id)
        except Exception as exc:
            logger.warning(
                "rollback_partial_writes: failed to delete wallet=%s pair=%s: %s",
                wallet,
                pair_id,
                exc,
            )
    if deleted:
        logger.info(
            "rollback_partial_writes: removed %d stale rows for pair=%s",
            deleted,
            pair_id,
        )
    return deleted
