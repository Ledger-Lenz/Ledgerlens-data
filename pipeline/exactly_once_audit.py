"""End-to-end exactly-once audit across pipeline stage boundaries (Issue #918).

``pipeline.exactly_once`` gives each stage a two-phase STAGED → COMMITTED
dedup protocol, but each stage builds its own ``ExactlyOnceStore`` from its
own configuration. A misconfigured stage (wrong backend, a TTL shorter than
the redelivery window, a backend that silently fails open) does not raise; it
quietly degrades that boundary to at-least-once (duplicates) or at-most-once
(drops). This module detects that.

``audit_pipeline`` traces a sample of synthetic records through every
configured ``StageBoundary`` in pipeline order and, at each boundary,
replays the delivery sequence a real record goes through:

1. first delivery          → must be ``NEW``        (the record is processed)
2. redelivery before commit → must be ``STAGED``     (redo, not skip or claim twice)
3. commit, then redelivery  → must be ``COMMITTED``  (duplicate, skipped)

It also checks the backend is reachable and that the stage's dedup TTL covers
the redelivery window. See ``docs/exactly_once_audit.md`` for the invariants
and what each violation means operationally.

Audit records use ``external_id`` values prefixed with ``AUDIT_PREFIX`` and a
per-run id, so they never collide with real traffic. They are released with
``mark_failed`` once probed (deleted on Redis, left ``FAILED`` on SQL).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from pipeline.exactly_once import (
    DedupBackendUnavailableError,
    DedupKey,
    DedupState,
    ExactlyOnceStore,
)
from utils.logging import get_logger

logger = get_logger(__name__)

AUDIT_PREFIX = "__eo_audit__"


class Invariant(StrEnum):
    BACKEND_AVAILABLE = "backend_available"
    FIRST_DELIVERY_PROCESSED = "first_delivery_processed"
    INFLIGHT_REDELIVERY_REDONE = "inflight_redelivery_redone"
    COMMITTED_REDELIVERY_SKIPPED = "committed_redelivery_skipped"
    TTL_COVERS_REDELIVERY_WINDOW = "ttl_covers_redelivery_window"


@dataclass(frozen=True)
class StageBoundary:
    """One stage's exactly-once boundary, as that stage is configured."""

    name: str
    store: ExactlyOnceStore
    source: str
    min_ttl_seconds: float = 3600.0


@dataclass(frozen=True)
class Violation:
    stage: str
    invariant: Invariant
    detail: str
    record_id: str | None = None


@dataclass
class AuditReport:
    run_id: str
    stages: list[str]
    records: list[str]
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def violations_for(self, stage: str) -> list[Violation]:
        return [v for v in self.violations if v.stage == stage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ok": self.ok,
            "stages": self.stages,
            "records": self.records,
            "violations": [asdict(v) for v in self.violations],
        }


# Expected state for first delivery, redelivery before commit and redelivery
# after commit, and what a different answer means for the boundary.
_EXPECTATIONS: tuple[tuple[Invariant, DedupState, str], ...] = (
    (
        Invariant.FIRST_DELIVERY_PROCESSED,
        DedupState.NEW,
        "a never-seen record was not treated as new; it would be skipped (at-most-once)",
    ),
    (
        Invariant.INFLIGHT_REDELIVERY_REDONE,
        DedupState.STAGED,
        "a redelivery before commit was not reported as STAGED; the staged claim is not "
        "durable (NEW: processed twice concurrently) or is treated as done (COMMITTED: "
        "dropped if the first attempt crashes)",
    ),
    (
        Invariant.COMMITTED_REDELIVERY_SKIPPED,
        DedupState.COMMITTED,
        "a redelivery after commit was not recognised as a duplicate; side effects run "
        "again (at-least-once)",
    ),
)


def _probe_record(boundary: StageBoundary, record_id: str) -> list[Violation]:
    """Replay the delivery sequence for one record at one boundary."""
    key = DedupKey(source=boundary.source, external_id=record_id)
    store = boundary.store
    violations: list[Violation] = []
    try:
        first = store.check_and_stage(key).state
        redelivered = store.check_and_stage(key).state
        store.commit(key)
        after_commit = store.check_and_stage(key).state
        observed = (first, redelivered, after_commit)
        for (invariant, expected, meaning), state in zip(_EXPECTATIONS, observed, strict=True):
            if state is not expected:
                violations.append(
                    Violation(
                        boundary.name,
                        invariant,
                        f"expected {expected.value}, got {state.value}: {meaning}",
                        record_id,
                    )
                )
        store.mark_failed(key)
    except DedupBackendUnavailableError as exc:
        violations.append(
            Violation(boundary.name, Invariant.BACKEND_AVAILABLE, str(exc), record_id)
        )
    return violations


def _check_boundary_config(boundary: StageBoundary) -> list[Violation]:
    violations: list[Violation] = []
    if not boundary.store.is_available():
        violations.append(
            Violation(
                boundary.name,
                Invariant.BACKEND_AVAILABLE,
                "dedup backend health check failed; the stage cannot confirm duplicates",
            )
        )
    ttl = boundary.store.ttl_seconds
    if ttl < boundary.min_ttl_seconds:
        violations.append(
            Violation(
                boundary.name,
                Invariant.TTL_COVERS_REDELIVERY_WINDOW,
                f"dedup TTL {ttl:.0f}s is shorter than the {boundary.min_ttl_seconds:.0f}s "
                "redelivery window; late redeliveries are processed again (at-least-once)",
            )
        )
    return violations


def audit_pipeline(
    boundaries: list[StageBoundary],
    *,
    sample_size: int = 5,
    run_id: str | None = None,
) -> AuditReport:
    """Trace *sample_size* synthetic records through *boundaries* in order.

    Every record is probed at every boundary, so one misconfigured stage is
    reported without masking problems further down the pipeline.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    run_id = run_id or uuid.uuid4().hex[:12]
    records = [f"{AUDIT_PREFIX}:{run_id}:{i}" for i in range(sample_size)]
    report = AuditReport(run_id=run_id, stages=[b.name for b in boundaries], records=records)

    for boundary in boundaries:
        config_violations = _check_boundary_config(boundary)
        report.violations.extend(config_violations)
        if any(v.invariant is Invariant.BACKEND_AVAILABLE for v in config_violations):
            continue
        for record_id in records:
            report.violations.extend(_probe_record(boundary, record_id))

    for violation in report.violations:
        logger.error(
            "Exactly-once audit violation at stage=%s invariant=%s record=%s: %s",
            violation.stage,
            violation.invariant.value,
            violation.record_id,
            violation.detail,
        )
    logger.info(
        "Exactly-once audit run_id=%s: %d stage(s), %d record(s), %d violation(s)",
        run_id,
        len(boundaries),
        len(records),
        len(report.violations),
    )
    return report
