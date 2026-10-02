"""Idempotency guarantees for repeatable pipeline jobs (Issue #435).

Overview
--------
Re-running the LedgerLens pipeline with the same inputs (same asset pair,
same time window, same feature snapshot) must produce the same observable
output and must *never* double-write to the risk-score store.

This module provides three independent but composable mechanisms:

1. ``CheckpointStore`` — durable SQLite-backed log of completed pipeline
   stages.  Keyed on ``(run_id, pair_id, stage)``.  ``run_id`` defaults to
   a SHA-256 of the canonical inputs so the same inputs always map to the
   same key without the caller needing to track it.

2. ``PipelineCheckpoint`` — context manager that wraps a single stage.
   On entry: checks whether the stage is already complete; if so, skips
   execution.  On successful exit: marks the stage done and persists any
   result payload.  On exception: marks the stage ``failed``.

3. ``idempotent_upsert`` — thin guard around ``RiskScoreStore.upsert`` that
   compares the *content hash* of the incoming risk score against the stored
   record and skips the write when nothing has changed.  This is the last
   line of defence against double-writes during partial pipeline replays.

Usage
-----
::

    from pipeline.idempotency import CheckpointStore, PipelineCheckpoint, idempotent_upsert

    store = CheckpointStore()           # defaults to IDEMPOTENCY_DB_URL
    run_id = CheckpointStore.make_run_id("USDC:.../XLM:native", since_iso)

    with PipelineCheckpoint(store, run_id, "USDC:.../XLM:native", "ingest") as cp:
        if cp.skip:
            trades_df = cp.result           # replay previous result
        else:
            trades_df = load_pair_to_dataframe(...)
            cp.set_result({"row_count": len(trades_df)})

    # ... later, at persist step ...
    idempotent_upsert(score_store, wallet, pair_id, risk_score_dict)

Environment variables
---------------------
``IDEMPOTENCY_DB_URL``
    SQLAlchemy URL for the checkpoint database.  Defaults to
    ``sqlite:///pipeline_checkpoints.db``.  Use an in-memory SQLite DB
    (``sqlite:///:memory:``) in unit tests.

``IDEMPOTENCY_TTL_HOURS``
    How many hours a checkpoint (idempotency key) is honoured.  Checkpoints
    older than this are treated as stale (re-run) and are evicted from the
    store, so it stays bounded.  Defaults to ``48``; see ``docs/idempotency.md``
    for the reasoning behind that value.

Metrics
-------
``ledgerlens_idempotency_store_entries{status}`` (gauge) tracks store size and
``ledgerlens_idempotency_store_evicted_total`` (counter) the TTL evictions;
both feed the "Idempotency Key Store" Grafana dashboard.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    func,
    or_,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import Counter, Gauge

    IDEMPOTENCY_STORE_ENTRIES = Gauge(
        "ledgerlens_idempotency_store_entries",
        "Pipeline idempotency-key (checkpoint) entries currently stored, by status",
        ["status"],
    )
    IDEMPOTENCY_STORE_EVICTED = Counter(
        "ledgerlens_idempotency_store_evicted_total",
        "Pipeline idempotency-key entries evicted after exceeding IDEMPOTENCY_TTL_HOURS",
    )
except ImportError:  # pragma: no cover - prometheus_client is a hard dependency in prod
    IDEMPOTENCY_STORE_ENTRIES = None
    IDEMPOTENCY_STORE_EVICTED = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_IDEMPOTENCY_DB_URL: str = getattr(
    config, "IDEMPOTENCY_DB_URL", "sqlite:///pipeline_checkpoints.db"
)
_IDEMPOTENCY_TTL_HOURS: int = int(getattr(config, "IDEMPOTENCY_TTL_HOURS", "48"))

# Minimum spacing between automatic TTL evictions triggered by stage starts.
_EVICTION_INTERVAL_SECONDS = 3600.0

# Statuses reported by the store-size gauge (see CheckpointRecord.status).
_STATUSES: tuple[str, ...] = ("pending", "running", "done", "failed")

# Stage names recognised by the pipeline — ordered by execution position.
PIPELINE_STAGES: tuple[str, ...] = (
    "ingest",
    "orderbook",
    "funding_graph",
    "features",
    "scoring",
    "persist",
    "onchain",
)

# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class CheckpointRecord(_Base):
    """One row per (run_id, pair_id, stage) triple."""

    __tablename__ = "pipeline_checkpoints"
    __table_args__ = (UniqueConstraint("run_id", "pair_id", "stage", name="uq_checkpoint"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pair_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    # "pending" | "running" | "done" | "failed"
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


class CheckpointStore:
    """Durable store for pipeline stage completion records.

    Thread-safe: each public method opens its own session.

    Entries older than the TTL are evicted when the store is opened and then
    at most once per hour as stages start, so the store only ever holds keys
    from the last ``ttl_hours``.

    Parameters
    ----------
    db_url:
        SQLAlchemy URL.  Defaults to ``IDEMPOTENCY_DB_URL`` from the
        environment (see module docstring).
    ttl_hours:
        Checkpoints older than this are treated as stale and evicted.
    """

    def __init__(
        self,
        db_url: str | None = None,
        ttl_hours: int | None = None,
    ) -> None:
        self._db_url = db_url or _IDEMPOTENCY_DB_URL
        self._ttl_hours = ttl_hours if ttl_hours is not None else _IDEMPOTENCY_TTL_HOURS
        self._lock = threading.Lock()

        engine = create_engine(
            self._db_url,
            connect_args={"check_same_thread": False} if "sqlite" in self._db_url else {},
        )
        _Base.metadata.create_all(engine, checkfirst=True)
        self._session_factory: sessionmaker[Session] = sessionmaker(bind=engine, future=True)
        self._last_eviction = 0.0
        self.evict_expired()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_run_id(*components: str) -> str:
        """Derive a deterministic 16-character run-id from canonical inputs.

        The run_id is a prefix of the SHA-256 of the joined components.
        Same components always produce the same run_id, ensuring that
        re-running the pipeline with identical inputs resumes the same
        checkpoint row.

        Example::

            run_id = CheckpointStore.make_run_id(pair_id, since_iso or "all")
        """
        raw = "|".join(components)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_complete(self, run_id: str, pair_id: str, stage: str) -> bool:
        """Return True if *stage* completed successfully within the TTL window."""
        record = self._get(run_id, pair_id, stage)
        if record is None or record.status != "done":
            return False
        if record.completed_at is None:
            return False
        # SQLite may store datetimes as naive; make them UTC-aware for comparison.
        completed_at = record.completed_at
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=UTC)
        age = datetime.now(UTC) - completed_at
        if age > timedelta(hours=self._ttl_hours):
            logger.info(
                "Checkpoint for stage=%s run_id=%s is stale (age=%s > ttl=%dh); re-running",
                stage,
                run_id,
                age,
                self._ttl_hours,
            )
            return False
        return True

    def get_result(self, run_id: str, pair_id: str, stage: str) -> Any:
        """Return the stored result payload for a completed stage, or None."""
        record = self._get(run_id, pair_id, stage)
        if record is None or record.result_json is None:
            return None
        try:
            return json.loads(record.result_json)
        except json.JSONDecodeError:
            logger.warning("Could not decode result_json for stage=%s run_id=%s", stage, run_id)
            return None

    def mark_started(self, run_id: str, pair_id: str, stage: str) -> None:
        """Record that a stage has begun (status → running)."""
        if time.monotonic() - self._last_eviction >= _EVICTION_INTERVAL_SECONDS:
            self.evict_expired()
        self._upsert_status(run_id, pair_id, stage, "running", started_at=datetime.now(UTC))

    def mark_done(
        self,
        run_id: str,
        pair_id: str,
        stage: str,
        result: Any = None,
    ) -> None:
        """Record that a stage completed successfully."""
        result_json = json.dumps(result) if result is not None else None
        self._upsert_status(
            run_id,
            pair_id,
            stage,
            "done",
            completed_at=datetime.now(UTC),
            result_json=result_json,
        )

    def mark_failed(self, run_id: str, pair_id: str, stage: str, error: str = "") -> None:
        """Record that a stage failed with an optional error message."""
        self._upsert_status(
            run_id,
            pair_id,
            stage,
            "failed",
            completed_at=datetime.now(UTC),
            error_message=error,
        )

    def list_stages(self, run_id: str, pair_id: str) -> list[CheckpointRecord]:
        """Return all checkpoint rows for a given (run_id, pair_id)."""
        with self._session_factory() as session:
            rows = list(
                session.scalars(
                    select(CheckpointRecord)
                    .where(
                        CheckpointRecord.run_id == run_id,
                        CheckpointRecord.pair_id == pair_id,
                    )
                    .order_by(CheckpointRecord.id)
                )
            )
            return rows

    def first_incomplete_stage(self, run_id: str, pair_id: str) -> str | None:
        """Return the name of the first PIPELINE_STAGE that is not yet 'done'.

        Returns None if all canonical stages are complete.
        """

        now = datetime.now(UTC)
        completed = set()
        for r in self.list_stages(run_id, pair_id):
            if r.status != "done" or r.completed_at is None:
                continue
            ts = r.completed_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if (now - ts) < timedelta(hours=self._ttl_hours):
                completed.add(r.stage)

        for stage in PIPELINE_STAGES:
            if stage not in completed:
                return stage
        return None

    def purge_old_checkpoints(self, older_than_hours: int | None = None) -> int:
        """Delete checkpoints older than *older_than_hours* (default: 7 × ttl).

        Returns the number of rows deleted.
        """
        cutoff_hours = older_than_hours if older_than_hours is not None else self._ttl_hours * 7
        cutoff = datetime.now(UTC) - timedelta(hours=cutoff_hours)
        with self._session_factory() as session:
            rows = list(
                session.scalars(
                    select(CheckpointRecord).where(
                        CheckpointRecord.completed_at < cutoff,
                    )
                )
            )
            for r in rows:
                session.delete(r)
            session.commit()
            return len(rows)

    def evict_expired(self) -> int:
        """Delete entries whose last update is older than the TTL.

        Such keys no longer block reprocessing (``is_complete`` already treats
        them as stale), so dropping them only bounds storage.  Returns the
        number of rows evicted and refreshes the store-size metrics.
        """
        cutoff = datetime.now(UTC) - timedelta(hours=self._ttl_hours)
        with self._lock, self._session_factory() as session:
            result = session.execute(
                delete(CheckpointRecord).where(
                    or_(
                        CheckpointRecord.completed_at < cutoff,
                        and_(
                            CheckpointRecord.completed_at.is_(None),
                            CheckpointRecord.started_at < cutoff,
                        ),
                    )
                )
            )
            session.commit()
            evicted = int(result.rowcount or 0)
        self._last_eviction = time.monotonic()
        if evicted:
            logger.info(
                "Evicted %d idempotency key(s) older than ttl=%dh", evicted, self._ttl_hours
            )
        if IDEMPOTENCY_STORE_EVICTED is not None:
            IDEMPOTENCY_STORE_EVICTED.inc(evicted)
        self.size()
        return evicted

    def size(self) -> dict[str, int]:
        """Return entry counts by status and publish them to the size gauge."""
        with self._session_factory() as session:
            counts = dict(
                session.execute(
                    select(CheckpointRecord.status, func.count()).group_by(CheckpointRecord.status)
                ).all()
            )
        sizes = {status: int(counts.get(status, 0)) for status in _STATUSES}
        if IDEMPOTENCY_STORE_ENTRIES is not None:
            for status, count in sizes.items():
                IDEMPOTENCY_STORE_ENTRIES.labels(status=status).set(count)
        return sizes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, run_id: str, pair_id: str, stage: str) -> CheckpointRecord | None:
        with self._session_factory() as session:
            return session.scalar(
                select(CheckpointRecord).where(
                    CheckpointRecord.run_id == run_id,
                    CheckpointRecord.pair_id == pair_id,
                    CheckpointRecord.stage == stage,
                )
            )

    def _upsert_status(
        self,
        run_id: str,
        pair_id: str,
        stage: str,
        status: str,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        result_json: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lock:
            with self._session_factory() as session:
                record = session.scalar(
                    select(CheckpointRecord).where(
                        CheckpointRecord.run_id == run_id,
                        CheckpointRecord.pair_id == pair_id,
                        CheckpointRecord.stage == stage,
                    )
                )
                if record is None:
                    record = CheckpointRecord(
                        run_id=run_id,
                        pair_id=pair_id,
                        stage=stage,
                    )
                    session.add(record)
                record.status = status
                if started_at is not None:
                    record.started_at = started_at
                if completed_at is not None:
                    record.completed_at = completed_at
                if result_json is not None:
                    record.result_json = result_json
                if error_message is not None:
                    record.error_message = error_message
                session.commit()


# ---------------------------------------------------------------------------
# PipelineCheckpoint context manager
# ---------------------------------------------------------------------------


class _CheckpointState:
    """Passed into the ``with PipelineCheckpoint(...)`` block via __enter__."""

    def __init__(self, skip: bool, result: Any) -> None:
        self.skip = skip
        """True when the stage was already completed — caller should skip work."""
        self.result = result
        """Stored result payload from the previous successful run (may be None)."""
        self._pending_result: Any = None

    def set_result(self, result: Any) -> None:
        """Record a result payload to be persisted when the context exits cleanly."""
        self._pending_result = result


class PipelineCheckpoint:
    """Context manager that wraps a single pipeline stage with idempotency.

    Usage::

        with PipelineCheckpoint(store, run_id, pair_id, "ingest") as cp:
            if cp.skip:
                trades_df = reload_from_somewhere(cp.result)
            else:
                trades_df = load_pair_to_dataframe(...)
                cp.set_result({"row_count": len(trades_df)})

    On clean exit the stage is marked ``done`` and any ``set_result`` payload
    is persisted.  On exception the stage is marked ``failed`` and the
    exception re-raised.

    Parameters
    ----------
    store:
        The ``CheckpointStore`` instance backing this checkpoint.
    run_id:
        Unique identifier for this pipeline run (use
        ``CheckpointStore.make_run_id`` to derive a deterministic value).
    pair_id:
        Asset-pair identifier (e.g. ``"USDC:.../XLM:native"``).
    stage:
        Stage name — should be one of ``PIPELINE_STAGES`` but is not enforced
        so custom stages work too.
    force:
        If True, the stage is always re-run even if already completed.
    """

    def __init__(
        self,
        store: CheckpointStore,
        run_id: str,
        pair_id: str,
        stage: str,
        force: bool = False,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._pair_id = pair_id
        self._stage = stage
        self._force = force
        self._state: _CheckpointState | None = None

    def __enter__(self) -> _CheckpointState:
        already_done = not self._force and self._store.is_complete(
            self._run_id, self._pair_id, self._stage
        )
        if already_done:
            prior_result = self._store.get_result(self._run_id, self._pair_id, self._stage)
            logger.info(
                "Skipping stage=%s run_id=%s pair=%s (already completed)",
                self._stage,
                self._run_id,
                self._pair_id,
            )
            self._state = _CheckpointState(skip=True, result=prior_result)
        else:
            self._store.mark_started(self._run_id, self._pair_id, self._stage)
            logger.debug(
                "Starting stage=%s run_id=%s pair=%s",
                self._stage,
                self._run_id,
                self._pair_id,
            )
            self._state = _CheckpointState(skip=False, result=None)
        return self._state

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        # If __enter__ raised before self._state was set, nothing to record.
        if self._state is None:
            return False
        if self._state.skip:
            # Stage was skipped — nothing to record.
            return False
        if exc_type is None:
            self._store.mark_done(
                self._run_id,
                self._pair_id,
                self._stage,
                result=self._state._pending_result,
            )
            logger.debug(
                "Completed stage=%s run_id=%s pair=%s",
                self._stage,
                self._run_id,
                self._pair_id,
            )
        else:
            error_msg = f"{exc_type.__name__}: {exc_val}"
            self._store.mark_failed(self._run_id, self._pair_id, self._stage, error=error_msg)
            logger.error(
                "Stage=%s run_id=%s pair=%s failed: %s",
                self._stage,
                self._run_id,
                self._pair_id,
                error_msg,
            )
        # Never suppress the exception.
        return False


# ---------------------------------------------------------------------------
# Idempotent upsert guard
# ---------------------------------------------------------------------------


def _risk_score_hash(risk_score: dict) -> str:
    """Return a stable hex digest of the *content* of a risk-score dict.

    Only the fields that belong to the on-chain RiskScore shape are hashed
    so incidental metadata changes don't trigger unnecessary writes.
    """
    canonical = {
        k: risk_score[k]
        for k in ("score", "benford_flag", "ml_flag", "confidence")
        if k in risk_score
    }
    # ring_id is optional; include when present.
    if "ring_id" in risk_score:
        canonical["ring_id"] = risk_score["ring_id"]
    serialised = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def idempotent_upsert(
    store: Any,
    wallet: str,
    asset_pair: str,
    risk_score: dict,
    *,
    finality: str = "provisional",
) -> tuple[bool, Any]:
    """Write *risk_score* only when the content differs from the stored record.

    Parameters
    ----------
    store:
        A ``RiskScoreStore`` instance (or any object with ``get`` and
        ``upsert`` methods matching that interface).
    wallet:
        Stellar wallet address.
    asset_pair:
        Canonical pair identifier (``"CODE:ISSUER/CODE:ISSUER"``).
    risk_score:
        Dict with at minimum ``score``, ``benford_flag``, ``ml_flag``,
        ``confidence`` keys.

    Returns
    -------
    (was_written, record)
        ``was_written`` is False when the existing record already matched;
        ``record`` is the current ``RiskScoreRecord`` after the call.
    """
    incoming_hash = _risk_score_hash(risk_score)

    existing = store.get(wallet, asset_pair)
    if existing is not None:
        existing_score_dict = {
            "score": existing.score,
            "benford_flag": existing.benford_flag,
            "ml_flag": existing.ml_flag,
            "confidence": existing.confidence,
        }
        if hasattr(existing, "ring_id") and existing.ring_id is not None:
            existing_score_dict["ring_id"] = existing.ring_id
        existing_hash = _risk_score_hash(existing_score_dict)
        if incoming_hash == existing_hash:
            logger.debug(
                "idempotent_upsert: no change for wallet=%s pair=%s — skipping write",
                wallet,
                asset_pair,
            )
            return False, existing

    record = store.upsert(wallet, asset_pair, risk_score, finality=finality)
    return True, record
