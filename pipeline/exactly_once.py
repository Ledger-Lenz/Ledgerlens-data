"""Unified exactly-once dedup primitive for the trade-processing pipeline (Issue #670).

See ``docs/adr/0001-unified-idempotency-finality.md`` for the full design
rationale. In short:

- ``DedupKey`` is the single canonical key format shared across the
  pipeline's dedup/idempotency mechanisms. It reserves a ``tenant_id`` slot
  (currently always ``None``) so a future multi-tenancy change (Grand 4)
  does not require a second migration of every dependent store.

- ``ExactlyOnceStore`` implements a two-phase STAGED → COMMITTED protocol:
  a caller stages a key *before* doing side-effecting work, and commits it
  *after* the work durably succeeds. A crash or exception between staging
  and commit leaves the record ``STAGED``, which is reported back as "redo
  this work", never conflated with "this was already done" (``COMMITTED``)
  nor with "never seen" (``NEW``). This is what makes redelivery safe: the
  caller can tell the difference between "duplicate, skip" and "prior
  attempt died mid-flight, retry".

- Two backends are provided. ``RedisExactlyOnceBackend`` is the low-latency
  backend for the streaming hot path (Kafka worker, trade ingestion dedup);
  it is explicitly **fail-closed** — a Redis outage raises
  ``DedupBackendUnavailableError`` rather than silently reporting "not a
  duplicate" (invariant 8 of Issue #670: a correctness-critical dedup path
  must never fail open). ``SqlExactlyOnceBackend`` is a durable backend for
  batch/offline use, built on the same SQLAlchemy engine pattern as
  ``detection.persistence``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy import (
    JSON,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
    select,
)
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import Gauge

    DEDUP_BACKEND_AVAILABLE = Gauge(
        "dedup_backend_available",
        "1 if the exactly-once dedup backend is reachable, 0 if degraded/blocked",
        ["source"],
    )
except ImportError:  # pragma: no cover - prometheus_client is a hard dependency in prod
    DEDUP_BACKEND_AVAILABLE = None


# ---------------------------------------------------------------------------
# Canonical key scheme
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupKey:
    """Canonical dedup/idempotency key: ``tenant_id`` is a reserved slot.

    ``source`` identifies the logical stream/job type (e.g. ``"kafka_trade"``,
    ``"horizon_trade"``, ``"pipeline_checkpoint:<run_id>:<pair_id>"``).
    ``external_id`` identifies the specific unit within that source (e.g. a
    trade id, or a stage name).
    """

    source: str
    external_id: str
    tenant_id: str | None = None

    def canonical(self) -> str:
        tenant = self.tenant_id if self.tenant_id is not None else "_"
        return f"{tenant}:{self.source}:{self.external_id}"


# ---------------------------------------------------------------------------
# States and errors
# ---------------------------------------------------------------------------


class DedupState(StrEnum):
    NEW = "new"
    STAGED = "staged"
    COMMITTED = "committed"
    FAILED = "failed"
    TTL_EXPIRED_REVERIFY = "ttl_expired_reverify"


class DedupError(Exception):
    """Base class for exactly-once store failures."""


class DedupBackendUnavailableError(DedupError):
    """Raised when the backend cannot answer authoritatively.

    Callers MUST treat this as "unknown", not "not a duplicate" — the whole
    point of this exception type is to make fail-open impossible by
    construction: there is no code path that maps a caught instance of this
    error to ``DedupState.NEW``.
    """

    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        super().__init__(f"dedup backend unavailable for source={source!r}: {reason}")


@dataclass
class DedupDecision:
    state: DedupState
    payload: Any = None


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class DedupBackend(Protocol):
    def check_and_stage(self, key: DedupKey, ttl_seconds: float) -> DedupDecision: ...

    def commit(self, key: DedupKey, payload: Any = None) -> None: ...

    def mark_failed(self, key: DedupKey) -> None: ...

    def get(self, key: DedupKey) -> DedupDecision: ...

    def health_check(self) -> bool: ...


# ---------------------------------------------------------------------------
# Redis backend — hot path, fail-closed
# ---------------------------------------------------------------------------


class RedisExactlyOnceBackend:
    """Redis-backed two-phase dedup. Fail-closed: raises on any Redis error.

    Staging uses ``SET key val NX`` (atomic claim). Commit overwrites the
    same key with a ``committed`` marker and a fresh TTL. There is no
    silent fallback mode — if the Redis client cannot be constructed or a
    command fails, every method raises ``DedupBackendUnavailableError``.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "ledgerlens:dedup:",
        client: Any = None,
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._redis = client
        if self._redis is None:
            try:
                import redis as redis_module

                self._redis = redis_module.from_url(
                    redis_url, decode_responses=True, socket_timeout=5
                )
            except Exception as exc:
                self._redis = None
                self._init_error = str(exc)
                return
        self._init_error: str | None = None

    def _require_redis(self, source: str) -> Any:
        if self._redis is None:
            if DEDUP_BACKEND_AVAILABLE is not None:
                DEDUP_BACKEND_AVAILABLE.labels(source=source).set(0)
            raise DedupBackendUnavailableError(
                source, self._init_error or "Redis client not initialised"
            )
        return self._redis

    def _redis_key(self, key: DedupKey) -> str:
        return f"{self._key_prefix}{key.canonical()}"

    def health_check(self) -> bool:
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    def check_and_stage(self, key: DedupKey, ttl_seconds: float) -> DedupDecision:
        """Atomically claim *key* via ``SET NX``.

        ``SET key STAGED NX`` is a single atomic Redis command: when N
        callers race on the same never-seen key, exactly one gets
        ``claimed=True`` (``DedupState.NEW`` — proceed with side effects),
        and every other caller observes the key already present. A prior
        GET-then-SET formulation here would have been a check-then-act race
        allowing every concurrent caller to see "not found" and each stage
        the key themselves — this is exactly the concurrency guarantee
        Issue #670's "50 concurrent duplicate submissions" acceptance
        criterion requires.
        """
        client = self._require_redis(key.source)
        redis_key = self._redis_key(key)
        try:
            claimed = client.set(redis_key, DedupState.STAGED.value, nx=True, ex=int(ttl_seconds))
            if DEDUP_BACKEND_AVAILABLE is not None:
                DEDUP_BACKEND_AVAILABLE.labels(source=key.source).set(1)
            if claimed:
                return DedupDecision(DedupState.NEW)

            existing = client.get(redis_key)
            if existing == DedupState.COMMITTED.value:
                return DedupDecision(DedupState.COMMITTED)
            # STAGED (or an unrecognised legacy value), or the key expired
            # between the failed NX and this GET: either way, safe to redo.
            return DedupDecision(DedupState.STAGED)
        except DedupBackendUnavailableError:
            raise
        except Exception as exc:
            if DEDUP_BACKEND_AVAILABLE is not None:
                DEDUP_BACKEND_AVAILABLE.labels(source=key.source).set(0)
            raise DedupBackendUnavailableError(key.source, str(exc)) from exc

    def commit(self, key: DedupKey, payload: Any = None, ttl_seconds: float = 86400) -> None:
        client = self._require_redis(key.source)
        redis_key = self._redis_key(key)
        try:
            client.set(redis_key, DedupState.COMMITTED.value, ex=int(ttl_seconds))
        except Exception as exc:
            raise DedupBackendUnavailableError(key.source, str(exc)) from exc

    def mark_failed(self, key: DedupKey) -> None:
        client = self._require_redis(key.source)
        try:
            client.delete(self._redis_key(key))
        except Exception as exc:
            raise DedupBackendUnavailableError(key.source, str(exc)) from exc

    def get(self, key: DedupKey) -> DedupDecision:
        client = self._require_redis(key.source)
        try:
            existing = client.get(self._redis_key(key))
        except Exception as exc:
            raise DedupBackendUnavailableError(key.source, str(exc)) from exc
        if existing is None:
            return DedupDecision(DedupState.NEW)
        if existing == DedupState.COMMITTED.value:
            return DedupDecision(DedupState.COMMITTED)
        return DedupDecision(DedupState.STAGED)

    def scan_count(self, pattern_source_prefix: str) -> int:
        """Count keys under a given ``source`` prefix (ops/introspection use).

        Raises ``DedupBackendUnavailableError`` if Redis is unreachable.
        """
        client = self._require_redis(pattern_source_prefix)
        pattern = f"{self._key_prefix}*:{pattern_source_prefix}:*"
        try:
            return sum(1 for _ in client.scan_iter(match=pattern))
        except Exception as exc:
            raise DedupBackendUnavailableError(pattern_source_prefix, str(exc)) from exc

    def scan_delete(self, pattern_source_prefix: str) -> int:
        """Delete all keys under a given ``source`` prefix (ops/introspection use).

        Raises ``DedupBackendUnavailableError`` if Redis is unreachable.
        """
        client = self._require_redis(pattern_source_prefix)
        pattern = f"{self._key_prefix}*:{pattern_source_prefix}:*"
        try:
            deleted = 0
            for k in list(client.scan_iter(match=pattern)):
                client.delete(k)
                deleted += 1
            return deleted
        except Exception as exc:
            raise DedupBackendUnavailableError(pattern_source_prefix, str(exc)) from exc


# ---------------------------------------------------------------------------
# SQL backend — durable, for batch/offline use
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class DedupRecord(_Base):
    __tablename__ = "exactly_once_dedup"
    __table_args__ = (UniqueConstraint("canonical_key", name="uq_exactly_once_canonical_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String, nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(String, nullable=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    staged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SqlExactlyOnceBackend:
    """SQLAlchemy-backed durable two-phase dedup for batch/offline pipelines.

    The row's ``UNIQUE(canonical_key)`` constraint is the actual concurrency
    guard: staging is an insert-or-detect-conflict, not a check-then-act race.
    """

    def __init__(self, db_url: str, *, ttl_hours: float = 48.0) -> None:
        self._db_url = db_url
        self._ttl_hours = ttl_hours
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        )
        if db_url.startswith("sqlite") and ":memory:" not in db_url:
            # WAL mode allows concurrent readers alongside a writer, which
            # matters here: check_and_stage() is called concurrently by
            # multiple worker threads/processes racing on the same key.
            # (In-memory sqlite URLs use a SingletonThreadPool and don't
            # benefit from WAL — genuine concurrent-connection testing
            # against this backend should use a file-backed database.)
            @event.listens_for(engine, "connect")
            def _wal(dbapi_conn, _rec):
                dbapi_conn.execute("PRAGMA journal_mode=WAL")
                dbapi_conn.execute("PRAGMA busy_timeout=30000")

        _Base.metadata.create_all(engine, checkfirst=True)
        self._session_factory: sessionmaker[Session] = sessionmaker(bind=engine, future=True)

    def health_check(self) -> bool:
        try:
            with self._session_factory() as session:
                session.execute(select(DedupRecord.id).limit(1))
            return True
        except Exception:
            return False

    def _get_row(self, session: Session, key: DedupKey) -> DedupRecord | None:
        return session.scalar(
            select(DedupRecord).where(DedupRecord.canonical_key == key.canonical())
        )

    def _with_retry(self, fn: Any) -> Any:
        """Retry *fn* (a zero-arg callable) on a transient ``OperationalError``.

        SQLite can raise ``OperationalError`` for genuinely transient
        conditions — lock contention (``database is locked``) or a momentary
        ``disk I/O error`` from the underlying filesystem — that are not
        application bugs and resolve on retry. Mirrors the existing
        exponential-backoff pattern in
        ``detection.risk_score_store.RiskScoreStore._upsert_impl``.
        """
        for attempt in range(5):
            try:
                return fn()
            except OperationalError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (2**attempt))

    def check_and_stage(self, key: DedupKey, ttl_seconds: float) -> DedupDecision:
        return self._with_retry(lambda: self._check_and_stage_impl(key, ttl_seconds))

    def _check_and_stage_impl(self, key: DedupKey, ttl_seconds: float) -> DedupDecision:
        with self._session_factory() as session:
            row = self._get_row(session, key)
            if row is None:
                row = DedupRecord(
                    canonical_key=key.canonical(),
                    source=key.source,
                    external_id=key.external_id,
                    tenant_id=key.tenant_id,
                    state=DedupState.STAGED.value,
                    staged_at=datetime.now(UTC),
                )
                session.add(row)
                try:
                    session.commit()
                except IntegrityError:
                    # A concurrent caller won the race and inserted first —
                    # the UNIQUE(canonical_key) constraint is the actual
                    # concurrency guard, not the SELECT above. Roll back and
                    # re-read what the winner committed: exactly one caller
                    # observes NEW, every other concurrent caller observes
                    # STAGED (or COMMITTED, if the winner finished first).
                    session.rollback()
                    row = self._get_row(session, key)
                    assert row is not None
                    if row.state == DedupState.COMMITTED.value:
                        return DedupDecision(DedupState.COMMITTED, row.payload_json)
                    return DedupDecision(DedupState.STAGED)
                return DedupDecision(DedupState.NEW)

            if row.state == DedupState.COMMITTED.value:
                age_hours = (datetime.now(UTC) - _as_aware(row.committed_at)).total_seconds() / 3600
                if age_hours > self._ttl_hours:
                    logger.info(
                        "Dedup record for key=%s is stale (age=%.1fh > ttl=%.1fh); "
                        "surfacing TTL_EXPIRED_REVERIFY, not silently re-running",
                        key.canonical(),
                        age_hours,
                        self._ttl_hours,
                    )
                    return DedupDecision(DedupState.TTL_EXPIRED_REVERIFY, row.payload_json)
                return DedupDecision(DedupState.COMMITTED, row.payload_json)

            if row.state == DedupState.FAILED.value:
                row.state = DedupState.STAGED.value
                row.staged_at = datetime.now(UTC)
                session.commit()
                return DedupDecision(DedupState.NEW)

            # STAGED — a prior attempt started but never committed/failed.
            return DedupDecision(DedupState.STAGED)

    def commit(self, key: DedupKey, payload: Any = None) -> None:
        self._with_retry(lambda: self._commit_impl(key, payload))

    def _commit_impl(self, key: DedupKey, payload: Any = None) -> None:
        with self._session_factory() as session:
            row = self._get_row(session, key)
            if row is None:
                row = DedupRecord(
                    canonical_key=key.canonical(),
                    source=key.source,
                    external_id=key.external_id,
                    tenant_id=key.tenant_id,
                    state=DedupState.STAGED.value,
                    staged_at=datetime.now(UTC),
                )
                session.add(row)
            row.state = DedupState.COMMITTED.value
            row.committed_at = datetime.now(UTC)
            if payload is not None:
                row.payload_json = payload
            try:
                session.commit()
            except IntegrityError:
                # Another caller staged the row between our SELECT and INSERT
                # (only possible if commit() is called without a prior
                # check_and_stage() on this key) — retry as an UPDATE.
                session.rollback()
                row = self._get_row(session, key)
                assert row is not None
                row.state = DedupState.COMMITTED.value
                row.committed_at = datetime.now(UTC)
                if payload is not None:
                    row.payload_json = payload
                session.commit()

    def mark_failed(self, key: DedupKey) -> None:
        self._with_retry(lambda: self._mark_failed_impl(key))

    def _mark_failed_impl(self, key: DedupKey) -> None:
        with self._session_factory() as session:
            row = self._get_row(session, key)
            if row is not None:
                row.state = DedupState.FAILED.value
                session.commit()

    def get(self, key: DedupKey) -> DedupDecision:
        return self._with_retry(lambda: self._get_impl(key))

    def _get_impl(self, key: DedupKey) -> DedupDecision:
        with self._session_factory() as session:
            row = self._get_row(session, key)
            if row is None:
                return DedupDecision(DedupState.NEW)
            return DedupDecision(DedupState(row.state), row.payload_json)

    def list_by_source_prefix(self, source_prefix: str) -> list[DedupRecord]:
        """Return all rows whose ``source`` starts with *source_prefix*.

        Used by higher-level facades (e.g. a checkpoint store keying on
        ``source=f"pipeline_checkpoint:{run_id}:{pair_id}"``) to answer
        "list everything for this run" without a second index table.
        """
        return self._with_retry(lambda: self._list_by_source_prefix_impl(source_prefix))

    def _list_by_source_prefix_impl(self, source_prefix: str) -> list[DedupRecord]:
        with self._session_factory() as session:
            rows = list(
                session.scalars(
                    select(DedupRecord)
                    .where(DedupRecord.source.like(f"{source_prefix}%"))
                    .order_by(DedupRecord.id)
                )
            )
            session.expunge_all()
            return rows


def _as_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class ExactlyOnceStore:
    """Facade combining a `DedupBackend` with the staged/committed protocol.

    Usage::

        store = ExactlyOnceStore(RedisExactlyOnceBackend(config.REDIS_URL))
        decision = store.check_and_stage(key)
        if decision.state is DedupState.COMMITTED:
            return  # duplicate — already fully processed
        # decision.state is NEW or STAGED (redo) or TTL_EXPIRED_REVERIFY
        do_side_effecting_work()
        store.commit(key)
    """

    def __init__(self, backend: DedupBackend, *, ttl_seconds: float = 86400.0) -> None:
        self._backend = backend
        self._ttl_seconds = ttl_seconds

    @property
    def ttl_seconds(self) -> float:
        """How long a staged key suppresses redelivery before expiring."""
        return self._ttl_seconds

    def check_and_stage(self, key: DedupKey) -> DedupDecision:
        return self._backend.check_and_stage(key, self._ttl_seconds)

    def commit(self, key: DedupKey, payload: Any = None) -> None:
        self._backend.commit(key, payload)

    def mark_failed(self, key: DedupKey) -> None:
        self._backend.mark_failed(key)

    def get(self, key: DedupKey) -> DedupDecision:
        return self._backend.get(key)

    def is_available(self) -> bool:
        return self._backend.health_check()


def now_ms() -> int:
    return int(time.time() * 1000)
