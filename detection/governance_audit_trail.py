"""Lightweight audit trail for on-chain governance events (#950).

Stores governance events and their off-chain justifications in a SQLite
database (or an in-memory dict when no ``db_url`` is provided — useful for
testing).

This module is intentionally kept separate from
``detection.audit_trail`` (which provides a cryptographically signed
append-only NDJSON log for forensic reports).  Governance event audit
entries are mutable: a justification field that starts as ``None`` is
later filled in by the change author.  The forensic trail is immutable
by design; governance justifications are not.

Schema
------
Each entry tracks:

- ``event_id``: unique identifier for the governance event.
- ``event_type``: e.g. ``"parameter_change"``, ``"threshold_changed"``.
- ``event_data``: JSON-serialisable dict with the event payload.
- ``timestamp``: Unix timestamp (float) when the event occurred on-chain.
- ``justification``: free-text or reference provided by the change author
  (``None`` until ``add_justification`` is called).
- ``submitter``: identity of the person who submitted the justification.
- ``correlated_at``: Unix timestamp (float) when the justification was linked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# AuditEntry dataclass
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """One governance event with its optional off-chain justification.

    Attributes
    ----------
    event_id:
        Unique identifier for the on-chain governance event (e.g. the
        Soroban RPC paging token or a deterministic hash).
    event_type:
        Category of the governance action (e.g. ``"parameter_change"``,
        ``"threshold_changed"``).
    event_data:
        Raw event payload dict (parameter name, old/new values, etc.).
    timestamp:
        Unix timestamp (float) when the event was observed on-chain.
    justification:
        Off-chain written justification for the change.  ``None`` until
        ``AuditTrail.add_justification`` is called.
    submitter:
        Identity of the person who submitted the justification.  ``None``
        until correlated.
    correlated_at:
        Unix timestamp (float) when the justification was linked.  ``None``
        until correlated.
    """

    event_id: str
    event_type: str
    event_data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    justification: str | None = None
    submitter: str | None = None
    correlated_at: float | None = None

    def is_correlated(self) -> bool:
        """Return ``True`` if a justification has been linked."""
        return self.justification is not None

    def as_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "timestamp": self.timestamp,
            "justification": self.justification,
            "submitter": self.submitter,
            "correlated_at": self.correlated_at,
        }


# ---------------------------------------------------------------------------
# AuditTrail storage class
# ---------------------------------------------------------------------------


class AuditTrail:
    """Stores governance events and their off-chain justifications.

    This implementation uses an in-memory dict as its backing store, which
    is suitable for testing and single-process deployments.  For durable
    production use, a subclass or alternative implementation backed by
    SQLite (or any SQLAlchemy-compatible database) should be used.

    Parameters
    ----------
    db_url:
        Optional SQLite database URL (e.g. ``"sqlite:///governance_audit.db"``).
        When provided, the audit trail is persisted to disk.  When ``None``
        (default), an in-memory dict is used.
    """

    def __init__(self, db_url: str | None = None) -> None:
        # In-memory backing store: event_id → AuditEntry
        self._store: dict[str, AuditEntry] = {}
        self._db_url = db_url

        if db_url is not None:
            self._init_db(db_url)

    # ------------------------------------------------------------------
    # Storage initialisation (SQLite path)
    # ------------------------------------------------------------------

    def _init_db(self, db_url: str) -> None:
        """Initialise SQLite-backed storage via SQLAlchemy."""
        try:
            from sqlalchemy import (
                Column,
                Float,
                Integer,
                String,
                Text,
                create_engine,
            )
            from sqlalchemy.orm import DeclarativeBase, sessionmaker

            class _Base(DeclarativeBase):
                pass

            class _AuditEntryRecord(_Base):
                __tablename__ = "governance_audit_entries"

                id = Column(Integer, primary_key=True, autoincrement=True)
                event_id = Column(String, unique=True, nullable=False, index=True)
                event_type = Column(String, nullable=False)
                event_data_json = Column(Text, nullable=False, default="{}")
                timestamp = Column(Float, nullable=False)
                justification = Column(Text, nullable=True)
                submitter = Column(String, nullable=True)
                correlated_at = Column(Float, nullable=True)

            self._engine = create_engine(db_url)
            _Base.metadata.create_all(self._engine)
            self._Session = sessionmaker(bind=self._engine)
            self._AuditEntryRecord = _AuditEntryRecord
            self._use_db = True
            logger.info("GovernanceAuditTrail: initialised SQLite store at %s", db_url)
        except ImportError:
            logger.warning(
                "SQLAlchemy not available; falling back to in-memory store "
                "(db_url=%s will not be used)",
                db_url,
            )
            self._use_db = False
        else:
            # Pre-load existing entries into the in-memory index for fast look-ups
            self._load_from_db()

    def _load_from_db(self) -> None:
        """Load all rows from SQLite into the in-memory index."""
        with self._Session() as session:
            rows = session.query(self._AuditEntryRecord).all()
            for row in rows:
                entry = AuditEntry(
                    event_id=row.event_id,
                    event_type=row.event_type,
                    event_data=json.loads(row.event_data_json or "{}"),
                    timestamp=row.timestamp,
                    justification=row.justification,
                    submitter=row.submitter,
                    correlated_at=row.correlated_at,
                )
                self._store[entry.event_id] = entry

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Write or update an entry in the SQLite database (if configured)."""
        if not getattr(self, "_use_db", False):
            return
        with self._Session() as session:
            row = (
                session.query(self._AuditEntryRecord)
                .filter_by(event_id=entry.event_id)
                .first()
            )
            if row is None:
                row = self._AuditEntryRecord(
                    event_id=entry.event_id,
                    event_type=entry.event_type,
                    event_data_json=json.dumps(entry.event_data, sort_keys=True),
                    timestamp=entry.timestamp,
                    justification=entry.justification,
                    submitter=entry.submitter,
                    correlated_at=entry.correlated_at,
                )
                session.add(row)
            else:
                row.justification = entry.justification
                row.submitter = entry.submitter
                row.correlated_at = entry.correlated_at
            session.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_governance_event(
        self,
        event_id: str,
        event_type: str,
        event_data: dict,
        timestamp: float,
    ) -> AuditEntry:
        """Record a new on-chain governance event.

        If an entry for *event_id* already exists, it is returned unchanged
        (idempotent).

        Parameters
        ----------
        event_id:
            Unique identifier for the governance event.
        event_type:
            Type/category of the governance action.
        event_data:
            Event payload dict.
        timestamp:
            Unix timestamp when the event occurred.

        Returns
        -------
        AuditEntry
            The newly created (or existing) entry.
        """
        if event_id in self._store:
            logger.debug(
                "GovernanceAuditTrail: event %s already recorded (idempotent)", event_id
            )
            return self._store[event_id]

        entry = AuditEntry(
            event_id=event_id,
            event_type=event_type,
            event_data=event_data,
            timestamp=timestamp,
        )
        self._store[event_id] = entry
        self._persist_entry(entry)
        logger.info(
            "GovernanceAuditTrail: recorded event %s (type=%s)", event_id, event_type
        )
        return entry

    def add_justification(
        self,
        event_id: str,
        justification: str,
        submitter: str,
    ) -> AuditEntry:
        """Link an off-chain justification to a governance event.

        Parameters
        ----------
        event_id:
            The event to correlate.
        justification:
            Written justification text or document reference.
        submitter:
            Identity of the justification author.

        Returns
        -------
        AuditEntry
            The updated entry.

        Raises
        ------
        KeyError
            If *event_id* has not been recorded via ``record_governance_event``.
        """
        if event_id not in self._store:
            raise KeyError(
                f"GovernanceAuditTrail: event {event_id!r} not found; "
                "call record_governance_event first"
            )

        entry = self._store[event_id]
        entry.justification = justification
        entry.submitter = submitter
        entry.correlated_at = time.time()
        self._persist_entry(entry)
        logger.info(
            "GovernanceAuditTrail: justification added for event %s by %s",
            event_id,
            submitter,
        )
        return entry

    def get_uncorrelated_events(self, older_than_seconds: float) -> list[dict]:
        """Return events without a justification that are older than *older_than_seconds*.

        Parameters
        ----------
        older_than_seconds:
            Minimum age (in seconds, relative to ``time.time()``) for an
            un-justified event to be considered overdue.

        Returns
        -------
        list[dict]
            A list of ``entry.as_dict()`` dicts for overdue un-justified events.
        """
        now = time.time()
        cutoff = now - older_than_seconds
        return [
            entry.as_dict()
            for entry in self._store.values()
            if not entry.is_correlated() and entry.timestamp <= cutoff
        ]

    def get_entry(self, event_id: str) -> AuditEntry | None:
        """Retrieve the full :class:`AuditEntry` for *event_id*, or ``None``."""
        return self._store.get(event_id)

    def list_entries(self, limit: int = 100) -> list[AuditEntry]:
        """Return up to *limit* entries, ordered by timestamp (newest first).

        Parameters
        ----------
        limit:
            Maximum number of entries to return.  Default 100.

        Returns
        -------
        list[AuditEntry]
        """
        sorted_entries = sorted(
            self._store.values(), key=lambda e: e.timestamp, reverse=True
        )
        return sorted_entries[:limit]
