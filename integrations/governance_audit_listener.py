"""Off-chain audit log correlation for on-chain governance events (#950).

Overview
--------
Every parameter change executed by the LedgerLens governance contract
(``integrations/governance_contract.rs``) must be accompanied by an
off-chain justification: a written explanation of *why* the change was
made, who authorised it, and which supporting analysis it was based on.

:class:`GovernanceAuditListener` enforces this requirement by:

1. Recording every ``t_changed`` (threshold-changed) governance event as
   soon as it is observed on-chain (via ``on_governance_event``).
2. Starting a **grace-period timer** (default 300 s / 5 min) during which
   the change author is expected to submit an off-chain justification via
   ``submit_justification``.
3. Calling ``alert_fn`` for any event whose grace period has expired without
   a linked justification (checked during each ``run_once()`` tick or
   ``check_missing_justifications()`` call).

Usage
-----
::

    from integrations.governance_audit_listener import GovernanceAuditListener
    from detection.audit_trail import AuditTrail

    store = AuditTrail()
    listener = GovernanceAuditListener(
        audit_store=store,
        alert_fn=my_alert_fn,
    )

    # When the event listener fires a governance event:
    listener.on_governance_event({
        "event_id": "gov-001",
        "event_type": "parameter_change",
        "event_data": {"parameter": "RISK_SCORE_FLAG_THRESHOLD", "new_value": 75},
        "timestamp": 1727544055.0,
    })

    # The change author links their justification:
    listener.submit_justification(
        event_id="gov-001",
        justification="Increased threshold to 75 based on Q3 2026 precision/recall analysis (see reports/q3_2026_threshold_review.pdf).",
        submitter="alice@example.com",
    )

    # Periodic reconciliation (call from a scheduler or background thread):
    listener.run_once()

See ``docs/governance_audit_process.md`` for the operational workflow.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from detection.governance_audit_trail import AuditTrail
from utils.logging import get_logger

logger = get_logger(__name__)


class GovernanceAuditListener:
    """Links on-chain governance events to off-chain justifications and alerts
    when a justification is missing after the grace period expires.

    Parameters
    ----------
    audit_store:
        An :class:`~detection.governance_audit_trail.AuditTrail` instance (or
        any object implementing the same interface) used to persist events and
        justifications.
    alert_fn:
        Callable with signature ``alert_fn(alert_details: dict) -> None``.
        Invoked for every event that exceeds the grace period without a
        justification.  The ``alert_details`` dict contains at minimum:
        ``event_id``, ``event_type``, ``event_data``, ``timestamp``,
        ``grace_period_seconds``, ``missing_for_seconds``, ``severity``.
    grace_period_seconds:
        How long (in seconds) after an on-chain governance event the change
        author has to submit a justification before an alert fires.
        Default 300 s (5 minutes).
    """

    def __init__(
        self,
        audit_store: AuditTrail,
        alert_fn: Callable[[dict], None],
        grace_period_seconds: int = 300,
    ) -> None:
        self.audit_store = audit_store
        self.alert_fn = alert_fn
        self.grace_period_seconds = grace_period_seconds

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def on_governance_event(self, event: dict) -> None:
        """Record an on-chain governance event and start the grace-period timer.

        The event dict should contain at minimum:

        - ``event_id`` (str): a unique identifier for this event (e.g. the
          Soroban paging token or a deterministic hash).
        - ``event_type`` (str): e.g. ``"parameter_change"``, ``"threshold_changed"``.
        - ``event_data`` (dict): the event payload (parameter name, values, etc.).
        - ``timestamp`` (float | str, optional): Unix timestamp or ISO string
          of when the event was observed.  Defaults to ``time.time()``.

        Parameters
        ----------
        event:
            Raw governance event dict as delivered by the event listener.
        """
        event_id: str = event.get("event_id", "")
        if not event_id:
            logger.warning("GovernanceAuditListener: received event without event_id, skipping")
            return

        event_type: str = event.get("event_type", "unknown")
        event_data: dict = event.get("event_data", {})
        timestamp_raw = event.get("timestamp")
        if isinstance(timestamp_raw, str):
            # Accept ISO strings: convert to float unix timestamp
            from datetime import UTC, datetime

            try:
                dt = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
            except ValueError:
                timestamp = time.time()
        elif isinstance(timestamp_raw, (int, float)):
            timestamp = float(timestamp_raw)
        else:
            timestamp = time.time()

        self.audit_store.record_governance_event(event_id, event_type, event_data, timestamp)
        logger.info(
            "GovernanceAuditListener: recorded governance event %s (type=%s) — "
            "justification required within %ds",
            event_id,
            event_type,
            self.grace_period_seconds,
        )

    # ------------------------------------------------------------------
    # Justification submission
    # ------------------------------------------------------------------

    def submit_justification(
        self,
        event_id: str,
        justification: str,
        submitter: str,
    ) -> None:
        """Link an off-chain justification to a governance event.

        Parameters
        ----------
        event_id:
            The event to correlate (must have been recorded via
            ``on_governance_event`` first).
        justification:
            The written justification text (or a reference to an external
            document, e.g. a report URL or Confluence link).
        submitter:
            Identity of the person submitting the justification
            (e.g. email, username, or Stellar G-address).
        """
        self.audit_store.add_justification(event_id, justification, submitter)
        logger.info(
            "GovernanceAuditListener: justification submitted for %s by %s",
            event_id,
            submitter,
        )

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def check_missing_justifications(self) -> list[dict]:
        """Find all governance events past the grace period without a justification
        and fire ``alert_fn`` for each one.

        Returns
        -------
        list[dict]
            The alert detail dicts that were dispatched (one per un-justified
            event past the grace period).
        """
        overdue = self.audit_store.get_uncorrelated_events(
            older_than_seconds=self.grace_period_seconds
        )
        alerts_fired: list[dict] = []
        now = time.time()

        for entry in overdue:
            missing_for = now - entry.get("timestamp", now)
            alert_details: dict = {
                "severity": "HIGH",
                "event_id": entry.get("event_id"),
                "event_type": entry.get("event_type"),
                "event_data": entry.get("event_data"),
                "timestamp": entry.get("timestamp"),
                "grace_period_seconds": self.grace_period_seconds,
                "missing_for_seconds": missing_for,
                "message": (
                    f"Governance event {entry.get('event_id')!r} "
                    f"(type={entry.get('event_type')!r}) has no off-chain "
                    f"justification after {missing_for:.0f}s "
                    f"(grace period: {self.grace_period_seconds}s). "
                    "See docs/governance_audit_process.md."
                ),
            }
            logger.warning(
                "GovernanceAuditListener: missing justification for event %s "
                "(overdue by %.0fs)",
                entry.get("event_id"),
                missing_for - self.grace_period_seconds,
            )
            self.alert_fn(alert_details)
            alerts_fired.append(alert_details)

        return alerts_fired

    def run_once(self) -> list[dict]:
        """Perform a single reconciliation tick.

        Equivalent to calling ``check_missing_justifications()``.

        Returns
        -------
        list[dict]
            Alert details fired during this tick.
        """
        return self.check_missing_justifications()
