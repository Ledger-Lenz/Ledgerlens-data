"""Durable, queryable record of every alert-dispatch attempt's terminal outcome.

Issue #670 (Grand 1), required scope E: ``validation/reconciliation.py``
previously had no way to trace a risk score at or above the alert threshold
through to a delivered (or dead-lettered) alert — a silently dropped alert
and a correctly-suppressed one (still within its per-wallet cooldown window)
were indistinguishable from the outside. ``AlertDeliveryLedger`` closes that
gap: :class:`~streaming.alert_dispatcher.AlertDispatcher` records one entry
per dispatch attempt that clears the threshold, tagged with its terminal
outcome — ``delivered``, ``dead_lettered``, or ``suppressed_cooldown`` — so
:func:`validation.reconciliation.reconcile_alert_delivery` can assert every
qualifying score is accounted for.

Keyed through the same canonical scheme as the rest of the pipeline's
exactly-once/idempotency stores (``pipeline.exactly_once.DedupKey``,
``source="alert_delivery"``) — this is not a dedup guard (repeated dispatch
attempts for the same wallet legitimately happen over time), but reusing the
same durable, race-safe commit primitive avoids inventing a second storage
pattern for what is structurally the same problem: "record that terminal
outcome X happened for identifier Y, exactly once, and let it be queried
later."
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from config import config
from pipeline.exactly_once import DedupKey, ExactlyOnceStore, SqlExactlyOnceBackend

AlertOutcome = Literal[
    "in_flight", "delivered", "dead_lettered", "suppressed_cooldown", "reconciled_skipped"
]

# Outcomes that close out an ``in_flight`` intent for a given idempotency key.
TERMINAL_OUTCOMES = frozenset({"delivered", "dead_lettered", "reconciled_skipped"})


@dataclass(frozen=True)
class AlertDeliveryRecord:
    wallet: str
    pair_id: str
    score: int | float | None
    outcome: AlertOutcome
    channel: str
    reason: str | None
    idempotency_key: str | None = None
    risk_score: dict | None = None


class AlertDeliveryLedger:
    """Durable SQL-backed ledger of alert-dispatch terminal outcomes."""

    def __init__(self, db_url: str | None = None) -> None:
        backend = SqlExactlyOnceBackend(db_url or config.RISK_SCORE_DB_URL)
        self._backend = backend
        self._store = ExactlyOnceStore(backend)

    @staticmethod
    def _key(wallet: str, pair_id: str, risk_score: dict) -> DedupKey:
        # A dispatch attempt is uniquely identified by (wallet, pair, the
        # exact score+timestamp that triggered it) — a wallet can legitimately
        # be dispatched-to many times over its lifetime, so external_id must
        # vary per attempt, not just per wallet/pair.
        external_id = f"{wallet}:{pair_id}:{risk_score.get('score')}:{time.time_ns()}"
        return DedupKey(source="alert_delivery", external_id=external_id)

    def record(
        self,
        wallet: str,
        pair_id: str,
        risk_score: dict,
        outcome: AlertOutcome,
        *,
        channel: str,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        key = self._key(wallet, pair_id, risk_score)
        self._store.commit(
            key,
            payload={
                "wallet": wallet,
                "pair_id": pair_id,
                "score": risk_score.get("score"),
                "outcome": outcome,
                "channel": channel,
                "reason": reason,
                "idempotency_key": idempotency_key,
                "risk_score": risk_score,
            },
        )

    def is_delivered(self, idempotency_key: str) -> bool:
        """True if a terminal outcome is already recorded for *idempotency_key*."""
        return any(
            r.idempotency_key == idempotency_key and r.outcome in TERMINAL_OUTCOMES
            for r in self.all_records()
        )

    def in_flight(self) -> list[AlertDeliveryRecord]:
        """Return ``in_flight`` intents with no matching terminal outcome.

        These are alerts whose external call may or may not have happened
        because the dispatcher died between recording intent and outcome.
        """
        records = self.all_records()
        closed = {
            r.idempotency_key
            for r in records
            if r.idempotency_key and r.outcome in TERMINAL_OUTCOMES
        }
        pending: dict[str, AlertDeliveryRecord] = {}
        for r in records:
            if r.outcome == "in_flight" and r.idempotency_key and r.idempotency_key not in closed:
                pending[r.idempotency_key] = r
        return list(pending.values())

    def for_wallet_pair(self, wallet: str, pair_id: str) -> list[AlertDeliveryRecord]:
        """Return every recorded dispatch outcome for a (wallet, pair_id)."""
        rows = self._backend.list_by_source_prefix("alert_delivery")
        out: list[AlertDeliveryRecord] = []
        for row in rows:
            payload: dict[str, Any] = row.payload_json or {}
            if payload.get("wallet") == wallet and payload.get("pair_id") == pair_id:
                out.append(
                    AlertDeliveryRecord(
                        wallet=wallet,
                        pair_id=pair_id,
                        score=payload.get("score"),
                        outcome=payload.get("outcome"),
                        channel=payload.get("channel", ""),
                        reason=payload.get("reason"),
                    )
                )
        return out

    def all_records(self) -> list[AlertDeliveryRecord]:
        """Return every recorded dispatch outcome (for batch reconciliation)."""
        rows = self._backend.list_by_source_prefix("alert_delivery")
        out: list[AlertDeliveryRecord] = []
        for row in rows:
            payload: dict[str, Any] = row.payload_json or {}
            out.append(
                AlertDeliveryRecord(
                    wallet=payload.get("wallet", ""),
                    pair_id=payload.get("pair_id", ""),
                    score=payload.get("score"),
                    outcome=payload.get("outcome"),
                    channel=payload.get("channel", ""),
                    reason=payload.get("reason"),
                    idempotency_key=payload.get("idempotency_key"),
                    risk_score=payload.get("risk_score"),
                )
            )
        return out
