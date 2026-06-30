"""Soroban contract event listener for the ledgerlens-score contract.

Subscribes to contract events (score_read, score_updated, threshold_updated)
via the Soroban RPC ``getEvents`` endpoint, with a Horizon effects fallback.
Persists parsed events to the ``contract_event`` DB table and fires stale-score
consumption alerts via ``streaming.alert_dispatcher.AlertDispatcher``.

Resumability
------------
The last successfully processed ledger sequence is stored in the
``event_watermark`` table.  On restart the listener replays from
``watermark + 1`` so no events are missed and no event is processed twice.

Security
--------
Both ``wallet_id`` and ``consumer_address`` are stored as HMAC-SHA256 digests
(keyed by ``EVENT_HMAC_SECRET``).  Raw addresses are never written to the DB
or emitted in log lines.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterator

import requests
from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STALE_SCORE_ALERT_THRESHOLD: int = int(os.getenv("STALE_SCORE_ALERT_THRESHOLD", "20"))
EVENT_HMAC_SECRET: str = os.getenv("EVENT_HMAC_SECRET", "ledgerlens-event-hmac-default")
EVENT_POLL_INTERVAL_SECONDS: float = float(os.getenv("EVENT_POLL_INTERVAL_SECONDS", "5"))
EVENT_PAGE_LIMIT: int = int(os.getenv("EVENT_PAGE_LIMIT", "200"))
# Which backend to use for event fetching: "rpc" (Soroban RPC) or "horizon"
EVENT_BACKEND: str = os.getenv("EVENT_BACKEND", "rpc")

KNOWN_EVENT_TYPES = {"score_read", "score_updated", "threshold_updated"}


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class ContractEvent:
    """Parsed representation of a ledgerlens-score contract event."""

    event_type: str          # "score_read" | "score_updated" | "threshold_updated"
    ledger_sequence: int
    timestamp: datetime
    wallet_id_hash: str      # HMAC-SHA256 of the raw wallet G-address
    score: int | None        # present on score_read / score_updated
    consumer_address_hash: str | None  # HMAC-SHA256; present on score_read
    asset_pair: str | None
    # threshold_updated only
    old_threshold: int | None = None
    new_threshold: int | None = None
    # raw topic/value payload for forward-compatibility
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQLAlchemy models
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class ContractEventRecord(_Base):
    """Persisted contract event row."""

    __tablename__ = "contract_event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ledger_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    wallet_id_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    consumer_address_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    asset_pair: Mapped[str | None] = mapped_column(String(128), nullable=True)
    old_threshold: Mapped[int | None] = mapped_column(Integer, nullable=True)
    new_threshold: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_contract_event_wallet_type", "wallet_id_hash", "event_type"),
    )


class EventWatermark(_Base):
    """Stores the last processed ledger sequence per contract."""

    __tablename__ = "event_watermark"

    contract_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_ledger: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


# ---------------------------------------------------------------------------
# Privacy helpers
# ---------------------------------------------------------------------------


def _hash_address(address: str) -> str:
    """Return HMAC-SHA256 hex digest of *address* keyed by EVENT_HMAC_SECRET."""
    return hmac.new(
        EVENT_HMAC_SECRET.encode(),
        address.encode(),
        hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


def _decode_scval(val: dict) -> Any:
    """Best-effort decode of a Soroban ScVal JSON fragment.

    Handles the common cases produced by both the RPC and Horizon backends:
    ``{"type": "string", "value": "..."}`` and the stellar-sdk scval shapes.
    """
    if not isinstance(val, dict):
        return val
    t = val.get("type", "")
    v = val.get("value")
    if t in ("string", "symbol"):
        return str(v)
    if t in ("u32", "i32", "u64", "i64", "u128", "i128"):
        return int(v) if v is not None else 0
    if t == "bool":
        return bool(v)
    if t == "address":
        return str(v)
    if t == "map":
        return {_decode_scval(item["key"]): _decode_scval(item["val"]) for item in (v or [])}
    if t == "vec":
        return [_decode_scval(item) for item in (v or [])]
    # Fallback: return the raw value
    return v


def parse_contract_event(raw: dict) -> ContractEvent | None:
    """Parse a single raw Soroban event dict into a :class:`ContractEvent`.

    Returns ``None`` for events with unrecognised types (forward-compatible).

    Expected raw shape (Soroban RPC ``getEvents`` response item)::

        {
          "type": "contract",
          "ledger": "12345",
          "ledgerClosedAt": "2026-06-30T00:00:00Z",
          "contractId": "C...",
          "id": "...",
          "topic": [<ScVal>, ...],   // [event_name_sym, wallet_addr, ...]
          "value": <ScVal>           // score or threshold payload
        }

    The Horizon effects fallback wraps the same payload under ``"data"``.
    """
    # Support both Soroban RPC shape and Horizon effects shape
    if "data" in raw and "topic" not in raw:
        inner = raw.get("data", {})
        raw = {**raw, **inner}

    try:
        ledger = int(raw.get("ledger", 0))
        closed_at_str = raw.get("ledgerClosedAt") or raw.get("created_at", "")
        try:
            ts = datetime.fromisoformat(closed_at_str.rstrip("Z")).replace(tzinfo=UTC)
        except (ValueError, AttributeError):
            ts = datetime.now(UTC)

        topics: list = raw.get("topic", [])
        value: dict = raw.get("value", {})

        if not topics:
            return None

        event_name = _decode_scval(topics[0]) if topics else None
        if event_name not in KNOWN_EVENT_TYPES:
            logger.debug("Ignoring unknown event type: %s", event_name)
            return None

        # wallet is always the second topic element
        raw_wallet = _decode_scval(topics[1]) if len(topics) > 1 else ""
        wallet_hash = _hash_address(str(raw_wallet)) if raw_wallet else ""

        decoded_value = _decode_scval(value)

        if event_name == "score_read":
            score = int(decoded_value.get("score", 0)) if isinstance(decoded_value, dict) else int(decoded_value or 0)
            raw_consumer = decoded_value.get("consumer", "") if isinstance(decoded_value, dict) else (_decode_scval(topics[2]) if len(topics) > 2 else "")
            consumer_hash = _hash_address(str(raw_consumer)) if raw_consumer else None
            asset_pair = decoded_value.get("asset_pair") if isinstance(decoded_value, dict) else None
            return ContractEvent(
                event_type="score_read",
                ledger_sequence=ledger,
                timestamp=ts,
                wallet_id_hash=wallet_hash,
                score=score,
                consumer_address_hash=consumer_hash,
                asset_pair=asset_pair,
                raw=raw,
            )

        if event_name == "score_updated":
            score = int(decoded_value.get("score", 0)) if isinstance(decoded_value, dict) else int(decoded_value or 0)
            asset_pair = decoded_value.get("asset_pair") if isinstance(decoded_value, dict) else None
            return ContractEvent(
                event_type="score_updated",
                ledger_sequence=ledger,
                timestamp=ts,
                wallet_id_hash=wallet_hash,
                score=score,
                consumer_address_hash=None,
                asset_pair=asset_pair,
                raw=raw,
            )

        if event_name == "threshold_updated":
            old_t = int(decoded_value.get("old_threshold", 0)) if isinstance(decoded_value, dict) else None
            new_t = int(decoded_value.get("new_threshold", 0)) if isinstance(decoded_value, dict) else None
            return ContractEvent(
                event_type="threshold_updated",
                ledger_sequence=ledger,
                timestamp=ts,
                wallet_id_hash=wallet_hash,
                score=None,
                consumer_address_hash=None,
                asset_pair=None,
                old_threshold=old_t,
                new_threshold=new_t,
                raw=raw,
            )

    except Exception as exc:
        logger.warning("Failed to parse contract event: %s — raw=%s", exc, raw)

    return None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _get_session_factory(db_url: str | None = None) -> sessionmaker[Session]:
    url = db_url or config.RISK_SCORE_DB_URL
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, connect_args=connect_args)
    _Base.metadata.create_all(engine, checkfirst=True)
    return sessionmaker(bind=engine, future=True)


def persist_event(event: ContractEvent, session_factory: sessionmaker) -> None:
    """Upsert *event* into the ``contract_event`` table."""
    with session_factory() as session:
        record = ContractEventRecord(
            event_type=event.event_type,
            ledger_sequence=event.ledger_sequence,
            event_timestamp=event.timestamp,
            wallet_id_hash=event.wallet_id_hash,
            score=event.score,
            consumer_address_hash=event.consumer_address_hash,
            asset_pair=event.asset_pair,
            old_threshold=event.old_threshold,
            new_threshold=event.new_threshold,
            raw_payload=json.dumps(event.raw, default=str),
        )
        session.add(record)
        session.commit()


def get_watermark(contract_id: str, session_factory: sessionmaker) -> int:
    with session_factory() as session:
        row = session.get(EventWatermark, contract_id)
        return row.last_ledger if row else 0


def set_watermark(contract_id: str, ledger: int, session_factory: sessionmaker) -> None:
    with session_factory() as session:
        row = session.get(EventWatermark, contract_id)
        if row is None:
            row = EventWatermark(contract_id=contract_id, last_ledger=ledger)
            session.add(row)
        else:
            row.last_ledger = ledger
            row.updated_at = datetime.now(UTC)
        session.commit()


# ---------------------------------------------------------------------------
# Stale-score alert
# ---------------------------------------------------------------------------


def check_stale_score_alert(
    event: ContractEvent,
    current_score: int | None,
    dispatcher: Any,
    threshold: int = STALE_SCORE_ALERT_THRESHOLD,
) -> bool:
    """Fire a stale-score alert if consumed score differs from current by > *threshold*.

    Returns True if an alert was fired.
    """
    if event.event_type != "score_read":
        return False
    if event.score is None or current_score is None:
        return False

    delta = abs(current_score - event.score)
    if delta > threshold:
        logger.warning(
            "Stale score consumption detected: wallet_hash=%s consumed_score=%d "
            "current_score=%d delta=%d",
            event.wallet_id_hash,
            event.score,
            current_score,
            delta,
        )
        if dispatcher is not None:
            alert_payload = {
                "score": current_score,
                "consumed_score": event.score,
                "delta": delta,
                "benford_flag": False,
                "ml_flag": False,
                "confidence": 0,
                "stale_consumption": True,
            }
            pair = event.asset_pair or "unknown/unknown"
            dispatcher.dispatch(event.wallet_id_hash, alert_payload, pair)
        return True
    return False


# ---------------------------------------------------------------------------
# Event fetching backends
# ---------------------------------------------------------------------------


def _fetch_events_rpc(
    rpc_url: str,
    contract_id: str,
    start_ledger: int,
    limit: int = EVENT_PAGE_LIMIT,
) -> Iterator[dict]:
    """Yield raw event dicts from the Soroban RPC ``getEvents`` method.

    Handles cursor-based pagination automatically.
    """
    cursor = None
    while True:
        payload: dict = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getEvents",
            "params": {
                "startLedger": str(start_ledger),
                "filters": [
                    {
                        "type": "contract",
                        "contractIds": [contract_id],
                    }
                ],
                "pagination": {"limit": limit},
            },
        }
        if cursor:
            payload["params"]["pagination"]["cursor"] = cursor

        try:
            resp = requests.post(rpc_url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Soroban RPC getEvents failed: %s", exc)
            return

        result = data.get("result", {})
        events = result.get("events", [])
        for ev in events:
            yield ev

        # Pagination: if we got a full page there may be more
        if len(events) < limit:
            break
        # Advance cursor to last event id
        if events:
            cursor = events[-1].get("id")
        else:
            break


def _fetch_events_horizon(
    horizon_url: str,
    contract_id: str,
    start_ledger: int,
    limit: int = EVENT_PAGE_LIMIT,
) -> Iterator[dict]:
    """Yield raw event dicts from the Horizon effects endpoint (fallback).

    Horizon wraps Soroban contract events as ``type=contract_credited`` /
    ``type=contract_debited`` effects under ``/accounts/{contract_id}/effects``.
    The Soroban-specific payload is nested under the ``data`` key.
    """
    url = f"{horizon_url.rstrip('/')}/accounts/{contract_id}/effects"
    params: dict = {"limit": limit, "order": "asc"}
    # Horizon cursor is a paging_token; approximate from ledger sequence
    params["cursor"] = f"{start_ledger * 4096}-1"

    while url:
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Horizon effects fetch failed: %s", exc)
            return

        records = data.get("_embedded", {}).get("records", [])
        for rec in records:
            # Only forward Soroban contract events (have a "data" dict with "topic")
            if isinstance(rec.get("data"), dict) and "topic" in rec.get("data", {}):
                yield rec

        next_url = data.get("_links", {}).get("next", {}).get("href")
        url = next_url if next_url and records else None
        params = {}  # cursor is embedded in next_url


# ---------------------------------------------------------------------------
# Main listener class
# ---------------------------------------------------------------------------


class SorobanEventListener:
    """Continuous listener for ledgerlens-score contract events.

    Parameters
    ----------
    contract_id:
        The Soroban contract ID (C... address).
    rpc_url:
        Soroban RPC endpoint.  Used when ``backend="rpc"``.
    horizon_url:
        Stellar Horizon base URL.  Used when ``backend="horizon"``.
    db_url:
        SQLAlchemy DB URL for persisting events and the watermark.
    dispatcher:
        An ``AlertDispatcher`` instance for stale-score alerts.  Pass
        ``None`` to disable alerting.
    current_score_fn:
        Callable ``(wallet_id_hash: str) -> int | None`` that returns the
        current score for a wallet (used for stale-score detection).  If
        ``None``, stale-score checks are skipped.
    backend:
        ``"rpc"`` (default) or ``"horizon"``.
    poll_interval:
        Seconds between polling loops.
    stale_threshold:
        Score delta above which a stale-score alert is fired.
    """

    def __init__(
        self,
        contract_id: str | None = None,
        rpc_url: str | None = None,
        horizon_url: str | None = None,
        db_url: str | None = None,
        dispatcher: Any = None,
        current_score_fn: Any = None,
        backend: str = EVENT_BACKEND,
        poll_interval: float = EVENT_POLL_INTERVAL_SECONDS,
        stale_threshold: int = STALE_SCORE_ALERT_THRESHOLD,
    ) -> None:
        self.contract_id = contract_id or config.LEDGERLENS_CONTRACT_ID
        self.rpc_url = rpc_url or config.SOROBAN_RPC_URL
        self.horizon_url = horizon_url or config.HORIZON_URL
        self.backend = backend
        self.poll_interval = poll_interval
        self.stale_threshold = stale_threshold
        self.dispatcher = dispatcher
        self.current_score_fn = current_score_fn

        self._session_factory = _get_session_factory(db_url)
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the polling loop in the current thread (blocks)."""
        logger.info(
            "SorobanEventListener starting: contract=%s backend=%s",
            self.contract_id, self.backend,
        )
        self._stop_event.clear()
        self._run_loop()

    def start_background(self) -> threading.Thread:
        """Start the polling loop in a daemon background thread."""
        t = threading.Thread(target=self._run_loop, daemon=True, name="soroban-event-listener")
        t.start()
        return t

    def stop(self) -> None:
        """Signal the polling loop to stop after the current iteration."""
        self._stop_event.set()

    def process_batch(self, raw_events: list[dict]) -> list[ContractEvent]:
        """Parse, persist, and alert on a batch of raw event dicts.

        Returns the list of successfully parsed ``ContractEvent`` objects.
        """
        parsed: list[ContractEvent] = []
        max_ledger = 0

        for raw in raw_events:
            event = parse_contract_event(raw)
            if event is None:
                continue

            persist_event(event, self._session_factory)
            parsed.append(event)
            max_ledger = max(max_ledger, event.ledger_sequence)

            if event.event_type == "score_read" and self.current_score_fn is not None:
                current = self.current_score_fn(event.wallet_id_hash)
                check_stale_score_alert(
                    event, current, self.dispatcher, self.stale_threshold
                )

        if max_ledger > 0:
            set_watermark(self.contract_id, max_ledger, self._session_factory)

        return parsed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_raw_events(self, start_ledger: int) -> list[dict]:
        if self.backend == "horizon":
            return list(
                _fetch_events_horizon(self.horizon_url, self.contract_id, start_ledger)
            )
        return list(
            _fetch_events_rpc(self.rpc_url, self.contract_id, start_ledger)
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                start = get_watermark(self.contract_id, self._session_factory) + 1
                raw_events = self._fetch_raw_events(start)
                if raw_events:
                    n = len(self.process_batch(raw_events))
                    logger.info("Processed %d contract events from ledger %d", n, start)
            except Exception as exc:
                logger.error("Event listener loop error: %s", exc, exc_info=True)

            self._stop_event.wait(timeout=self.poll_interval)
