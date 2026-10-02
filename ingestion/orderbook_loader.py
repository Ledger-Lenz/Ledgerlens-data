"""Order-book event ingestion via Horizon's operations endpoint.

Horizon doesn't expose a dedicated "order cancelled" effect, but
`manage_buy_offer` / `manage_sell_offer` / `create_passive_sell_offer`
operations encode the lifecycle of an offer:

  - `offer_id == "0"` and `amount != "0"` -> a new offer was created
  - `amount == "0"` (with a non-zero `offer_id`) -> the existing offer was
    cancelled (fully consumed or withdrawn)
  - otherwise -> an existing offer was updated (amount/price changed)

  A withdrawal surfaced as a `cancelled` operation is either a pure
  cancellation (an offer removed while still fully unfilled) or the
  cancellation of the remainder of an offer that was previously partially
  filled. Both cases look identical here: partial fills are recorded in the
  trade stream, not as manage-offer operations, so the loader reports them
  as the same `cancelled` action.

This module pages through an account's manage-offer operations and maps
them to `OrderBookEvent` records, which `feature_engineering.py` uses to
compute `order_cancellation_rate`.
"""

from collections.abc import Iterable, Iterator

import pandas as pd
from pydantic import ValidationError
from stellar_sdk import Server

from config import config
from ingestion.data_models import Asset, OrderBookEvent
from ingestion.exceptions import record_context
from ingestion.untrusted_input import UntrustedInputError, validate_orderbook_event
from utils.logging import get_logger
from utils.retry import retry_with_backoff

logger = get_logger(__name__)

_MANAGE_OFFER_OPERATION_TYPES = {
    "manage_buy_offer",
    "manage_sell_offer",
    "create_passive_sell_offer",
}


@retry_with_backoff(exceptions=(ConnectionError, TimeoutError, OSError))
def _fetch_page(call_builder):
    return call_builder.call()


def _asset_from_operation(record: dict, prefix: str) -> Asset:
    asset_type = record.get(f"{prefix}_asset_type", "native")
    if asset_type == "native":
        return Asset(code="XLM", issuer=None)
    return Asset(code=record[f"{prefix}_asset_code"], issuer=record.get(f"{prefix}_asset_issuer"))


def _action_for_operation(record: dict) -> str | None:
    """Classify a manage-offer operation as created/cancelled/updated.

    Returns `None` for no-op operations (e.g. amount "0" with offer_id "0").
    """
    if record["type"] == "create_passive_sell_offer":
        return "created"

    amount = record.get("amount", "0")
    offer_id = str(record.get("offer_id", "0"))

    if amount == "0":
        return "cancelled" if offer_id != "0" else None
    return "created" if offer_id == "0" else "updated"


def _to_orderbook_event(record: dict) -> OrderBookEvent | None:
    """Map a manage-offer operation record to an :class:`OrderBookEvent`.

    Returns ``None`` for no-op operations.

    Raises:
        RecordValidationError: If the record is missing fields, has wrong-typed
            values, or otherwise fails ``OrderBookEvent`` validation.
    """
    with record_context("orderbook_loader._to_orderbook_event", record):
        action = _action_for_operation(record)
        if action is None:
            return None

        price = record.get("price")
        if price is None:
            n, d = record.get("price_r", {"n": 0, "d": 1}).values()
            price = float(n) / float(d) if d else 0.0
        else:
            price = float(price)

        return OrderBookEvent(
            event_id=record["id"],
            account=record["source_account"],
            ledger_close_time=record["created_at"],
            selling=_asset_from_operation(record, "selling"),
            buying=_asset_from_operation(record, "buying"),
            amount=float(record.get("amount", "0")),
            price=price,
            action=action,
        )


def load_orderbook_events(account_id: str, limit_per_page: int = 200) -> Iterator[OrderBookEvent]:
    """Page through an account's manage-offer operations from Horizon."""
    server = Server(horizon_url=config.HORIZON_URL)

    call_builder = (
        server.operations().for_account(account_id).limit(limit_per_page).order(desc=False)
    )

    while True:
        page = _fetch_page(call_builder)
        records = page["_embedded"]["records"]
        if not records:
            break

        for record in records:
            if record.get("type") not in _MANAGE_OFFER_OPERATION_TYPES:
                continue
            try:
                event = _to_orderbook_event(record)
                if event is not None:
                    validate_orderbook_event(event, source="orderbook_loader")
            except (UntrustedInputError, ValidationError, KeyError, ValueError) as exc:
                logger.warning(
                    "Rejected malformed orderbook record from Horizon (id=%s): %s",
                    record.get("id", "?"),
                    exc,
                )
                continue
            if event is not None:
                yield event

        next_url = page["_links"]["next"]["href"]
        if not next_url:
            break
        call_builder = call_builder.cursor(records[-1]["paging_token"])


def orderbook_events_to_dataframe(events: Iterable[OrderBookEvent]) -> pd.DataFrame:
    """Flatten `OrderBookEvent` records into a DataFrame keyed by `account`."""
    rows = []
    for e in events:
        rows.append(
            {
                "event_id": e.event_id,
                "account": e.account,
                "ledger_close_time": e.ledger_close_time,
                "selling": f"{e.selling.code}:{e.selling.issuer or 'native'}",
                "buying": f"{e.buying.code}:{e.buying.issuer or 'native'}",
                "amount": e.amount,
                "price": e.price,
                "action": e.action,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "account",
            "ledger_close_time",
            "selling",
            "buying",
            "amount",
            "price",
            "action",
        ],
    )


def load_accounts_orderbook_events(account_ids: list[str]) -> pd.DataFrame:
    """Load and combine order-book events for a set of accounts."""
    frames = [orderbook_events_to_dataframe(load_orderbook_events(a)) for a in account_ids]
    if not frames:
        return orderbook_events_to_dataframe(iter([]))
    return pd.concat(frames, ignore_index=True)


class SequenceGapError(Exception):
    """Raised internally when an incremental update skips a sequence number."""


class OrderBookState:
    """Order book maintained from a snapshot + incremental update stream.

    Every update must carry `sequence == last_sequence + 1`. On a gap the
    current book is discarded and a fresh snapshot is fetched via
    `snapshot_fetcher()` before incremental application resumes. Updates
    older than the snapshot sequence are ignored.

    Snapshot shape: {"sequence": int, "bids": {price: qty}, "asks": {price: qty}}
    Update shape:   {"sequence": int, "side": "bids"|"asks", "price": float, "qty": float}
    (qty == 0 removes the level).
    """

    def __init__(self, snapshot_fetcher, resync_alert_threshold: int = 5):
        self._fetch_snapshot = snapshot_fetcher
        self.resync_alert_threshold = resync_alert_threshold
        self.metrics = {"resync_count": 0, "updates_applied": 0, "updates_skipped": 0}
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.sequence = -1
        self._load_snapshot()

    def _load_snapshot(self) -> None:
        snap = self._fetch_snapshot()
        self.bids = dict(snap.get("bids", {}))
        self.asks = dict(snap.get("asks", {}))
        self.sequence = int(snap["sequence"])

    def resync(self) -> None:
        self.metrics["resync_count"] += 1
        logger.warning(
            "Order book sequence gap at %s; resyncing (resync #%d)",
            self.sequence,
            self.metrics["resync_count"],
        )
        if self.metrics["resync_count"] >= self.resync_alert_threshold:
            logger.error(
                "Order book resynced %d times; investigate upstream feed",
                self.metrics["resync_count"],
            )
        self._load_snapshot()

    def _apply(self, update: dict) -> None:
        seq = int(update["sequence"])
        if seq <= self.sequence:
            self.metrics["updates_skipped"] += 1
            return
        if seq != self.sequence + 1:
            raise SequenceGapError(f"expected {self.sequence + 1}, got {seq}")
        book = self.bids if update["side"] == "bids" else self.asks
        if float(update["qty"]) == 0:
            book.pop(update["price"], None)
        else:
            book[update["price"]] = float(update["qty"])
        self.sequence = seq
        self.metrics["updates_applied"] += 1

    def apply_update(self, update: dict) -> None:
        try:
            self._apply(update)
        except SequenceGapError:
            self.resync()
            # The fresh snapshot may already include this update.
            if int(update["sequence"]) == self.sequence + 1:
                self._apply(update)
            elif int(update["sequence"]) <= self.sequence:
                self.metrics["updates_skipped"] += 1

    def apply_updates(self, updates: Iterable[dict]) -> None:
        for u in updates:
            self.apply_update(u)
