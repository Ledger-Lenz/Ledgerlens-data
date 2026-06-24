"""Order-book event ingestion via Horizon's operations endpoint.

Horizon doesn't expose a dedicated "order cancelled" effect, but
`manage_buy_offer` / `manage_sell_offer` / `create_passive_sell_offer`
operations encode the lifecycle of an offer:

  - `offer_id == "0"` and `amount != "0"` -> a new offer was created
  - `amount == "0"` (with a non-zero `offer_id`) -> the existing offer was
    cancelled (fully consumed or withdrawn)
  - otherwise -> an existing offer was updated (amount/price changed)

This module pages through an account's manage-offer operations and maps
them to `OrderBookEvent` records, which `feature_engineering.py` uses to
compute `order_cancellation_rate`.
"""

from collections.abc import Iterable, Iterator

import pandas as pd
from stellar_sdk import Server

from config import config
from ingestion.data_models import Asset, OrderBookEvent
from utils.retry import retry_with_backoff

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
            event = _to_orderbook_event(record)
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
