"""AMM liquidity pool trade ingestion via Horizon's paginated and SSE endpoints.

Supports bulk historical load and real-time streaming of AMM pool trades.
Pool IDs are validated as 64-character hex strings before any API call.
"""

import re
from collections.abc import Generator
from datetime import datetime

import pandas as pd
from typing import Any, cast

import requests
from stellar_sdk import Server

from config import config
from ingestion.data_models import Asset, Trade
from utils.logging import get_logger
from utils.retry import retry_with_backoff

logger = get_logger(__name__)

_POOL_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class PoolNotFoundError(Exception):
    """Raised when a liquidity pool ID is not found on Horizon (HTTP 404)."""


def _validate_pool_id(pool_id: str) -> None:
    if not _POOL_ID_RE.match(pool_id):
        raise ValueError(
            f"Invalid pool ID {pool_id!r} — must be a 64-character lowercase hex string"
        )


def _amm_record_to_trade(record: dict) -> Trade:
    price_raw = record.get("price", {})
    try:
        price = float(price_raw["n"]) / float(price_raw["d"])
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        price = 0.0

    return Trade(
        trade_id=record.get("id", record["paging_token"]),
        ledger_close_time=record["ledger_close_time"],
        base_account=record.get("base_account", ""),
        counter_account=record.get("counter_account", ""),
        base_asset=Asset(
            code=record.get("base_asset_code") or "XLM",
            issuer=record.get("base_asset_issuer"),
        ),
        counter_asset=Asset(
            code=record.get("counter_asset_code") or "XLM",
            issuer=record.get("counter_asset_issuer"),
        ),
        base_amount=float(record.get("base_amount", 0.0)),
        counter_amount=float(record.get("counter_amount", 0.0)),
        price=price,
    )


@retry_with_backoff(exceptions=(ConnectionError, TimeoutError, OSError))
def _fetch_page(session: requests.Session, url: str, params: dict) -> dict:
    resp = session.get(url, params=params, timeout=30)
    if resp.status_code == 404:
        raise PoolNotFoundError(f"Liquidity pool not found: {url}")
    resp.raise_for_status()
    return cast(dict[Any, Any], resp.json())


def load_amm_pool_trades(
    pool_id: str,
    since: datetime,
    until: datetime,
    limit_per_page: int = 200,
) -> pd.DataFrame:
    """Bulk-load historical trades for a liquidity pool from Horizon.

    Returns a DataFrame with the same column schema as
    ``historical_loader.trades_to_dataframe``:
    trade_id, ledger_close_time, base_account, counter_account,
    base_asset, counter_asset, amount, price.

    Raises:
        ValueError: If pool_id is not a valid 64-character hex string.
        PoolNotFoundError: If the pool does not exist on Horizon (HTTP 404).
    """
    _validate_pool_id(pool_id)

    url = f"{config.HORIZON_URL.rstrip('/')}/liquidity_pools/{pool_id}/trades"
    session = requests.Session()

    seen_paging_tokens: set[str] = set()
    rows: list[dict] = []
    cursor = None

    while True:
        params: dict = {"limit": limit_per_page, "order": "asc"}
        if cursor:
            params["cursor"] = cursor

        page = _fetch_page(session, url, params)
        records = page.get("_embedded", {}).get("records", [])

        if not records:
            break

        for record in records:
            paging_token = record["paging_token"]
            if paging_token in seen_paging_tokens:
                continue
            seen_paging_tokens.add(paging_token)

            trade = _amm_record_to_trade(record)

            ledger_time = pd.to_datetime(trade.ledger_close_time, utc=True)
            since_ts = (
                pd.Timestamp(since, tz="UTC") if since.tzinfo is None else pd.Timestamp(since)
            )
            until_ts = (
                pd.Timestamp(until, tz="UTC") if until.tzinfo is None else pd.Timestamp(until)
            )

            if ledger_time < since_ts:
                cursor = paging_token
                continue
            if ledger_time > until_ts:
                return _records_to_dataframe(rows)

            rows.append(
                {
                    "trade_id": trade.trade_id,
                    "ledger_close_time": trade.ledger_close_time,
                    "base_account": trade.base_account,
                    "counter_account": trade.counter_account,
                    "base_asset": f"{trade.base_asset.code}:{trade.base_asset.issuer or 'native'}",
                    "counter_asset": f"{trade.counter_asset.code}:{trade.counter_asset.issuer or 'native'}",
                    "amount": trade.base_amount,
                    "price": trade.price,
                }
            )
            cursor = paging_token

        next_href = page.get("_links", {}).get("next", {}).get("href", "")
        if not next_href:
            break

    return _records_to_dataframe(rows)


def _records_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "ledger_close_time",
                "base_account",
                "counter_account",
                "base_asset",
                "counter_asset",
                "amount",
                "price",
            ]
        )
    return pd.DataFrame(rows)


def stream_amm_pool_trades(pool_id: str) -> Generator[Trade, None, None]:
    """Stream real-time AMM pool trades via Horizon SSE.

    Yields ``Trade`` objects as they occur. Uses cursor="now" so only
    new trades from the moment of subscription are returned.

    Raises:
        ValueError: If pool_id is not a valid 64-character hex string.
        PoolNotFoundError: If the pool does not exist on Horizon (HTTP 404).
    """
    _validate_pool_id(pool_id)

    server = Server(horizon_url=config.HORIZON_URL)
    cursor = "now"

    while True:
        try:
            call_builder = server.trades().for_liquidity_pool(pool_id).cursor(cursor)
            for record in call_builder.stream():
                trade = _amm_record_to_trade(record)
                cursor = record["paging_token"]
                yield trade
        except Exception as exc:
            logger.warning("AMM stream error for pool %s: %s — reconnecting", pool_id, exc)


def list_active_pools(asset_code: str, asset_issuer: str) -> list[str]:
    """Return pool IDs for all active liquidity pools containing the given asset.

    Queries ``GET /liquidity_pools?reserves[]=<asset>`` and returns a list
    of 64-character pool ID hex strings.
    """
    if asset_issuer == "native" or asset_issuer == "XLM":
        reserve_param = "native"
    else:
        reserve_param = f"{asset_code}:{asset_issuer}"

    url = f"{config.HORIZON_URL.rstrip('/')}/liquidity_pools"
    session = requests.Session()
    pool_ids: list[str] = []
    cursor = None

    while True:
        params: dict = {"reserves[]": reserve_param, "limit": 200, "order": "asc"}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError:
            break

        page = resp.json()
        records = page.get("_embedded", {}).get("records", [])
        if not records:
            break

        for record in records:
            pool_ids.append(record["id"])
            cursor = record["paging_token"]

        next_href = page.get("_links", {}).get("next", {}).get("href", "")
        if not next_href:
            break

    return pool_ids
