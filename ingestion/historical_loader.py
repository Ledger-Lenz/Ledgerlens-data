"""Bulk historical trade ingestion via Horizon's paginated trades endpoint."""

from collections.abc import Iterator
from datetime import datetime

import pandas as pd
from stellar_sdk import Asset as SdkAsset
from stellar_sdk import Server

from config import config
from ingestion.data_models import Trade
from ingestion.horizon_streamer import _to_trade
from utils.retry import retry_with_backoff


@retry_with_backoff(exceptions=(ConnectionError, TimeoutError, OSError))
def _fetch_page(call_builder):
    return call_builder.call()


def load_trades(
    base_asset: SdkAsset,
    counter_asset: SdkAsset,
    start_time: datetime | None = None,
    limit_per_page: int = 200,
) -> Iterator[Trade]:
    """Page through historical trades for an asset pair from Horizon.

    If `start_time` is provided, records before it are skipped. Horizon
    paginates results in ascending order by default.
    """
    server = Server(horizon_url=config.HORIZON_URL)

    call_builder = (
        server.trades()
        .for_asset_pair(base_asset, counter_asset)
        .limit(limit_per_page)
        .order(desc=False)
    )

    while True:
        page = _fetch_page(call_builder)
        records = page["_embedded"]["records"]
        if not records:
            break

        for record in records:
            trade = _to_trade(record)
            if start_time and trade.ledger_close_time < start_time:
                continue
            yield trade

        next_url = page["_links"]["next"]["href"]
        if not next_url:
            break
        call_builder = call_builder.cursor(records[-1]["paging_token"])


def trades_to_dataframe(trades: Iterator[Trade]) -> pd.DataFrame:
    """Flatten an iterable of `Trade` objects into a DataFrame for feature
    engineering and the Benford engine."""
    rows = []
    for t in trades:
        rows.append(
            {
                "trade_id": t.trade_id,
                "ledger_close_time": t.ledger_close_time,
                "base_account": t.base_account,
                "counter_account": t.counter_account,
                "base_asset": f"{t.base_asset.code}:{t.base_asset.issuer or 'native'}",
                "counter_asset": f"{t.counter_asset.code}:{t.counter_asset.issuer or 'native'}",
                "amount": t.amount,
                "price": t.price,
            }
        )
    return pd.DataFrame(rows)


def load_watched_pairs_to_dataframe(start_time: datetime | None = None) -> pd.DataFrame:
    """Load historical trades for every pair configured in
    `WATCHED_ASSET_PAIRS` and combine them into a single DataFrame."""
    frames = []
    xlm = SdkAsset.native()

    for code, issuer in config.WATCHED_ASSET_PAIRS:
        asset = xlm if issuer == "native" else SdkAsset(code, issuer)
        if asset == xlm:
            continue
        frames.append(trades_to_dataframe(load_trades(asset, xlm, start_time=start_time)))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
