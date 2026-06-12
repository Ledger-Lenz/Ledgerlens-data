"""Real-time trade ingestion via Horizon's Server-Sent Events stream.

Streams trades for each watched asset pair and yields `Trade` objects as
they occur on the ledger.
"""

from collections.abc import Iterator

from stellar_sdk import Asset as SdkAsset
from stellar_sdk import Server

from config import config
from ingestion.data_models import Asset, Trade


def _to_trade(record: dict) -> Trade:
    return Trade(
        trade_id=record["id"],
        ledger_close_time=record["ledger_close_time"],
        base_account=record["base_account"],
        counter_account=record["counter_account"],
        base_asset=Asset(
            code=record["base_asset_code"] or "XLM",
            issuer=record.get("base_asset_issuer"),
        ),
        counter_asset=Asset(
            code=record["counter_asset_code"] or "XLM",
            issuer=record.get("counter_asset_issuer"),
        ),
        base_amount=float(record["base_amount"]),
        counter_amount=float(record["counter_amount"]),
        price=float(record["price"]["n"]) / float(record["price"]["d"]),
    )


def stream_trades(
    base_asset: SdkAsset,
    counter_asset: SdkAsset,
    cursor: str = "now",
) -> Iterator[Trade]:
    """Yield `Trade` objects as they are streamed from Horizon.

    This is a blocking generator intended to be run in its own worker
    process/thread per watched asset pair.
    """
    server = Server(horizon_url=config.HORIZON_URL)

    call_builder = server.trades().for_asset_pair(base_asset, counter_asset).cursor(cursor)

    for response in call_builder.stream():
        yield _to_trade(response)


def stream_all_watched_pairs() -> Iterator[Trade]:
    """Convenience generator that round-robins through configured pairs.

    NOTE: for production use, run one `stream_trades` generator per pair in
    its own task/thread rather than interleaving here.
    """
    if not config.WATCHED_ASSET_PAIRS:
        raise ValueError("WATCHED_ASSET_PAIRS is not configured")

    streams = []
    for code, issuer in config.WATCHED_ASSET_PAIRS:
        asset = SdkAsset.native() if issuer == "native" else SdkAsset(code, issuer)
        xlm = SdkAsset.native()
        if asset == xlm:
            continue
        streams.append(stream_trades(asset, xlm))

    for stream in streams:
        yield from stream
