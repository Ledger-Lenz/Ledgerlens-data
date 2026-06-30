"""Real-time trade ingestion via Horizon's Server-Sent Events stream.

Streams trades for each watched asset pair and yields `Trade` objects as
they occur on the ledger.
"""

from collections.abc import Iterator

from stellar_sdk import Asset as SdkAsset
from stellar_sdk import Server

from config import config
from ingestion.data_models import Asset, Trade
from utils.logging import get_logger
from utils.tracing import get_tracer, hash_span_id, inject_trace_context

logger = get_logger(__name__)
_tracer = get_tracer(__name__)


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
    max_reconnect_attempts: int = 5,
) -> Iterator[Trade]:
    """Yield `Trade` objects as they are streamed from Horizon.

    This is a blocking generator intended to be run in its own worker
    process/thread per watched asset pair. On a transient connection error
    the stream is re-opened from the last successfully processed cursor, up
    to `max_reconnect_attempts` consecutive failures.
    """
    server = Server(horizon_url=config.HORIZON_URL)
    attempts = 0

    while True:
        call_builder = server.trades().for_asset_pair(base_asset, counter_asset).cursor(cursor)
        try:
            for response in call_builder.stream():
                trade = _to_trade(response)
                with _tracer.start_as_current_span("trade_event.received") as span:
                    span.set_attribute("trade.id", hash_span_id(trade.trade_id))
                    span.set_attribute("trade.base_asset", trade.base_asset.code)
                    span.set_attribute("trade.counter_asset", trade.counter_asset.code)
                yield trade
                cursor = response["paging_token"]
                attempts = 0
        except (ConnectionError, TimeoutError, OSError) as exc:
            attempts += 1
            if attempts >= max_reconnect_attempts:
                raise
            logger.warning(
                "Trade stream disconnected (attempt %d/%d): %s — reconnecting from cursor %s",
                attempts,
                max_reconnect_attempts,
                exc,
                cursor,
            )


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
