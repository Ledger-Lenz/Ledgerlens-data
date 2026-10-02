"""Bulk historical trade ingestion via Horizon's paginated trades endpoint.

Large backfills should go through :func:`backfill_trades`, which processes the
trade history in fixed-size chunks and persists a checkpoint after each one so
an interrupted run resumes from the last completed chunk. See
``docs/ingestion.md`` ("Resumable chunked backfill").
"""

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError
from stellar_sdk import Asset as SdkAsset
from stellar_sdk import Server

from config import config
from ingestion.data_models import Trade
from ingestion.horizon_fetcher import fetch as horizon_fetch
from ingestion.horizon_streamer import _to_trade
from ingestion.untrusted_input import UntrustedInputError, validate_trade
from utils.checkpoint import PipelineCheckpoint
from utils.logging import get_logger
from utils.retry import retry_with_backoff

logger = get_logger(__name__)

#: Pipeline name stamped into backfill checkpoint files.
BACKFILL_PIPELINE = "historical_backfill"

#: Raw Horizon records per checkpointed chunk (see docs/ingestion.md for sizing).
DEFAULT_BACKFILL_CHUNK_SIZE = 10_000

#: Receives ``(chunk_index, trades)`` and durably writes them. Must be idempotent
#: per ``chunk_index`` (a chunk interrupted before its checkpoint is written is
#: re-delivered on resume). May return an artifact path to record in the checkpoint.
ChunkSink = Callable[[int, list[Trade]], str | None]


@retry_with_backoff(exceptions=(ConnectionError, TimeoutError, OSError))
def _fetch_page(call_builder):
    return horizon_fetch(call_builder.call)


def _iter_records(
    base_asset: SdkAsset,
    counter_asset: SdkAsset,
    *,
    cursor: str | None = None,
    limit_per_page: int = 200,
    horizon_url: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw Horizon trade records in ascending order, starting after *cursor*."""
    server = Server(horizon_url=horizon_url or config.HORIZON_URL)

    call_builder = (
        server.trades()
        .for_asset_pair(base_asset, counter_asset)
        .limit(limit_per_page)
        .order(desc=False)
    )
    if cursor:
        call_builder = call_builder.cursor(cursor)

    while True:
        page = _fetch_page(call_builder)
        records = page["_embedded"]["records"]
        if not records:
            break

        yield from records

        next_url = page["_links"]["next"]["href"]
        if not next_url:
            break
        call_builder = call_builder.cursor(records[-1]["paging_token"])


def _parse_record(record: dict[str, Any], start_time: datetime | None) -> Trade | None:
    """Convert a raw record to a validated `Trade`, or None if rejected/filtered."""
    try:
        trade = _to_trade(record)
        validate_trade(trade, source="historical_loader")
    except (UntrustedInputError, ValidationError, KeyError, ValueError) as exc:
        logger.warning(
            "Rejected malformed trade record from Horizon (id=%s): %s",
            record.get("id", "?"),
            exc,
        )
        return None
    if start_time and trade.ledger_close_time < start_time:
        return None
    return trade


def load_trades(
    base_asset: SdkAsset,
    counter_asset: SdkAsset,
    start_time: datetime | None = None,
    limit_per_page: int = 200,
    *,
    cursor: str | None = None,
    horizon_url: str | None = None,
) -> Iterator[Trade]:
    """Page through historical trades for an asset pair from Horizon.

    If `start_time` is provided, records before it are skipped. Horizon
    paginates results in ascending order by default.
    """
    for record in _iter_records(
        base_asset,
        counter_asset,
        cursor=cursor,
        limit_per_page=limit_per_page,
        horizon_url=horizon_url,
    ):
        trade = _parse_record(record, start_time)
        if trade is not None:
            yield trade


def _asset_id(asset: SdkAsset) -> str:
    return f"{asset.code}:{asset.issuer or 'native'}"


def _chunk_id(index: int) -> str:
    return f"chunk-{index:06d}"


def _last_completed_chunk(
    completed: dict[str, dict[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    """Return ``(index, entry)`` of the highest completed chunk, if any."""
    indices = [int(unit[len("chunk-") :]) for unit in completed if unit.startswith("chunk-")]
    if not indices:
        return None
    index = max(indices)
    return index, completed[_chunk_id(index)]


def parquet_chunk_sink(output_dir: str | Path) -> ChunkSink:
    """Sink writing each chunk to ``<output_dir>/chunk-NNNNNN.parquet``.

    Re-delivering a chunk overwrites the same file, so a resumed backfill
    never duplicates rows on disk.
    """
    directory = Path(output_dir)

    def _write(index: int, trades: list[Trade]) -> str | None:
        if not trades:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_chunk_id(index)}.parquet"
        trades_to_dataframe(trades).to_parquet(path, index=False)
        return str(path)

    return _write


def backfill_trades(
    base_asset: SdkAsset,
    counter_asset: SdkAsset,
    checkpoint_path: str | Path,
    sink: ChunkSink,
    *,
    start_time: datetime | None = None,
    chunk_size: int = DEFAULT_BACKFILL_CHUNK_SIZE,
    limit_per_page: int = 200,
    fresh: bool = False,
) -> dict[str, Any]:
    """Resumable, chunked backfill of an asset pair's trade history.

    Raw Horizon records are grouped into chunks of *chunk_size*. Each chunk's
    valid trades are handed to *sink*; only once the sink returns is the chunk
    recorded in the checkpoint at *checkpoint_path*, together with the paging
    token of its last record. On restart the backfill resumes from that token,
    so an interruption loses at most the in-flight chunk and — because sinks
    are idempotent per chunk index — never produces gaps or duplicates.

    A sink failure is recorded against the chunk and re-raised; the next run
    retries the same chunk. Returns the checkpoint summary.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    pair_id = f"{_asset_id(base_asset)}/{_asset_id(counter_asset)}"
    ckpt = PipelineCheckpoint.load_or_create(
        path=checkpoint_path,
        pipeline=BACKFILL_PIPELINE,
        fingerprint_inputs={
            "pair": pair_id,
            "start_time": start_time.isoformat() if start_time else None,
        },
        fresh=fresh,
    )

    cursor: str | None = None
    index = 0
    last = _last_completed_chunk(ckpt.completed)
    if last is not None:
        cursor = last[1]["metadata"]["cursor"]
        index = last[0] + 1
        logger.info("Resuming backfill of %s at chunk %d (cursor=%s)", pair_id, index, cursor)

    def _commit(trades: list[Trade], raw_count: int, last_token: str) -> None:
        chunk = _chunk_id(index)
        try:
            artifact = sink(index, trades)
        except Exception as exc:
            ckpt.record_failure(chunk, exc)
            raise
        ckpt.record_success(
            chunk,
            artifact_path=artifact,
            metadata={
                "cursor": last_token,
                "raw_records": raw_count,
                "trades": len(trades),
                "last_ledger_close_time": (
                    trades[-1].ledger_close_time.isoformat() if trades else None
                ),
            },
        )
        logger.info("Backfill %s: committed %s (%d trades)", pair_id, chunk, len(trades))

    trades: list[Trade] = []
    raw_count = 0
    last_token = ""
    for record in _iter_records(
        base_asset, counter_asset, cursor=cursor, limit_per_page=limit_per_page
    ):
        raw_count += 1
        last_token = record["paging_token"]
        trade = _parse_record(record, start_time)
        if trade is not None:
            trades.append(trade)
        if raw_count == chunk_size:
            _commit(trades, raw_count, last_token)
            index += 1
            trades, raw_count = [], 0

    if raw_count:
        _commit(trades, raw_count, last_token)

    return ckpt.summary()


def backfill_status(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read-only progress snapshot of a backfill checkpoint, for operators.

    Safe to call while the backfill is running: checkpoint files are replaced
    atomically, so a reader always sees the last committed chunk.

    Raises:
        FileNotFoundError: no checkpoint exists at *checkpoint_path*.
        ValueError: the file is not a historical backfill checkpoint.
    """
    path = Path(checkpoint_path)
    raw = json.loads(path.read_text())
    if raw.get("pipeline") != BACKFILL_PIPELINE:
        raise ValueError(f"{path} is not a {BACKFILL_PIPELINE} checkpoint")

    completed: dict[str, dict[str, Any]] = raw.get("completed", {})
    chunks = [entry["metadata"] for unit, entry in completed.items() if unit.startswith("chunk-")]
    last = _last_completed_chunk(completed)
    last_meta = last[1]["metadata"] if last else {}
    last_close_times = [m["last_ledger_close_time"] for m in chunks if m["last_ledger_close_time"]]
    inputs = raw.get("fingerprint_inputs", {})
    return {
        "checkpoint": str(path),
        "pair": inputs.get("pair"),
        "start_time": inputs.get("start_time"),
        "chunks_completed": len(chunks),
        "last_chunk": last[0] if last else None,
        "raw_records": sum(m["raw_records"] for m in chunks),
        "trades": sum(m["trades"] for m in chunks),
        "cursor": last_meta.get("cursor"),
        "last_ledger_close_time": max(last_close_times) if last_close_times else None,
        "failed_chunks": raw.get("failed", {}),
        "started_at": raw.get("started_at"),
        "updated_at": raw.get("updated_at"),
    }


def trades_to_dataframe(trades: Iterable[Trade]) -> pd.DataFrame:
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


def load_pair_to_dataframe(
    base_asset: SdkAsset,
    counter_asset: SdkAsset,
    start_time: datetime | None = None,
) -> pd.DataFrame:
    """Load historical trades for a single asset pair into a DataFrame."""
    return trades_to_dataframe(load_trades(base_asset, counter_asset, start_time=start_time))


def load_watched_pairs_to_dataframe(start_time: datetime | None = None) -> pd.DataFrame:
    """Load historical trades for every pair configured in
    `WATCHED_ASSET_PAIRS` and combine them into a single DataFrame."""
    frames = []
    xlm = SdkAsset.native()

    for code, issuer in config.WATCHED_ASSET_PAIRS:
        asset = xlm if issuer == "native" else SdkAsset(code, issuer)
        if asset == xlm:
            continue
        frames.append(load_pair_to_dataframe(asset, xlm, start_time=start_time))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_trades_file(path: str, scanner=None) -> pd.DataFrame:
    """Load historical trades from a CSV/Parquet file after pre-parse scanning."""
    from ingestion.secure_file_handler import check_file

    checked = check_file(path, scanner)
    if checked.suffix == ".parquet":
        return pd.read_parquet(checked)
    return pd.read_csv(checked)
