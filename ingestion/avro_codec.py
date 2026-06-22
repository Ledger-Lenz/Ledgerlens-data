"""Avro (de)serialisation helpers shared by the Kafka producer and worker.

The wire format is a *schemaless* Avro binary encoding of the ``Trade`` record
defined in ``data/trade_avro_schema.json``.  Both sides load the same schema so
no external Schema Registry is required for the default deployment, while the
encoding remains compatible with ``kafkacat -s avro`` when a registry is wired
in.

Centralising the codec here keeps the producer (``ingestion/kafka_producer.py``)
and the worker (``streaming/kafka_worker.py``) in lock-step on field names,
types, and the canonical ``asset_pair`` string format.
"""

from __future__ import annotations

import io
import json
import time
from datetime import UTC, datetime
from functools import lru_cache

import fastavro

from config import config
from ingestion.data_models import Asset, Trade


@lru_cache(maxsize=4)
def load_schema(schema_path: str | None = None) -> dict:
    """Parse and cache the Avro schema from *schema_path* (or the configured default)."""
    path = schema_path or config.TRADE_AVRO_SCHEMA_PATH
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return fastavro.parse_schema(raw)


def trade_to_record(trade: Trade) -> dict:
    """Convert a :class:`Trade` to the Avro record dict.

    ``ledger_close_time`` is kept as a timezone-aware ``datetime`` so fastavro's
    ``timestamp-millis`` logical type encodes it; ``ingestion_timestamp_ms`` is
    the wall-clock time the trade entered the producer (epoch milliseconds).
    """
    return {
        "trade_id": trade.trade_id,
        "base_account": trade.base_account,
        "counter_account": trade.counter_account,
        "base_amount": float(trade.base_amount),
        "counter_amount": float(trade.counter_amount),
        "price": float(trade.price),
        "asset_pair": trade.base_asset.pair_id(trade.counter_asset),
        "ledger_close_time": trade.ledger_close_time,
        "ingestion_timestamp_ms": int(time.time() * 1000),
    }


def record_to_trade(record: dict) -> Trade:
    """Rebuild a :class:`Trade` from a decoded Avro record dict.

    The ``asset_pair`` string ("CODE:ISSUER/CODE:ISSUER") is split back into its
    two :class:`Asset` operands.
    """
    base_part, _, counter_part = record["asset_pair"].partition("/")
    base_code, _, base_issuer = base_part.partition(":")
    counter_code, _, counter_issuer = counter_part.partition(":")

    close_time = record["ledger_close_time"]
    if isinstance(close_time, int):
        close_time = datetime.fromtimestamp(close_time / 1000, tz=UTC)

    return Trade(
        trade_id=record["trade_id"],
        ledger_close_time=close_time,
        base_account=record["base_account"],
        counter_account=record["counter_account"],
        base_asset=Asset(
            code=base_code,
            issuer=None if base_issuer in ("", "native") else base_issuer,
        ),
        counter_asset=Asset(
            code=counter_code,
            issuer=None if counter_issuer in ("", "native") else counter_issuer,
        ),
        base_amount=record["base_amount"],
        counter_amount=record["counter_amount"],
        price=record["price"],
    )


def serialize(record: dict, schema: dict) -> bytes:
    """Encode *record* to schemaless Avro binary bytes.

    Raises if *record* is missing fields or has wrong-typed values — this is the
    first line of defence against poison-pill messages.
    """
    fastavro.validation.validate(record, schema, raise_errors=True)
    buffer = io.BytesIO()
    fastavro.schemaless_writer(buffer, schema, record)
    return buffer.getvalue()


def deserialize(value: bytes, schema: dict) -> dict:
    """Decode schemaless Avro binary *value* back into a record dict."""
    return fastavro.schemaless_reader(io.BytesIO(value), schema)


def validate(record: dict, schema: dict) -> None:
    """Raise ``fastavro`` validation error if *record* does not match *schema*."""
    fastavro.validation.validate(record, schema, raise_errors=True)
