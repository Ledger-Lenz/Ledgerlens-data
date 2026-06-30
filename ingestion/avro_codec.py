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

from typing import Any, cast
import io
import json
import struct
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
    return cast(dict[Any, Any], fastavro.parse_schema(raw))


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
    return cast(bytes, buffer.getvalue())


def deserialize(value: bytes, schema: dict) -> dict:
    """Decode schemaless Avro binary *value* back into a record dict."""
    return cast(dict[Any, Any], fastavro.schemaless_reader(io.BytesIO(value), schema))


def validate(record: dict, schema: dict) -> None:
    """Raise ``fastavro`` validation error if *record* does not match *schema*."""
    fastavro.validation.validate(record, schema, raise_errors=True)


# ---------------------------------------------------------------------------
# Avro CRC32 canonical fingerprinting (#201)
# ---------------------------------------------------------------------------

def _avro_crc32_fingerprint(schema_dict: dict) -> int:
    """Compute the 64-bit Avro CRC-64-AVRO fingerprint of the canonical schema JSON.

    Avro's canonical fingerprinting algorithm applies a specific CRC-64 over
    the schema's Parsing Canonical Form (PCF).  We use fastavro's built-in
    ``fingerprint`` helper when available and fall back to CRC-32 (from
    ``struct``) for environments without the optional dependency.

    The returned value is a signed 64-bit integer for consistency with the
    Avro specification.
    """
    canonical = json.dumps(schema_dict, sort_keys=True, separators=(",", ":"))
    data = canonical.encode("utf-8")
    try:
        # fastavro >= 1.6 exposes rabin fingerprint
        return fastavro.schema.fingerprint(data, "CRC-64-AVRO")  # type: ignore[attr-defined]
    except AttributeError:
        pass
    # Fallback: CRC-32 packed as a signed 64-bit integer
    import zlib
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return struct.unpack(">i", struct.pack(">I", crc)[:4])[0]


# ---------------------------------------------------------------------------
# Schema compatibility checks (#201)
# ---------------------------------------------------------------------------

def _field_map(schema: dict) -> dict[str, dict]:
    """Return {field_name: field_def} for a parsed Avro record schema."""
    return {f["name"]: f for f in schema.get("fields", [])}


def _has_default(field: dict) -> bool:
    return "default" in field


def _is_nullable(field: dict) -> bool:
    ftype = field.get("type")
    if isinstance(ftype, list):
        return "null" in ftype
    return ftype == "null"


def _field_is_optional(field: dict) -> bool:
    return _has_default(field) or _is_nullable(field)


def check_backward_compatibility(old_schema: dict, new_schema: dict) -> tuple[bool, list[str]]:
    """Check whether messages written with *old_schema* can be read with *new_schema*.

    Backward compatibility rules (Avro spec):
    - Fields added to *new_schema* must have a default value.
    - Fields removed from *new_schema* (present in *old_schema*) must have been
      optional (had a default or nullable type) in *old_schema*.

    Returns:
        (is_compatible: bool, violations: list[str])
    """
    old_fields = _field_map(old_schema)
    new_fields = _field_map(new_schema)
    violations = []

    # Added fields must have defaults so old messages (missing the field) are valid
    for name, field in new_fields.items():
        if name not in old_fields and not _field_is_optional(field):
            violations.append(
                f"Added field '{name}' has no default — old messages cannot supply a value"
            )

    # Removed fields: readers skip them unless they were optional
    for name, field in old_fields.items():
        if name not in new_fields and not _field_is_optional(field):
            violations.append(
                f"Removed required field '{name}' — new reader cannot reconstruct it"
            )

    return len(violations) == 0, violations


def check_forward_compatibility(old_schema: dict, new_schema: dict) -> tuple[bool, list[str]]:
    """Check whether messages written with *new_schema* can be read with *old_schema*.

    Forward compatibility rules (Avro spec):
    - Fields added in *new_schema* must have a default so *old_schema* readers
      can supply a value when the field is missing from the reader's perspective.
    - Fields present in *old_schema* but missing from *new_schema* must have
      defaults in *old_schema* so the reader can supply them.

    Returns:
        (is_compatible: bool, violations: list[str])
    """
    old_fields = _field_map(old_schema)
    new_fields = _field_map(new_schema)
    violations = []

    # New writer writes new fields; old reader must be able to ignore them
    for name, field in new_fields.items():
        if name not in old_fields and not _field_is_optional(field):
            violations.append(
                f"New field '{name}' has no default — old reader cannot supply it when missing"
            )

    # Old reader expects fields that the new writer omitted
    for name, field in old_fields.items():
        if name not in new_fields and not _field_is_optional(field):
            violations.append(
                f"Field '{name}' expected by old reader is absent from new schema "
                "and has no default — old reader cannot reconstruct it"
            )

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# SchemaRegistry (#201)
# ---------------------------------------------------------------------------

class SchemaRegistry:
    """In-process registry of Avro schema versions with fingerprint-based lookup.

    All schemas loaded through this registry are sourced exclusively from the
    bundled ``data/`` directory — schemas from untrusted external sources are
    never accepted at runtime.

    Usage::

        registry = SchemaRegistry()
        v1 = registry.register(load_schema("data/trade_avro_schema.json"))
        v2 = registry.register(new_schema_dict)
        ok, errors = registry.check_backward_compatibility(v1, v2)
    """

    def __init__(self) -> None:
        # fingerprint -> (version_number, raw_schema_dict)
        self._versions: dict[int, tuple[int, dict]] = {}
        self._counter: int = 0

    def register(self, schema: dict) -> int:
        """Register *schema* and return its fingerprint.

        If the schema is already registered its existing fingerprint is returned
        without incrementing the version counter.
        """
        fp = _avro_crc32_fingerprint(schema)
        if fp not in self._versions:
            self._counter += 1
            self._versions[fp] = (self._counter, schema)
        return fp

    def get_schema(self, fingerprint: int) -> dict | None:
        """Return the raw schema dict for *fingerprint*, or None if unknown."""
        entry = self._versions.get(fingerprint)
        return entry[1] if entry else None

    def get_version(self, fingerprint: int) -> int | None:
        """Return the sequential version number for *fingerprint*, or None."""
        entry = self._versions.get(fingerprint)
        return entry[0] if entry else None

    def latest_fingerprint(self) -> int | None:
        """Return the fingerprint of the most recently registered schema."""
        if not self._versions:
            return None
        return max(self._versions, key=lambda fp: self._versions[fp][0])

    def check_backward_compatibility(
        self, old_fp: int, new_fp: int
    ) -> tuple[bool, list[str]]:
        """Backward-compat check between two registered schemas by fingerprint."""
        old = self.get_schema(old_fp)
        new = self.get_schema(new_fp)
        if old is None:
            raise KeyError(f"Unknown fingerprint (old): {old_fp}")
        if new is None:
            raise KeyError(f"Unknown fingerprint (new): {new_fp}")
        return check_backward_compatibility(old, new)

    def check_forward_compatibility(
        self, old_fp: int, new_fp: int
    ) -> tuple[bool, list[str]]:
        """Forward-compat check between two registered schemas by fingerprint."""
        old = self.get_schema(old_fp)
        new = self.get_schema(new_fp)
        if old is None:
            raise KeyError(f"Unknown fingerprint (old): {old_fp}")
        if new is None:
            raise KeyError(f"Unknown fingerprint (new): {new_fp}")
        return check_forward_compatibility(old, new)

    def all_fingerprints(self) -> list[tuple[int, int]]:
        """Return [(version, fingerprint)] sorted by version ascending."""
        return sorted(
            [(v, fp) for fp, (v, _) in self._versions.items()],
            key=lambda x: x[0],
        )


# Module-level default registry populated with the bundled schema on first use.
_default_registry: SchemaRegistry | None = None


def get_default_registry() -> SchemaRegistry:
    """Return (and lazily initialise) the module-level SchemaRegistry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SchemaRegistry()
        raw = json.load(open(config.TRADE_AVRO_SCHEMA_PATH, encoding="utf-8"))
        _default_registry.register(raw)
    return _default_registry
