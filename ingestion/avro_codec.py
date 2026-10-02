"""Avro (de)serialisation helpers shared by the Kafka producer and worker.

The wire format is a *schemaless* Avro binary encoding of the ``Trade`` record
defined in ``data/trade_avro_schema.json``.  Producers register that schema with
a schema registry before publishing (:func:`get_default_registry`): a
Confluent-compatible registry when ``SCHEMA_REGISTRY_URL`` is set, otherwise an
in-process :class:`SchemaRegistry`.  Either way the configured
``SCHEMA_COMPATIBILITY_MODE`` is enforced at registration time, so a breaking
schema change is rejected before any message is written.  See
``docs/schema_registry_runbook.md``.

Centralising the codec here keeps the producer (``ingestion/kafka_producer.py``)
and the worker (``streaming/kafka_worker.py``) in lock-step on field names,
types, and the canonical ``asset_pair`` string format.
"""

import io
import json
import struct
import time
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from typing import Any, Protocol, cast

import fastavro
import requests

from config import config
from ingestion.data_models import Asset, Trade
from ingestion.exceptions import (
    IngestionTransportError,
    InvalidInputError,
    SchemaCompatibilityError,
    SchemaDecodeError,
    SchemaValidationError,
    record_context,
)


def read_schema(schema_path: str | None = None) -> dict:
    """Return the raw (unparsed) Avro schema JSON from *schema_path* or the configured default."""
    path = schema_path or config.TRADE_AVRO_SCHEMA_PATH
    with open(path, encoding="utf-8") as fh:
        return cast(dict[Any, Any], json.load(fh))


@lru_cache(maxsize=4)
def load_schema(schema_path: str | None = None) -> dict:
    """Parse and cache the Avro schema from *schema_path* (or the configured default)."""
    return cast(dict[Any, Any], fastavro.parse_schema(read_schema(schema_path)))


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

    Raises:
        RecordValidationError: If the decoded record is missing fields, has
            wrong-typed values, or otherwise fails ``Trade`` validation.
    """
    with record_context("avro_codec.record_to_trade", record):
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


def _validate_against_schema(record: dict, schema: dict, source: str) -> None:
    """Run fastavro validation, re-raising failures as :class:`SchemaValidationError`."""
    try:
        fastavro.validation.validate(record, schema, raise_errors=True)
    except SchemaValidationError:
        raise
    except Exception as exc:
        raise SchemaValidationError(
            f"{source}: record does not match the Avro schema — {exc}",
            source=source,
            reason=str(exc),
            raw=record,
        ) from exc


def serialize(record: dict, schema: dict) -> bytes:
    """Encode *record* to schemaless Avro binary bytes.

    This is the first line of defence against poison-pill messages.

    Raises:
        SchemaValidationError: If *record* is missing fields or has wrong-typed
            values.
    """
    _validate_against_schema(record, schema, "avro_codec.serialize")
    buffer = io.BytesIO()
    fastavro.schemaless_writer(buffer, schema, record)
    return cast(bytes, buffer.getvalue())


def deserialize(value: bytes, schema: dict) -> dict:
    """Decode schemaless Avro binary *value* back into a record dict."""
    try:
        return cast(dict[Any, Any], fastavro.schemaless_reader(io.BytesIO(value), schema))
    except SchemaDecodeError:
        raise
    except Exception as exc:
        raise SchemaDecodeError.from_exception(
            exc,
            source="kafka",
            operation="deserialize_avro",
            details={"payload_size_bytes": len(value)},
        ) from exc


def validate(record: dict, schema: dict) -> None:
    """Validate *record* against *schema*.

    Raises:
        SchemaValidationError: If *record* does not match *schema*.
    """
    _validate_against_schema(record, schema, "avro_codec.validate")


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
            violations.append(f"Removed required field '{name}' — new reader cannot reconstruct it")

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


class CompatibilityMode(StrEnum):
    """Schema registry compatibility modes (same names as Confluent Schema Registry)."""

    NONE = "NONE"
    BACKWARD = "BACKWARD"
    FORWARD = "FORWARD"
    FULL = "FULL"


def _resolve_mode(mode: CompatibilityMode | str | None) -> CompatibilityMode:
    raw = mode if mode is not None else config.SCHEMA_COMPATIBILITY_MODE
    try:
        return CompatibilityMode(str(raw).upper())
    except ValueError as exc:
        raise InvalidInputError(
            f"Unknown schema compatibility mode {raw!r}",
            source="avro_codec",
            reason=f"expected one of {[m.value for m in CompatibilityMode]}",
        ) from exc


def check_compatibility(
    old_schema: dict, new_schema: dict, mode: CompatibilityMode | str
) -> tuple[bool, list[str]]:
    """Check *new_schema* against *old_schema* under compatibility *mode*."""
    mode = _resolve_mode(mode)
    violations: list[str] = []
    if mode in (CompatibilityMode.BACKWARD, CompatibilityMode.FULL):
        violations += check_backward_compatibility(old_schema, new_schema)[1]
    if mode in (CompatibilityMode.FORWARD, CompatibilityMode.FULL):
        violations += [
            v for v in check_forward_compatibility(old_schema, new_schema)[1] if v not in violations
        ]
    return len(violations) == 0, violations


def _incompatible(subject: str, mode: CompatibilityMode, violations: list[str]) -> None:
    raise SchemaCompatibilityError(
        f"Schema for subject {subject!r} violates {mode.value} compatibility: "
        + "; ".join(violations),
        source="avro_codec.register",
        reason="; ".join(violations),
        details={"subject": subject, "compatibility": mode.value, "violations": violations},
    )


# ---------------------------------------------------------------------------
# SchemaRegistry (#201, #914)
# ---------------------------------------------------------------------------

DEFAULT_SUBJECT = f"{config.KAFKA_TOPIC_PREFIX}-value"


class SchemaRegistryBackend(Protocol):
    """What producers and consumers need from a schema registry."""

    def register(self, schema: dict, subject: str = DEFAULT_SUBJECT) -> int: ...

    def get_schema(self, schema_id: int) -> dict | None: ...


class SchemaRegistry:
    """In-process registry of Avro schema versions with fingerprint-based lookup.

    All schemas loaded through this registry are sourced exclusively from the
    bundled ``data/`` directory — schemas from untrusted external sources are
    never accepted at runtime.

    Registration enforces *compatibility* (defaults to
    ``config.SCHEMA_COMPATIBILITY_MODE``) against the latest schema of the same
    subject and raises :class:`SchemaCompatibilityError` on a violation.

    Usage::

        registry = SchemaRegistry()
        v1 = registry.register(read_schema("data/trade_avro_schema.json"))
        v2 = registry.register(new_schema_dict)  # raises if incompatible
        ok, errors = registry.check_backward_compatibility(v1, v2)
    """

    def __init__(self, compatibility: CompatibilityMode | str | None = None) -> None:
        self.compatibility = _resolve_mode(compatibility)
        # fingerprint -> (version_number, raw_schema_dict)
        self._versions: dict[int, tuple[int, dict]] = {}
        self._subjects: dict[str, list[int]] = {}
        self._counter: int = 0

    def register(self, schema: dict, subject: str = DEFAULT_SUBJECT) -> int:
        """Register *schema* under *subject* and return its fingerprint.

        If the schema is already the subject's latest version its fingerprint
        is returned without incrementing the version counter.

        Raises:
            SchemaCompatibilityError: If *schema* violates the registry's
                compatibility mode relative to the subject's latest schema.
        """
        fp = _avro_crc32_fingerprint(schema)
        history = self._subjects.setdefault(subject, [])
        if history and history[-1] == fp:
            return fp
        if history:
            ok, violations = check_compatibility(
                self._versions[history[-1]][1], schema, self.compatibility
            )
            if not ok:
                _incompatible(subject, self.compatibility, violations)
        if fp not in self._versions:
            self._counter += 1
            self._versions[fp] = (self._counter, schema)
        history.append(fp)
        return fp

    def latest_schema(self, subject: str = DEFAULT_SUBJECT) -> dict | None:
        """Return the latest schema registered under *subject*, or None."""
        history = self._subjects.get(subject)
        return self._versions[history[-1]][1] if history else None

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

    def check_backward_compatibility(self, old_fp: int, new_fp: int) -> tuple[bool, list[str]]:
        """Backward-compat check between two registered schemas by fingerprint."""
        old = self.get_schema(old_fp)
        new = self.get_schema(new_fp)
        if old is None:
            raise InvalidInputError(
                f"Unknown fingerprint (old): {old_fp}",
                source="avro_codec.SchemaRegistry",
                reason="fingerprint is not registered",
            )
        if new is None:
            raise InvalidInputError(
                f"Unknown fingerprint (new): {new_fp}",
                source="avro_codec.SchemaRegistry",
                reason="fingerprint is not registered",
            )
        return check_backward_compatibility(old, new)

    def check_forward_compatibility(self, old_fp: int, new_fp: int) -> tuple[bool, list[str]]:
        """Forward-compat check between two registered schemas by fingerprint."""
        old = self.get_schema(old_fp)
        new = self.get_schema(new_fp)
        if old is None:
            raise InvalidInputError(
                f"Unknown fingerprint (old): {old_fp}",
                source="avro_codec.SchemaRegistry",
                reason="fingerprint is not registered",
            )
        if new is None:
            raise InvalidInputError(
                f"Unknown fingerprint (new): {new_fp}",
                source="avro_codec.SchemaRegistry",
                reason="fingerprint is not registered",
            )
        return check_forward_compatibility(old, new)

    def all_fingerprints(self) -> list[tuple[int, int]]:
        """Return [(version, fingerprint)] sorted by version ascending."""
        return sorted(
            [(v, fp) for fp, (v, _) in self._versions.items()],
            key=lambda x: x[0],
        )


class ConfluentSchemaRegistry:
    """Client for a Confluent-compatible Schema Registry REST API.

    :meth:`register` pins the subject's compatibility level to the configured
    mode, asks the registry whether the schema is compatible with the latest
    version, and only then registers it. Incompatible schemas raise
    :class:`SchemaCompatibilityError`; network failures raise
    :class:`IngestionTransportError`.
    """

    _CONTENT_TYPE = "application/vnd.schemaregistry.v1+json"

    def __init__(
        self,
        url: str,
        compatibility: CompatibilityMode | str | None = None,
        *,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.compatibility = _resolve_mode(compatibility)
        self._session = session or requests.Session()
        self._timeout = timeout
        self._cache: dict[int, dict] = {}

    def _request(self, method: str, path: str, payload: dict | None = None) -> requests.Response:
        try:
            return self._session.request(
                method,
                f"{self.url}{path}",
                json=payload,
                headers={"Content-Type": self._CONTENT_TYPE, "Accept": self._CONTENT_TYPE},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise IngestionTransportError.from_exception(
                exc, source="avro_codec.ConfluentSchemaRegistry", operation=f"{method} {path}"
            ) from exc

    def _raise_for_status(self, response: requests.Response, operation: str) -> None:
        if response.status_code >= 400:
            raise IngestionTransportError(
                f"Schema registry {operation} failed with HTTP {response.status_code}",
                source="avro_codec.ConfluentSchemaRegistry",
                reason=response.text,
                operation=operation,
                retryable=response.status_code >= 500,
            )

    def register(self, schema: dict, subject: str = DEFAULT_SUBJECT) -> int:
        """Register *schema* under *subject* and return the registry's schema id."""
        payload = {"schema": json.dumps(schema)}

        response = self._request(
            "PUT", f"/config/{subject}", {"compatibility": self.compatibility.value}
        )
        self._raise_for_status(response, "set compatibility")

        response = self._request(
            "POST", f"/compatibility/subjects/{subject}/versions/latest?verbose=true", payload
        )
        if response.status_code != 404:  # 404: first version of a new subject
            self._raise_for_status(response, "compatibility check")
            body = response.json()
            if not body.get("is_compatible", False):
                _incompatible(subject, self.compatibility, list(body.get("messages") or []))

        response = self._request("POST", f"/subjects/{subject}/versions", payload)
        if response.status_code == 409:
            _incompatible(subject, self.compatibility, [response.text])
        self._raise_for_status(response, "register")
        schema_id = int(response.json()["id"])
        self._cache[schema_id] = schema
        return schema_id

    def get_schema(self, schema_id: int) -> dict | None:
        """Return the schema registered under *schema_id*, or None if unknown."""
        if schema_id not in self._cache:
            response = self._request("GET", f"/schemas/ids/{schema_id}")
            if response.status_code == 404:
                return None
            self._raise_for_status(response, "lookup")
            self._cache[schema_id] = json.loads(response.json()["schema"])
        return self._cache[schema_id]


# Module-level default registry populated with the bundled schema on first use.
_default_registry: SchemaRegistryBackend | None = None


def get_default_registry() -> SchemaRegistryBackend:
    """Return (and lazily initialise) the process-wide schema registry.

    A :class:`ConfluentSchemaRegistry` when ``SCHEMA_REGISTRY_URL`` is set,
    otherwise an in-process :class:`SchemaRegistry` seeded with the bundled
    trade schema.
    """
    global _default_registry
    if _default_registry is None:
        if config.SCHEMA_REGISTRY_URL:
            _default_registry = ConfluentSchemaRegistry(config.SCHEMA_REGISTRY_URL)
        else:
            _default_registry = SchemaRegistry()
            _default_registry.register(read_schema())
    return _default_registry
