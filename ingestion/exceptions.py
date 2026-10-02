"""Typed exceptions for the ingestion and validation layer.

The hierarchy is rooted at :class:`~utils.exceptions.LedgerLensError` so other
packages can eventually share a single base without depending on ``ingestion/``:

    LedgerLensError                     (utils/exceptions.py)
    └── IngestionError
        ├── InvalidInputError           (also a ValueError)
        ├── RecordValidationError
        │   └── SchemaValidationError
        └── SourceUnavailableError
            ├── HorizonRateLimitExceeded    (ingestion/horizon_fetcher.py)
            └── PoolNotFoundError           (ingestion/amm_pool_loader.py)

Context fields
--------------
``IngestionError`` carries ``source`` / ``reason`` / ``raw``, mirroring the two
conventions already in the codebase:

* ``ingestion/kafka_producer.py::_produce_to_dlq`` writes a dead-letter envelope
  of ``{"reason": ..., "raw": ...}``.
* ``utils/circuit_breaker.py::CircuitOpenError`` stores a ``component``
  attribute naming what failed; ``source`` plays that role here.

``raw`` is scrubbed through :func:`safe_raw` so an exception never holds an
unserialisable payload.

Logging
-------
Pass the context through as a logging ``extra`` so it survives into structured
output::

    logger.error("%s", exc, extra={"context": exc.context})

``utils/logging.py`` emits these fields when ``LOG_FORMAT=json`` (via
``python-json-logger``). The plain-text fallback formatter does **not** render
``extra`` fields — they are attached to the ``LogRecord`` but invisible in
dev-mode text logs. That is a pre-existing property of the formatter.

Tracing
-------
``exc.context["raw"]`` may contain wallet addresses. ``utils/tracing.py``
requires that spans never carry raw wallet addresses or trade amounts, so hash
identifiers with ``hash_span_id`` before attaching anything to a span; never
pass ``exc.context`` straight into span attributes.

ValueError compatibility
------------------------
``InvalidInputError`` also inherits ``ValueError`` so existing callers and tests
using ``except ValueError`` / ``pytest.raises(ValueError)`` keep working.
``RecordValidationError`` deliberately does **not** inherit ``KeyError``:
pydantic-originated failures would otherwise satisfy unrelated ``except
KeyError`` control flow elsewhere in the codebase (e.g.
``ingestion/sketches.py``'s hot-path wallet-lock lookup).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

from utils.decimal_guards import PrecisionError
from utils.exceptions import LedgerLensError

# Failure modes raised while turning an untrusted upstream record into a typed
# model. pydantic's ValidationError subclasses ValueError, so it is covered.
RECORD_ERRORS: tuple[type[BaseException], ...] = (
    KeyError,
    TypeError,
    ValueError,
    ZeroDivisionError,
    PrecisionError,
)


def safe_raw(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a best-effort JSON-serialisable copy of *record*.

    Mirrors ``ingestion/kafka_producer.py::_safe_raw`` so an exception's ``raw``
    payload and a dead-letter envelope describe a failed record identically.
    Non-primitive values are stringified rather than dropped.
    """
    if record is None:
        return None
    return {
        k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
        for k, v in record.items()
    }


class IngestionError(LedgerLensError):
    """Base class for ingestion and validation failures.

    Args:
        message: Human-readable description; becomes ``str(exc)``.
        source: Module/function that raised it, e.g.
            ``"horizon_streamer._to_trade"``. Optional so subclasses stay
            constructible from a single positional message.
        reason: Underlying cause, typically ``str(original_exception)``.
        raw: The offending record; scrubbed via :func:`safe_raw`.
    """

    error_code: ClassVar[str] = "ingestion_error"
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        reason: str | None = None,
        raw: Mapping[str, Any] | None = None,
        operation: str | None = None,
        retryable: bool | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.source = source
        self.reason = reason
        self.raw = safe_raw(raw)
        self.operation = operation
        self.retryable = self.default_retryable if retryable is None else retryable
        self.details = dict(details or {})

        context: dict[str, Any] = {}
        if source is not None:
            context["source"] = source
        if reason is not None:
            context["reason"] = reason
        if self.raw is not None:
            context["raw"] = self.raw

        super().__init__(message, context=context)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable machine-readable representation for logs and DLQs."""
        payload: dict[str, Any] = {
            "error_code": self.error_code,
            "error_type": type(self).__name__,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.source is not None:
            payload["source"] = self.source
        if self.operation is not None:
            payload["operation"] = self.operation
        if self.details:
            payload["details"] = dict(self.details)
        return payload

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        source: str | None = None,
        operation: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> IngestionError:
        return cls(str(error), source=source, operation=operation, details=details)


class InvalidInputError(IngestionError, ValueError):
    """Caller-supplied input failed validation before any I/O was attempted.

    Also a ``ValueError`` so existing ``except ValueError`` handlers and tests
    continue to catch it.
    """

    error_code = "ingestion_validation_error"


class RecordValidationError(IngestionError):
    """An upstream record could not be turned into a typed model.

    Covers missing fields, wrong types, and pydantic validation failures.
    """

    error_code = "ingestion_record_invalid"


class SchemaValidationError(RecordValidationError):
    """A record failed schema validation (Avro wire format)."""

    error_code = "ingestion_schema_invalid"


class SchemaCompatibilityError(SchemaValidationError):
    """A schema change violates the configured registry compatibility mode."""

    error_code = "ingestion_schema_incompatible"


class SourceUnavailableError(IngestionError):
    """An upstream data source was unavailable or exhausted its retry budget."""


class IngestionTransportError(SourceUnavailableError, RuntimeError):
    """A network or broker failure that is normally safe to retry."""

    error_code = "ingestion_transport_error"
    default_retryable = True


class IngestionRateLimitError(IngestionTransportError):
    error_code = "ingestion_rate_limit_exceeded"


class IngestionNotFoundError(IngestionError):
    error_code = "ingestion_resource_not_found"


class IngestionValidationError(InvalidInputError):
    """Compatibility base for caller-supplied validation failures."""


class SchemaDecodeError(IngestionValidationError):
    error_code = "ingestion_payload_decode_failed"


@contextmanager
def record_context(
    source: str,
    record: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Translate raw record-construction failures into :class:`RecordValidationError`.

    Wraps the exception types in :data:`RECORD_ERRORS`, attaching *source* and
    the offending *record* as context. Exceptions already in this hierarchy pass
    through unchanged so a more specific type is never downgraded.

    Usage::

        with record_context("horizon_streamer._to_trade", record):
            return Trade(...)
    """
    try:
        yield
    except IngestionError:
        raise
    except RECORD_ERRORS as exc:
        raise RecordValidationError(
            f"{source}: could not build record — {exc}",
            source=source,
            reason=str(exc),
            raw=record,
        ) from exc
