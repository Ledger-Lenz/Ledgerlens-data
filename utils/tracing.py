"""OpenTelemetry distributed tracing helpers for LedgerLens.

Provides a single :func:`get_tracer` entry point that returns a configured
OTel tracer pointing at the OTLP endpoint defined by the
``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable.

Sampling is controlled by ``OTEL_SAMPLING_RATE`` (default 0.1 = 10 %).

Security: spans must NOT include raw wallet addresses or trade amounts.
Use :func:`hash_span_id` to produce hashed identifiers for span attributes.

Kafka trace context is propagated using W3C TraceContext headers via
:func:`inject_trace_context` and :func:`extract_trace_context`.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.propagate import extract, inject
    from opentelemetry.propagators.b3 import B3Format
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False

_provider_initialized = False


def _ensure_provider() -> None:
    global _provider_initialized
    if _provider_initialized or not _OTEL_AVAILABLE:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    sampling_rate = float(os.getenv("OTEL_SAMPLING_RATE", "0.1"))

    resource = Resource.create({"service.name": "ledgerlens"})
    sampler = ParentBased(root=TraceIdRatioBased(sampling_rate))
    provider = TracerProvider(resource=resource, sampler=sampler)

    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _provider_initialized = True
    logger.info(
        "OTel tracer provider initialized: endpoint=%s sampling_rate=%.2f",
        endpoint,
        sampling_rate,
    )


def get_tracer(name: str):
    """Return a configured OTel tracer for *name*.

    Falls back to a no-op tracer when ``opentelemetry-sdk`` is not installed,
    so instrumented code is safe to run without the OTel dependency.
    """
    if not _OTEL_AVAILABLE:
        return _NoopTracer()
    _ensure_provider()
    return trace.get_tracer(name)


def hash_span_id(value: str) -> str:
    """Return a SHA-256 hex digest of *value* for use as a span attribute.

    Spans must not expose raw wallet addresses or trade amounts; pass them
    through this function first.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def inject_trace_context(headers: dict[str, str]) -> None:
    """Inject W3C TraceContext headers into *headers* for Kafka propagation."""
    if not _OTEL_AVAILABLE:
        return
    inject(headers)


def extract_trace_context(headers: dict[str, str]) -> Any:
    """Extract W3C TraceContext from *headers* received from Kafka.

    Returns an OTel context object suitable for passing to ``tracer.start_as_current_span``.
    """
    if not _OTEL_AVAILABLE:
        return None
    return extract(headers)


# ---------------------------------------------------------------------------
# No-op fallback so callers work without opentelemetry-sdk installed
# ---------------------------------------------------------------------------

class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def set_attribute(self, *_):
        pass

    def record_exception(self, *_):
        pass


class _NoopTracer:
    def start_as_current_span(self, name: str, **_kwargs):
        return _NoopSpan()

    def start_span(self, name: str, **_kwargs):
        return _NoopSpan()
