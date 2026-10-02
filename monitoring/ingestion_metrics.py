"""Prometheus emitters for ingestion throughput, latency, and failures.

The module exposes a process-wide emitter for production call sites and an
``IngestionMetricsEmitter`` class that accepts a custom registry for isolated
consumers. Metric labels deliberately contain only bounded values: source,
pipeline stage, and exception class.

In addition to throughput, the emitter tracks end-to-end freshness: the
latency from the originating on-chain event time to the moment an alert is
dispatched. A source event timestamp is propagated through the pipeline
(ingestion -> feature -> scoring -> alert) so that per-stage and combined
freshness can be attributed for root-causing.
"""

from __future__ import annotations

import time
from typing import Any

try:
    from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional observability dependency
    REGISTRY = None
    CollectorRegistry = Any  # type: ignore[misc,assignment]
    _PROM_AVAILABLE = False


# Ordered pipeline stages used for freshness attribution. The order defines the
# sequence in which a source event timestamp flows through the pipeline.
FRESHNESS_STAGES = ("ingestion", "feature", "scoring", "alert")

# Default end-to-end freshness SLO in seconds. A regression beyond this budget
# raises the freshness SLO gauge so alerting rules can fire.
DEFAULT_FRESHNESS_SLO_SECONDS = 60.0


class IngestionMetricsEmitter:
    """Emit low-cardinality metrics for any ingestion source and stage."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or REGISTRY
        self.records = None
        self.failures = None
        self.throughput = None
        self.duration = None
        self.last_success = None
        self.freshness = None
        self.end_to_end_freshness = None
        self.freshness_slo_seconds = None
        self.freshness_slo_breached = None
        if not _PROM_AVAILABLE:
            return

        self.records = self._metric(
            Counter,
            "ledgerlens_ingestion_records_total",
            "Total records successfully emitted by ingestion",
            ["source", "stage"],
        )
        self.failures = self._metric(
            Counter,
            "ledgerlens_ingestion_failures_total",
            "Total ingestion failures grouped by exception type",
            ["source", "stage", "error_type"],
        )
        self.throughput = self._metric(
            Gauge,
            "ledgerlens_ingestion_throughput_records_per_second",
            "Most recently observed ingestion batch throughput",
            ["source", "stage"],
        )
        self.duration = self._metric(
            Histogram,
            "ledgerlens_ingestion_duration_seconds",
            "Ingestion operation duration in seconds",
            ["source", "stage"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
        )
        self.last_success = self._metric(
            Gauge,
            "ledgerlens_ingestion_last_success_timestamp_seconds",
            "Unix timestamp of the last successful ingestion operation",
            ["source", "stage"],
        )
        self.freshness = self._metric(
            Histogram,
            "ledgerlens_pipeline_freshness_seconds",
            "Per-stage latency from source event time to stage completion",
            ["source", "stage"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
        )
        self.end_to_end_freshness = self._metric(
            Histogram,
            "ledgerlens_pipeline_end_to_end_freshness_seconds",
            "End-to-end data-to-alert latency from source event time to alert dispatch",
            ["source"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
        )
        self.freshness_slo_seconds = self._metric(
            Gauge,
            "ledgerlens_pipeline_freshness_slo_seconds",
            "Configured end-to-end freshness SLO budget in seconds",
            ["source"],
        )
        self.freshness_slo_breached = self._metric(
            Gauge,
            "ledgerlens_pipeline_freshness_slo_breached",
            "1 when the last end-to-end freshness observation exceeded the SLO",
            ["source"],
        )

    def _metric(self, metric_type, name: str, description: str, labels: list[str], **kwargs):
        """Create a collector, reusing one already present in the same registry."""
        existing = getattr(self._registry, "_names_to_collectors", {}).get(name)
        if existing is not None:
            return existing
        return metric_type(name, description, labels, registry=self._registry, **kwargs)

    def emit_success(
        self,
        source: str,
        *,
        stage: str = "fetch",
        record_count: int = 1,
        duration_seconds: float | None = None,
    ) -> None:
        """Record a successful operation and its observed batch throughput."""
        count = max(0, int(record_count))
        if self.records is not None:
            self.records.labels(source=source, stage=stage).inc(count)
        if self.last_success is not None:
            self.last_success.labels(source=source, stage=stage).set(time.time())
        if duration_seconds is not None:
            duration = max(0.0, float(duration_seconds))
            if self.duration is not None:
                self.duration.labels(source=source, stage=stage).observe(duration)
            if self.throughput is not None:
                rate = count / duration if duration > 0 else float(count)
                self.throughput.labels(source=source, stage=stage).set(rate)

    def emit_failure(
        self,
        source: str,
        error: BaseException | type[BaseException] | str,
        *,
        stage: str = "fetch",
        duration_seconds: float | None = None,
    ) -> None:
        """Record a failed operation without exposing exception messages as labels."""
        if isinstance(error, str):
            error_type = error
        elif isinstance(error, type):
            error_type = error.__name__
        else:
            error_type = type(error).__name__
        if self.failures is not None:
            self.failures.labels(source=source, stage=stage, error_type=error_type).inc()
        if duration_seconds is not None and self.duration is not None:
            self.duration.labels(source=source, stage=stage).observe(
                max(0.0, float(duration_seconds))
            )

    def emit_freshness(
        self,
        source: str,
        *,
        stage: str,
        event_timestamp: float,
        observed_at: float | None = None,
    ) -> float:
        """Record per-stage freshness for a propagated source event timestamp.

        ``event_timestamp`` is the originating on-chain event time (Unix
        seconds) carried through the pipeline. ``observed_at`` defaults to the
        current time and represents when the given ``stage`` completed. Returns
        the computed latency in seconds so callers can chain stages.
        """
        now = time.time() if observed_at is None else float(observed_at)
        latency = max(0.0, now - float(event_timestamp))
        if self.freshness is not None:
            self.freshness.labels(source=source, stage=stage).observe(latency)
        return latency

    def emit_end_to_end_freshness(
        self,
        source: str,
        *,
        event_timestamp: float,
        alert_timestamp: float | None = None,
        slo_seconds: float = DEFAULT_FRESHNESS_SLO_SECONDS,
    ) -> float:
        """Record combined data-to-alert latency and evaluate the freshness SLO.

        ``event_timestamp`` is the source event time and ``alert_timestamp`` is
        the alert-dispatch time (defaults to now). The end-to-end latency is
        observed and compared against ``slo_seconds``; the SLO breach gauge is
        set so alerting rules can fire on a freshness regression. Returns the
        end-to-end latency in seconds.
        """
        dispatched = time.time() if alert_timestamp is None else float(alert_timestamp)
        latency = max(0.0, dispatched - float(event_timestamp))
        if self.end_to_end_freshness is not None:
            self.end_to_end_freshness.labels(source=source).observe(latency)
        if self.freshness_slo_seconds is not None:
            self.freshness_slo_seconds.labels(source=source).set(max(0.0, float(slo_seconds)))
        if self.freshness_slo_breached is not None:
            breached = 1.0 if latency > float(slo_seconds) else 0.0
            self.freshness_slo_breached.labels(source=source).set(breached)
        return latency


INGESTION_METRICS = IngestionMetricsEmitter()


def emit_ingestion_success(
    source: str,
    *,
    stage: str = "fetch",
    record_count: int = 1,
    duration_seconds: float | None = None,
) -> None:
    """Emit success through the process-wide ingestion metrics collector."""
    INGESTION_METRICS.emit_success(
        source,
        stage=stage,
        record_count=record_count,
        duration_seconds=duration_seconds,
    )


def emit_ingestion_failure(
    source: str,
    error: BaseException | type[BaseException] | str,
    *,
    stage: str = "fetch",
    duration_seconds: float | None = None,
) -> None:
    """Emit failure through the process-wide ingestion metrics collector."""
    INGESTION_METRICS.emit_failure(
        source,
        error,
        stage=stage,
        duration_seconds=duration_seconds,
    )


def emit_stage_freshness(
    source: str,
    *,
    stage: str,
    event_timestamp: float,
    observed_at: float | None = None,
) -> float:
    """Emit per-stage freshness through the process-wide collector."""
    return INGESTION_METRICS.emit_freshness(
        source,
        stage=stage,
        event_timestamp=event_timestamp,
        observed_at=observed_at,
    )


def emit_end_to_end_freshness(
    source: str,
    *,
    event_timestamp: float,
    alert_timestamp: float | None = None,
    slo_seconds: float = DEFAULT_FRESHNESS_SLO_SECONDS,
) -> float:
    """Emit combined data-to-alert freshness through the process-wide collector."""
    return INGESTION_METRICS.emit_end_to_end_freshness(
        source,
        event_timestamp=event_timestamp,
        alert_timestamp=alert_timestamp,
        slo_seconds=slo_seconds,
    )
