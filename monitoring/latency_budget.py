"""Per-stage latency budgets with SLO-burn alerting (Issue #921).

The detection pipeline is a chain of five stages::

    ingestion -> streaming -> feature -> scoring -> alert

Until now only the end-to-end latency (``ledgerlens_e2e_latency_seconds``)
and a few stage-local timers were exported, so when the overall detection
latency SLO degraded there was no single view of *which* stage consumed the
budget. This module adds:

* :data:`STAGE_LATENCY_SLOS` -- an explicit latency budget per stage. The
  budgets sum exactly to :data:`E2E_DETECTION_LATENCY_SLO_SECONDS` (the
  documented p99 < 10s end-to-end target, see ``docs/load_testing.md``);
  :func:`validate_stage_slos` enforces this and runs at import time.
* ``ledgerlens_stage_latency_seconds{stage}`` -- one histogram, tagged
  consistently by ``stage``, that every stage emits into. Because every
  stage records into the same metric, ``sum by (stage)`` of the histogram
  sums adds up to the end-to-end total on the dashboard
  (``monitoring/grafana/dashboards/latency_budget.json``).
* :class:`LatencyBudgetTracker` -- in-process rolling windows per stage
  that compute an SLO *burn rate* (observed fraction of events over the
  stage budget divided by the allowed error budget). Burn rate is computed
  **per stage against that stage's own budget**, so a slow stage raises
  only its own alert and never pushes a neighbouring stage over.
  Alerts are routed through :class:`alerts.router.AlertRouter`.

See ``docs/latency_slos.md`` for the SLO values and their rationale.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from alerts.router import AlertRouter, RouteDestination
from utils.logging import get_logger

logger = get_logger(__name__)

# Documented end-to-end detection latency target (p99), in seconds.
E2E_DETECTION_LATENCY_SLO_SECONDS = 10.0

# Ordered pipeline stages. Order matters for dashboard/report rendering.
PIPELINE_LATENCY_STAGES: tuple[str, ...] = (
    "ingestion",
    "streaming",
    "feature",
    "scoring",
    "alert",
)

SLO_BURN_DETECTOR = "latency_slo_burn"


@dataclass(frozen=True)
class StageSLO:
    """Latency budget for one pipeline stage.

    Attributes:
        stage: Stage name (one of :data:`PIPELINE_LATENCY_STAGES`).
        budget_seconds: p99 latency the stage may contribute.
        objective: Fraction of events that must finish within budget.
    """

    stage: str
    budget_seconds: float
    objective: float = 0.99

    @property
    def error_budget(self) -> float:
        return 1.0 - self.objective


# Rationale for each value lives in docs/latency_slos.md.
STAGE_LATENCY_SLOS: dict[str, StageSLO] = {
    "ingestion": StageSLO("ingestion", 2.0),
    "streaming": StageSLO("streaming", 1.5),
    "feature": StageSLO("feature", 2.5),
    "scoring": StageSLO("scoring", 3.0),
    "alert": StageSLO("alert", 1.0),
}


def validate_stage_slos(
    slos: Mapping[str, StageSLO],
    e2e_target_seconds: float = E2E_DETECTION_LATENCY_SLO_SECONDS,
) -> None:
    """Raise ``ValueError`` unless *slos* cover every stage and sum to target."""
    missing = [s for s in PIPELINE_LATENCY_STAGES if s not in slos]
    if missing:
        raise ValueError(f"stage latency SLOs missing for stage(s): {missing}")
    for name, slo in slos.items():
        if slo.budget_seconds <= 0:
            raise ValueError(f"stage {name!r} budget must be positive")
        if not 0.0 < slo.objective < 1.0:
            raise ValueError(f"stage {name!r} objective must be in (0, 1)")
    total = sum(slo.budget_seconds for slo in slos.values())
    if not math.isclose(total, e2e_target_seconds, abs_tol=1e-9):
        raise ValueError(
            f"stage latency budgets sum to {total}s, expected the end-to-end "
            f"detection latency target of {e2e_target_seconds}s"
        )


validate_stage_slos(STAGE_LATENCY_SLOS)


try:
    from prometheus_client import Counter, Gauge, Histogram

    stage_latency_seconds = Histogram(
        "ledgerlens_stage_latency_seconds",
        "Per-stage contribution to end-to-end detection latency",
        ["stage"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0),
    )
    stage_latency_budget_breaches_total = Counter(
        "ledgerlens_stage_latency_budget_breaches_total",
        "Events whose stage latency exceeded that stage's SLO budget",
        ["stage"],
    )
    stage_latency_slo_burn_rate = Gauge(
        "ledgerlens_stage_latency_slo_burn_rate",
        "Rolling SLO burn rate per stage (1.0 = consuming error budget exactly on pace)",
        ["stage"],
    )
    stage_latency_budget_seconds = Gauge(
        "ledgerlens_stage_latency_budget_seconds",
        "Configured latency budget per stage",
        ["stage"],
    )
    for _slo in STAGE_LATENCY_SLOS.values():
        stage_latency_budget_seconds.labels(stage=_slo.stage).set(_slo.budget_seconds)
except Exception:  # pragma: no cover - prometheus optional in tests
    stage_latency_seconds = None  # type: ignore[assignment]
    stage_latency_budget_breaches_total = None  # type: ignore[assignment]
    stage_latency_slo_burn_rate = None  # type: ignore[assignment]
    stage_latency_budget_seconds = None  # type: ignore[assignment]


@dataclass
class SLOBurnAlert:
    """A fired per-stage SLO-burn alert and where it was routed."""

    stage: str
    burn_rate: float
    alert: dict[str, Any]
    destinations: list[RouteDestination] = field(default_factory=list)


class LatencyBudgetTracker:
    """Records per-stage latency and raises SLO-burn alerts per stage.

    Parameters
    ----------
    router:
        ``AlertRouter`` used to route SLO-burn alerts. ``None`` computes
        alerts without routing them.
    dispatch:
        Optional callback ``(alert, destinations)`` performing delivery.
    slos:
        Stage budgets; defaults to :data:`STAGE_LATENCY_SLOS`.
    window_size:
        Number of most recent observations per stage used for burn rate.
    burn_rate_threshold:
        Burn rate at or above which a stage alert fires (2.0 = consuming
        error budget at twice the sustainable pace).
    min_samples:
        Minimum observations in a stage window before it can alert, so a
        single cold-start outlier cannot page anyone.
    """

    def __init__(
        self,
        router: AlertRouter | None = None,
        dispatch: Callable[[dict[str, Any], list[RouteDestination]], None] | None = None,
        slos: Mapping[str, StageSLO] | None = None,
        window_size: int = 1000,
        burn_rate_threshold: float = 2.0,
        min_samples: int = 20,
    ) -> None:
        self._slos = dict(slos or STAGE_LATENCY_SLOS)
        validate_stage_slos(self._slos, sum(s.budget_seconds for s in self._slos.values()))
        self._router = router
        self._dispatch = dispatch
        self._burn_rate_threshold = burn_rate_threshold
        self._min_samples = min_samples
        self._lock = threading.Lock()
        self._windows: dict[str, deque[float]] = {s: deque(maxlen=window_size) for s in self._slos}
        self._totals: dict[str, float] = {s: 0.0 for s in self._slos}
        self._counts: dict[str, int] = {s: 0 for s in self._slos}
        self._e2e_total = 0.0
        self._e2e_count = 0

    # ------------------------------------------------------------------ record

    def record(self, stage: str, seconds: float) -> None:
        """Record one observation of *stage* taking *seconds*."""
        slo = self._slos.get(stage)
        if slo is None:
            raise KeyError(f"unknown pipeline stage {stage!r}")
        seconds = max(0.0, float(seconds))
        with self._lock:
            self._windows[stage].append(seconds)
            self._totals[stage] += seconds
            self._counts[stage] += 1
        if stage_latency_seconds is not None:
            stage_latency_seconds.labels(stage=stage).observe(seconds)
            if seconds > slo.budget_seconds:
                stage_latency_budget_breaches_total.labels(stage=stage).inc()

    def record_event(self, stage_latencies: Mapping[str, float]) -> float:
        """Record every stage of one event; returns its end-to-end latency."""
        for stage, seconds in stage_latencies.items():
            self.record(stage, seconds)
        e2e = sum(max(0.0, float(v)) for v in stage_latencies.values())
        with self._lock:
            self._e2e_total += e2e
            self._e2e_count += 1
        return e2e

    @contextmanager
    def time_stage(self, stage: str) -> Generator[None, None, None]:
        """Context manager recording the wall-clock time of the wrapped block."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.record(stage, time.monotonic() - t0)

    # --------------------------------------------------------------- analysis

    def burn_rate(self, stage: str) -> float:
        """Fraction of the stage window over budget, divided by error budget."""
        slo = self._slos[stage]
        with self._lock:
            window = list(self._windows[stage])
        if not window:
            return 0.0
        bad = sum(1 for v in window if v > slo.budget_seconds)
        return (bad / len(window)) / slo.error_budget

    def breakdown(self) -> dict[str, Any]:
        """Per-stage mean latency and share of the end-to-end total.

        Mirrors the dashboard: ``sum(stage means)`` equals the mean end-to-end
        latency of events recorded via :meth:`record_event`.
        """
        with self._lock:
            means = {
                s: (self._totals[s] / self._counts[s]) if self._counts[s] else 0.0
                for s in self._slos
            }
            e2e_mean = self._e2e_total / self._e2e_count if self._e2e_count else 0.0
        total = sum(means.values())
        return {
            "stages": {
                s: {
                    "mean_seconds": means[s],
                    "budget_seconds": self._slos[s].budget_seconds,
                    "share_of_total": (means[s] / total) if total else 0.0,
                    "burn_rate": self.burn_rate(s),
                }
                for s in self._slos
            },
            "stage_sum_seconds": total,
            "e2e_mean_seconds": e2e_mean,
        }

    def check_slo_burn(self) -> list[SLOBurnAlert]:
        """Evaluate every stage and route an alert for each one burning budget."""
        fired: list[SLOBurnAlert] = []
        for stage, slo in self._slos.items():
            rate = self.burn_rate(stage)
            if stage_latency_slo_burn_rate is not None:
                stage_latency_slo_burn_rate.labels(stage=stage).set(rate)
            with self._lock:
                samples = len(self._windows[stage])
                window = sorted(self._windows[stage])
            if samples < self._min_samples or rate < self._burn_rate_threshold:
                continue
            p99 = window[min(len(window) - 1, math.ceil(0.99 * len(window)) - 1)]
            alert = {
                "alert_type": "latency_slo_burn",
                "detectors": [SLO_BURN_DETECTOR],
                "asset_pair": f"pipeline/{stage}",
                "stage": stage,
                "burn_rate": rate,
                "budget_seconds": slo.budget_seconds,
                "observed_p99_seconds": p99,
                "objective": slo.objective,
                "samples": samples,
            }
            destinations = self._router.route(alert) if self._router is not None else []
            logger.warning(
                "latency SLO burn: stage=%s burn_rate=%.2f p99=%.3fs budget=%.3fs "
                "samples=%d destinations=%s",
                stage,
                rate,
                p99,
                slo.budget_seconds,
                samples,
                [d.channel for d in destinations],
            )
            if self._dispatch is not None and destinations:
                self._dispatch(alert, destinations)
            fired.append(SLOBurnAlert(stage, rate, alert, list(destinations)))
        return fired
