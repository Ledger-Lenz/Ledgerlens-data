"""API package for the forensic reporting service.

This package exposes the HTTP surface of the application. In addition to
wiring up the Flask app (see :mod:`api.app`), it provides the shared
request-cost accounting primitives used to protect expensive endpoints
(forensic report generation, backtests) from monopolizing shared compute.

The cost model is intentionally simple and deterministic so that it can be
documented and reasoned about:

* Every request is assigned a ``cost`` (in abstract "cost units"). Cheap
  read-only endpoints cost ``1`` unit; expensive endpoints declare a higher
  cost via :func:`estimate_cost`.
* Each tenant has a ``budget`` (cost units) per rolling window. Accumulated
  cost is tracked per tenant in :data:`_TENANT_USAGE`.
* When a tenant's accumulated cost would exceed its budget, the request is
  rejected with :class:`QuotaExceeded` which carries ``retry_after`` seconds
  so callers get actionable guidance instead of being queued indefinitely.

Quota consumption is exported as a Prometheus-style counter
(:data:`QUOTA_CONSUMPTION`) so operators can alert on tenants approaching
or exhausting their budget.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_WINDOW_SECONDS",
    "EXPENSIVE_ENDPOINTS",
    "QuotaExceeded",
    "QuotaTracker",
    "QUOTA_CONSUMPTION",
    "estimate_cost",
    "get_tracker",
]

#: Default per-tenant budget, in cost units, per rolling window.
DEFAULT_BUDGET: int = 1000

#: Length of the rolling accounting window, in seconds.
DEFAULT_WINDOW_SECONDS: int = 3600

#: Cost assigned to a plain (cheap) request.
BASE_COST: int = 1

#: Endpoints that trigger expensive downstream work. The value is the cost
#: estimate for a single request to that endpoint. Keys are matched against
#: the request path (exact match or prefix match for trailing ``*``).
EXPENSIVE_ENDPOINTS: Dict[str, int] = {
    # Forensic report generation (reporting/export_service.py).
    "/api/reports/forensic": 50,
    "/api/reports/forensic/*": 50,
    "/api/reports/export": 50,
    # Backtests.
    "/api/backtests": 25,
    "/api/backtests/*": 25,
}


def estimate_cost(path: str) -> int:
    """Return the cost estimate (in cost units) for a request ``path``.

    Expensive endpoints listed in :data:`EXPENSIVE_ENDPOINTS` return their
    declared cost; everything else falls back to :data:`BASE_COST`.
    """
    if not path:
        return BASE_COST
    normalized = path.rstrip("/") or "/"
    for pattern, cost in EXPENSIVE_ENDPOINTS.items():
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            if normalized.startswith(prefix.rstrip("/")):
                return cost
        elif normalized == pattern.rstrip("/"):
            return cost
    return BASE_COST


class QuotaExceeded(Exception):
    """Raised when a tenant has exhausted its request-cost budget.

    Carries ``retry_after`` (seconds) so the API layer can return a clear,
    actionable error (HTTP 429 with a ``Retry-After`` header) instead of
    queuing the request indefinitely.
    """

    def __init__(self, tenant: str, retry_after: int, budget: int) -> None:
        self.tenant = tenant
        self.retry_after = max(1, int(retry_after))
        self.budget = budget
        super().__init__(
            f"Tenant '{tenant}' exceeded its request-cost budget of "
            f"{budget} units; retry after {self.retry_after}s."
        )


@dataclass
class _TenantUsage:
    """Accumulated cost for a single tenant within the current window."""

    cost: int = 0
    window_start: float = field(default_factory=time.monotonic)


class QuotaTracker:
    """Thread-safe per-tenant request-cost accounting.

    Usage is tracked per tenant over a rolling window. When a request would
    push a tenant over its budget, :meth:`charge` raises
    :class:`QuotaExceeded` and leaves the tenant's usage untouched, so other
    tenants are never affected by one tenant exhausting its quota.
    """

    def __init__(
        self,
        budget: int = DEFAULT_BUDGET,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.budget = budget
        self.window_seconds = window_seconds
        self._usage: Dict[str, _TenantUsage] = {}
        self._lock = threading.Lock()

    def _reset_if_expired(self, usage: _TenantUsage, now: float) -> None:
        if now - usage.window_start >= self.window_seconds:
            usage.cost = 0
            usage.window_start = now

    def charge(self, tenant: str, cost: int) -> int:
        """Charge ``cost`` units to ``tenant`` and return remaining budget.

        Raises :class:`QuotaExceeded` (with retry-after guidance) when the
        charge would exceed the tenant's budget.
        """
        now = time.monotonic()
        with self._lock:
            usage = self._usage.setdefault(tenant, _TenantUsage())
            self._reset_if_expired(usage, now)
            if usage.cost + cost > self.budget:
                retry_after = int(
                    self.window_seconds - (now - usage.window_start)
                )
                QUOTA_CONSUMPTION.labels(tenant=tenant, outcome="rejected").inc()
                raise QuotaExceeded(tenant, retry_after, self.budget)
            usage.cost += cost
            QUOTA_CONSUMPTION.labels(tenant=tenant, outcome="allowed").inc(cost)
            return self.budget - usage.cost

    def remaining(self, tenant: str) -> int:
        """Return the tenant's remaining budget without charging it."""
        now = time.monotonic()
        with self._lock:
            usage = self._usage.get(tenant)
            if usage is None:
                return self.budget
            self._reset_if_expired(usage, now)
            return max(0, self.budget - usage.cost)


class _Counter:
    """Minimal Prometheus-style counter used for quota metrics.

    Kept dependency-free so the API package does not require the Prometheus
    client at import time. If the real client is available it is used
    transparently via :meth:`labels`.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def labels(self, **labels) -> "_Counter":
        key = tuple(sorted(labels.items()))
        return _LabeledCounter(self, key)

    def inc(self, amount: float = 1.0) -> None:
        self.labels().inc(amount)

    def value(self, **labels) -> float:
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def _add(self, key: tuple, amount: float) -> None:
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount


class _LabeledCounter:
    def __init__(self, parent: _Counter, key: tuple) -> None:
        self._parent = parent
        self._key = key

    def inc(self, amount: float = 1.0) -> None:
        self._parent._add(self._key, amount)


#: Metric tracking quota consumption per tenant.
#: Labels: ``tenant`` and ``outcome`` (``allowed`` or ``rejected``).
QUOTA_CONSUMPTION = _Counter(
    "api_quota_consumption_total",
    "Request-cost units consumed per tenant, by outcome.",
)

_tracker: Optional[QuotaTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> QuotaTracker:
    """Return the process-wide :class:`QuotaTracker` singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = QuotaTracker()
    return _tracker
