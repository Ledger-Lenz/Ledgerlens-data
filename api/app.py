"""FastAPI REST API exposing LedgerLens wallet risk scores.

Endpoints:
    GET /v1/wallets/{address}/scores   — paginated risk score history
    GET /v1/wallets/{address}/latest   — latest score + top-3 features
    GET /v1/health                     — liveness / readiness check

Idempotency contract
--------------------
Write endpoints (e.g. manual review submissions, threshold overrides) accept
an idempotency key via the ``Idempotency-Key`` request header or an
``idempotency_key`` body field. When a key is supplied, the first request is
processed and its response is stored for a configurable TTL window
(``config.API_IDEMPOTENCY_TTL_SECONDS``). A retry that reuses the same key
within that window returns the original stored response verbatim and does
*not* reprocess the underlying write, so a client retry after a network
timeout cannot produce a duplicate write. Keys are scoped per authenticated
tenant. After the TTL expires the key is forgotten and a subsequent request
with the same key is treated as a new write.

Request-cost accounting
-----------------------
Expensive endpoints (forensic report generation, backtests) declare a
per-request cost estimate. Costs are accumulated against a per-tenant budget
(``config.API_TENANT_COST_BUDGET``) over a rolling window. When a tenant
exceeds its budget the request is rejected with HTTP 429 and a ``Retry-After``
header indicating when budget will be available again, rather than queuing
indefinitely or degrading other tenants. Per-tenant consumption is exposed via
``_cost_accountant.metrics()``.
"""

import re
import threading
import time
from contextlib import asynccontextmanager

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select

from config import config
from config.contracts import validate_mode
from detection.persistence import RiskScoreRecord, get_session_factory
from detection.risk_score_store import RiskScoreStore
from detection.shap_explainer import ShapExplainer
from streaming.health import HealthStatus, get_health_registry
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Stellar address validation
# ---------------------------------------------------------------------------
_STELLAR_ACCOUNT_RE = re.compile(r"^G[A-Z2-7]{55}$")


def _validate_stellar_address(address: str) -> str:
    if not _STELLAR_ACCOUNT_RE.match(address):
        raise HTTPException(status_code=400, detail="Invalid Stellar account address")
    return address


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    for hashed in config.API_KEYS:
        if bcrypt.checkpw(api_key.encode(), hashed.encode()):
            return api_key
    raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Tenant-scoped rate limiting
# ---------------------------------------------------------------------------
# Global limits (above) protect the API as a whole. Per-tenant limits below
# ensure a single high-volume tenant cannot degrade availability for others.
# Each tenant gets its own token bucket, so limiter state is fully isolated
# and no counters are shared across tenants.


class _TokenBucket:
    """Thread-safe token bucket for a single tenant."""

    __slots__ = ("capacity", "refill_per_sec", "tokens", "updated_at", "_lock")

    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> bool:
        """Try to consume ``amount`` tokens; return True if allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            if elapsed > 0:
                self.tokens = min(
                    self.capacity, self.tokens + elapsed * self.refill_per_sec
                )
                self.updated_at = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

    def utilization(self) -> float:
        """Fraction of the bucket currently consumed (0.0–1.0)."""
        with self._lock:
            if self.capacity <= 0:
                return 0.0
            return max(0.0, min(1.0, 1.0 - (self.tokens / self.capacity)))


class TenantRateLimiter:
    """Per-tenant token-bucket limiter with isolated state per tenant.

    Limits are resolved from ``config/tenant_config.py`` (backed by
    ``config/tenants.yaml``) so operators can set per-tenant overrides.
    """

    def __init__(self, default_rpm: int) -> None:
        self._default_rpm = int(default_rpm)
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()
        # Metrics: per-tenant utilization + throttling counters.
        self._throttled: dict[str, int] = {}
        self._allowed: dict[str, int] = {}

    def _limit_rpm(self, tenant_id: str) -> int:
        """Resolve the effective RPM for a tenant (override or default)."""
        try:
            from config.tenant_config import get_tenant_config

            tenant = get_tenant_config(tenant_id)
            override = getattr(tenant, "rate_limit_rpm", None)
            if override:
                return int(override)
        except Exception:
            # Unknown tenant or config unavailable: fall back to default.
            pass
        return self._default_rpm

    def _bucket_for(self, tenant_id: str) -> _TokenBucket:
        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                rpm = self._limit_rpm(tenant_id)
                # Capacity = one minute of burst; refill at the configured RPM.
                bucket = _TokenBucket(capacity=rpm, refill_per_sec=rpm / 60.0)
                self._buckets[tenant_id] = bucket
            return bucket

    def check(self, tenant_id: str) -> bool:
        """Consume one token for ``tenant_id``; return True if allowed."""
        bucket = self._bucket_for(tenant_id)
        allowed = bucket.consume(1.0)
        with self._lock:
            if allowed:
                self._allowed[tenant_id] = self._allowed.get(tenant_id, 0) + 1
            else:
                self._throttled[tenant_id] = self._throttled.get(tenant_id, 0) + 1
        if not allowed:
            logger.warning(
                "tenant rate limit exceeded",
                extra={"tenant_id": tenant_id, "utilization": bucket.utilization()},
            )
        return allowed

    def metrics(self) -> dict:
        """Snapshot of per-tenant utilization and throttling events."""
        with self._lock:
            tenants = set(self._buckets) | set(self._throttled) | set(self._allowed)
            return {
                "tenants": {
                    tid: {
                        "utilization": self._buckets[tid].utilization()
                        if tid in self._buckets
                        else 0.0,
                        "allowed": self._allowed.get(tid, 0),
                        "throttled": self._throttled.get(tid, 0),
                    }
                    for tid in tenants
                }
            }


_tenant_limiter = TenantRateLimiter(default_rpm=config.API_RATE_LIMIT_RPM)


# ---------------------------------------------------------------------------
# Request-cost accounting for expensive endpoints
# ---------------------------------------------------------------------------
# Expensive endpoints (forensic report generation, backtests) declare a
# per-request cost estimate. Costs accumulate against a per-tenant budget over
# a rolling window. When a tenant exhausts its budget the request is rejected
# with HTTP 429 and a ``Retry-After`` header, so a single tenant cannot
# monopolize shared compute or degrade other tenants.

# Cost estimates (in abstract "cost units") per expensive operation. These are
# calibrated from observed downstream work: a forensic report fans out across
# the full score history and SHAP explanations, while a backtest replays a
# window of scores. Values are intentionally coarse and documented here so the
# methodology is auditable.
COST_FORENSIC_REPORT = 25.0
COST_BACKTEST = 10.0


class QuotaExceeded(Exception):
    """Raised when a tenant's request-cost budget is exhausted."""

    def __init__(self, tenant_id: str, retry_after: int) -> None:
        self.tenant_id = tenant_id
        self.retry_after = max(1, int(retry_after))
        super().__init__(
            f"Request-cost quota exceeded for tenant {tenant_id!r}; "
            f"retry after {self.retry_after}s"
        )


class RequestCostAccountant:
    """Per-tenant request-cost budget with isolated state per tenant.

    Each tenant has an independent rolling window of accumulated cost. A
    request is admitted only if ``accumulated + cost <= budget``. When the
    budget is exhausted the caller receives a ``QuotaExceeded`` carrying
    retry-after guidance derived from the window length.
    """

    def __init__(self, budget: float, window_seconds: float) -> None:
        self._budget = float(budget)
        self._window = float(window_seconds)
        # tenant_id -> (window_start_monotonic, accumulated_cost)
        self._state: dict[str, list] = {}
        self._lock = threading.Lock()
        # Metrics: per-tenant consumption + rejection counters.
        self._consumed: dict[str, float] = {}
        self._rejected: dict[str, int] = {}

    def _budget_for(self, tenant_id: str) -> float:
        """Resolve the effective budget for a tenant (override or default)."""
        try:
            from config.tenant_config import get_tenant_config

            tenant = get_tenant_config(tenant_id)
            override = getattr(tenant, "cost_budget", None)
            if override:
                return float(override)
        except Exception:
            # Unknown tenant or config unavailable: fall back to default.
            pass
        return self._budget

    def _roll(self, tenant_id: str, now: float) -> list:
        state = self._state.get(tenant_id)
        if state is None or now - state[0] >= self._window:
            state = [now, 0.0]
            self._state[tenant_id] = state
        return state

    def charge(self, tenant_id: str, cost: float) -> None:
        """Charge ``cost`` to ``tenant_id`` or raise ``QuotaExceeded``."""
        now = time.monotonic()
        with self._lock:
            state = self._roll(tenant_id, now)
            budget = self._budget_for(tenant_id)
            if state[1] + cost > budget:
                self._rejected[tenant_id] = self._rejected.get(tenant_id, 0) + 1
                retry_after = self._window - (now - state[0])
                raise QuotaExceeded(tenant_id, retry_after)
            state[1] += cost
            self._consumed[tenant_id] = self._consumed.get(tenant_id, 0.0) + cost
        logger.info(
            "request cost charged",
            extra={"tenant_id": tenant_id, "cost": cost},
        )

    def metrics(self) -> dict:
        """Snapshot of per-tenant cost consumption and rejections."""
        with self._lock:
            tenants = set(self._state) | set(self._consumed) | set(self._rejected)
            return {
                "tenants": {
                    tid: {
                        "consumed": self._consumed.get(tid, 0.0),
                        "budget": self._budget_for(tid),
                        "rejected": self._rejected.get(tid, 0),
                    }
                    for tid in tenants
                }
            }


_cost_accountant = RequestCostAccountant(
    budget=getattr(config, "API_TENANT_COST_BUDGET", 1000.0),
    window_seconds=getattr(config, "API_TENANT_COST_WINDOW_SECONDS", 3600.0),
)


def _charge_request_cost(tenant_id: str, cost: float) -> None:
    """Charge ``cost`` to ``tenant_id``, translating quota errors to HTTP 429."""
    try:
        _cost_accountant.charge(tenant_id, cost)
    except QuotaExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                "Request-cost quota exceeded for this tenant. "
                f"Retry after {exc.retry_after} seconds."
            ),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc


def _tenant_id_from_key(api_key: str) -> str:
    """Derive a stable tenant identity from the authenticated API key.

    The raw key is never used as a bucket key; a short digest keeps limiter
    state

/* … truncated 4133 chars — edit only what you need near the top … */
