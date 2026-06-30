"""Three-state circuit breaker for resilient downstream calls.

States: CLOSED (normal) → OPEN (failing, rejecting) → HALF_OPEN (testing recovery) → CLOSED.

Thread-safe: all state transitions are protected by a single lock.
When a circuit opens, a ``ledgerlens_circuit_open`` Prometheus gauge is emitted
(labelled by component name).  Falls back to an in-process dict if
``prometheus_client`` is not installed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Prometheus gauge (optional dependency)
# ---------------------------------------------------------------------------

# In-process fallback — always populated regardless of prometheus_client.
_OPEN_GAUGES: dict[str, int] = {}
_GAUGES_LOCK = threading.Lock()

try:
    from prometheus_client import Gauge as _PromGauge

    _CIRCUIT_OPEN_GAUGE = _PromGauge(
        "ledgerlens_circuit_open",
        "1 when the circuit breaker for a component is OPEN, 0 when CLOSED",
        ["component"],
    )
except ImportError:  # pragma: no cover
    _CIRCUIT_OPEN_GAUGE = None  # type: ignore[assignment]


def _set_gauge(component: str, value: int) -> None:
    with _GAUGES_LOCK:
        _OPEN_GAUGES[component] = value
    if _CIRCUIT_OPEN_GAUGE is not None:  # pragma: no cover
        _CIRCUIT_OPEN_GAUGE.labels(component=component).set(value)


def get_open_gauges() -> dict[str, int]:
    """Snapshot of current ``ledgerlens_circuit_open`` gauge values (keyed by component)."""
    with _GAUGES_LOCK:
        return dict(_OPEN_GAUGES)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the named circuit is OPEN.

    Error messages never include internal stack traces or connection strings.
    """

    def __init__(self, component: str) -> None:
        super().__init__(f"Circuit '{component}' is OPEN — call rejected")
        self.component = component


class CircuitBreaker:
    """Thread-safe three-state circuit breaker.

    Parameters
    ----------
    name:
        Label for the protected component; used in logs and as the Prometheus
        ``component`` label.
    failure_threshold:
        Consecutive failures required to move CLOSED → OPEN (default 5).
    timeout_seconds:
        Seconds the circuit stays OPEN before entering HALF_OPEN (default 60).
    success_threshold:
        Consecutive successes in HALF_OPEN required to move → CLOSED (default 2).
    _clock:
        Monotonic clock function; override in tests to avoid real sleeps.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        success_threshold: int = 2,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        self._clock = _clock or time.monotonic

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (accounts for elapsed timeout)."""
        with self._lock:
            return self._effective_state()

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Invoke *func* through the circuit breaker.

        Raises ``CircuitOpenError`` when the circuit is OPEN and the timeout
        has not elapsed.  In HALF_OPEN the call is allowed through as a
        recovery probe; a failure immediately re-opens the circuit.
        """
        with self._lock:
            effective = self._effective_state()
            if effective == CircuitState.OPEN:
                raise CircuitOpenError(self.name)
            if self._state == CircuitState.OPEN:
                # Timeout elapsed → transition to HALF_OPEN
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes = 0
                logger.info("Circuit '%s' → HALF_OPEN (timeout elapsed)", self.name)

        try:
            result = func(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise

        self._record_success()
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _effective_state(self) -> CircuitState:
        """Resolve OPEN → HALF_OPEN when the timeout has elapsed (call under lock)."""
        if self._state == CircuitState.OPEN:
            assert self._opened_at is not None
            if self._clock() - self._opened_at >= self.timeout_seconds:
                return CircuitState.HALF_OPEN
        return self._state

    def _record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
                    self._opened_at = None
                    _set_gauge(self.name, 0)
                    logger.info("Circuit '%s' → CLOSED (recovered)", self.name)
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._consecutive_successes = 0
                _set_gauge(self.name, 1)
                logger.warning("Circuit '%s' → OPEN (probe failed in HALF_OPEN)", self.name)
                return

            self._consecutive_failures += 1
            if (
                self._state == CircuitState.CLOSED
                and self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                _set_gauge(self.name, 1)
                logger.warning(
                    "Circuit '%s' → OPEN after %d consecutive failure(s)",
                    self.name,
                    self._consecutive_failures,
                )
