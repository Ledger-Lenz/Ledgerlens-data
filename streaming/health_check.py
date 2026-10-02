"""Worker health check management.

Provides an HTTP health endpoint to verify long-running processes (Kafka workers,
SSE streams) are active. Workers call `heartbeat()` periodically. If any worker
fails to heartbeat within `HEALTH_CHECK_TIMEOUT_SECONDS`, the `/health` endpoint
returns 503 instead of 200.

Liveness vs readiness (Issue #902)
----------------------------------
- ``/livez`` (and legacy ``/health``): *liveness* — the process and its
  workers are running.  Never depends on external systems, so an orchestrator
  will not restart a healthy instance just because Kafka is briefly down.
- ``/readyz``: *readiness* — every registered dependency (Kafka broker,
  feature store, model artifact) is reachable right now.  Checks run live on
  each probe with a short timeout, so readiness fails fast during an outage
  and recovers automatically once the dependency is back — no restart needed.

Register dependencies with :func:`register_dependency`, e.g.::

    register_dependency("kafka", kafka_broker_check("broker:9092"))
    register_dependency("model", model_artifact_check("models/latest.joblib"))

See ``docs/deployment_modes.md`` for Kubernetes probe configuration.
"""

import json
import os
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


class WorkerHealthManager:
    """Tracks heartbeat timestamps for multiple workers."""

    def __init__(self, timeout_seconds: float = 120.0):
        self._workers: dict[str, float] = {}
        self._timeout = timeout_seconds
        self._lock = threading.Lock()

    def heartbeat(self, worker_id: str) -> None:
        """Register a heartbeat for the given worker."""
        with self._lock:
            self._workers[worker_id] = time.time()

    def is_healthy(self) -> tuple[bool, dict[str, Any]]:
        """Return True if all registered workers have heartbeated recently."""
        with self._lock:
            if not self._workers:
                # If no workers registered yet, assume healthy to avoid premature failure.
                return True, {"status": "ok", "message": "no workers registered"}

            now = time.time()
            details = {}
            all_healthy = True
            for wid, last_hb in self._workers.items():
                if now - last_hb > self._timeout:
                    details[wid] = "stalled"
                    all_healthy = False
                else:
                    details[wid] = "ok"

            return all_healthy, details


# Global instance
_health_manager = WorkerHealthManager(
    timeout_seconds=float(os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", "120.0"))
)


def heartbeat(worker_id: str) -> None:
    """Global convenience for registering a heartbeat."""
    _health_manager.heartbeat(worker_id)


DependencyCheck = Callable[[], bool]


class ReadinessChecker:
    """Evaluates dependency connectivity live on every readiness probe."""

    def __init__(self, timeout_seconds: float = 2.0) -> None:
        self._checks: dict[str, DependencyCheck] = {}
        self._timeout = timeout_seconds
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="readiness")

    def register(self, name: str, check: DependencyCheck) -> None:
        with self._lock:
            self._checks[name] = check

    def unregister(self, name: str) -> None:
        with self._lock:
            self._checks.pop(name, None)

    def is_ready(self) -> tuple[bool, dict[str, str]]:
        """Return (ready, per-dependency detail). A check that raises or times out fails."""
        with self._lock:
            checks = dict(self._checks)
        futures = {name: self._pool.submit(fn) for name, fn in checks.items()}
        details: dict[str, str] = {}
        for name, fut in futures.items():
            try:
                details[name] = "ok" if fut.result(timeout=self._timeout) else "unavailable"
            except FutureTimeout:
                details[name] = "timeout"
            except Exception as exc:
                details[name] = f"error: {exc}"
        return all(v == "ok" for v in details.values()), details


def kafka_broker_check(bootstrap_servers: str, timeout: float = 1.0) -> DependencyCheck:
    """TCP-connect check against the first reachable Kafka bootstrap server."""

    def _check() -> bool:
        for server in bootstrap_servers.split(","):
            host, _, port = server.strip().rpartition(":")
            try:
                with socket.create_connection((host or "localhost", int(port or 9092)), timeout):
                    return True
            except OSError:
                continue
        return False

    return _check


def model_artifact_check(path: str) -> DependencyCheck:
    """Readiness check that the model artifact file is present and non-empty."""
    return lambda: os.path.isfile(path) and os.path.getsize(path) > 0


_readiness = ReadinessChecker(
    timeout_seconds=float(os.getenv("READINESS_CHECK_TIMEOUT_SECONDS", "2.0"))
)


def register_dependency(name: str, check: DependencyCheck) -> None:
    """Register a dependency check evaluated by ``/readyz``."""
    _readiness.register(name, check)


def unregister_dependency(name: str) -> None:
    _readiness.unregister(name)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler serving a /health endpoint."""

    def do_GET(self) -> None:
        if self.path in ("/health", "/livez"):
            healthy, details = _health_manager.is_healthy()
            self._send(healthy, {"status": "ok" if healthy else "unhealthy", "workers": details})
        elif self.path == "/readyz":
            ready, deps = _readiness.is_ready()
            self._send(ready, {"status": "ready" if ready else "not_ready", "dependencies": deps})
        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, ok: bool, body: dict[str, Any]) -> None:
        self.send_response(200 if ok else 503)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP logging."""
        pass


def start_health_server(port: int = 8080) -> None:
    """Start the health check HTTP server in a daemon thread."""

    def run_server() -> None:
        try:
            # Bind to all interfaces for Kubernetes / Docker checks
            server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
            logger.info("Health check server started on port %d", port)
            server.serve_forever()
        except Exception as exc:
            logger.error("Failed to start health check server: %s", exc)

    thread = threading.Thread(target=run_server, daemon=True, name="health-check-server")
    thread.start()
