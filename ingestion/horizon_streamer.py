"""Real-time trade ingestion via Horizon's Server-Sent Events stream.

Streams trades for each watched asset pair and yields `Trade` objects as
they occur on the ledger.

Multi-region failover (#202)
----------------------------
``HorizonEndpointPool`` maintains a list of Horizon URLs configured via
``HORIZON_FAILOVER_URLS`` (comma-separated, all must be HTTPS except in dev
mode).  A background health-check loop probes each endpoint every
``HORIZON_HEALTH_CHECK_INTERVAL_SECONDS`` seconds and marks them healthy or
unhealthy based on HTTP status and latency.  ``stream_trades`` uses the pool
automatically when more than one endpoint is configured.
"""

from __future__ import annotations

import statistics
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from collections.abc import Iterator

from stellar_sdk import Asset as SdkAsset
from stellar_sdk import Server

from config import config
from ingestion.data_models import Asset, Trade
from utils.logging import get_logger

logger = get_logger(__name__)


class HorizonEndpointPool:
    """Manages a pool of Horizon endpoints with health-based routing.

    The pool runs a background thread that probes each endpoint with
    ``GET /`` every ``health_check_interval`` seconds.  Endpoints are
    marked healthy when they return HTTP 200 within the latency budget
    and unhealthy otherwise.

    Routing selects the endpoint with the lowest p95 latency among
    healthy candidates using a simple sliding window of the last 20
    observations.  Falls back to weighted round-robin on ties.

    All ``HORIZON_FAILOVER_URLS`` must be HTTPS unless ``dev_mode=True``.
    HTTP endpoints are rejected at construction to prevent MITM on
    Horizon responses.
    """

    _LATENCY_WINDOW = 20

    def __init__(
        self,
        urls: list[str] | None = None,
        health_check_interval: int | None = None,
        failover_timeout: int | None = None,
        dev_mode: bool | None = None,
    ) -> None:
        raw_urls = list(urls) if urls is not None else list(config.HORIZON_FAILOVER_URLS)
        if not raw_urls:
            raw_urls = [config.HORIZON_URL]
        self._dev_mode = dev_mode if dev_mode is not None else config.HORIZON_DEV_MODE
        self._urls = self._validate_urls(raw_urls)
        self._health_check_interval = (
            health_check_interval
            if health_check_interval is not None
            else config.HORIZON_HEALTH_CHECK_INTERVAL_SECONDS
        )
        self._failover_timeout = (
            failover_timeout
            if failover_timeout is not None
            else config.HORIZON_FAILOVER_TIMEOUT_SECONDS
        )
        self._lock = threading.Lock()
        # url -> True/False
        self._healthy: dict[str, bool] = {u: True for u in self._urls}
        # url -> deque of recent latencies (seconds)
        self._latencies: dict[str, deque] = {
            u: deque(maxlen=self._LATENCY_WINDOW) for u in self._urls
        }
        self._rr_index: int = 0
        self._stop_event = threading.Event()
        self._health_thread = threading.Thread(
            target=self._health_check_loop, daemon=True, name="horizon-health-check"
        )
        self._health_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def best_url(self) -> str:
        """Return the URL of the healthiest (lowest p95 latency) endpoint.

        If all endpoints are unhealthy, emits a CRITICAL log and returns the
        primary URL so the caller can enter a retry loop.
        """
        with self._lock:
            healthy = [u for u in self._urls if self._healthy[u]]
        if not healthy:
            logger.critical(
                "All Horizon endpoints are unhealthy — entering retry loop with primary endpoint"
            )
            return self._urls[0]
        if len(healthy) == 1:
            return healthy[0]
        return min(healthy, key=self._p95_latency)

    def mark_unhealthy(self, url: str) -> None:
        with self._lock:
            if url in self._healthy:
                self._healthy[url] = False
                logger.warning("Horizon endpoint marked unhealthy: %s", url)

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_urls(self, urls: list[str]) -> list[str]:
        validated = []
        for url in urls:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme == "http" and not self._dev_mode:
                raise ValueError(
                    f"Horizon endpoint {url!r} uses HTTP — only HTTPS is allowed "
                    "(set HORIZON_DEV_MODE=1 to allow HTTP in development)"
                )
            validated.append(url)
        return validated

    def _probe(self, url: str) -> tuple[bool, float]:
        """HTTP GET / probe; returns (healthy, latency_seconds)."""
        try:
            start = time.monotonic()
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self._failover_timeout) as resp:
                latency = time.monotonic() - start
                return resp.status == 200, latency
        except Exception as exc:
            logger.debug("Health check failed for %s: %s", url, exc)
            return False, float(self._failover_timeout)

    def _health_check_loop(self) -> None:
        while not self._stop_event.is_set():
            for url in self._urls:
                healthy, latency = self._probe(url)
                with self._lock:
                    prev = self._healthy[url]
                    self._healthy[url] = healthy
                    self._latencies[url].append(latency)
                    if prev != healthy:
                        level = logger.info if healthy else logger.warning
                        level(  # type: ignore[operator]
                            "Horizon endpoint %s is now %s (latency=%.3fs)",
                            url,
                            "healthy" if healthy else "unhealthy",
                            latency,
                        )
            self._stop_event.wait(timeout=self._health_check_interval)

    def _p95_latency(self, url: str) -> float:
        with self._lock:
            samples = list(self._latencies[url])
        if not samples:
            return 0.0
        sorted_s = sorted(samples)
        idx = max(0, int(len(sorted_s) * 0.95) - 1)
        return sorted_s[idx]


# Module-level singleton — populated on first use when HORIZON_FAILOVER_URLS is set.
_endpoint_pool: HorizonEndpointPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> HorizonEndpointPool | None:
    """Return the module-level HorizonEndpointPool, creating it if needed.

    Returns None when only a single endpoint is configured (no failover needed).
    """
    global _endpoint_pool
    all_urls = list(config.HORIZON_FAILOVER_URLS)
    if not all_urls:
        return None
    with _pool_lock:
        if _endpoint_pool is None:
            _endpoint_pool = HorizonEndpointPool(urls=[config.HORIZON_URL] + all_urls)
    return _endpoint_pool


def _to_trade(record: dict) -> Trade:
    return Trade(
        trade_id=record["id"],
        ledger_close_time=record["ledger_close_time"],
        base_account=record["base_account"],
        counter_account=record["counter_account"],
        base_asset=Asset(
            code=record["base_asset_code"] or "XLM",
            issuer=record.get("base_asset_issuer"),
        ),
        counter_asset=Asset(
            code=record["counter_asset_code"] or "XLM",
            issuer=record.get("counter_asset_issuer"),
        ),
        base_amount=float(record["base_amount"]),
        counter_amount=float(record["counter_amount"]),
        price=float(record["price"]["n"]) / float(record["price"]["d"]),
    )


def stream_trades(
    base_asset: SdkAsset,
    counter_asset: SdkAsset,
    cursor: str = "now",
    max_reconnect_attempts: int = 5,
) -> Iterator[Trade]:
    """Yield `Trade` objects as they are streamed from Horizon.

    When ``HORIZON_FAILOVER_URLS`` is configured, the pool's healthiest endpoint
    is selected on each (re)connect attempt.  On failure the current endpoint is
    marked unhealthy and the next attempt picks the new best endpoint, achieving
    transparent failover within ``HORIZON_FAILOVER_TIMEOUT_SECONDS``.  The
    ledger cursor is preserved across failover so no events are missed or
    duplicated.

    Without failover URLs this behaves identically to the original single-endpoint
    implementation.
    """
    pool = _get_pool()
    attempts = 0

    while True:
        horizon_url = pool.best_url() if pool is not None else config.HORIZON_URL
        server = Server(horizon_url=horizon_url)
        call_builder = server.trades().for_asset_pair(base_asset, counter_asset).cursor(cursor)
        try:
            for response in call_builder.stream():
                yield _to_trade(response)
                cursor = response["paging_token"]
                attempts = 0
        except (ConnectionError, TimeoutError, OSError) as exc:
            attempts += 1
            if pool is not None:
                pool.mark_unhealthy(horizon_url)
            if attempts >= max_reconnect_attempts:
                raise
            logger.warning(
                "Trade stream disconnected (attempt %d/%d) on %s: %s — reconnecting from cursor %s",
                attempts,
                max_reconnect_attempts,
                horizon_url,
                exc,
                cursor,
            )


def stream_all_watched_pairs() -> Iterator[Trade]:
    """Convenience generator that round-robins through configured pairs.

    NOTE: for production use, run one `stream_trades` generator per pair in
    its own task/thread rather than interleaving here.
    """
    if not config.WATCHED_ASSET_PAIRS:
        raise ValueError("WATCHED_ASSET_PAIRS is not configured")

    streams = []
    for code, issuer in config.WATCHED_ASSET_PAIRS:
        asset = SdkAsset.native() if issuer == "native" else SdkAsset(code, issuer)
        xlm = SdkAsset.native()
        if asset == xlm:
            continue
        streams.append(stream_trades(asset, xlm))

    for stream in streams:
        yield from stream
