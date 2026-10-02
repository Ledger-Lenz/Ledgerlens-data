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

Checkpoint resume (#904)
------------------------
When ``stream_trades`` is given a ``WatermarkTracker``, it resumes from the
pair's last committed watermark on startup and advances the watermark after
each yielded trade.  If the watermark's ledger has been pruned from the
queried Horizon node (the node's oldest available trade is newer than the
watermark), a gap alert is raised and the missing range is backfilled via
``historical_loader.load_trades`` against ``HORIZON_HISTORY_URL`` (a
full-history node) before live streaming continues.  See
``docs/stream_replay_runbook.md``.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING

from stellar_sdk import Asset as SdkAsset
from stellar_sdk import Server

from config import config
from ingestion.data_models import Asset, Trade
from ingestion.exceptions import InvalidInputError, RecordValidationError, record_context
from streaming.cursor_store import BaseCursorStore, get_cursor_store
from streaming.health import HealthStatus, WorkerHealthMonitor, get_health_registry
from utils.logging import get_logger
from utils.tracing import get_tracer, hash_span_id

if TYPE_CHECKING:
    from ingestion.watermark_tracker import WatermarkTracker

logger = get_logger(__name__)
_tracer = get_tracer(__name__)


def _token_key(token: str) -> tuple[int, ...]:
    """Order Horizon paging tokens (``"<toid>-<index>"``) numerically."""
    return tuple(int(p) for p in str(token).split("-") if p.isdigit())


def _oldest_available_token(horizon_url: str, base: SdkAsset, counter: SdkAsset) -> str | None:
    """Return the paging token of the oldest trade the Horizon node still retains."""
    page = (
        Server(horizon_url=horizon_url)
        .trades()
        .for_asset_pair(base, counter)
        .order(desc=False)
        .limit(1)
        .call()
    )
    records = page["_embedded"]["records"]
    return records[0]["paging_token"] if records else None


def _alert_history_gap(pair_id: str, watermark: str, oldest: str, horizon_url: str) -> None:
    """Operator alert: the resume point was pruned from the Horizon node."""
    logger.critical(
        "HORIZON_HISTORY_GAP pair=%s watermark=%s oldest_available=%s node=%s — "
        "falling back to historical backfill (see docs/stream_replay_runbook.md)",
        pair_id,
        watermark,
        oldest,
        horizon_url,
        extra={"alert": "horizon_history_gap", "pair_id": pair_id},
    )


def _backfill_pruned_range(
    base: SdkAsset, counter: SdkAsset, pair_id: str, watermark: str
) -> Iterator[Trade]:
    """Replay trades after *watermark* from a full-history node (dedup by token)."""
    from ingestion.historical_loader import load_trades

    history_url = getattr(config, "HORIZON_HISTORY_URL", None) or config.HORIZON_URL
    wm_key = _token_key(watermark)
    for trade in load_trades(base, counter, cursor=watermark, horizon_url=history_url):
        if _token_key(trade.trade_id) > wm_key:
            yield trade


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

        self._health_monitor = WorkerHealthMonitor(
            "horizon_endpoint_pool",
            health_check_fn=self._get_health_status,
        )
        get_health_registry().register(self._health_monitor)

    def _get_health_status(self) -> tuple[HealthStatus, str | None]:
        with self._lock:
            healthy_count = sum(1 for h in self._healthy.values() if h)
            total = len(self._urls)

        if healthy_count == total:
            return HealthStatus.HEALTHY, f"All {total} endpoints healthy"
        elif healthy_count > 0:
            return HealthStatus.DEGRADED, f"{healthy_count}/{total} endpoints healthy"
        else:
            return HealthStatus.UNHEALTHY, "All Horizon endpoints unhealthy"

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
        self._health_monitor.mark_stopped("HorizonEndpointPool stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_urls(self, urls: list[str]) -> list[str]:
        validated = []
        for url in urls:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme == "http" and not self._dev_mode:
                raise InvalidInputError(
                    f"Horizon endpoint {url!r} uses HTTP — only HTTPS is allowed "
                    "(set HORIZON_DEV_MODE=1 to allow HTTP in development)",
                    source="horizon_streamer.HorizonEndpointPool._validate_urls",
                    reason="non-HTTPS Horizon endpoint rejected outside dev mode",
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
            # Broad catch justified: health checks must not crash on any network error
            # (DNS failures, socket errors, timeouts, etc.). Probe failure just marks
            # endpoint unhealthy; the pool will select another one on next call.
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
    """Build a :class:`Trade` from a raw Horizon trade record.

    Raises:
        RecordValidationError: If the record is missing fields, has wrong-typed
            values, or otherwise fails ``Trade`` validation.
    """
    with record_context("horizon_streamer._to_trade", record):
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
    cursor_store: BaseCursorStore | None = None,
    stream_id: str | None = None,
    watermark_tracker: WatermarkTracker | None = None,
) -> Iterator[Trade]:
    """Yield `Trade` objects as they are streamed from Horizon.

    When ``HORIZON_FAILOVER_URLS`` is configured, the pool's healthiest endpoint
    is selected on each (re)connect attempt.  On failure the current endpoint is
    marked unhealthy and the next attempt picks the new best endpoint, achieving
    transparent failover within ``HORIZON_FAILOVER_TIMEOUT_SECONDS``.  The
    ledger cursor is preserved across failover so no events are missed or
    duplicated.

    When a ``cursor_store`` is provided (or if default config cursor store is active),
    the streaming cursor is automatically restored on restart and saved on each event.

    When ``watermark_tracker`` is provided, its committed watermark for the pair
    takes precedence as the resume point, and the pruned-history fallback
    described in the module docstring applies (#904).
    """
    pool = _get_pool()
    attempts = 0

    active_stream_id = stream_id or f"trades:{base_asset.code}:{counter_asset.code}"
    active_store = cursor_store or get_cursor_store()

    # Restore persisted cursor if available
    cursor = active_store.get_cursor(active_stream_id, default=cursor)

    pair_id = f"{base_asset.code}:{counter_asset.code}"

    def _commit(trade: Trade, token: str) -> None:
        active_store.save_cursor(
            active_stream_id,
            token,
            metadata={"trade_id": trade.trade_id, "timestamp": time.time()},
        )
        if watermark_tracker is not None:
            watermark_tracker.advance(pair_id, token, trade.ledger_close_time)

    if watermark_tracker is not None:
        wm_cursor = watermark_tracker.get_cursor(pair_id)
        if wm_cursor:
            cursor = wm_cursor
            probe_url = pool.best_url() if pool is not None else config.HORIZON_URL
            oldest = _oldest_available_token(probe_url, base_asset, counter_asset)
            if oldest is not None and _token_key(oldest) > _token_key(wm_cursor):
                _alert_history_gap(pair_id, wm_cursor, oldest, probe_url)
                for trade in _backfill_pruned_range(base_asset, counter_asset, pair_id, wm_cursor):
                    yield trade
                    cursor = trade.trade_id
                    _commit(trade, cursor)
            else:
                logger.info("Resuming %s from watermark %s", pair_id, wm_cursor)

    while True:
        horizon_url = pool.best_url() if pool is not None else config.HORIZON_URL
        server = Server(horizon_url=horizon_url)
        call_builder = server.trades().for_asset_pair(base_asset, counter_asset).cursor(cursor)
        try:
            for response in call_builder.stream():
                with _tracer.start_as_current_span("trade_event.received") as span:
                    try:
                        trade = _to_trade(response)
                    except RecordValidationError as exc:
                        # Only scrubbed fields reach the span: this module's
                        # tracing contract forbids raw wallet addresses and
                        # amounts, and a pydantic validation message embeds the
                        # offending input value. Full context goes to the log.
                        span.set_attribute("error", True)
                        span.set_attribute("error.type", type(exc).__name__)
                        span.set_attribute("error.source", exc.source or "")
                        logger.error(
                            "Malformed trade record on %s: %s",
                            horizon_url,
                            exc,
                            extra={"context": exc.context},
                        )
                        raise
                    span.set_attribute("trade.id", hash_span_id(trade.trade_id))
                    span.set_attribute("trade.base_asset", trade.base_asset.code)
                    span.set_attribute("trade.counter_asset", trade.counter_asset.code)
                yield trade
                cursor = response["paging_token"]
                _commit(trade, cursor)
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
        raise InvalidInputError(
            "WATCHED_ASSET_PAIRS is not configured",
            source="horizon_streamer.stream_all_watched_pairs",
            reason="WATCHED_ASSET_PAIRS is empty",
        )

    streams = []
    for code, issuer in config.WATCHED_ASSET_PAIRS:
        asset = SdkAsset.native() if issuer == "native" else SdkAsset(code, issuer)
        xlm = SdkAsset.native()
        if asset == xlm:
            continue
        streams.append(stream_trades(asset, xlm))

    for stream in streams:
        yield from stream
