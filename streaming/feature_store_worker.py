"""Background feature-store refresh worker (#183).

Listens for new trade events on the ``PubSubRouter`` and incrementally
updates cached feature vectors in the ``RedisFeatureStore`` so the
streaming scorer always has a fresh entry ready before a score request
arrives.

Architecture
------------
The worker runs in a dedicated daemon thread inside the streaming pipeline.
It subscribes to a wallet-level pub/sub channel (``wallet/{wallet_id}``)
via the existing ``PubSubRouter``, receives trade notifications as plain
Python dicts (forwarded by ``streaming/pipeline.py``), rebuilds the feature
row for the affected (wallet, pair), and writes it back to Redis with the
appropriate per-window TTL.

Fallback
--------
When Redis is unavailable the worker logs a warning and skips the write.
The streaming scorer's fallback path (direct recomputation) takes over
transparently.

Usage
-----
    worker = FeatureStoreWorker(feature_store, feature_buffer)
    worker.start()          # starts background daemon thread
    # ... at shutdown:
    worker.stop()
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import pandas as pd

from config import config
from streaming.feature_store import RedisFeatureStore
from utils.logging import get_logger

if TYPE_CHECKING:
    from streaming.feature_buffer import FeatureBuffer
    from streaming.pubsub_router import PubSubRouter

logger = get_logger(__name__)

# Internal sentinel to signal the worker thread to stop.
_STOP = object()

# Prometheus counter (optional)
try:
    from prometheus_client import Counter

    _refreshed_total = Counter(
        "ledgerlens_feature_store_refreshed_total",
        "Number of feature vectors refreshed in Redis by the background worker",
    )
    _refresh_errors_total = Counter(
        "ledgerlens_feature_store_refresh_errors_total",
        "Number of errors during background feature refresh",
    )
except Exception:  # pragma: no cover
    _refreshed_total = None  # type: ignore[assignment]
    _refresh_errors_total = None  # type: ignore[assignment]


class FeatureStoreWorker:
    """Background thread that keeps Redis feature vectors up to date.

    Parameters
    ----------
    feature_store : RedisFeatureStore
        The store to write refreshed features into.
    feature_buffer : FeatureBuffer
        The per-wallet rolling trade buffer used to recompute features.
    pubsub_router : PubSubRouter | None
        When provided, the worker registers itself as a subscriber on the
        ``"internal/trade"`` channel and processes events from there.
        When ``None``, callers drive the worker directly via
        :meth:`submit_trade_event`.
    max_queue_depth : int
        Maximum number of pending refresh tasks.  Oldest items are dropped
        when the queue is full to prevent unbounded memory growth.
    min_trade_threshold : int
        Minimum number of buffered trades before a refresh is attempted.
        Mirrors ``config.MIN_TRADES_FOR_SCORING`` by default.
    """

    INTERNAL_TRADE_CHANNEL = "internal/trade"

    def __init__(
        self,
        feature_store: RedisFeatureStore,
        feature_buffer: "FeatureBuffer",
        pubsub_router: "PubSubRouter | None" = None,
        max_queue_depth: int = 1000,
        min_trade_threshold: int | None = None,
    ) -> None:
        self._store = feature_store
        self._buffer = feature_buffer
        self._router = pubsub_router
        self._min_trades = (
            min_trade_threshold
            if min_trade_threshold is not None
            else config.MIN_TRADES_FOR_SCORING
        )
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_depth)
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._worker_id = "feature_store_worker"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background refresh thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run,
            name="FeatureStoreWorker",
            daemon=True,
        )
        self._thread.start()

        # Register with the pubsub router if provided
        if self._router is not None:
            self._router.subscribe(
                self._worker_id, [self.INTERNAL_TRADE_CHANNEL]
            )

        logger.info("FeatureStoreWorker started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it to drain."""
        self._running.clear()
        if self._router is not None:
            try:
                self._router.unsubscribe(self._worker_id, [self.INTERNAL_TRADE_CHANNEL])
            except Exception:
                pass
        # Unblock the queue.get() in _run
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("FeatureStoreWorker stopped.")

    # ------------------------------------------------------------------
    # Event submission (called by the streaming pipeline / pubsub dispatch)
    # ------------------------------------------------------------------

    def submit_trade_event(
        self,
        wallet_id: str,
        pair_id: str,
        event_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Queue a refresh task for ``(wallet_id, pair_id)``.

        This is non-blocking: if the queue is full the task is silently
        dropped (preventing backpressure from a slow Redis from stalling
        the ingestion path).
        """
        task = {
            "wallet_id": wallet_id,
            "pair_id": pair_id,
            "queued_at": time.monotonic(),
            "metadata": event_metadata or {},
        }
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            logger.debug(
                "FeatureStoreWorker queue full — dropping refresh for wallet=%s pair=%s",
                wallet_id,
                pair_id,
            )

    # ------------------------------------------------------------------
    # Internal worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop: drain the queue and refresh feature vectors."""
        while self._running.is_set():
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is _STOP:
                break

            try:
                self._process_task(task)
            except Exception as exc:
                logger.warning(
                    "FeatureStoreWorker: unhandled error processing task %s: %s",
                    task,
                    exc,
                    exc_info=True,
                )
                if _refresh_errors_total is not None:
                    try:
                        _refresh_errors_total.inc()
                    except Exception:
                        pass
            finally:
                self._queue.task_done()

    def _process_task(self, task: dict) -> None:
        """Recompute features for one (wallet, pair) and write to Redis.

        Skips wallets with fewer than ``min_trade_threshold`` buffered trades
        to avoid caching low-quality feature rows during warm-up.
        """
        wallet_id: str = task["wallet_id"]
        pair_id: str = task["pair_id"]

        trade_count = self._buffer.wallet_trade_count(wallet_id)
        if trade_count < self._min_trades:
            logger.debug(
                "FeatureStoreWorker: skipping %s (%d trades < min %d)",
                wallet_id,
                trade_count,
                self._min_trades,
            )
            return

        # Rebuild the feature row from the buffer
        feature_row = self._buffer.get_feature_row(wallet_id)
        if feature_row is None:
            return

        # Convert pd.Series to plain dict with JSON-safe scalar types
        features = _series_to_safe_dict(feature_row)

        # Determine TTL from the default window map (use the 1h window as a
        # conservative choice; callers can customise via window_hours if needed)
        success = self._store.put(
            wallet_id=wallet_id,
            pair_id=pair_id,
            features=features,
            window_hours=1,
        )

        if success:
            logger.debug(
                "FeatureStoreWorker: refreshed wallet=%s pair=%s (%d trades)",
                wallet_id,
                pair_id,
                trade_count,
            )
            if _refreshed_total is not None:
                try:
                    _refreshed_total.inc()
                except Exception:
                    pass
        else:
            logger.debug(
                "FeatureStoreWorker: Redis write failed for wallet=%s pair=%s (fallback ok)",
                wallet_id,
                pair_id,
            )

    @property
    def queue_depth(self) -> int:
        """Current number of pending refresh tasks."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._running.is_set() and (
            self._thread is not None and self._thread.is_alive()
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series_to_safe_dict(series: pd.Series) -> dict[str, Any]:
    """Convert a pd.Series to a dict with msgpack-safe scalar types.

    numpy scalars and NaN/Inf are normalised to Python floats or 0.0.
    """
    import math

    result: dict[str, Any] = {}
    for k, v in series.items():
        # Convert numpy scalars
        if hasattr(v, "item"):
            v = v.item()
        # Normalise NaN/Inf to 0.0 (avoid msgpack failures)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = 0.0
        result[str(k)] = v
    return result
