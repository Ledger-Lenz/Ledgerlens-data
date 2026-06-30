"""Top-level orchestrator for the real-time detection pipeline.

StreamingPipeline starts one daemon thread per watched asset pair, drives
each through stream_trades(), and wires the FeatureBuffer → StreamingScorer →
AlertDispatcher chain.

Reconnection on Horizon SSE failures is handled at two levels:
  1. stream_trades() retries internally (up to max_reconnect_attempts).
  2. _stream_pair() restarts the generator if stream_trades() raises after
     exhausting its own retries.

Circuit breakers
----------------
Pass ``circuit_breakers`` to wrap the model-inference call with independent
per-component ``CircuitBreaker`` instances.  Supported keys:
  - ``"inference"``  — wraps ``scorer.score_wallet()``
  - ``"db_write"``   — wraps ``dispatcher.dispatch()``
  - ``"benford"``    — wraps any explicit Benford computation (future extension)

When a circuit is OPEN the pipeline falls back to the last cached score for
the wallet-pair so alert delivery can continue uninterrupted.

Shutdown
--------
Call pipeline.run() from the main thread.  SIGINT (Ctrl-C) sets the internal
stop event via a signal handler; the main loop wakes up, joins all worker
threads with a 5-second timeout, and returns.
"""

from __future__ import annotations

import signal
import threading
import time
from typing import TYPE_CHECKING

from prometheus_client import Histogram
from stellar_sdk import Asset as SdkAsset

from config import config
from ingestion.amm_pool_loader import PoolNotFoundError, stream_amm_pool_trades
from ingestion.horizon_streamer import stream_trades
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer
from streaming.streaming_scorer import StreamingScorer
from utils.logging import get_logger

if TYPE_CHECKING:
    from streaming.account_metadata_stream import AccountMetadataUpdate

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metric — join lag histogram
# ---------------------------------------------------------------------------

#: Time (seconds) from a metadata update arriving in ``MetadataJoinState``
#: to the next scoring cycle in which the updated metadata is actually used.
#: Measures the end-to-end freshness of the join.
METADATA_JOIN_LAG = Histogram(
    "metadata_join_lag_seconds",
    "Seconds between an account metadata update arriving and the affected "
    "wallet's features being re-scored with the new metadata",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)


# ---------------------------------------------------------------------------
# Stateful join state
# ---------------------------------------------------------------------------


class MetadataJoinState:
    """Thread-safe store of the latest account metadata per wallet.

    Join window semantics
    ~~~~~~~~~~~~~~~~~~~~~
    An ``AccountMetadataUpdate`` enriches trade events for
    ``METADATA_JOIN_WINDOW_SECONDS`` after it arrives.  Once that window
    expires the entry is treated as *stale* and removed from active state.

    Late-arrival handling
    ~~~~~~~~~~~~~~~~~~~~~
    If a metadata update arrives for a wallet that has no active join entry
    (either the wallet has never been seen, or its previous entry expired),
    the update is stored in ``_pending_updates``.  On the next call to
    ``get_metadata`` for that wallet the pending update is promoted to active
    state so it enriches the next scoring cycle.

    Memory bounding
    ~~~~~~~~~~~~~~~
    ``evict_inactive_wallets()`` should be called periodically (e.g., every
    5 minutes).  It removes join state for wallets whose last trade timestamp
    is older than ``METADATA_ACTIVE_WALLET_TTL_SECONDS``.  This prevents
    unbounded memory growth for wallets that stopped trading.
    """

    def __init__(
        self,
        join_window_seconds: int | None = None,
        active_wallet_ttl_seconds: int | None = None,
    ) -> None:
        self._join_window = (
            join_window_seconds
            if join_window_seconds is not None
            else config.METADATA_JOIN_WINDOW_SECONDS
        )
        self._active_wallet_ttl = (
            active_wallet_ttl_seconds
            if active_wallet_ttl_seconds is not None
            else config.METADATA_ACTIVE_WALLET_TTL_SECONDS
        )
        self._lock = threading.RLock()
        # wallet_id → (AccountMetadataUpdate, arrival_time: float)
        self._active: dict[str, tuple[AccountMetadataUpdate, float]] = {}
        # wallet_id → (AccountMetadataUpdate, arrival_time: float)
        # Holds updates that arrived after the join window; promoted on next
        # get_metadata() call for the wallet.
        self._pending_updates: dict[str, tuple[AccountMetadataUpdate, float]] = {}
        # wallet_id → last trade timestamp (monotonic clock, seconds).
        self._last_trade_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Apply incoming metadata update
    # ------------------------------------------------------------------

    def apply_update(self, update: AccountMetadataUpdate) -> None:
        """Store *update* in active or pending state.

        If the wallet has an active join entry that is still within the
        join window, the entry is *replaced* with the newer update.

        If the wallet has no active entry (first time seen, or previous
        entry expired), the update goes into ``_pending_updates`` so it
        does not discard in-flight join work for other wallets sharing state.
        """
        wallet = update.account_id
        arrival = time.monotonic()

        with self._lock:
            existing = self._active.get(wallet)
            if existing is not None:
                # Replace active entry with the newer update regardless of
                # whether the window has expired — this ensures the latest
                # metadata is always reflected.
                self._active[wallet] = (update, arrival)
                logger.debug(
                    "Metadata join: updated active entry for wallet %s (effect=%s)",
                    wallet,
                    update.effect_type,
                )
            else:
                # No active entry — queue as pending; will be promoted on
                # next get_metadata() call for this wallet.
                self._pending_updates[wallet] = (update, arrival)
                logger.debug(
                    "Metadata join: queued pending update for wallet %s (effect=%s)",
                    wallet,
                    update.effect_type,
                )

    # ------------------------------------------------------------------
    # Retrieve metadata for a wallet (called on every trade event)
    # ------------------------------------------------------------------

    def get_metadata(self, wallet: str) -> AccountMetadataUpdate | None:
        """Return the latest metadata for *wallet*, or ``None`` if unavailable.

        Side effects
        ------------
        * Records a trade event for *wallet* (updates ``_last_trade_at``).
        * Promotes any pending update to active state.
        * Evicts the active entry if its join window has expired.
        * Observes join lag on the Prometheus histogram when a pending or
          newly-promoted entry is used for the first time.
        """
        now_mono = time.monotonic()

        with self._lock:
            self._last_trade_at[wallet] = now_mono

            # 1. Promote pending update if present.
            if wallet in self._pending_updates:
                pending, pending_arrival = self._pending_updates.pop(wallet)
                self._active[wallet] = (pending, pending_arrival)
                lag = now_mono - pending_arrival
                METADATA_JOIN_LAG.observe(lag)
                logger.debug(
                    "Metadata join: promoted pending update for wallet %s "
                    "(lag=%.2fs, effect=%s)",
                    wallet,
                    lag,
                    pending.effect_type,
                )

            # 2. Check whether the active entry is still within its window.
            entry = self._active.get(wallet)
            if entry is None:
                return None

            active_update, arrival = entry
            age = now_mono - arrival
            if age > self._join_window:
                # Window expired — evict and return None.  The next metadata
                # update for this wallet will be admitted via _pending_updates.
                del self._active[wallet]
                logger.debug(
                    "Metadata join: join window expired for wallet %s (age=%.0fs > window=%ds)",
                    wallet,
                    age,
                    self._join_window,
                )
                return None

            return active_update

    # ------------------------------------------------------------------
    # Re-score trigger: wallets with fresh metadata updates
    # ------------------------------------------------------------------

    def wallets_needing_rescore(self) -> list[str]:
        """Return wallets whose metadata changed since the last scoring cycle.

        A wallet is returned when its active metadata was updated (arrival
        time updated) within the last ``_join_window`` seconds and the
        wallet has had at least one trade event.

        The caller is responsible for re-computing features and scoring the
        returned wallets.
        """
        now_mono = time.monotonic()
        result = []
        with self._lock:
            for wallet, (_, arrival) in list(self._active.items()):
                if wallet in self._last_trade_at and (now_mono - arrival) <= self._join_window:
                    result.append(wallet)
        return result

    # ------------------------------------------------------------------
    # Housekeeping: evict stale wallet state
    # ------------------------------------------------------------------

    def evict_inactive_wallets(self) -> int:
        """Remove join state for wallets inactive for > active_wallet_ttl seconds.

        Returns the number of wallets evicted.  Intended to be called
        periodically (e.g., every 5 minutes) from a housekeeping thread.
        """
        now_mono = time.monotonic()
        evicted = 0
        with self._lock:
            inactive = [
                w
                for w, last in self._last_trade_at.items()
                if now_mono - last > self._active_wallet_ttl
            ]
            for wallet in inactive:
                self._active.pop(wallet, None)
                self._pending_updates.pop(wallet, None)
                del self._last_trade_at[wallet]
                evicted += 1
        if evicted:
            logger.info(
                "Metadata join: evicted %d inactive wallet(s) from join state", evicted
            )
        return evicted

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def active_wallet_count(self) -> int:
        with self._lock:
            return len(self._active)

    def pending_update_count(self) -> int:
        with self._lock:
            return len(self._pending_updates)


class StreamingPipeline:
    """Orchestrates one SSE thread per pair and wires the scoring pipeline.

    Metadata join
    ~~~~~~~~~~~~~
    When *metadata_join_state* is provided, each trade event triggers a lookup
    of the involved wallets' latest ``AccountMetadataUpdate`` from join state.
    The update is passed to ``FeatureBuffer.apply_metadata`` so wallet-graph
    features reflect the current on-chain account state rather than the static
    snapshot loaded at pipeline startup.

    A background housekeeping thread runs every
    ``_HOUSEKEEPING_INTERVAL_SECONDS`` (300 s) to evict stale join state for
    wallets that have been inactive for more than
    ``METADATA_ACTIVE_WALLET_TTL_SECONDS``.

    When *metadata_stream* is provided, the pipeline will also watch every
    wallet that appears in trade events, dynamically subscribing to their
    Horizon effects feed via ``AccountMetadataStream.add_wallet()``.
    """

    _HOUSEKEEPING_INTERVAL_SECONDS = 300

    def __init__(
        self,
        buffer: FeatureBuffer,
        scorer: StreamingScorer | None,
        dispatcher: AlertDispatcher,
        pairs: list[tuple[str, str]] | None = None,
        amm_pools: list[str] | None = None,
        role: str = "all",
        metadata_join_state: MetadataJoinState | None = None,
        metadata_stream=None,  # AccountMetadataStream | None
    ):
        if role not in ("all", "producer", "worker"):
            raise ValueError(f"Unknown role: {role!r}")
        self._role = role
        self._buffer = buffer
        self._scorer = scorer
        self._dispatcher = dispatcher
        self._pairs = list(pairs) if pairs is not None else list(config.WATCHED_ASSET_PAIRS)
        self._amm_pools = (
            list(amm_pools) if amm_pools is not None else list(config.WATCHED_AMM_POOLS)
        )
        self._stop_event = threading.Event()
        self._worker_threads: list[threading.Thread] = []
        self._circuit_breakers: dict[str, CircuitBreaker] = circuit_breakers or {}
        # Wallet → last successfully computed score (used as fallback when OPEN)
        self._score_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the pipeline using the configured backend.

        ``config.STREAMING_BACKEND`` selects the transport:
          * ``"sse"`` (default) — the threaded Horizon SSE pipeline below.
          * ``"kafka"`` — a Kafka producer per pair + a :class:`KafkaWorker`.

        The Kafka modules are imported lazily so the default ``sse`` path never
        touches ``confluent_kafka`` (operators without Kafka can run unchanged).
        """
        if config.STREAMING_BACKEND == "kafka":
            self._run_kafka()
        else:
            self._run_sse()

    def _run_sse(self) -> None:
        """Start one thread per pair, block until KeyboardInterrupt or stop()."""
        sdk_pairs = self._build_sdk_pairs()
        if not sdk_pairs:
            logger.warning("No asset pairs configured — streaming pipeline has nothing to do")
            return

        # Install SIGINT handler when called from the main thread so that
        # Ctrl-C sets the stop event rather than raising KeyboardInterrupt
        # mid-iteration inside a worker thread.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())

        self._worker_threads = []
        for base_asset, counter_asset in sdk_pairs:
            t = threading.Thread(
                target=self._stream_pair,
                args=(base_asset, counter_asset),
                daemon=True,
            )
            t.start()
            self._worker_threads.append(t)

        for pool_id in self._amm_pools:
            t = threading.Thread(
                target=self._stream_amm_pool,
                args=(pool_id,),
                daemon=True,
            )
            t.start()
            self._worker_threads.append(t)

        # Start the metadata join housekeeping thread (evicts stale wallets).
        if self._metadata_join_state is not None:
            hk = threading.Thread(
                target=self._run_housekeeping,
                name="metadata-housekeeping",
                daemon=True,
            )
            hk.start()
            self._worker_threads.append(hk)

        logger.info(
            "Streaming pipeline running with %d SDEX pair(s) and %d AMM pool(s)",
            len(sdk_pairs),
            len(self._amm_pools),
        )

        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            logger.info("Shutting down — joining worker threads (timeout=5s)")
            for t in self._worker_threads:
                t.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_kafka(self) -> None:
        """Kafka backend: produce SSE trades to per-pair topics, score via worker.

        One daemon producer thread per pair forwards Horizon SSE trades into
        Kafka; a :class:`KafkaWorker` consumes them, scores wallets, and
        dispatches alerts. Imports are local so the ``sse`` path stays Kafka-free.
        """
        # Local imports — only reached when STREAMING_BACKEND == "kafka".
        from ingestion.kafka_producer import HorizonKafkaProducer
        from streaming.kafka_worker import KafkaWorker

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())

        self._worker_threads = []
        producer = None
        worker = None

        if self._role in ("all", "producer"):
            sdk_pairs = self._build_sdk_pairs()
            if not sdk_pairs:
                logger.warning("No asset pairs configured — producer has nothing to do")
            else:
                producer = HorizonKafkaProducer()
                for base_asset, counter_asset in sdk_pairs:
                    t = threading.Thread(
                        target=self._produce_pair,
                        args=(producer, base_asset, counter_asset),
                        daemon=True,
                    )
                    t.start()
                    self._worker_threads.append(t)
                logger.info("Kafka producer running with %d pair(s)", len(sdk_pairs))

        if self._role in ("all", "worker"):
            assert self._scorer is not None
            worker = KafkaWorker(
                self._scorer,
                self._dispatcher,
                self._buffer,
                metrics_port=config.KAFKA_METRICS_PORT,
            )
            worker_thread = threading.Thread(target=worker.run, daemon=True)
            worker_thread.start()
            self._worker_threads.append(worker_thread)
            logger.info("Kafka scoring worker running (group=%s)", config.KAFKA_CONSUMER_GROUP)

        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            logger.info("Shutting down Kafka pipeline")
            if worker is not None:
                worker.stop()
            if producer is not None:
                producer.flush()
            for t in self._worker_threads:
                t.join(timeout=5)

    def _produce_pair(self, producer, base_asset: SdkAsset, counter_asset: SdkAsset) -> None:
        pair_label = (
            f"{base_asset.code}:{getattr(base_asset, 'issuer', None) or 'native'}"
            f"/{counter_asset.code}:{getattr(counter_asset, 'issuer', None) or 'native'}"
        )
        while not self._stop_event.is_set():
            try:
                for trade in stream_trades(base_asset, counter_asset):
                    if self._stop_event.is_set():
                        return
                    producer.produce_trade(trade)
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                logger.warning(
                    "Producer stream error for pair %s: %s — will reconnect",
                    pair_label,
                    exc,
                )

    def _build_sdk_pairs(self) -> list[tuple[SdkAsset, SdkAsset]]:
        xlm = SdkAsset.native()
        pairs = []
        for code, issuer in self._pairs:
            asset = SdkAsset.native() if issuer == "native" else SdkAsset(code, issuer)
            if asset == xlm:
                continue
            pairs.append((asset, xlm))
        return pairs

    def _subscribe_wallet(self, wallet: str) -> None:
        """Subscribe *wallet* to Horizon effects via ``AccountMetadataStream``.

        No-ops when the metadata stream is not configured or the wallet is
        already subscribed.  Safe to call from any thread.
        """
        if self._metadata_stream is None:
            return
        with self._subscribed_wallets_lock:
            if wallet in self._subscribed_wallets:
                return
            self._subscribed_wallets.add(wallet)
        self._metadata_stream.add_wallet(wallet)

    def _enrich_from_metadata(self, wallet: str) -> None:
        """Look up latest metadata for *wallet* and apply it to the feature buffer.

        Called on every trade event involving *wallet*.  When join state has
        a valid, in-window metadata entry the buffer's ``apply_metadata``
        method (if it exists) is called so wallet-graph features reflect the
        current on-chain account state.

        Also ensures the wallet is subscribed to Horizon effects so future
        updates are received promptly.
        """
        self._subscribe_wallet(wallet)
        if self._metadata_join_state is None:
            return
        metadata = self._metadata_join_state.get_metadata(wallet)
        if metadata is not None and hasattr(self._buffer, "apply_metadata"):
            try:
                self._buffer.apply_metadata(wallet, metadata)
            except Exception as exc:
                logger.warning(
                    "apply_metadata failed for wallet %s: %s", wallet, exc
                )

    def _trigger_rescore_from_metadata(self, pair_id: str) -> None:
        """Re-score wallets whose metadata changed since the last trade event.

        This is called periodically (or after each metadata update) to ensure
        wallets whose account state changed mid-window are re-scored promptly
        even if no new trade has arrived for them.
        """
        if self._metadata_join_state is None or self._scorer is None:
            return
        for wallet in self._metadata_join_state.wallets_needing_rescore():
            start = time.perf_counter()
            score = self._scorer.score_wallet(wallet, self._buffer)
            lag = time.perf_counter() - start
            METADATA_JOIN_LAG.observe(lag)
            if score is not None:
                self._dispatcher.dispatch(wallet, score, pair_id)

    def _run_housekeeping(self) -> None:
        """Background thread: evict stale join state every housekeeping interval."""
        while not self._stop_event.is_set():
            # Sleep in short increments so the thread exits promptly on stop.
            for _ in range(self._HOUSEKEEPING_INTERVAL_SECONDS * 10):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)
            if self._metadata_join_state is not None:
                self._metadata_join_state.evict_inactive_wallets()

    def _stream_pair(self, base_asset: SdkAsset, counter_asset: SdkAsset) -> None:
        pair_label = (
            f"{base_asset.code}:{getattr(base_asset, 'issuer', None) or 'native'}"
            f"/{counter_asset.code}:{getattr(counter_asset, 'issuer', None) or 'native'}"
        )
        while not self._stop_event.is_set():
            try:
                for trade in stream_trades(base_asset, counter_asset):
                    if self._stop_event.is_set():
                        return
                    self._buffer.update(trade)
                    pair_id = trade.base_asset.pair_id(trade.counter_asset)
                    assert self._scorer is not None
                    for wallet in (trade.base_account, trade.counter_account):
                        # Enrich with latest metadata before scoring.
                        self._enrich_from_metadata(wallet)
                        score = self._scorer.score_wallet(wallet, self._buffer)
                        if score is not None:
                            self._dispatch_with_cb(wallet, score, pair_id)
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                logger.warning(
                    "Stream error for pair %s: %s — will reconnect",
                    pair_label,
                    exc,
                )

    def _stream_amm_pool(self, pool_id: str) -> None:
        while not self._stop_event.is_set():
            try:
                for trade in stream_amm_pool_trades(pool_id):
                    if self._stop_event.is_set():
                        return
                    self._buffer.update(trade)
                    pair_id = trade.base_asset.pair_id(trade.counter_asset)
                    assert self._scorer is not None
                    for wallet in (trade.base_account, trade.counter_account):
                        if not wallet:
                            continue
                        # Enrich with latest metadata before scoring.
                        self._enrich_from_metadata(wallet)
                        score = self._scorer.score_wallet(wallet, self._buffer)
                        if score is not None:
                            self._dispatcher.dispatch(wallet, score, pair_id)
            except PoolNotFoundError as exc:
                logger.error("AMM pool %s not found — stopping stream: %s", pool_id, exc)
                return
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                logger.warning(
                    "AMM stream error for pool %s: %s — will reconnect",
                    pool_id,
                    exc,
                )
