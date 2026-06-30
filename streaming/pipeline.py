"""Top-level orchestrator for the real-time detection pipeline.

StreamingPipeline starts one daemon thread per watched asset pair, drives
each through stream_trades(), and wires the FeatureBuffer → StreamingScorer →
AlertDispatcher chain.

Reconnection on Horizon SSE failures is handled at two levels:
  1. stream_trades() retries internally (up to max_reconnect_attempts).
  2. _stream_pair() restarts the generator if stream_trades() raises after
     exhausting its own retries.

Shutdown
--------
Call pipeline.run() from the main thread.  SIGINT (Ctrl-C) sets the internal
stop event via a signal handler; the main loop wakes up, joins all worker
threads with a 5-second timeout, and returns.
"""

import signal
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime

from stellar_sdk import Asset as SdkAsset

from config import config
from ingestion.amm_pool_loader import PoolNotFoundError, stream_amm_pool_trades
from ingestion.horizon_streamer import stream_trades
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer
from streaming.streaming_scorer import StreamingScorer
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import Gauge

    _WATERMARK_LAG = Gauge(
        "watermark_lag_seconds",
        "Seconds between current wall-clock time and the per-(wallet,pair) watermark",
        ["wallet", "pair"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


class WatermarkManager:
    """Tracks per-(wallet, pair) event-time watermarks and buffers late events.

    The watermark for a (wallet, pair) key is defined as:
        max_observed_timestamp(wallet, pair) - allowed_lateness_seconds

    Events whose ``ledger_close_time`` falls before the current watermark are
    classified as *late* and stored in ``_late_buffer`` so they can be replayed
    into the correct historical window before its final aggregation.

    Timestamps are sourced exclusively from ``trade.ledger_close_time``
    (the Stellar ledger close time) to prevent client-supplied timestamp
    spoofing from altering window assignment.

    The watermark advances monotonically — a late event never causes it to
    decrease.
    """

    def __init__(self, allowed_lateness_seconds: int | None = None) -> None:
        self._allowed_lateness: int = (
            allowed_lateness_seconds
            if allowed_lateness_seconds is not None
            else config.WATERMARK_ALLOWED_LATENESS_SECONDS
        )
        # (wallet, pair) -> max observed ledger_close_time as epoch seconds
        self._max_ts: dict[tuple[str, str], float] = defaultdict(float)
        # (wallet, pair) -> list of late Trade objects pending replay
        self._late_buffer: dict[tuple[str, str], list] = defaultdict(list)
        self._lock = threading.Lock()

    def _pair_key(self, trade) -> tuple[str, str]:
        return (trade.base_account, trade.base_asset.pair_id(trade.counter_asset))

    def _ts(self, trade) -> float:
        """Return epoch seconds from trade.ledger_close_time (always ledger time)."""
        t = trade.ledger_close_time
        if isinstance(t, datetime):
            return t.timestamp()
        # Already a numeric epoch value
        return float(t)

    def watermark(self, wallet: str, pair: str) -> float:
        """Return the current watermark (epoch seconds) for (wallet, pair)."""
        with self._lock:
            return self._max_ts[(wallet, pair)] - self._allowed_lateness

    def process(self, trade) -> tuple[bool, list]:
        """Classify *trade* as on-time or late and update the watermark.

        Returns:
            (is_late, replayed_trades)
            - is_late: True when trade.ledger_close_time < current watermark
            - replayed_trades: list of previously buffered late trades that are
              now safe to replay (their timestamps are newer than earlier late
              events but still within the allowed lateness window)
        """
        wallet = trade.base_account
        pair = trade.base_asset.pair_id(trade.counter_asset)
        ts = self._ts(trade)

        with self._lock:
            current_max = self._max_ts[(wallet, pair)]
            wm = current_max - self._allowed_lateness

            if ts < wm:
                # Late event — buffer it, watermark must not decrease
                self._late_buffer[(wallet, pair)].append(trade)
                self._emit_lag(wallet, pair, current_max)
                return True, []

            # On-time event — advance watermark monotonically
            if ts > current_max:
                self._max_ts[(wallet, pair)] = ts

            new_wm = self._max_ts[(wallet, pair)] - self._allowed_lateness
            replayed = self._drain_late_buffer(wallet, pair, new_wm)
            self._emit_lag(wallet, pair, self._max_ts[(wallet, pair)])
            return False, replayed

    def _drain_late_buffer(self, wallet: str, pair: str, watermark: float) -> list:
        """Return buffered late events whose timestamps are >= watermark."""
        key = (wallet, pair)
        buf = self._late_buffer[key]
        if not buf:
            return []
        still_late = [t for t in buf if self._ts(t) < watermark]
        replayable = [t for t in buf if self._ts(t) >= watermark]
        self._late_buffer[key] = still_late
        return sorted(replayable, key=self._ts)

    def _emit_lag(self, wallet: str, pair: str, max_ts: float) -> None:
        if not _PROMETHEUS_AVAILABLE:
            return
        lag = time.time() - (max_ts - self._allowed_lateness)
        try:
            _WATERMARK_LAG.labels(wallet=wallet, pair=pair).set(max(0.0, lag))
        except Exception:
            pass


class StreamingPipeline:
    """Orchestrates one SSE thread per pair and wires the scoring pipeline."""

    def __init__(
        self,
        buffer: FeatureBuffer,
        scorer: StreamingScorer | None,
        dispatcher: AlertDispatcher,
        pairs: list[tuple[str, str]] | None = None,
        amm_pools: list[str] | None = None,
        role: str = "all",
        watermark_manager: WatermarkManager | None = None,
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
        self._role = role
        self._stop_event = threading.Event()
        self._worker_threads: list[threading.Thread] = []
        self._watermark = watermark_manager if watermark_manager is not None else WatermarkManager()

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

    def _process_trade(self, trade) -> None:
        """Apply watermark classification then score and dispatch."""
        is_late, replayed = self._watermark.process(trade)
        pair_id = trade.base_asset.pair_id(trade.counter_asset)
        assert self._scorer is not None

        if is_late:
            logger.debug(
                "Late event buffered: trade_id=%s ledger_close_time=%s",
                trade.trade_id,
                trade.ledger_close_time,
            )
            return

        # Replay any late events that are now within the window before processing
        # the on-time event so historical windows are updated in chronological order.
        for late_trade in replayed:
            self._buffer.update(late_trade)
            for wallet in (late_trade.base_account, late_trade.counter_account):
                score = self._scorer.score_wallet(wallet, self._buffer)
                if score is not None:
                    self._dispatcher.dispatch(wallet, score, pair_id)

        self._buffer.update(trade)
        for wallet in (trade.base_account, trade.counter_account):
            score = self._scorer.score_wallet(wallet, self._buffer)
            if score is not None:
                self._dispatcher.dispatch(wallet, score, pair_id)

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
                    self._process_trade(trade)
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
                    is_late, replayed = self._watermark.process(trade)
                    pair_id = trade.base_asset.pair_id(trade.counter_asset)
                    assert self._scorer is not None

                    if is_late:
                        logger.debug(
                            "Late AMM event buffered: pool=%s trade_id=%s",
                            pool_id,
                            trade.trade_id,
                        )
                        continue

                    for late_trade in replayed:
                        self._buffer.update(late_trade)
                        for wallet in (late_trade.base_account, late_trade.counter_account):
                            if not wallet:
                                continue
                            score = self._scorer.score_wallet(wallet, self._buffer)
                            if score is not None:
                                self._dispatcher.dispatch(wallet, score, pair_id)

                    self._buffer.update(trade)
                    for wallet in (trade.base_account, trade.counter_account):
                        if not wallet:
                            continue
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
