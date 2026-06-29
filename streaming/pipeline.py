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

import signal
import threading
import time

from stellar_sdk import Asset as SdkAsset

from config import config
from ingestion.amm_pool_loader import PoolNotFoundError, stream_amm_pool_trades
from ingestion.horizon_streamer import stream_trades
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer
from streaming.streaming_scorer import StreamingScorer
from utils.logging import get_logger

logger = get_logger(__name__)


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
