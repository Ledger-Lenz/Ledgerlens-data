"""Per-partition Kafka worker for parallel trade processing.

Each worker handles a fixed set of partitions and maintains per-pair Benford
window state. On startup, workers assign themselves to a subset of partitions
(either via round-robin or explicit partition list). Rebalancing is handled
gracefully by committing offsets before partition revocation.

Architecture:
    - Consumer group (e.g., "ledgerlens-workers") with multiple members
    - Each worker subscribes to same topic; Kafka assigns partitions
    - Per-worker state: FeatureBuffer, StreamingScorer, Benford windows
    - Offset commit on interval + on rebalance

Usage:
    worker = KafkaWorker(
        topic="trades",
        group_id="ledgerlens-workers",
        bootstrap_servers=["localhost:9092"],
    )
    worker.install_signal_handlers()  # SIGTERM → drain in-flight, commit, exit
    worker.run()  # Blocks until shutdown

Kafka consumer + scoring worker -- the scale-out half of the streaming backend.

``KafkaWorker`` subscribes (via regex) to every ``ledgerlens.trades.*`` topic,
rebuilds each Avro message into a :class:`~ingestion.data_models.Trade`, feeds an
in-process :class:`~streaming.feature_buffer.FeatureBuffer`, scores the affected
wallets, and dispatches alerts.

Delivery semantics
-------------------
Exactly-once *effects*, at-least-once *delivery* (Issue #670): the consumer
commits a message's offset **only after** its dedup key has been durably
committed in :class:`~pipeline.exactly_once.ExactlyOnceStore`, which in turn
only happens **after** scoring and alert dispatch have completed for both
wallets. The dedup key is *staged* (not committed) before processing begins.
If :meth:`AlertDispatcher.dispatch` raises, neither the dedup key nor the
Kafka offset is committed, so the message is redelivered after a
restart/rebalance and is correctly recognised as "redo this", not "already
done" — see :meth:`KafkaWorker._process_correlated_message` and
``docs/adr/0001-unified-idempotency-finality.md`` for the full ordering
argument and the bug this replaced.

Poison-pill protection
-----------------------
Every message is Avro-validated before it reaches the scorer. Messages that fail
to decode or validate are logged, counted, and their offset committed (skipped)
so a single bad record cannot wedge a partition. The dead-letter topic is never
processed.

Lag alerting
------------
Per-partition consumer lag is published as a Prometheus gauge. When lag exceeds
``KAFKA_LAG_ALERT_THRESHOLD`` a CRITICAL log is emitted — the worker keeps
running rather than crashing.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from collections import defaultdict

from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition
from prometheus_client import Counter, Gauge, Histogram

from config import config
from ingestion.avro_codec import deserialize, load_schema, record_to_trade, validate
from monitoring.capacity_metrics import record_e2e_latency
from monitoring.emergency_watchdog import EmergencyWatchdog
from pipeline.exactly_once import (
    DedupBackendUnavailableError,
    DedupKey,
    DedupState,
    ExactlyOnceStore,
    RedisExactlyOnceBackend,
)
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer
from streaming.health import WorkerHealthMonitor, get_health_registry
from streaming.streaming_scorer import StreamingScorer
from utils.logging import correlation_id_from_headers, get_logger, log_context

logger = get_logger(__name__)

# Graceful shutdown (#892): max seconds to wait for the in-flight message to
# finish after SIGTERM. On expiry the process exits with DRAIN_TIMEOUT_EXIT_CODE
# WITHOUT committing the in-flight offset — it is redelivered on restart and the
# staged-but-uncommitted dedup key prevents double-processing.
KAFKA_DRAIN_TIMEOUT_SECONDS: float = float(os.getenv("KAFKA_DRAIN_TIMEOUT_SECONDS", "30"))
DRAIN_TIMEOUT_EXIT_CODE = 75

# Module-level metrics — registered once and shared across worker instances.
KAFKA_MESSAGES_CONSUMED = Counter(
    "kafka_messages_consumed_total",
    "Total trade messages fully processed by the scoring worker",
)
KAFKA_LAG_BY_PARTITION = Gauge(
    "kafka_lag_by_partition",
    "Consumer lag (messages behind the high watermark) per topic partition",
    ["topic", "partition"],
)
SCORING_LATENCY_MS = Histogram(
    "scoring_latency_ms",
    "Per-wallet scoring latency in milliseconds",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000),
)
ALERTS_DISPATCHED = Counter(
    "alerts_dispatched_total",
    "Total alerts dispatched by the scoring worker",
)
KAFKA_POISON_MESSAGES = Counter(
    "kafka_poison_messages_total",
    "Total messages dropped because they failed Avro decode/validation",
)
KAFKA_BACKPRESSURE_PAUSES = Counter(
    "kafka_backpressure_pauses_total",
    "Total times a partition was paused due to high-water-mark back-pressure",
    ["topic", "partition"],
)
KAFKA_BACKPRESSURE_RESUMES = Counter(
    "kafka_backpressure_resumes_total",
    "Total times a partition was resumed after draining below low-water-mark",
    ["topic", "partition"],
)
KAFKA_DEDUP_BACKEND_DEGRADED = Counter(
    "kafka_dedup_backend_degraded_total",
    "Total times the exactly-once dedup backend was unreachable while processing a message",
)


class DeduplicationCache:
    """Two-phase exactly-once dedup for the Kafka worker, keyed by
    ``(ledger_sequence, trade_id)`` via :class:`~pipeline.exactly_once.ExactlyOnceStore`.

    This is a thin, source-scoped wrapper around the shared library (Issue
    #670) — it replaces the previous single-phase Redis ``SETNX`` cache,
    which marked a message "seen" *before* its side effects were known to
    have completed and fell back to "not a duplicate" on any Redis error.
    Both of those were the root cause of the critical silent-drop bug this
    issue fixes: see ``docs/adr/0001-unified-idempotency-finality.md``.

    Raises ``DedupBackendUnavailableError`` (imported from
    ``pipeline.exactly_once``) from :meth:`check_and_stage`/:meth:`commit`
    when Redis is unreachable — there is no silent "allow through" fallback.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else config.KAFKA_DEDUP_TTL_SECONDS
        backend = RedisExactlyOnceBackend(
            redis_url or config.REDIS_URL, key_prefix="ledgerlens:kafka_dedup:"
        )
        self._store = ExactlyOnceStore(backend, ttl_seconds=float(ttl))

    @staticmethod
    def _key(ledger_sequence: int | str, trade_id: str) -> DedupKey:
        return DedupKey(source="kafka_trade", external_id=f"{ledger_sequence}:{trade_id}")

    def check_and_stage(self, ledger_sequence: int | str, trade_id: str) -> DedupState:
        """Stage the key for this message and return its dedup state.

        ``DedupState.COMMITTED`` means the message's side effects already
        durably completed — skip reprocessing. ``DedupState.NEW`` or
        ``DedupState.STAGED`` mean it must be (re)processed.
        """
        return self._store.check_and_stage(self._key(ledger_sequence, trade_id)).state

    def commit(self, ledger_sequence: int | str, trade_id: str) -> None:
        """Durably mark this message's side effects as complete."""
        self._store.commit(self._key(ledger_sequence, trade_id))

    def is_available(self) -> bool:
        return self._store.is_available()


class BackPressureController:
    """Per-partition back-pressure: pauses/resumes the Kafka consumer partition
    assignment when the in-flight queue depth crosses HWM/LWM thresholds.

    Pause/resume are issued on the *consumer* object so other partitions continue
    flowing while a slow partition drains.  A fire-and-forget DLT producer routes
    messages that have exceeded ``max_retries`` without successful processing.
    """

    def __init__(
        self,
        consumer: Consumer,
        hwm: int | None = None,
        lwm: int | None = None,
        max_retries: int | None = None,
        dead_letter_topic: str | None = None,
        bootstrap_servers: str | None = None,
    ) -> None:
        self._consumer = consumer
        self._hwm = hwm if hwm is not None else config.KAFKA_BACKPRESSURE_HWM
        self._lwm = lwm if lwm is not None else config.KAFKA_BACKPRESSURE_LWM
        self._max_retries = max_retries if max_retries is not None else config.KAFKA_MAX_RETRIES
        self._dlt_topic = dead_letter_topic or config.KAFKA_DEAD_LETTER_TOPIC
        self._paused: set[tuple[str, int]] = set()
        self._retry_counts: dict[str, int] = defaultdict(int)

        dlt_conf: dict = {
            "bootstrap.servers": bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS,
        }
        dlt_conf.update(_sasl_conf())
        self._dlt_producer = Producer(dlt_conf)

    def check_and_apply(self, topic_partition: TopicPartition, queue_depth: int) -> None:
        """Pause or resume *topic_partition* based on *queue_depth*."""
        key = (topic_partition.topic, topic_partition.partition)
        if queue_depth >= self._hwm and key not in self._paused:
            self._consumer.pause([topic_partition])
            self._paused.add(key)
            KAFKA_BACKPRESSURE_PAUSES.labels(
                topic=topic_partition.topic, partition=topic_partition.partition
            ).inc()
            logger.info(
                "Back-pressure: paused %s[%d] at queue depth %d (HWM=%d)",
                topic_partition.topic,
                topic_partition.partition,
                queue_depth,
                self._hwm,
            )
        elif queue_depth <= self._lwm and key in self._paused:
            self._consumer.resume([topic_partition])
            self._paused.discard(key)
            KAFKA_BACKPRESSURE_RESUMES.labels(
                topic=topic_partition.topic, partition=topic_partition.partition
            ).inc()
            logger.info(
                "Back-pressure: resumed %s[%d] at queue depth %d (LWM=%d)",
                topic_partition.topic,
                topic_partition.partition,
                queue_depth,
                self._lwm,
            )

    def record_failure(self, msg, reason: str) -> None:
        """Increment retry counter for *msg*; route to DLT when max_retries exceeded.

        The DLT producer is fire-and-forget (poll(0)) so it never blocks the
        main consumer loop.  All original message headers are preserved.
        """
        msg_key = f"{msg.topic()}:{msg.partition()}:{msg.offset()}"
        self._retry_counts[msg_key] += 1
        if self._retry_counts[msg_key] > self._max_retries:
            headers = list(msg.headers() or []) + [
                ("reason", reason.encode("utf-8")),
                ("original_topic", msg.topic().encode("utf-8")),
                ("original_partition", str(msg.partition()).encode("utf-8")),
                ("original_offset", str(msg.offset()).encode("utf-8")),
            ]
            try:
                self._dlt_producer.produce(
                    topic=self._dlt_topic,
                    value=msg.value(),
                    key=msg.key(),
                    headers=headers,
                )
                self._dlt_producer.poll(0)
                logger.info(
                    "Routed %s[%d]@%d to DLT after %d retries",
                    msg.topic(),
                    msg.partition(),
                    msg.offset(),
                    self._retry_counts[msg_key],
                )
            except Exception as exc:  # pragma: no cover - best-effort DLT delivery
                logger.error("Failed to produce to DLT: %s", exc)
            del self._retry_counts[msg_key]

    def flush(self) -> None:
        self._dlt_producer.flush(5.0)


class KafkaWorker:
    """Consumes Avro trade messages, scores wallets, and dispatches alerts."""

    def __init__(
        self,
        scorer: StreamingScorer,
        dispatcher: AlertDispatcher,
        buffer: FeatureBuffer | None = None,
        watchdog: EmergencyWatchdog | None = None,
        *,
        consumer: Consumer | None = None,
        bootstrap_servers: str | None = None,
        group_id: str | None = None,
        topic_pattern: str | None = None,
        dlq_topic: str | None = None,
        lag_threshold: int | None = None,
        schema_path: str | None = None,
        metrics_port: int | None = None,
        enable_backpressure: bool = True,
        dedup_cache: DeduplicationCache | None = None,
        drain_timeout: float | None = None,
    ) -> None:
        self._scorer = scorer
        self._drain_timeout = (
            drain_timeout if drain_timeout is not None else KAFKA_DRAIN_TIMEOUT_SECONDS
        )
        self._shutdown_requested = threading.Event()
        self._drain_timer: threading.Timer | None = None
        self._dispatcher = dispatcher
        self._buffer = buffer if buffer is not None else FeatureBuffer()
        self._watchdog = watchdog
        self._schema = load_schema(schema_path)
        self._dlq_topic = dlq_topic or config.KAFKA_DLQ_TOPIC
        self._lag_threshold = (
            lag_threshold if lag_threshold is not None else config.KAFKA_LAG_ALERT_THRESHOLD
        )
        if self._lag_threshold is None or not isinstance(self._lag_threshold, int):
            raise ValueError(
                f"KAFKA_LAG_ALERT_THRESHOLD must be a positive integer, got "
                f"{self._lag_threshold!r}"
            )
        if self._lag_threshold <= 0:
            raise ValueError(
                f"KAFKA_LAG_ALERT_THRESHOLD must be a positive integer, got "
                f"{self._lag_threshold!r}"
            )
        self._metrics_port = metrics_port
        self._running = False
        self._in_flight: dict[tuple[str, int], int] = defaultdict(int)

        servers = bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS
        if consumer is not None:
            self._consumer = consumer
        else:
            self._consumer = Consumer(
                _build_consumer_conf(servers, group_id or config.KAFKA_CONSUMER_GROUP)
            )
            self._consumer.subscribe([topic_pattern or config.KAFKA_TOPIC_PATTERN])

        self._backpressure: BackPressureController | None = (
            BackPressureController(self._consumer, bootstrap_servers=servers)
            if enable_backpressure
            else None
        )
        self._dedup_cache = dedup_cache if dedup_cache is not None else DeduplicationCache()
        self._health_monitor = WorkerHealthMonitor("kafka_worker")
        get_health_registry().register(self._health_monitor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Poll-and-process loop. Blocks until :meth:`stop` is called."""
        if self._metrics_port is not None:
            from prometheus_client import start_http_server

            start_http_server(self._metrics_port)
            logger.info("Prometheus metrics exposed on :%d", self._metrics_port)

        self._running = True
        logger.info("KafkaWorker started — consuming trade topics")

        from streaming.health_check import heartbeat

        try:
            while self._running and not self._shutdown_requested.is_set():
                heartbeat(threading.current_thread().name)
                msg = self._consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.warning("Consumer error: %s", msg.error())
                    self._health_monitor.update_details("last_consumer_error", str(msg.error()))
                    continue
                try:
                    self.process_message(msg)
                except Exception as exc:
                    # Offset deliberately not committed → redelivered later.
                    logger.error(
                        "Processing failed for %s[%d]@%d — offset not committed: %s",
                        msg.topic(),
                        msg.partition(),
                        msg.offset(),
                        exc,
                    )
                    self._health_monitor.record_heartbeat(error_message=str(exc))
        finally:
            self.close()

    def install_signal_handlers(self) -> None:
        """Route SIGTERM/SIGINT to :meth:`request_shutdown` (main thread only)."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, _frame: self.request_shutdown(signum))

    def request_shutdown(self, signum: int | None = None) -> None:
        """Stop polling; let the in-flight message finish and commit, then exit.

        Offsets are committed synchronously per message inside
        :meth:`process_message`, so once the current message returns there is
        no partial state. A drain timer enforces ``drain_timeout``.
        """
        if self._shutdown_requested.is_set():
            return
        logger.info("Graceful shutdown requested (signal=%s) — draining in-flight work", signum)
        self._shutdown_requested.set()
        self._running = False
        if self._drain_timeout > 0:
            self._drain_timer = threading.Timer(self._drain_timeout, self._on_drain_timeout)
            self._drain_timer.daemon = True
            self._drain_timer.start()

    def _on_drain_timeout(self) -> None:
        """Fallback: drain exceeded — exit without committing the in-flight offset."""
        logger.error(
            "Drain timeout (%.1fs) exceeded — exiting without committing in-flight offset; "
            "message will be redelivered and deduplicated",
            self._drain_timeout,
        )
        os._exit(DRAIN_TIMEOUT_EXIT_CODE)

    def stop(self) -> None:
        self._running = False
        self._health_monitor.mark_stopped("KafkaWorker stopped")

    def close(self) -> None:
        if self._drain_timer is not None:
            self._drain_timer.cancel()
        self._health_monitor.mark_stopped("KafkaWorker closed")
        if self._backpressure is not None:
            self._backpressure.flush()
        try:
            self._consumer.close()
        except Exception:  # pragma: no cover - best-effort shutdown
            pass

    def process_message(self, msg) -> None:
        """Decode, score, dispatch, then commit the offset (at-least-once)."""
        correlation_id = correlation_id_from_headers(msg.headers())
        if correlation_id is None:
            correlation_id = f"kafka-{msg.topic()}-{msg.partition()}-{msg.offset()}"
        with log_context(
            correlation_id=correlation_id,
            pipeline_stage="streaming.kafka_worker",
            kafka_topic=msg.topic(),
            kafka_partition=msg.partition(),
            kafka_offset=msg.offset(),
        ):
            self._process_correlated_message(msg)

    def _process_correlated_message(self, msg) -> None:
        """Process one Kafka message with its transport correlation context bound."""
        # Never process the dead-letter topic — DLQ/DLT requires human review.
        if msg.topic() in (self._dlq_topic, config.KAFKA_DEAD_LETTER_TOPIC):
            self._consumer.commit(message=msg, asynchronous=False)
            return

        tp_key = (msg.topic(), msg.partition())
        self._in_flight[tp_key] += 1
        if self._backpressure is not None:
            self._backpressure.check_and_apply(
                TopicPartition(msg.topic(), msg.partition()),
                self._in_flight[tp_key],
            )

        try:
            record = deserialize(msg.value(), self._schema)
            validate(record, self._schema)
        except Exception as exc:
            logger.error(
                "Poison-pill message on %s[%d]@%d dropped: %s",
                msg.topic(),
                msg.partition(),
                msg.offset(),
                exc,
            )
            KAFKA_POISON_MESSAGES.inc()
            if self._backpressure is not None:
                self._backpressure.record_failure(msg, str(exc))
            self._consumer.commit(message=msg, asynchronous=False)
            self._in_flight[tp_key] = max(0, self._in_flight[tp_key] - 1)
            return

        self._check_lag(msg)

        # Exactly-once dedup: stage the key BEFORE any side effect runs, and
        # only commit it AFTER scoring + dispatch durably succeed for both
        # wallets (see the docstring at the top of this module and
        # docs/adr/0001-unified-idempotency-finality.md). Staging before
        # processing — rather than the previous single-phase "mark seen then
        # process" — is what makes a crash between staging and commit safe:
        # redelivery sees STAGED ("redo this"), never a false COMMITTED.
        ledger_seq = record.get("ledger_sequence", msg.offset())
        trade_id = record.get("trade_id", "")
        try:
            dedup_state = self._dedup_cache.check_and_stage(ledger_seq, trade_id)
        except DedupBackendUnavailableError as exc:
            # Fail closed: never treat "unknown" as "not a duplicate". Leave
            # the offset uncommitted so this message is retried (halting
            # effective progress on this partition) rather than risking a
            # silent duplicate-processing or silent-drop outcome.
            KAFKA_DEDUP_BACKEND_DEGRADED.inc()
            self._health_monitor.update_details("dedup_backend_degraded", str(exc))
            logger.error(
                "Dedup backend unavailable for %s[%d]@%d — refusing to process "
                "without a dedup guarantee: %s",
                msg.topic(),
                msg.partition(),
                msg.offset(),
                exc,
            )
            self._in_flight[tp_key] = max(0, self._in_flight[tp_key] - 1)
            raise

        if dedup_state is DedupState.COMMITTED:
            logger.debug(
                "Duplicate trade %s (ledger=%s) skipped — dedup key already committed",
                trade_id,
                ledger_seq,
            )
            self._consumer.commit(message=msg, asynchronous=False)
            self._in_flight[tp_key] = max(0, self._in_flight[tp_key] - 1)
            return

        # dedup_state is NEW or STAGED (a prior attempt staged this key but
        # crashed before committing it) — process (or reprocess) the message.
        # FeatureBuffer.update is idempotent per (wallet, trade_id), so
        # reprocessing a STAGED message never double-counts feature state.
        self._buffer.update(record_to_trade(record))
        pair_id = record["asset_pair"]

        ingestion_timestamp_ms = None
        if msg.headers():
            for k, v in msg.headers():
                if k == "ingestion_timestamp_ms":
                    try:
                        ingestion_timestamp_ms = int(v.decode("utf-8"))
                    except Exception:
                        pass

        for wallet in (record["base_account"], record["counter_account"]):
            start = time.perf_counter()
            score = self._scorer.score_wallet(wallet, self._buffer)
            SCORING_LATENCY_MS.observe((time.perf_counter() - start) * 1000)
            if score is not None:
                # If dispatch raises, we never reach commit() below → redelivery.
                self._dispatcher.dispatch(wallet, score, pair_id)
                ALERTS_DISPATCHED.inc()

                e2e_latency_ms = None
                if ingestion_timestamp_ms is not None:
                    e2e_latency_ms = (time.time() * 1000) - ingestion_timestamp_ms
                    record_e2e_latency(e2e_latency_ms / 1000.0)

                if self._watchdog is not None:
                    self._watchdog.record_score(wallet, score, e2e_latency_ms)

        # Commit the dedup key BEFORE the Kafka offset: if the process dies
        # between these two commits, redelivery sees COMMITTED and just
        # advances the offset without reprocessing — no double dispatch, no
        # dropped wallet. If dispatch raised above, neither commit is
        # reached, and the message is correctly redelivered as STAGED.
        try:
            self._dedup_cache.commit(ledger_seq, trade_id)
        except DedupBackendUnavailableError as exc:
            KAFKA_DEDUP_BACKEND_DEGRADED.inc()
            self._health_monitor.update_details("dedup_backend_degraded", str(exc))
            logger.error(
                "Dedup backend unavailable while committing %s[%d]@%d — offset left "
                "uncommitted; side effects already ran and are safe to redo on retry: %s",
                msg.topic(),
                msg.partition(),
                msg.offset(),
                exc,
            )
            self._in_flight[tp_key] = max(0, self._in_flight[tp_key] - 1)
            raise

        self._consumer.commit(message=msg, asynchronous=False)
        KAFKA_MESSAGES_CONSUMED.inc()
        self._in_flight[tp_key] = max(0, self._in_flight[tp_key] - 1)
        if self._backpressure is not None:
            self._backpressure.check_and_apply(
                TopicPartition(msg.topic(), msg.partition()),
                self._in_flight[tp_key],
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_lag(self, msg) -> None:
        try:
            _, high = self._consumer.get_watermark_offsets(
                TopicPartition(msg.topic(), msg.partition()), timeout=1.0, cached=True
            )
        except Exception:  # pragma: no cover - watermark unavailable
            return

        lag = max(0, high - (msg.offset() + 1))
        KAFKA_LAG_BY_PARTITION.labels(topic=msg.topic(), partition=msg.partition()).set(lag)
        if lag > self._lag_threshold:
            logger.critical(
                "Kafka consumer lag %d on %s[%d] exceeds threshold %d",
                lag,
                msg.topic(),
                msg.partition(),
                self._lag_threshold,
            )


def _sasl_conf() -> dict:
    """SASL_SSL/PLAIN credentials for Kafka clients, or {} when unset."""
    user = config.KAFKA_SASL_USERNAME
    password = config.KAFKA_SASL_PASSWORD
    if user and password:
        return {
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": user,
            "sasl.password": password,
        }
    return {}


def _build_consumer_conf(bootstrap_servers: str, group_id: str) -> dict:
    conf: dict = {
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        # Manual commits only — we commit after successful dispatch.
        "enable.auto.commit": False,
    }
    conf.update(_sasl_conf())
    return conf
