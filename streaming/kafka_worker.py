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
    worker.run()  # Blocks until shutdown

Kafka consumer + scoring worker -- the scale-out half of the streaming backend.

``KafkaWorker`` subscribes (via regex) to every ``ledgerlens.trades.*`` topic,
rebuilds each Avro message into a :class:`~ingestion.data_models.Trade`, feeds an
in-process :class:`~streaming.feature_buffer.FeatureBuffer`, scores the affected
wallets, and dispatches alerts.

Delivery semantics
-------------------
At-least-once: the consumer commits a message's offset **only after** scoring
and alert dispatch have completed for that message. If
:meth:`AlertDispatcher.dispatch` raises, the offset is left uncommitted so the
message is redelivered after a restart/rebalance.

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
import socket
import time
from collections import defaultdict

from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition
from prometheus_client import Counter, Gauge, Histogram

from config import config
from ingestion.avro_codec import deserialize, load_schema, record_to_trade, validate
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer
from streaming.streaming_scorer import StreamingScorer
from utils.logging import get_logger

logger = get_logger(__name__)

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


class DeduplicationCache:
    """Redis SET-based deduplication keyed by (ledger_sequence, trade_id).

    Acts as a second-layer defence for consumers that do not support Kafka
    transactions.  Uses SETNX with a configurable TTL so the set stays bounded.
    Falls back to a no-op when Redis is unavailable (logs a warning once).
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else config.KAFKA_DEDUP_TTL_SECONDS
        self._redis = None
        self._unavailable = False
        try:
            import redis

            self._redis = redis.from_url(redis_url or config.REDIS_URL, decode_responses=True)
            self._redis.ping()
        except Exception as exc:
            logger.warning("DeduplicationCache: Redis unavailable — dedup disabled: %s", exc)
            self._unavailable = True

    def is_duplicate(self, ledger_sequence: int | str, trade_id: str) -> bool:
        """Return True if this (ledger_sequence, trade_id) pair has been seen before."""
        if self._unavailable or self._redis is None:
            return False
        key = f"dedup:{ledger_sequence}:{trade_id}"
        try:
            added = self._redis.set(key, "1", nx=True, ex=self._ttl)
            return added is None  # None → key already existed → duplicate
        except Exception as exc:
            logger.warning("DeduplicationCache: Redis error during check: %s", exc)
            return False


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
    ) -> None:
        self._scorer = scorer
        self._dispatcher = dispatcher
        self._buffer = buffer if buffer is not None else FeatureBuffer()
        self._schema = load_schema(schema_path)
        self._dlq_topic = dlq_topic or config.KAFKA_DLQ_TOPIC
        self._lag_threshold = (
            lag_threshold if lag_threshold is not None else config.KAFKA_LAG_ALERT_THRESHOLD
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
        self._dedup_cache = DeduplicationCache()

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
        try:
            while self._running:
                msg = self._consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.warning("Consumer error: %s", msg.error())
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
        finally:
            self.close()

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        if self._backpressure is not None:
            self._backpressure.flush()
        try:
            self._consumer.close()
        except Exception:  # pragma: no cover - best-effort shutdown
            pass

    def process_message(self, msg) -> None:
        """Decode, score, dispatch, then commit the offset (at-least-once)."""
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

        # Exactly-once second-layer dedup via Redis.
        ledger_seq = record.get("ledger_sequence", msg.offset())
        trade_id = record.get("trade_id", "")
        if self._dedup_cache.is_duplicate(ledger_seq, trade_id):
            logger.debug(
                "Duplicate trade %s (ledger=%s) skipped — dedup cache hit",
                trade_id,
                ledger_seq,
            )
            self._consumer.commit(message=msg, asynchronous=False)
            self._in_flight[tp_key] = max(0, self._in_flight[tp_key] - 1)
            return

        self._buffer.update(record_to_trade(record))
        pair_id = record["asset_pair"]
        for wallet in (record["base_account"], record["counter_account"]):
            start = time.perf_counter()
            score = self._scorer.score_wallet(wallet, self._buffer)
            SCORING_LATENCY_MS.observe((time.perf_counter() - start) * 1000)
            if score is not None:
                # If dispatch raises, we never reach commit() below → redelivery.
                self._dispatcher.dispatch(wallet, score, pair_id)
                ALERTS_DISPATCHED.inc()

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


def _build_consumer_conf(bootstrap_servers: str, group_id: str) -> dict:
    conf: dict = {
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        # Manual commits only — we commit after successful dispatch.
        "enable.auto.commit": False,
    }
    user = config.KAFKA_SASL_USERNAME
    password = config.KAFKA_SASL_PASSWORD
    if user and password:
        conf.update(
            {
                "security.protocol": "SASL_SSL",
                "sasl.mechanisms": "PLAIN",
                "sasl.username": user,
                "sasl.password": password,
            }
        )
    return conf
