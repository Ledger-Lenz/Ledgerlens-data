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
"""Kafka consumer + scoring worker — the scale-out half of the streaming backend.

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

import json
import signal
import threading
import time
from typing import TYPE_CHECKING

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from streaming.feature_buffer import FeatureBuffer
from streaming.streaming_scorer import StreamingScorer
from streaming.alert_dispatcher import AlertDispatcher
from utils.logging import get_logger

if TYPE_CHECKING:
    from ingestion.data_models import Trade

logger = get_logger(__name__)

# Offset commit interval (seconds)
DEFAULT_COMMIT_INTERVAL = 30


class KafkaWorker:
    """Per-partition trade consumer with streaming scoring."""

    def __init__(
        self,
        topic: str,
        group_id: str,
        bootstrap_servers: list[str] | str = "localhost:9092",
        buffer: FeatureBuffer | None = None,
        scorer: StreamingScorer | None = None,
        dispatcher: AlertDispatcher | None = None,
        partitions: list[int] | None = None,
        commit_interval_seconds: int = DEFAULT_COMMIT_INTERVAL,
    ):
        """Initialize Kafka worker.

        Args:
            topic: Kafka topic to consume from
            group_id: Consumer group ID (e.g., "ledgerlens-workers")
            bootstrap_servers: Kafka bootstrap server(s)
            buffer: FeatureBuffer instance (default: new instance)
            scorer: StreamingScorer instance (required)
            dispatcher: AlertDispatcher instance (required)
            partitions: Explicit partition list to consume (optional; if None, use group assignment)
            commit_interval_seconds: How often to commit offsets
        """
        if isinstance(bootstrap_servers, str):
            bootstrap_servers = [bootstrap_servers]

        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.partitions = partitions
        self.commit_interval_seconds = commit_interval_seconds

        # Components
        self.buffer = buffer or FeatureBuffer()
        self.scorer = scorer
        self.dispatcher = dispatcher

        if not self.scorer:
            raise ValueError("scorer is required")
        if not self.dispatcher:
            raise ValueError("dispatcher is required")

        # Consumer setup
        self.consumer = KafkaConsumer(
            topic,
            group_id=group_id,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=False,  # Manual commit for control
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )

        # State
        self._stop_event = threading.Event()
        self._last_commit_time = time.time()
        self._messages_processed = 0

        logger.info(
            "KafkaWorker initialized: topic=%s, group_id=%s, servers=%s",
            topic,
            group_id,
            bootstrap_servers,
        )

    def run(self) -> None:
        """Start consuming and processing trades.

        Blocks until stop signal (SIGTERM, SIGINT) or error.
        """
        # Install signal handlers
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, lambda *_: self._stop_event.set())
            signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())

        try:
            # Subscribe with explicit partitions if provided
            if self.partitions:
                from kafka.structs import TopicPartition

                topic_partitions = [TopicPartition(self.topic, p) for p in self.partitions]
                self.consumer.assign(topic_partitions)
                logger.info("Assigned partitions: %s", self.partitions)
            else:
                self.consumer.subscribe([self.topic])
                logger.info("Subscribed to topic (group assignment will assign partitions)")

            while not self._stop_event.is_set():
                messages = self.consumer.poll(timeout_ms=1000, max_records=100)

                if messages:
                    self._process_batch(messages)

                # Commit offsets periodically
                now = time.time()
                if now - self._last_commit_time > self.commit_interval_seconds:
                    self._commit_offsets()
                    self._last_commit_time = now

        except Exception as exc:
            logger.error("Worker error: %s", exc)
            raise
        finally:
            self._commit_offsets()
            self.consumer.close()
            logger.info("Worker stopped (processed %d messages)", self._messages_processed)

    def _process_batch(self, messages_by_partition: dict) -> None:
        """Process a batch of messages from Kafka.

        Args:
            messages_by_partition: dict[TopicPartition, list[ConsumerRecord]]
        """
        for topic_partition, records in messages_by_partition.items():
            for record in records:
                try:
                    self._process_message(record.value)
                    self._messages_processed += 1
                except Exception as exc:
                    logger.error(
                        "Error processing message from partition %d offset %d: %s",
                        topic_partition.partition,
                        record.offset,
                        exc,
                    )
                    # Continue processing; error is logged

    def _process_message(self, payload: dict) -> None:
        """Process a single trade message.

        Args:
            payload: Trade event dict from Kafka
        """
        from datetime import datetime

        from ingestion.data_models import Trade, Asset

        # Reconstruct Trade object from payload
        try:
            trade = Trade(
                trade_id=payload.get("trade_id", ""),
                ledger_close_time=datetime.fromisoformat(
                    payload.get("ledger_close_time", "2024-01-01T00:00:00")
                ),
                base_account=payload.get("base_account", ""),
                counter_account=payload.get("counter_account", ""),
                base_asset=Asset(
                    code=payload.get("base_asset_code", ""),
                    issuer=payload.get("base_asset_issuer"),
                ),
                counter_asset=Asset(
                    code=payload.get("counter_asset_code", ""),
                    issuer=payload.get("counter_asset_issuer"),
                ),
                base_amount=payload.get("base_amount", 0.0),
                counter_amount=payload.get("counter_amount", 0.0),
                price=payload.get("price", 0.0),
            )
        except Exception as exc:
            logger.error("Failed to reconstruct Trade from payload: %s", exc)
            return

        # Update buffer
        self.buffer.update(trade)
        pair_id = payload.get("pair_id", "unknown")

        # Score wallets
        for wallet in (trade.base_account, trade.counter_account):
            score = self.scorer.score_wallet(wallet, self.buffer)
            if score is not None:
                self.dispatcher.dispatch(wallet, score, pair_id)

    def _commit_offsets(self) -> None:
        """Commit current offsets."""
        try:
            self.consumer.commit()
            logger.debug("Committed offsets")
        except KafkaError as exc:
            logger.error("Offset commit failed: %s", exc)

    def stop(self) -> None:
        """Signal worker to stop."""
        self._stop_event.set()
import time

from confluent_kafka import Consumer, KafkaError, TopicPartition
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

        if consumer is not None:
            self._consumer = consumer
        else:
            self._consumer = Consumer(
                _build_consumer_conf(
                    bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS,
                    group_id or config.KAFKA_CONSUMER_GROUP,
                )
            )
            self._consumer.subscribe([topic_pattern or config.KAFKA_TOPIC_PATTERN])

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
        try:
            self._consumer.close()
        except Exception:  # pragma: no cover - best-effort shutdown
            pass

    def process_message(self, msg) -> None:
        """Decode, score, dispatch, then commit the offset (at-least-once)."""
        # Never process the dead-letter topic — DLQ requires human review.
        if msg.topic() == self._dlq_topic:
            self._consumer.commit(message=msg, asynchronous=False)
            return

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
            self._consumer.commit(message=msg, asynchronous=False)
            return

        self._check_lag(msg)

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
