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
