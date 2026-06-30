"""Kafka producer for trade event partitioning by asset pair ID.

Trades are produced to a Kafka topic with the partition key set to the canonical
asset_pair_id (sorted alphabetically). This ensures all trades for a given pair
land on the same partition, enabling independent per-partition consumers to
compute Benford metrics in parallel.

Partition Key Format:
    CODE:ISSUER/CODE:ISSUER (alphabetically sorted by CODE:ISSUER)
    Example: USDC:GA.../XLM:native

Dead-letter Topic:
    Malformed asset pair IDs are routed to {topic}-dlq for validation failure.

Kafka producer that publishes Horizon SSE trades as Avro to per-pair topics.

``HorizonKafkaProducer`` is the ingestion half of the Kafka streaming backend
(Issue #36).  For every trade it:

  1. Converts the :class:`~ingestion.data_models.Trade` to the Avro record
     defined in ``data/trade_avro_schema.json``.
  2. Serialises it to schemaless Avro binary.
  3. Produces it to ``ledgerlens.trades.{asset_pair_sanitised}`` keyed by the
     base account (``wallet_id``) so every trade for a wallet lands in the same
     partition — preserving per-wallet ordering for feature computation.

Failure handling
----------------
* Serialisation failures (poison-pill input) are routed to the dead-letter
  queue ``ledgerlens.trades.dlq`` with the raw payload and a ``reason`` — they
  are **never** retried automatically and require human review.
* Transient ``KafkaException`` errors on produce are retried with exponential
  backoff via :func:`utils.retry.retry_with_backoff`.
"""

from __future__ import annotations

import os
import socket

from confluent_kafka import KafkaException, Producer

from config import config
from ingestion.avro_codec import load_schema, serialize, trade_to_record
from ingestion.data_models import Trade
from utils.logging import get_logger
from utils.retry import retry_with_backoff

logger = get_logger(__name__)

_SANITISE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitise_pair(asset_pair: str) -> str:
    """Turn an ``asset_pair`` string into a Kafka-topic-safe suffix."""
    return _SANITISE_RE.sub("_", asset_pair).strip("_")


def _build_transactional_id() -> str:
    """Build a unique transactional ID per worker instance (hostname + PID).

    In multi-tenant deployments the prefix can be overridden via
    ``KAFKA_TRANSACTIONAL_ID_PREFIX`` to avoid exposing hostnames.
    """
    prefix = config.KAFKA_TRANSACTIONAL_ID_PREFIX
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{prefix}-{hostname}-{pid}"


class HorizonKafkaProducer:
    """Serialises trades to Avro and produces them to per-pair Kafka topics."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        *,
        topic_prefix: str | None = None,
        dlq_topic: str | None = None,
        schema_path: str | None = None,
        producer: Producer | None = None,
        transactional: bool = False,
    ) -> None:
        self._topic_prefix = topic_prefix or config.KAFKA_TOPIC_PREFIX
        self._dlq_topic = dlq_topic or config.KAFKA_DLQ_TOPIC
        self._schema = load_schema(schema_path)
        self._transactional = transactional

        if producer is not None:
            self._producer = producer
        else:
            servers = bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS
            conf = _build_producer_conf(servers)
            if transactional:
                conf["transactional.id"] = _build_transactional_id()
                conf["transaction.timeout.ms"] = config.KAFKA_TRANSACTION_TIMEOUT_MS
            self._producer = Producer(conf)
            if transactional:
                self._producer.init_transactions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def topic_for_pair(self, asset_pair: str) -> str:
        """Return the topic name for *asset_pair*."""
        return f"{self._topic_prefix}.{sanitise_pair(asset_pair)}"

    def produce_trade(self, trade: Trade) -> None:
        """Serialise and produce *trade*; route serialisation failures to the DLQ.

        Each Kafka message carries an ``avro-schema-version`` header whose value
        is the hex-encoded CRC-32 fingerprint of the schema used for encoding.
        Consumers can use this header to select the correct deserialisation schema
        without requiring an external Schema Registry.
        """
        from ingestion.avro_codec import _avro_crc32_fingerprint

        record = trade_to_record(trade)
        try:
            value = serialize(record, self._schema)
        except Exception as exc:  # serialisation / validation failure → DLQ
            logger.error(
                "Serialisation failed for trade %s — routing to DLQ: %s",
                record.get("trade_id"),
                exc,
            )
            self._produce_to_dlq(record, reason=str(exc))
            return

        topic = self.topic_for_pair(record["asset_pair"])
        # Partition key = wallet_id (base account) → per-wallet ordering.
        key = record["base_account"].encode("utf-8")
        # Schema version header: hex CRC-32 fingerprint of the encoding schema.
        schema_fp = hex(_avro_crc32_fingerprint(self._schema) & 0xFFFFFFFF).encode("utf-8")
        self._produce_with_headers(topic, value, key, schema_fp)
        self._producer.poll(0)

    def begin_transaction(self) -> None:
        """Begin a Kafka transaction (requires transactional=True at construction)."""
        self._producer.begin_transaction()

    def commit_transaction(self, consumer, positions: list | None = None) -> None:
        """Commit in-flight transaction and optionally send consumer offsets.

        *positions* is a list of ``TopicPartition`` objects with the offsets to
        commit atomically inside the transaction.
        """
        if positions:
            self._producer.send_offsets_to_transaction(
                positions, consumer.consumer_group_metadata()
            )
        self._producer.commit_transaction()

    def abort_transaction(self, transaction_id: str, offset: int) -> None:
        """Abort the current transaction and log for replay investigation."""
        try:
            self._producer.abort_transaction()
        except KafkaException as exc:
            logger.error(
                "abort_transaction failed — txn_id=%s offset=%d err=%s",
                transaction_id,
                offset,
                exc,
            )

    def flush(self, timeout: float = 10.0) -> int:
        """Block until all queued messages are delivered; returns # still in queue."""
        return int(self._producer.flush(timeout))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @retry_with_backoff(
        max_attempts=5,
        base_delay_seconds=0.5,
        exceptions=(KafkaException, BufferError),
    )
    def _produce(self, topic: str, value: bytes, key: bytes) -> None:
        self._producer.produce(topic=topic, value=value, key=key, on_delivery=_on_delivery)

    @retry_with_backoff(
        max_attempts=5,
        base_delay_seconds=0.5,
        exceptions=(KafkaException, BufferError),
    )
    def _produce_with_headers(
        self, topic: str, value: bytes, key: bytes, schema_fp: bytes
    ) -> None:
        self._producer.produce(
            topic=topic,
            value=value,
            key=key,
            headers=[("avro-schema-version", schema_fp)],
            on_delivery=_on_delivery,
        )

    def _produce_to_dlq(self, record: dict, reason: str) -> None:
        """Produce a poison-pill envelope to the DLQ — raw payload + reason.

        DLQ messages carry the raw (best-effort JSON) payload and the failure
        reason both as the value envelope and as a Kafka header. They are never
        consumed by the scoring worker and must be triaged by a human.
        """
        envelope = json.dumps(
            {"reason": reason, "raw": _safe_raw(record)},
            default=str,
        ).encode("utf-8")
        try:
            self._producer.produce(
                topic=self._dlq_topic,
                value=envelope,
                headers=[("reason", reason.encode("utf-8"))],
            )
            self._producer.poll(0)
        except (KafkaException, BufferError) as exc:
            logger.critical("Failed to write to DLQ topic %s: %s", self._dlq_topic, exc)


def _safe_raw(record: dict) -> dict:
    """Best-effort JSON-serialisable copy of the original record for the DLQ."""
    return {
        k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
        for k, v in record.items()
    }


def _on_delivery(err, msg) -> None:
    if err is not None:
        logger.warning("Delivery failed for topic %s: %s", msg.topic() if msg else "?", err)


def _build_producer_conf(bootstrap_servers: str) -> dict:
    """Build the librdkafka producer config, adding SASL only when credentials exist."""
    conf: dict = {
        "bootstrap.servers": bootstrap_servers,
        "enable.idempotence": True,
        "acks": "all",
        "linger.ms": 5,
    }
    conf.update(_sasl_conf())
    return conf


def _sasl_conf() -> dict:
    """SASL_SSL/PLAIN config when KAFKA_SASL_USERNAME/PASSWORD are set (env only)."""
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
