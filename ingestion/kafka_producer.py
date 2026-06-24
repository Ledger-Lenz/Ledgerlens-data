"""Kafka producer that publishes Horizon SSE trades as Avro to per-pair topics.

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

import json
import re

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
    ) -> None:
        self._topic_prefix = topic_prefix or config.KAFKA_TOPIC_PREFIX
        self._dlq_topic = dlq_topic or config.KAFKA_DLQ_TOPIC
        self._schema = load_schema(schema_path)
        self._producer = (
            producer
            if producer is not None
            else Producer(_build_producer_conf(bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def topic_for_pair(self, asset_pair: str) -> str:
        """Return the topic name for *asset_pair*."""
        return f"{self._topic_prefix}.{sanitise_pair(asset_pair)}"

    def produce_trade(self, trade: Trade) -> None:
        """Serialise and produce *trade*; route serialisation failures to the DLQ."""
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
        self._produce(topic, value, key)
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> int:
        """Block until all queued messages are delivered; returns # still in queue."""
        return self._producer.flush(timeout)

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
