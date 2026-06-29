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
from typing import TYPE_CHECKING

from kafka import KafkaProducer
from kafka.errors import KafkaError

from utils.logging import get_logger

if TYPE_CHECKING:
    from ingestion.data_models import Trade

logger = get_logger(__name__)

# Canonical format: CODE:ISSUER or CODE:native
ASSET_PAIR_PATTERN = re.compile(r"^([A-Z0-9]+):(native|[A-Z0-9]{56})$")

DLQ_SUFFIX = "-dlq"


def _validate_asset_code(code: str) -> bool:
    """Check if asset code matches expected format."""
    return bool(re.match(r"^[A-Z0-9]+$", code)) and 1 <= len(code) <= 12


def _validate_issuer(issuer: str) -> bool:
    """Check if issuer is 'native' or a 56-char Stellar account ID."""
    if issuer == "native":
        return True
    return bool(re.match(r"^[A-Z0-9]{56}$", issuer))


def _to_canonical_pair_id(code_a: str, issuer_a: str, code_b: str, issuer_b: str) -> str:
    """Generate canonical asset pair ID (alphabetically sorted).

    Returns:
        str: "CODE1:ISSUER1/CODE2:ISSUER2" (sorted alphabetically)

    Raises:
        ValueError: If asset format is invalid.
    """
    if not _validate_asset_code(code_a) or not _validate_issuer(issuer_a):
        raise ValueError(f"Invalid asset A: {code_a}:{issuer_a}")
    if not _validate_asset_code(code_b) or not _validate_issuer(issuer_b):
        raise ValueError(f"Invalid asset B: {code_b}:{issuer_b}")

    asset_a = f"{code_a}:{issuer_a}"
    asset_b = f"{code_b}:{issuer_b}"

    # Sort alphabetically to ensure deterministic ordering
    pair_parts = sorted([asset_a, asset_b])
    return "/".join(pair_parts)


class KafkaTradeProducer:
    """Produces trades to Kafka topic with asset_pair_id partition key."""

    def __init__(
        self,
        topic: str,
        bootstrap_servers: list[str] | str = "localhost:9092",
    ):
        """Initialize Kafka producer.

        Args:
            topic: Kafka topic name for trade events
            bootstrap_servers: Kafka bootstrap server(s)
        """
        self.topic = topic
        self.dlq_topic = f"{topic}{DLQ_SUFFIX}"

        if isinstance(bootstrap_servers, str):
            bootstrap_servers = [bootstrap_servers]

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )

        logger.info(
            "KafkaTradeProducer initialized: topic=%s, dlq_topic=%s, servers=%s",
            self.topic,
            self.dlq_topic,
            bootstrap_servers,
        )

    def produce_trade(self, trade: Trade) -> None:
        """Produce a single trade to Kafka.

        Args:
            trade: Trade object to produce

        Raises:
            ValueError: If asset pair ID validation fails (sent to DLQ)
        """
        try:
            # Generate deterministic partition key
            partition_key = _to_canonical_pair_id(
                trade.base_asset.code,
                trade.base_asset.issuer or "native",
                trade.counter_asset.code,
                trade.counter_asset.issuer or "native",
            )
        except ValueError as exc:
            # Validation failed: send to DLQ
            logger.warning("Invalid asset pair for trade %s: %s", trade.trade_id, exc)
            self._send_to_dlq(trade, str(exc))
            return

        # Trade payload
        payload = {
            "trade_id": trade.trade_id,
            "ledger_close_time": trade.ledger_close_time,
            "base_account": trade.base_account,
            "counter_account": trade.counter_account,
            "base_asset_code": trade.base_asset.code,
            "base_asset_issuer": trade.base_asset.issuer,
            "counter_asset_code": trade.counter_asset.code,
            "counter_asset_issuer": trade.counter_asset.issuer,
            "base_amount": trade.base_amount,
            "counter_amount": trade.counter_amount,
            "price": trade.price,
            "pair_id": partition_key,  # Include canonical pair ID in payload
        }

        # Send to main topic with partition key
        future = self.producer.send(
            self.topic,
            value=payload,
            key=partition_key,
        )

        try:
            record_metadata = future.get(timeout=10)
            logger.debug(
                "Produced trade %s to partition %d offset %d",
                trade.trade_id,
                record_metadata.partition,
                record_metadata.offset,
            )
        except KafkaError as exc:
            logger.error("Failed to produce trade %s: %s", trade.trade_id, exc)
            raise

    def _send_to_dlq(self, trade: Trade, error_reason: str) -> None:
        """Send malformed trade to dead-letter queue.

        Args:
            trade: The invalid trade
            error_reason: Description of the validation error
        """
        dlq_payload = {
            "trade_id": trade.trade_id,
            "error": error_reason,
            "original_trade": {
                "base_asset_code": trade.base_asset.code,
                "base_asset_issuer": trade.base_asset.issuer,
                "counter_asset_code": trade.counter_asset.code,
                "counter_asset_issuer": trade.counter_asset.issuer,
            },
        }
        try:
            self.producer.send(self.dlq_topic, value=dlq_payload, key=None)
            logger.info("Sent invalid trade %s to DLQ %s", trade.trade_id, self.dlq_topic)
        except KafkaError as exc:
            logger.error("Failed to send trade to DLQ: %s", exc)

    def flush(self, timeout_ms: int = 30000) -> None:
        """Flush pending messages."""
        self.producer.flush(timeout_ms=timeout_ms)

    def close(self) -> None:
        """Close producer."""
        self.producer.close()

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
