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
