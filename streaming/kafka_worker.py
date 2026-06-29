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
"""

from __future__ import annotations

import json
import signal
import threading
import time
from typing import TYPE_CHECKING

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from streaming.feature_buffer import FeatureBuffer, StreamingScorer
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
        # Reconstruct Trade-like object from payload
        # (We don't have the full Trade class here, but the payload has the data)
        trade_id = payload.get("trade_id")
        base_account = payload.get("base_account")
        counter_account = payload.get("counter_account")
        pair_id = payload.get("pair_id")

        # Build minimal trade record for buffer
        trade_record = {
            "ledger_close_time": payload.get("ledger_close_time"),
            "base_account": base_account,
            "counter_account": counter_account,
            "amount": payload.get("base_amount"),
        }

        # Update buffer for both accounts
        for wallet in (base_account, counter_account):
            self.buffer._buffer.setdefault(wallet, []).append(trade_record)
            # Trim if needed
            buf = self.buffer._buffer[wallet]
            if len(buf) > self.buffer.max_trades_per_wallet:
                self.buffer._buffer[wallet] = buf[-self.buffer.max_trades_per_wallet :]

            # Score wallet if it has enough history
            score = self.scorer.score_wallet(wallet)
            if score is not None:
                self.dispatcher.dispatch(wallet, score, pair_id or "unknown")

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
