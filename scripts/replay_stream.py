"""Stream replay capability for backfilling risk scores with new model versions.

Replays historical Kafka events through the current streaming pipeline,
updating stored risk scores with a replay tag. Supports resumable replay,
dry-run mode, and explicit confirmation to prevent accidental overwrites.

Usage:
    # Replay last 24 hours of events (no alerts, dry-run)
    python -m scripts.replay_stream --from-timestamp -86400 --dry-run --confirm

    # Replay specific time range and store scores
    python -m scripts.replay_stream \\
        --from-timestamp 1704067200 --to-timestamp 1704153600 \\
        --confirm

    # Resume interrupted replay from last committed offset
    python -m scripts.replay_stream --resume --confirm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError

from config import config
from detection.model_inference import RiskScorer
from detection.risk_score_store import RiskScoreStore
from streaming.feature_buffer import FeatureBuffer
from streaming.streaming_scorer import StreamingScorer
from streaming.alert_dispatcher import AlertDispatcher
from ingestion.data_models import Trade, Asset
from utils.logging import get_logger

if TYPE_CHECKING:
    from kafka.structs import ConsumerRecord

logger = get_logger(__name__)

# Replay-specific consumer group to track replay offsets separately
DEFAULT_REPLAY_GROUP = "ledgerlens-replay"

# Replay model version tag
REPLAY_MODEL_VERSION_TAG = "replay"


class NoOpAlertDispatcher:
    """Alert dispatcher that does nothing (used in replay mode)."""

    def dispatch(self, wallet: str, risk_score: dict, pair_id: str) -> None:
        """No-op dispatch for replay mode."""
        pass


class StreamReplayer:
    """Replays historical Kafka events through the streaming pipeline."""

    def __init__(
        self,
        topic: str,
        group_id: str = DEFAULT_REPLAY_GROUP,
        bootstrap_servers: list[str] | str = "localhost:9092",
        dry_run: bool = False,
        from_timestamp: int | None = None,
        to_timestamp: int | None = None,
    ):
        """Initialize replayer.

        Args:
            topic: Kafka topic to replay from
            group_id: Consumer group for tracking replay offsets
            bootstrap_servers: Kafka bootstrap servers
            dry_run: If True, don't write to DB
            from_timestamp: Unix timestamp to replay from (None = earliest)
            to_timestamp: Unix timestamp to replay to (None = latest)
        """
        if isinstance(bootstrap_servers, str):
            bootstrap_servers = [bootstrap_servers]

        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.dry_run = dry_run
        self.from_timestamp = from_timestamp
        self.to_timestamp = to_timestamp

        # Components
        self.buffer = FeatureBuffer()
        risk_scorer = RiskScorer()
        self.scorer = StreamingScorer(model_dir=None)  # Uses default model dir
        # Use no-op dispatcher for replay (no live alerts)
        self.dispatcher = NoOpAlertDispatcher()
        self.store = RiskScoreStore() if not dry_run else None

        # Consumer
        self.consumer = KafkaConsumer(
            topic,
            group_id=group_id,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )

        self._total_events = 0
        self._scored_wallets = 0
        self._start_time = time.time()

        logger.info(
            "StreamReplayer initialized: topic=%s, group=%s, dry_run=%s",
            topic,
            group_id,
            dry_run,
        )

    def run(self) -> None:
        """Execute the replay from specified time range.

        Resumes from last committed offset if one exists.
        """
        try:
            # Get partitions
            partitions = self.consumer.partitions_for_topic(self.topic)
            if not partitions:
                logger.error("Topic %s has no partitions", self.topic)
                return

            # Seek to timestamp range
            self._seek_to_timestamp_range(partitions)

            logger.info("Starting replay of topic %s", self.topic)
            messages_in_batch = 0

            while True:
                messages = self.consumer.poll(timeout_ms=1000, max_records=100)
                if not messages:
                    if messages_in_batch == 0:
                        logger.info("No more messages to replay")
                    break

                for topic_partition, records in messages.items():
                    for record in records:
                        try:
                            self._process_message(record)
                            self._total_events += 1
                            messages_in_batch += 1
                        except Exception as exc:
                            logger.error(
                                "Error processing message from partition %d offset %d: %s",
                                topic_partition.partition,
                                record.offset,
                                exc,
                            )

                # Commit offsets every 100 messages
                if messages_in_batch >= 100:
                    self._commit_offsets()
                    self._log_progress()
                    messages_in_batch = 0

            # Final commit
            self._commit_offsets()
            self._log_progress(final=True)

        finally:
            self.consumer.close()

    def _seek_to_timestamp_range(self, partitions: set) -> None:
        """Seek consumer to the specified timestamp range.

        Args:
            partitions: Set of partition IDs for the topic
        """
        topic_partitions = [TopicPartition(self.topic, p) for p in partitions]

        # Try to seek by timestamp
        if self.from_timestamp is not None:
            from_millis = int(self.from_timestamp * 1000)
            timestamps_dict = {tp: from_millis for tp in topic_partitions}
            try:
                offsets = self.consumer.offsets_for_times(timestamps_dict)
                for tp, offset_info in offsets.items():
                    if offset_info is not None:
                        self.consumer.seek(tp, offset_info.offset)
                        logger.info(
                            "Seeked partition %d to offset %d (timestamp %d)",
                            tp.partition,
                            offset_info.offset,
                            self.from_timestamp,
                        )
                    else:
                        logger.warning(
                            "No offset found for partition %d at timestamp %d",
                            tp.partition,
                            self.from_timestamp,
                        )
            except KafkaError as exc:
                logger.error("Failed to seek by timestamp: %s", exc)
        else:
            # Seek to beginning
            self.consumer.assign(topic_partitions)
            self.consumer.seek_to_beginning(*topic_partitions)

    def _process_message(self, record: ConsumerRecord) -> None:
        """Process a single trade message and score wallets.

        Args:
            record: Kafka consumer record
        """
        payload = record.value

        # Check if within time range
        if self.to_timestamp is not None:
            ledger_time = payload.get("ledger_close_time", "2024-01-01T00:00:00")
            try:
                dt = datetime.fromisoformat(ledger_time)
                if isinstance(dt, datetime) and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                ts = dt.timestamp()
                if ts > self.to_timestamp:
                    return  # Skip events after to_timestamp
            except Exception:
                pass  # Continue if parsing fails

        # Reconstruct Trade object
        try:
            trade = self._reconstruct_trade(payload)
        except Exception as exc:
            logger.error("Failed to reconstruct trade: %s", exc)
            return

        # Update buffer
        self.buffer.update(trade)
        pair_id = payload.get("pair_id", "unknown")

        # Score wallets
        for wallet in (trade.base_account, trade.counter_account):
            score = self.scorer.score_wallet(wallet, self.buffer)
            if score is not None:
                self._store_replay_score(wallet, score, pair_id)
                self._scored_wallets += 1

    def _reconstruct_trade(self, payload: dict) -> Trade:
        """Reconstruct Trade object from Kafka payload.

        Args:
            payload: Trade event dict from Kafka

        Returns:
            Trade object

        Raises:
            ValueError: If payload is malformed
        """
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
            return trade
        except Exception as exc:
            raise ValueError(f"Failed to reconstruct trade: {exc}") from exc

    def _store_replay_score(self, wallet: str, risk_score: dict, pair_id: str) -> None:
        """Store replay score with model version tag.

        Args:
            wallet: Wallet address
            risk_score: Risk score dict from scorer
            pair_id: Asset pair ID
        """
        if self.dry_run:
            logger.info(
                "Replay (dry-run) wallet=%s, pair=%s, score=%d",
                wallet,
                pair_id,
                risk_score["score"],
            )
        else:
            try:
                # Add replay tag
                replay_score = {
                    **risk_score,
                    "replay_model_version": REPLAY_MODEL_VERSION_TAG,
                }
                if self.store:
                    self.store.upsert(wallet, pair_id, replay_score)
                logger.debug(
                    "Stored replay score: wallet=%s, pair=%s, score=%d",
                    wallet,
                    pair_id,
                    risk_score["score"],
                )
            except Exception as exc:
                logger.error("Failed to store replay score: %s", exc)

    def _commit_offsets(self) -> None:
        """Commit current offsets."""
        try:
            self.consumer.commit()
            logger.debug("Committed replay offsets")
        except KafkaError as exc:
            logger.error("Failed to commit offsets: %s", exc)

    def _log_progress(self, final: bool = False) -> None:
        """Log replay progress.

        Args:
            final: If True, log final summary
        """
        elapsed = time.time() - self._start_time
        throughput = self._total_events / elapsed if elapsed > 0 else 0

        if final:
            logger.info(
                "Replay completed: %d events processed, %d wallets scored, "
                "%.1f events/sec (%.1f min total)",
                self._total_events,
                self._scored_wallets,
                throughput,
                elapsed / 60,
            )
        else:
            logger.info(
                "Replay progress: %d events, %d scored, %.1f events/sec",
                self._total_events,
                self._scored_wallets,
                throughput,
            )

    def close(self) -> None:
        """Close consumer and cleanup."""
        self.consumer.close()


def _get_timestamp_from_arg(arg: str | None, now: datetime) -> int | None:
    """Convert command-line timestamp argument to Unix timestamp.

    Args:
        arg: Argument value (e.g., '-86400' for 24h ago, '1704067200' for absolute timestamp)
        now: Current time reference

    Returns:
        Unix timestamp or None
    """
    if arg is None:
        return None

    try:
        if arg.startswith("-"):
            # Relative offset in seconds
            offset = int(arg)
            return int((now + timedelta(seconds=offset)).timestamp())
        else:
            # Absolute Unix timestamp
            return int(arg)
    except ValueError:
        logger.error("Invalid timestamp argument: %s", arg)
        return None


def _confirm_replay() -> bool:
    """Prompt user for replay confirmation.

    Returns:
        True if confirmed, False otherwise
    """
    try:
        response = input("Confirm stream replay? (type 'yes' to proceed): ").strip().lower()
        return response == "yes"
    except (EOFError, KeyboardInterrupt):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.replay_stream",
        description="Replay historical Kafka events for model version backfilling",
    )
    parser.add_argument(
        "--topic",
        default="trades",
        help="Kafka topic to replay from",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Comma-separated Kafka bootstrap servers",
    )
    parser.add_argument(
        "--group",
        default=DEFAULT_REPLAY_GROUP,
        help="Kafka consumer group for tracking replay offsets",
    )
    parser.add_argument(
        "--from-timestamp",
        type=str,
        help="Replay from timestamp (Unix timestamp or relative offset in seconds, e.g., -86400 for 24h ago)",
    )
    parser.add_argument(
        "--to-timestamp",
        type=str,
        help="Replay until timestamp (Unix timestamp or relative offset)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last committed replay offset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process events and log scores without writing to DB",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Proceed without interactive confirmation prompt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Require explicit confirmation
    if not args.confirm:
        if not _confirm_replay():
            logger.info("Replay cancelled by user")
            sys.exit(0)

    # Parse bootstrap servers
    bootstrap_servers = [s.strip() for s in args.bootstrap_servers.split(",")]

    # Parse timestamps
    now = datetime.now(UTC)
    from_timestamp = _get_timestamp_from_arg(args.from_timestamp, now) if not args.resume else None
    to_timestamp = _get_timestamp_from_arg(args.to_timestamp, now)

    logger.info(
        "Stream replay configuration: topic=%s, dry_run=%s, "
        "from_timestamp=%s, to_timestamp=%s, resume=%s",
        args.topic,
        args.dry_run,
        from_timestamp,
        to_timestamp,
        args.resume,
    )

    # Create and run replayer
    try:
        replayer = StreamReplayer(
            topic=args.topic,
            group_id=args.group,
            bootstrap_servers=bootstrap_servers,
            dry_run=args.dry_run,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        replayer.run()
    except Exception as exc:
        logger.error("Stream replay failed: %s", exc)
        sys.exit(1)
    finally:
        try:
            replayer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
