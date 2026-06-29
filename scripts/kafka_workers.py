"""Kafka worker pool orchestration script.

Starts N workers in parallel, each handling a subset of partitions from the
configured Kafka topic. Workers communicate through Kafka's consumer group
rebalancing protocol.

Usage:
    python -m scripts.kafka_workers --num-workers 4
    python -m scripts.kafka_workers --num-workers 4 --topic trades --group ledgerlens-workers
    python -m scripts.kafka_workers --num-workers 4 --bootstrap-servers localhost:9092,localhost:9093
"""

import argparse
import os
import signal
import sys
import threading
import time

from config import config
from detection.model_inference import RiskScorer
from streaming.kafka_worker import KafkaWorker
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer, StreamingScorer
from utils.logging import get_logger

logger = get_logger(__name__)


def create_worker(
    topic: str,
    group_id: str,
    worker_id: int,
    num_workers: int,
    bootstrap_servers: list[str],
) -> KafkaWorker:
    """Create a Kafka worker instance.

    Args:
        topic: Kafka topic
        group_id: Consumer group ID
        worker_id: Worker index (0-based)
        num_workers: Total number of workers
        bootstrap_servers: Kafka bootstrap servers

    Returns:
        Configured KafkaWorker instance
    """
    buffer = FeatureBuffer()
    risk_scorer = RiskScorer()
    scorer = StreamingScorer(risk_scorer, buffer, min_trades=20)
    dispatcher = AlertDispatcher(
        channel=os.getenv("ALERT_CHANNEL", "stdout"),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        alert_cooldown_seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", "3600")),
    )

    worker = KafkaWorker(
        topic=topic,
        group_id=group_id,
        bootstrap_servers=bootstrap_servers,
        buffer=buffer,
        scorer=scorer,
        dispatcher=dispatcher,
        commit_interval_seconds=30,
    )

    logger.info(
        "Created worker %d/%d (group=%s, topic=%s)",
        worker_id + 1,
        num_workers,
        group_id,
        topic,
    )

    return worker


def run_worker_pool(
    num_workers: int,
    topic: str,
    group_id: str,
    bootstrap_servers: list[str],
) -> None:
    """Start and manage a pool of workers.

    Args:
        num_workers: Number of workers to spawn
        topic: Kafka topic
        group_id: Consumer group ID
        bootstrap_servers: Kafka bootstrap servers
    """
    workers = []
    threads = []
    stop_event = threading.Event()

    def shutdown_handler(*_):
        """Handle SIGTERM/SIGINT."""
        logger.info("Shutdown signal received, stopping workers...")
        stop_event.set()

    # Install signal handlers
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        # Create and start workers
        logger.info("Starting %d workers...", num_workers)
        for worker_id in range(num_workers):
            worker = create_worker(
                topic=topic,
                group_id=group_id,
                worker_id=worker_id,
                num_workers=num_workers,
                bootstrap_servers=bootstrap_servers,
            )
            workers.append(worker)

            # Start worker in daemon thread
            thread = threading.Thread(
                target=_run_worker_with_error_handling,
                args=(worker, worker_id, stop_event),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        logger.info("All %d workers started. Press Ctrl+C to stop.", num_workers)

        # Wait for shutdown signal
        while not stop_event.is_set():
            time.sleep(1)

    finally:
        logger.info("Stopping all workers...")
        stop_event.set()

        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=5)

        logger.info("Worker pool stopped")


def _run_worker_with_error_handling(
    worker: KafkaWorker,
    worker_id: int,
    stop_event: threading.Event,
) -> None:
    """Run a worker with error handling and graceful shutdown.

    Args:
        worker: KafkaWorker instance
        worker_id: Worker index
        stop_event: Shutdown signal
    """
    try:
        while not stop_event.is_set():
            try:
                # Run until stop signal
                worker.run()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("Worker %d encountered error: %s", worker_id, exc)
                time.sleep(5)  # Backoff before restart
    finally:
        worker.stop()
        logger.info("Worker %d stopped", worker_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.kafka_workers",
        description="Start a pool of Kafka workers for parallel trade processing",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="Number of worker threads to start",
    )
    parser.add_argument(
        "--topic",
        default="trades",
        help="Kafka topic to consume from",
    )
    parser.add_argument(
        "--group",
        default="ledgerlens-workers",
        help="Kafka consumer group ID",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Comma-separated Kafka bootstrap servers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse bootstrap servers
    bootstrap_servers = [s.strip() for s in args.bootstrap_servers.split(",")]

    logger.info(
        "Kafka Worker Pool Configuration: num_workers=%d, topic=%s, group=%s, servers=%s",
        args.num_workers,
        args.topic,
        args.group,
        bootstrap_servers,
    )

    # Validate config
    if not config.WATCHED_ASSET_PAIRS:
        logger.warning("WATCHED_ASSET_PAIRS is not configured")

    # Try to load models
    try:
        RiskScorer()
    except Exception as exc:
        logger.error("Failed to initialize RiskScorer: %s", exc)
        sys.exit(1)

    # Run worker pool
    try:
        run_worker_pool(
            num_workers=args.num_workers,
            topic=args.topic,
            group_id=args.group,
            bootstrap_servers=bootstrap_servers,
        )
    except Exception as exc:
        logger.error("Worker pool error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
