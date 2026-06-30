"""Real-time Stellar DEX risk-scoring pipeline.

Streams trades from Horizon SSE, maintains a rolling FeatureBuffer per
wallet, scores wallets once they have sufficient history, and dispatches
alerts via stdout, webhook, or WebSocket.

Usage
-----
    python -m scripts.stream
    python -m scripts.stream --alert-channel webhook
    python -m scripts.stream --alert-channel websocket --cooldown-seconds 1800
    python -m scripts.stream --alert-channel websocket --no-ws  # skip WS server

Environment variables (see .env.example)
-----------------------------------------
WATCHED_ASSET_PAIRS, ALERT_CHANNEL, ALERT_WEBHOOK_URL,
ALERT_COOLDOWN_SECONDS, WS_BIND_HOST, WS_PORT, WS_ALLOW_EXTERNAL
"""

import argparse
import os
import sys

from config import config
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer
from streaming.pipeline import StreamingPipeline
from streaming.streaming_scorer import StreamingScorer
from utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.stream",
        description="LedgerLens real-time streaming risk-scoring pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alert-channel",
        default=os.getenv("ALERT_CHANNEL", "stdout"),
        choices=["stdout", "webhook", "websocket"],
        help="Alert delivery channel",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=int(os.getenv("ALERT_COOLDOWN_SECONDS", "3600")),
        help="Per-wallet alert deduplication window (seconds)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=20,
        help="Minimum buffered trades before a wallet is scored",
    )
    parser.add_argument(
        "--no-ws",
        action="store_true",
        help="Disable the WebSocket broadcast server even when channel=websocket",
    )
    parser.add_argument(
        "--backend",
        default=config.STREAMING_BACKEND,
        choices=["sse", "kafka"],
        help="Ingestion transport (overrides STREAMING_BACKEND)",
    )
    parser.add_argument(
        "--role",
        default="all",
        choices=["all", "producer", "worker"],
        help="Kafka role: 'producer' (SSE→Kafka), 'worker' (scorer), or 'all'",
    )
    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Disable adaptive micro-batch sizing and use a fixed batch of N. "
            "Useful for debugging PID oscillation (Issue #243)."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # --backend overrides the configured transport for this process.
    config.STREAMING_BACKEND = args.backend

    # --- Config validation ---
    # A producer needs pairs to stream; a worker discovers topics dynamically.
    if args.role != "worker" and not config.WATCHED_ASSET_PAIRS:
        logger.error("WATCHED_ASSET_PAIRS is not configured — set it in .env before streaming")
        sys.exit(1)

    # --- Load ensemble models (not needed for a pure Kafka producer) ---
    scorer = None
    if not (args.backend == "kafka" and args.role == "producer"):
        scorer = StreamingScorer()
        scorer.min_trades = args.min_trades
        if args.fixed_batch_size is not None:
            logger.info("Adaptive batch sizing disabled; fixed batch size = %d", args.fixed_batch_size)
            scorer._fixed_batch_size = args.fixed_batch_size

        if not scorer._risk_scorer.models:
            logger.error(
                "No trained models found in %s. " "Run 'python -m detection.model_training' first.",
                config.MODEL_DIR,
            )
            sys.exit(1)

    # --- Optional WebSocket server ---
    ws_client = None
    ws_addr: str | None = None

    if args.alert_channel == "websocket" and not args.no_ws:
        from streaming.ws_server import _WsClientAdapter, start_ws_server_thread

        host = os.getenv("WS_BIND_HOST", "127.0.0.1")
        port = int(os.getenv("WS_PORT", "8765"))
        start_ws_server_thread(host, port)
        ws_client = _WsClientAdapter()
        ws_addr = f"ws://{host}:{port}"

    # --- Wire up components ---
    buffer = FeatureBuffer()
    dispatcher = AlertDispatcher(
        channel=args.alert_channel,
        webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        ws_client=ws_client,
        alert_cooldown_seconds=args.cooldown_seconds,
    )
    pipeline = StreamingPipeline(buffer, scorer, dispatcher, role=args.role)

    # --- Startup banner ---
    pair_count = len(config.WATCHED_ASSET_PAIRS)
    logger.info(
        "LedgerLens streaming pipeline starting — backend=%s role=%s, "
        "%d pair(s), channel=%s, min_trades=%d",
        args.backend,
        args.role,
        pair_count,
        args.alert_channel,
        args.min_trades,
    )
    if ws_addr:
        logger.info("WebSocket server: %s", ws_addr)

    pipeline.run()


if __name__ == "__main__":
    main()
