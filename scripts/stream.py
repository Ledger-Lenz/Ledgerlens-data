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
from detection.model_inference import RiskScorer
from streaming.alert_dispatcher import AlertDispatcher
from streaming.feature_buffer import FeatureBuffer, StreamingScorer
from streaming.pipeline import StreamingPipeline
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
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # --- Config validation ---
    if not config.WATCHED_ASSET_PAIRS:
        logger.error("WATCHED_ASSET_PAIRS is not configured — set it in .env before streaming")
        sys.exit(1)

    # --- Load ensemble models ---
    try:
        risk_scorer = RiskScorer()
    except Exception as exc:
        logger.error("Failed to initialise RiskScorer from %s: %s", config.MODEL_DIR, exc)
        sys.exit(1)

    if not risk_scorer.models:
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
    scorer = StreamingScorer(risk_scorer, buffer, min_trades=args.min_trades)
    dispatcher = AlertDispatcher(
        channel=args.alert_channel,
        webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        ws_client=ws_client,
        alert_cooldown_seconds=args.cooldown_seconds,
    )
    pipeline = StreamingPipeline(buffer, scorer, dispatcher)

    # --- Startup banner ---
    pair_count = len(config.WATCHED_ASSET_PAIRS)
    logger.info(
        "LedgerLens streaming pipeline starting — %d pair(s), channel=%s, min_trades=%d",
        pair_count,
        args.alert_channel,
        args.min_trades,
    )
    if ws_addr:
        logger.info("WebSocket server: %s", ws_addr)

    pipeline.run()


if __name__ == "__main__":
    main()
