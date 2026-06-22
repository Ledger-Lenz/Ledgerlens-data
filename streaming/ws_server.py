"""Production-grade WebSocket server with pub/sub, auth, and rate limiting.

Features:
- JWT bearer token authentication (RS256)
- Channel subscriptions: wallet/{id}, pair/{pair}, all (admin)
- Per-client async queue with backpressure
- Per-client rate limiting (token-bucket)
- Sequence numbers with replay buffer
- Prometheus metrics
- Max connection limit
- Token logging with redaction
"""

import asyncio
import json
import os
import re
import threading
import time
from collections import deque
from typing import Any

import websockets
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import Counter, Gauge

from config import config
from streaming.pubsub_router import PubSubRouter
from streaming.ws_auth import JWTAuthenticator
from utils.logging import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Pydantic Message Schemas
# ─────────────────────────────────────────────────────────────────────────


class ScoreUpdateMessage(BaseModel):
    """Message schema for score updates."""

    type: str = Field(default="score_update", const=True)
    seq: int
    channel: str
    wallet: str
    asset_pair: str
    score: int = Field(ge=0, le=100)
    score_lower: int = Field(ge=0, le=100)
    score_upper: int = Field(ge=0, le=100)
    bft_divergence: bool = False
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: str  # ISO 8601 format


class SubscribeMessage(BaseModel):
    """Client message to subscribe to channels."""

    type: str = Field(default="subscribe", const=True)
    channels: list[str]


class UnsubscribeMessage(BaseModel):
    """Client message to unsubscribe from channels."""

    type: str = Field(default="unsubscribe", const=True)
    channels: list[str]


class ReplayMessage(BaseModel):
    """Client message to request message replay."""

    type: str = Field(default="replay", const=True)
    channel: str
    since_seq: int


class ErrorMessage(BaseModel):
    """Server error message."""

    type: str = Field(default="error", const=True)
    code: str
    message: str = ""
    retry_after_ms: int | None = None


class DroppedMessage(BaseModel):
    """Notification of dropped messages due to backpressure."""

    type: str = Field(default="dropped", const=True)
    count: int


# ─────────────────────────────────────────────────────────────────────────
# Rate Limiter (Token Bucket)
# ─────────────────────────────────────────────────────────────────────────


class TokenBucket:
    """Token-bucket rate limiter."""

    def __init__(self, rate: int):
        """Initialize with rate (messages per second).

        Args:
            rate: Max messages per second
        """
        self.rate = float(rate)
        self.tokens = float(rate)
        self.last_update = time.monotonic()

    def is_allowed(self) -> bool:
        """Check if next request is allowed.

        Returns:
            True if allowed (token available), False if rate limit exceeded.
        """
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def retry_after_ms(self) -> int:
        """Estimate milliseconds until next token is available.

        Returns:
            Milliseconds to wait
        """
        if self.tokens >= 1.0:
            return 0
        return int((1.0 - self.tokens) / self.rate * 1000) + 1


# ─────────────────────────────────────────────────────────────────────────
# Sequence Numbers (Thread-safe Global)
# ─────────────────────────────────────────────────────────────────────────


class SequenceCounter:
    """Thread-safe monotonically increasing sequence numbers."""

    def __init__(self):
        self._seq = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        """Get next sequence number.

        Returns:
            Monotonically increasing integer
        """
        with self._lock:
            self._seq += 1
            return self._seq

    def current(self) -> int:
        """Get current sequence number without incrementing.

        Returns:
            Current sequence number
        """
        with self._lock:
            return self._seq


# ─────────────────────────────────────────────────────────────────────────
# Replay Buffer (Ring Buffer per Channel)
# ─────────────────────────────────────────────────────────────────────────


class ReplayBuffer:
    """Ring buffer storing last N messages per channel."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffers: dict[str, deque] = {}
        self._lock = threading.Lock()

    def append(self, channel: str, seq: int, message: dict) -> None:
        """Add message to replay buffer for channel.

        Args:
            channel: Channel name (e.g., "wallet/GXXX")
            seq: Sequence number
            message: Message dict
        """
        with self._lock:
            if channel not in self._buffers:
                self._buffers[channel] = deque(maxlen=self.max_size)
            self._buffers[channel].append((seq, message))

    def get_since(self, channel: str, since_seq: int) -> list[dict]:
        """Get all messages with seq >= since_seq.

        Args:
            channel: Channel name
            since_seq: Minimum sequence number (inclusive)

        Returns:
            List of messages in ascending seq order
        """
        with self._lock:
            if channel not in self._buffers:
                return []
            return [msg for seq, msg in self._buffers[channel] if seq >= since_seq]


# ─────────────────────────────────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────────────────────────────────

ws_connected_clients = Gauge("ws_connected_clients", "Number of connected WebSocket clients")
ws_messages_published_total = Counter(
    "ws_messages_published_total",
    "Total messages published",
    labelnames=["channel_type"],
)
ws_messages_dropped_total = Counter(
    "ws_messages_dropped_total",
    "Total messages dropped due to backpressure",
)
ws_auth_failures_total = Counter(
    "ws_auth_failures_total",
    "Total authentication failures",
)

# ─────────────────────────────────────────────────────────────────────────
# Module-level State
# ─────────────────────────────────────────────────────────────────────────

_clients: dict[str, Any] = {}  # client_id -> client_state
_clients_lock = threading.Lock()
_seq_counter = SequenceCounter()
_replay_buffer = ReplayBuffer(config.WS_REPLAY_BUFFER_SIZE)
_router = PubSubRouter()
_auth = JWTAuthenticator(config.JWT_PUBLIC_KEY_PATH)
_loop: asyncio.AbstractEventLoop | None = None


# ─────────────────────────────────────────────────────────────────────────
# Channel Validation
# ─────────────────────────────────────────────────────────────────────────

# Regex for valid channel names
CHANNEL_REGEX = re.compile(r"^(wallet/G[A-Z2-7]{55}|pair/[A-Z0-9:\/\-]+|all)$")


def _validate_channel(channel: str) -> bool:
    """Validate channel name format.

    Args:
        channel: Channel name to validate

    Returns:
        True if valid, False otherwise
    """
    return bool(CHANNEL_REGEX.match(channel))


def _redact_token(token: str) -> str:
    """Redact token for logging.

    Args:
        token: Token to redact

    Returns:
        Redacted token string (first 10 and last 10 chars with ... in middle)
    """
    if len(token) <= 20:
        return "[REDACTED]"
    return f"{token[:10]}...{token[-10:]}"


# ─────────────────────────────────────────────────────────────────────────
# WebSocket Connection Handler
# ─────────────────────────────────────────────────────────────────────────


async def _handler(websocket) -> None:
    """Handle WebSocket client connection.

    Connection lifecycle:
    1. Extract and verify JWT token
    2. Register client with subscriptions
    3. Process inbound messages (subscribe/unsubscribe/replay)
    4. Receive outbound messages from router
    5. Clean up on disconnect
    """
    client_id = None
    permissions = set()

    try:
        # ─────────────────────────────────────────────────────────────────
        # 1. AUTHENTICATION: Extract and verify JWT token
        # ─────────────────────────────────────────────────────────────────

        token = await _extract_token(websocket)
        if not token:
            ws_auth_failures_total.inc()
            await websocket.close(code=1008, reason="Unauthorized: missing token")
            logger.warning("WebSocket connection rejected: no token provided")
            return

        claims = _auth.verify(token)
        if not claims:
            ws_auth_failures_total.inc()
            await websocket.close(code=1008, reason="Unauthorized: invalid token")
            logger.warning("WebSocket connection rejected: invalid token (%s)", _redact_token(token))
            return

        client_id = claims.get("sub", "unknown")
        permissions = _auth.extract_permissions(claims)

        # ─────────────────────────────────────────────────────────────────
        # 2. CONNECTION LIMITS: Check max clients
        # ─────────────────────────────────────────────────────────────────

        with _clients_lock:
            if len(_clients) >= config.WS_MAX_CLIENTS:
                await websocket.close(code=1008, reason="Server at capacity")
                logger.warning(
                    "WebSocket connection rejected: max clients reached (%d)",
                    config.WS_MAX_CLIENTS,
                )
                return
            ws_connected_clients.set(len(_clients) + 1)

        # ─────────────────────────────────────────────────────────────────
        # 3. REGISTER CLIENT: Create queue and rate limiter
        # ─────────────────────────────────────────────────────────────────

        client_queue = asyncio.Queue(maxlen=config.WS_CLIENT_QUEUE_DEPTH)
        rate_limiter = TokenBucket(config.WS_RATE_LIMIT_MSGS_PER_SECOND)

        with _clients_lock:
            _clients[client_id] = {
                "websocket": websocket,
                "queue": client_queue,
                "rate_limiter": rate_limiter,
                "permissions": permissions,
            }

        logger.info("WebSocket client connected (client_id=%s, total=%d)", client_id, len(_clients))

        # ─────────────────────────────────────────────────────────────────
        # 4. HANDLE MESSAGES: Inbound (subscribe/unsubscribe/replay) and
        #    outbound (from router queue)
        # ─────────────────────────────────────────────────────────────────

        # Task 1: Process inbound messages
        inbound_task = asyncio.create_task(_process_inbound(websocket, client_id, permissions))

        # Task 2: Process outbound messages from router
        outbound_task = asyncio.create_task(
            _process_outbound(websocket, client_queue, client_id, rate_limiter)
        )

        # Wait for either task to complete or for client to disconnect
        done, pending = await asyncio.wait(
            [inbound_task, outbound_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as exc:
        logger.error("Unexpected error in WebSocket handler: %s", str(exc))

    finally:
        # ─────────────────────────────────────────────────────────────────
        # 5. CLEANUP: Unsubscribe and remove client
        # ─────────────────────────────────────────────────────────────────

        if client_id:
            _router.disconnect(client_id)
            with _clients_lock:
                _clients.pop(client_id, None)
                ws_connected_clients.set(len(_clients))
            logger.info("WebSocket client disconnected (client_id=%s, total=%d)", client_id, len(_clients))


async def _extract_token(websocket) -> str | None:
    """Extract JWT token from websocket connection.

    Tries two sources:
    1. Authorization header: "Authorization: Bearer <token>"
    2. Query parameter: ?token=<token>

    Args:
        websocket: WebSocket connection object

    Returns:
        Token string if found, None otherwise
    """
    # Try Authorization header
    auth_header = websocket.request_headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    # Try query parameter
    if websocket.request_headers.get("Path"):
        path = websocket.request_headers.get("Path", "")
        if "?token=" in path:
            try:
                token = path.split("?token=")[1].split("&")[0]
                return token if token else None
            except IndexError:
                pass

    return None


async def _process_inbound(websocket, client_id: str, permissions: set[str]) -> None:
    """Process inbound client messages (subscribe, unsubscribe, replay).

    Args:
        websocket: WebSocket connection
        client_id: Client ID
        permissions: Set of permission strings
    """
    try:
        async for message_str in websocket:
            try:
                payload = json.loads(message_str)
                msg_type = payload.get("type")

                if msg_type == "subscribe":
                    await _handle_subscribe(websocket, client_id, permissions, payload)
                elif msg_type == "unsubscribe":
                    await _handle_unsubscribe(websocket, client_id, payload)
                elif msg_type == "replay":
                    await _handle_replay(websocket, client_id, permissions, payload)
                else:
                    error = ErrorMessage(code="unknown_type", message=f"Unknown message type: {msg_type}")
                    await websocket.send(error.model_dump_json())

            except json.JSONDecodeError:
                error = ErrorMessage(code="invalid_json", message="Message must be valid JSON")
                await websocket.send(error.model_dump_json())
            except ValidationError as exc:
                error = ErrorMessage(code="validation_error", message=str(exc))
                await websocket.send(error.model_dump_json())
            except Exception as exc:
                logger.warning("Error processing inbound message from %s: %s", client_id, str(exc))

    except asyncio.CancelledError:
        pass
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as exc:
        logger.error("Unexpected error in _process_inbound: %s", str(exc))


async def _handle_subscribe(websocket, client_id: str, permissions: set[str], payload: dict) -> None:
    """Handle subscribe message from client.

    Args:
        websocket: WebSocket connection
        client_id: Client ID
        permissions: Set of permission strings
        payload: Message payload dict
    """
    try:
        msg = SubscribeMessage(**payload)
        channels = msg.channels

        # Validate channels
        invalid_channels = [ch for ch in channels if not _validate_channel(ch)]
        if invalid_channels:
            error = ErrorMessage(
                code="invalid_channel",
                message=f"Invalid channel format: {invalid_channels}",
            )
            await websocket.send(error.model_dump_json())
            return

        # Check permissions
        forbidden_channels = [
            ch for ch in channels if not _auth.is_permitted_channel(permissions, ch)
        ]
        if forbidden_channels:
            error = ErrorMessage(
                code="forbidden",
                message=f"Not permitted to subscribe to: {forbidden_channels}",
            )
            await websocket.send(error.model_dump_json())
            logger.warning(
                "Client %s attempted to subscribe to forbidden channels: %s",
                client_id,
                forbidden_channels,
            )
            return

        # Subscribe
        _router.subscribe(client_id, channels)
        logger.info("Client %s subscribed to channels: %s", client_id, channels)

    except ValidationError as exc:
        error = ErrorMessage(code="validation_error", message=str(exc))
        await websocket.send(error.model_dump_json())


async def _handle_unsubscribe(websocket, client_id: str, payload: dict) -> None:
    """Handle unsubscribe message from client.

    Args:
        websocket: WebSocket connection
        client_id: Client ID
        payload: Message payload dict
    """
    try:
        msg = UnsubscribeMessage(**payload)
        channels = msg.channels

        # Validate channels
        invalid_channels = [ch for ch in channels if not _validate_channel(ch)]
        if invalid_channels:
            error = ErrorMessage(
                code="invalid_channel",
                message=f"Invalid channel format: {invalid_channels}",
            )
            await websocket.send(error.model_dump_json())
            return

        _router.unsubscribe(client_id, channels)
        logger.info("Client %s unsubscribed from channels: %s", client_id, channels)

    except ValidationError as exc:
        error = ErrorMessage(code="validation_error", message=str(exc))
        await websocket.send(error.model_dump_json())


async def _handle_replay(websocket, client_id: str, permissions: set[str], payload: dict) -> None:
    """Handle replay message from client.

    Args:
        websocket: WebSocket connection
        client_id: Client ID
        permissions: Set of permission strings
        payload: Message payload dict
    """
    try:
        msg = ReplayMessage(**payload)
        channel = msg.channel
        since_seq = msg.since_seq

        # Validate channel
        if not _validate_channel(channel):
            error = ErrorMessage(code="invalid_channel", message=f"Invalid channel: {channel}")
            await websocket.send(error.model_dump_json())
            return

        # Check permission
        if not _auth.is_permitted_channel(permissions, channel):
            error = ErrorMessage(code="forbidden", message=f"Not permitted to replay: {channel}")
            await websocket.send(error.model_dump_json())
            logger.warning("Client %s attempted to replay forbidden channel: %s", client_id, channel)
            return

        # Get messages from replay buffer
        messages = _replay_buffer.get_since(channel, since_seq)
        logger.info(
            "Client %s replayed %d messages from channel %s since seq %d",
            client_id,
            len(messages),
            channel,
            since_seq,
        )

        # Send replayed messages
        for msg_dict in messages:
            await websocket.send(json.dumps(msg_dict))

    except ValidationError as exc:
        error = ErrorMessage(code="validation_error", message=str(exc))
        await websocket.send(error.model_dump_json())


async def _process_outbound(websocket, queue: asyncio.Queue, client_id: str, rate_limiter: TokenBucket) -> None:
    """Process outbound messages from router queue to client.

    Handles rate limiting and backpressure notifications.

    Args:
        websocket: WebSocket connection
        queue: Client's message queue
        client_id: Client ID
        rate_limiter: Token-bucket rate limiter for this client
    """
    try:
        while True:
            try:
                # Get next message from queue (non-blocking, short timeout)
                message = queue.get_nowait()
            except asyncio.QueueEmpty:
                # Wait a bit for messages
                await asyncio.sleep(0.01)
                continue

            # Rate limiting
            if not rate_limiter.is_allowed():
                retry_ms = rate_limiter.retry_after_ms()
                error = ErrorMessage(code="rate_limit", retry_after_ms=retry_ms)
                try:
                    await websocket.send(error.model_dump_json())
                except Exception as exc:
                    logger.debug("Failed to send rate limit error: %s", str(exc))
                continue

            # Send message
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as exc:
                logger.warning("Error sending message to client %s: %s", client_id, str(exc))
                break

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("Unexpected error in _process_outbound: %s", str(exc))


# ─────────────────────────────────────────────────────────────────────────
# Public API: Publishing and Server Management
# ─────────────────────────────────────────────────────────────────────────


async def publish_score_update(score_event: dict) -> None:
    """Publish a score update to subscribed clients.

    This must be called from within the server's asyncio event loop.
    Determines which clients are subscribed to the event's wallet/pair
    channels and enqueues the message.

    Args:
        score_event: Score event dict with keys:
            - wallet: wallet ID (e.g., "GXXX...")
            - asset_pair: asset pair (e.g., "XLM:native/USDC:...")
            - score: risk score (0-100)
            - score_lower: lower bound
            - score_upper: upper bound
            - bft_divergence: bool
            - top_features: list of feature dicts
            - timestamp: ISO 8601 timestamp
    """
    try:
        wallet_id = score_event.get("wallet")
        asset_pair = score_event.get("asset_pair")
        if not wallet_id or not asset_pair:
            logger.warning("Score event missing wallet or asset_pair: %s", score_event)
            return

        # Get next sequence number
        seq = _seq_counter.next()

        # Create message
        channel_subscribers = set()
        wallet_channel = f"wallet/{wallet_id}"
        pair_channel = f"pair/{asset_pair}"

        # Route to subscribers
        # Send to wallet channel subscribers
        for client_id in _router.get_subscribers(wallet_channel):
            message = {
                **score_event,
                "type": "score_update",
                "seq": seq,
                "channel": wallet_channel,
            }
            _enqueue_for_client(client_id, message)
            channel_subscribers.add(client_id)
            ws_messages_published_total.labels(channel_type="wallet").inc()

        # Send to pair channel subscribers
        for client_id in _router.get_subscribers(pair_channel):
            if client_id not in channel_subscribers:  # Don't send twice
                message = {
                    **score_event,
                    "type": "score_update",
                    "seq": seq,
                    "channel": pair_channel,
                }
                _enqueue_for_client(client_id, message)
                ws_messages_published_total.labels(channel_type="pair").inc()

        # Send to admin (all) subscribers
        for client_id in _router.get_subscribers("all"):
            if client_id not in channel_subscribers:  # Don't send twice
                message = {
                    **score_event,
                    "type": "score_update",
                    "seq": seq,
                    "channel": "all",
                }
                _enqueue_for_client(client_id, message)
                ws_messages_published_total.labels(channel_type="all").inc()

        # Store in replay buffer for each channel
        message_base = {**score_event, "type": "score_update", "seq": seq}

        if channel_subscribers:
            msg_wallet = {**message_base, "channel": wallet_channel}
            _replay_buffer.append(wallet_channel, seq, msg_wallet)

            msg_pair = {**message_base, "channel": pair_channel}
            _replay_buffer.append(pair_channel, seq, msg_pair)

            msg_all = {**message_base, "channel": "all"}
            _replay_buffer.append("all", seq, msg_all)

    except Exception as exc:
        logger.error("Error publishing score update: %s", str(exc))


def _enqueue_for_client(client_id: str, message: dict) -> None:
    """Enqueue message for client, handling backpressure.

    If queue is full, drops oldest message and sends notification.

    Args:
        client_id: Client ID
        message: Message dict to enqueue
    """
    with _clients_lock:
        if client_id not in _clients:
            return
        queue = _clients[client_id]["queue"]

    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        # Queue is full; drop oldest message
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        queue.put_nowait(message)

        # Notify client of dropped message
        dropped_msg = DroppedMessage(count=1)
        try:
            queue.put_nowait(dropped_msg.model_dump())
        except asyncio.QueueFull:
            pass

        ws_messages_dropped_total.inc()
        logger.warning("Dropped queued WebSocket message for client %s due to backpressure", client_id)


def push_alert_sync(payload: dict) -> None:
    """Thread-safe: schedule a score update from any thread.

    This is the legacy interface for compatibility with AlertDispatcher.

    Args:
        payload: Score event dict
    """
    if _loop is not None and _loop.is_running():
        asyncio.run_coroutine_threadsafe(publish_score_update(payload), _loop)


# ─────────────────────────────────────────────────────────────────────────
# Server Lifecycle
# ─────────────────────────────────────────────────────────────────────────


async def run_ws_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the WebSocket server and block until cancelled.

    Args:
        host: Bind host (default: 127.0.0.1)
        port: Bind port (default: 8765)
    """
    effective_host = os.getenv("WS_BIND_HOST", host)
    effective_port = int(os.getenv("WS_PORT", str(port)))

    if effective_host == "0.0.0.0" and not os.getenv("WS_ALLOW_EXTERNAL"):
        raise ValueError("Binding WebSocket server to 0.0.0.0 requires WS_ALLOW_EXTERNAL=1")

    logger.info("WebSocket server listening on %s:%d", effective_host, effective_port)
    logger.info(
        "WebSocket limits configured: max_clients=%d, queue_depth=%d, replay_buffer=%d",
        config.WS_MAX_CLIENTS,
        config.WS_CLIENT_QUEUE_DEPTH,
        config.WS_REPLAY_BUFFER_SIZE,
    )
    async with websockets.serve(_handler, effective_host, effective_port):
        await asyncio.Future()  # run until cancelled


def start_ws_server_thread(host: str = "127.0.0.1", port: int = 8765) -> threading.Thread:
    """Launch the WebSocket server in a daemon thread.

    Returns:
        Threading.Thread object (daemon=True)
    """
    global _loop

    ready = threading.Event()

    def _run() -> None:
        global _loop
        loop = asyncio.new_event_loop()
        _loop = loop
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_until_complete(run_ws_server(host, port))

    t = threading.Thread(target=_run, daemon=True, name="ws-server")
    t.start()
    ready.wait()
    return t


# ─────────────────────────────────────────────────────────────────────────
# Backward Compatibility Adapter
# ─────────────────────────────────────────────────────────────────────────


class _WsClientAdapter:
    """Adapts ws_client.send(msg) to push_alert_sync for backward compatibility."""

    def send(self, message: str) -> None:
        push_alert_sync(json.loads(message))
