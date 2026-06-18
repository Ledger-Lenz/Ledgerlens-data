"""Minimal asyncio WebSocket server for broadcasting risk alerts.

The server runs in a background daemon thread so it does not block the SSE
ingestion threads. External code pushes alerts via push_alert_sync(), which
schedules a coroutine on the server's event loop thread-safely.

Security defaults
-----------------
- Binds to 127.0.0.1 by default (loopback-only).
- Raises ValueError if asked to bind to 0.0.0.0 without WS_ALLOW_EXTERNAL=1.
"""

import asyncio
import json
import os
import threading

import websockets

from utils.logging import get_logger

logger = get_logger(__name__)

# Module-level state managed exclusively inside the asyncio event loop.
_clients: set = set()
_loop: asyncio.AbstractEventLoop | None = None


async def _handler(websocket) -> None:
    _clients.add(websocket)
    logger.debug("WebSocket client connected (%d total)", len(_clients))
    try:
        async for _ in websocket:
            pass  # server-push only; ignore any inbound messages
    finally:
        _clients.discard(websocket)
        logger.debug("WebSocket client disconnected (%d remaining)", len(_clients))


async def run_ws_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve connected WebSocket clients and block until cancelled."""
    effective_host = os.getenv("WS_BIND_HOST", host)
    effective_port = int(os.getenv("WS_PORT", str(port)))

    if effective_host == "0.0.0.0" and not os.getenv("WS_ALLOW_EXTERNAL"):
        raise ValueError("Binding WebSocket server to 0.0.0.0 requires WS_ALLOW_EXTERNAL=1")

    logger.info("WebSocket server listening on %s:%d", effective_host, effective_port)
    async with websockets.serve(_handler, effective_host, effective_port):
        await asyncio.Future()  # run until cancelled


async def send_alert(payload: dict) -> None:
    """Broadcast *payload* to all currently connected WebSocket clients.

    Must be called from within the server's asyncio event loop.
    """
    if not _clients:
        return
    message = json.dumps(payload)
    dead: set = set()
    for client in list(_clients):
        try:
            await client.send(message)
        except Exception:
            dead.add(client)
    _clients.difference_update(dead)


def push_alert_sync(payload: dict) -> None:
    """Thread-safe: schedule a WebSocket broadcast from any thread."""
    if _loop is not None and _loop.is_running():
        asyncio.run_coroutine_threadsafe(send_alert(payload), _loop)


def start_ws_server_thread(host: str = "127.0.0.1", port: int = 8765) -> threading.Thread:
    """Launch the WebSocket server in a daemon thread and return it."""
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


class _WsClientAdapter:
    """Adapts ws_client.send(msg) to the server's push_alert_sync mechanism."""

    def send(self, message: str) -> None:
        push_alert_sync(json.loads(message))
