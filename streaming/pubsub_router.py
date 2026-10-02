"""Pub/Sub router for WebSocket message distribution.

Routes published messages to subscribers based on channel subscriptions.
Thread-safe for concurrent operations.
"""

import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DEAD_LETTER_PATH = os.getenv("PUBSUB_DEAD_LETTER_PATH", "data/pubsub_dead_letter.jsonl")
DEFAULT_MAX_DELIVERY_ATTEMPTS = int(os.getenv("PUBSUB_MAX_DELIVERY_ATTEMPTS", "3"))


class DeadLetterStore:
    """Append-only JSONL store for poison messages.

    Each entry carries the original message plus failure context: the last
    error, retry count, first/last failure timestamps and target client.
    """

    def __init__(self, path: str = DEFAULT_DEAD_LETTER_PATH):
        self.path = path
        self._lock = threading.Lock()

    def append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def entries(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with self._lock, open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def remove(self, entry_ids: set[str]) -> None:
        """Drop entries (e.g. after a successful replay)."""
        keep = [e for e in self.entries() if e.get("id") not in entry_ids]
        with self._lock, open(self.path, "w", encoding="utf-8") as f:
            for e in keep:
                f.write(json.dumps(e, default=str) + "\n")


class PubSubRouter:
    """Routes messages to subscribers based on channel subscriptions.

    Channels:
    - wallet/{wallet_id}: all scores for a specific wallet
    - pair/{asset_pair}: all scores for a specific asset pair
    - all: admin channel for all messages
    """

    def __init__(
        self,
        max_delivery_attempts: int = DEFAULT_MAX_DELIVERY_ATTEMPTS,
        dead_letter_store: DeadLetterStore | None = None,
    ):
        self.max_delivery_attempts = max(1, max_delivery_attempts)
        self.dead_letter_store = dead_letter_store or DeadLetterStore()
        # client_id -> set of subscribed channels
        self._subscriptions: dict[str, set[str]] = defaultdict(set)
        # channel -> set of subscribed client_ids
        self._channel_subscribers: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def subscribe(self, client_id: str, channels: list[str]) -> None:
        """Register client subscriptions to channels.

        Args:
            client_id: Unique client identifier
            channels: List of channel names (e.g., ["wallet/GXXX", "pair/..."])
        """
        with self._lock:
            for channel in channels:
                self._subscriptions[client_id].add(channel)
                self._channel_subscribers[channel].add(client_id)
            logger.debug(
                "Client %s subscribed to %d channel(s): %s",
                client_id,
                len(channels),
                channels,
            )

    def unsubscribe(self, client_id: str, channels: list[str]) -> None:
        """Remove client subscriptions from channels.

        Args:
            client_id: Unique client identifier
            channels: List of channel names to unsubscribe from
        """
        with self._lock:
            for channel in channels:
                self._subscriptions[client_id].discard(channel)
                self._channel_subscribers[channel].discard(client_id)
                # Clean up empty entries
                if not self._channel_subscribers[channel]:
                    del self._channel_subscribers[channel]
            logger.debug(
                "Client %s unsubscribed from %d channel(s)",
                client_id,
                len(channels),
            )

    def disconnect(self, client_id: str) -> None:
        """Remove client and all its subscriptions.

        Args:
            client_id: Unique client identifier
        """
        with self._lock:
            if client_id not in self._subscriptions:
                return
            channels = list(self._subscriptions[client_id])
            for channel in channels:
                self._channel_subscribers[channel].discard(client_id)
                if not self._channel_subscribers[channel]:
                    del self._channel_subscribers[channel]
            del self._subscriptions[client_id]
            logger.debug(
                "Client %s disconnected (was subscribed to %d channel(s))", client_id, len(channels)
            )

    def get_subscribers(self, channel: str) -> set[str]:
        """Get set of client IDs subscribed to a channel.

        Args:
            channel: Channel name (e.g., "wallet/GXXX" or "pair/...")

        Returns:
            Set of subscribed client_ids (may be empty).
        """
        with self._lock:
            return set(self._channel_subscribers.get(channel, set()))

    def get_clients_for_event(self, wallet_id: str, asset_pair: str) -> set[str]:
        """Determine which clients should receive a score event.

        Clients subscribed to:
        - wallet/{wallet_id}
        - pair/{asset_pair}
        - all (admin subscribers)

        Args:
            wallet_id: Wallet ID (e.g., "GXXX...")
            asset_pair: Asset pair (e.g., "XLM:native/USDC:...")

        Returns:
            Set of client_ids that should receive the message.
        """
        with self._lock:
            clients = set()

            # Wallet-specific subscribers
            wallet_channel = f"wallet/{wallet_id}"
            clients.update(self._channel_subscribers.get(wallet_channel, set()))

            # Pair-specific subscribers
            pair_channel = f"pair/{asset_pair}"
            clients.update(self._channel_subscribers.get(pair_channel, set()))

            # Admin subscribers
            clients.update(self._channel_subscribers.get("all", set()))

            return clients

    def get_subscriptions(self, client_id: str) -> set[str]:
        """Get set of channels for a client.

        Args:
            client_id: Unique client identifier

        Returns:
            Set of channel names the client is subscribed to.
        """
        with self._lock:
            return set(self._subscriptions.get(client_id, set()))

    def stats(self) -> dict[str, Any]:
        """Return router statistics for monitoring.

        Returns:
            Dict with keys: total_clients, total_channels, subscriptions_per_client.
        """
        with self._lock:
            return {
                "total_clients": len(self._subscriptions),
                "total_channels": len(self._channel_subscribers),
                "subscriptions_per_client": {
                    client_id: len(channels) for client_id, channels in self._subscriptions.items()
                },
                "subscribers_per_channel": {
                    channel: len(clients) for channel, clients in self._channel_subscribers.items()
                },
            }

    def deliver(
        self,
        client_id: str,
        channel: str,
        message: dict[str, Any],
        handler: Callable[[str, dict[str, Any]], None],
    ) -> bool:
        """Deliver *message* to *client_id* via *handler* with bounded retries.

        After ``max_delivery_attempts`` consecutive failures the message is
        routed to the dead-letter store with failure context instead of being
        retried indefinitely or silently dropped. Returns True on success.
        """
        first_failure: float | None = None
        last_error: Exception | None = None
        for attempt in range(1, self.max_delivery_attempts + 1):
            try:
                handler(client_id, message)
                return True
            except Exception as exc:  # noqa: BLE001 — any downstream failure counts
                last_error = exc
                first_failure = first_failure or time.time()
                logger.warning(
                    "PubSub delivery to %s failed (attempt %d/%d): %s",
                    client_id, attempt, self.max_delivery_attempts, exc,
                )
        entry = {
            "id": f"{client_id}:{channel}:{time.time_ns()}",
            "client_id": client_id,
            "channel": channel,
            "message": message,
            "error": f"{type(last_error).__name__}: {last_error}",
            "retry_count": self.max_delivery_attempts,
            "first_failure_at": first_failure,
            "dead_lettered_at": time.time(),
        }
        self.dead_letter_store.append(entry)
        logger.error("PubSub message dead-lettered: %s", entry["id"])
        return False

    def replay_dead_letters(
        self,
        handler: Callable[[str, dict[str, Any]], None],
        entry_ids: set[str] | None = None,
    ) -> dict[str, int]:
        """Re-deliver dead-lettered messages; successes are removed from the store.

        Messages that still fail are re-dead-lettered with fresh context.
        """
        entries = [
            e for e in self.dead_letter_store.entries() if entry_ids is None or e["id"] in entry_ids
        ]
        self.dead_letter_store.remove({e["id"] for e in entries})
        ok = 0
        for e in entries:
            if self.deliver(e["client_id"], e["channel"], e["message"], handler):
                ok += 1
        return {"replayed": ok, "failed": len(entries) - ok}
