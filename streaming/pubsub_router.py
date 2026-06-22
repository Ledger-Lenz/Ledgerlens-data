"""Pub/Sub router for WebSocket message distribution.

Routes published messages to subscribers based on channel subscriptions.
Thread-safe for concurrent operations.
"""

import threading
from collections import defaultdict
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


class PubSubRouter:
    """Routes messages to subscribers based on channel subscriptions.

    Channels:
    - wallet/{wallet_id}: all scores for a specific wallet
    - pair/{asset_pair}: all scores for a specific asset pair
    - all: admin channel for all messages
    """

    def __init__(self):
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
            logger.debug("Client %s disconnected (was subscribed to %d channel(s))", client_id, len(channels))

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
                    channel: len(clients)
                    for channel, clients in self._channel_subscribers.items()
                },
            }
