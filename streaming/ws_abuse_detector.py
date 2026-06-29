"""WebSocket abuse detection for LedgerLens (issue #223).

Detects two abuse patterns:
1. Request-rate abuse — a client exceeds WS_ABUSE_MAX_REQUESTS_PER_MINUTE
   in a rolling 60-second window.
2. Wallet-targeting abuse — a client queries an unusually high number of
   distinct wallet IDs within a rolling window (model-extraction signal).

Usage::

    detector = AbuseDetector()          # one shared instance
    verdict = detector.record(client_id, channel)
    if verdict.blocked:
        # send error and close connection
        ...
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

_WALLET_RE = re.compile(r"^wallet/(G[A-Z2-7]{55})$")


@dataclass
class AbuseVerdict:
    blocked: bool
    reason: str = ""
    retry_after_seconds: int = 60


@dataclass
class _ClientState:
    request_times: deque = field(default_factory=deque)
    wallet_window_start: float = field(default_factory=time.monotonic)
    wallet_ids_seen: set = field(default_factory=set)
    blocked_until: float = 0.0


class AbuseDetector:
    """Thread-safe per-client rate-limit and wallet-targeting detector."""

    def __init__(
        self,
        max_requests_per_minute: int | None = None,
        max_distinct_wallets_per_window: int | None = None,
        wallet_window_seconds: int | None = None,
        block_duration_seconds: int | None = None,
    ) -> None:
        self._max_rpm = max_requests_per_minute or config.WS_ABUSE_MAX_REQUESTS_PER_MINUTE
        self._max_wallets = (
            max_distinct_wallets_per_window or config.WS_ABUSE_MAX_DISTINCT_WALLETS
        )
        self._wallet_window = wallet_window_seconds or config.WS_ABUSE_WALLET_WINDOW_SECONDS
        self._block_duration = block_duration_seconds or config.WS_ABUSE_BLOCK_DURATION_SECONDS
        self._states: dict[str, _ClientState] = defaultdict(_ClientState)
        self._lock = threading.Lock()

    def record(self, client_id: str, channel: str) -> AbuseVerdict:
        """Record one message and return a verdict.

        Args:
            client_id: Authenticated client identifier (JWT sub).
            channel:   Channel string, e.g. ``"wallet/GXXX..."`` or ``"pair/..."``.

        Returns:
            :class:`AbuseVerdict` — ``blocked=True`` if the client tripped a
            threshold or is currently banned.
        """
        now = time.monotonic()
        with self._lock:
            state = self._states[client_id]

            # Already blocked?
            if state.blocked_until > now:
                remaining = int(state.blocked_until - now) + 1
                return AbuseVerdict(
                    blocked=True,
                    reason="abuse_block_active",
                    retry_after_seconds=remaining,
                )

            # Request-rate check (sliding 60 s window)
            cutoff = now - 60.0
            while state.request_times and state.request_times[0] < cutoff:
                state.request_times.popleft()
            state.request_times.append(now)

            if len(state.request_times) > self._max_rpm:
                state.blocked_until = now + self._block_duration
                logger.warning(
                    "WS abuse: rate limit exceeded (client=%s, rpm=%d, limit=%d)",
                    client_id,
                    len(state.request_times),
                    self._max_rpm,
                )
                return AbuseVerdict(
                    blocked=True,
                    reason="rate_limit_exceeded",
                    retry_after_seconds=self._block_duration,
                )

            # Wallet-targeting check
            m = _WALLET_RE.match(channel)
            if m:
                wallet_id = m.group(1)
                if now - state.wallet_window_start > self._wallet_window:
                    state.wallet_window_start = now
                    state.wallet_ids_seen.clear()
                state.wallet_ids_seen.add(wallet_id)

                if len(state.wallet_ids_seen) > self._max_wallets:
                    state.blocked_until = now + self._block_duration
                    logger.warning(
                        "WS abuse: wallet-targeting detected "
                        "(client=%s, distinct_wallets=%d, limit=%d, window=%ds)",
                        client_id,
                        len(state.wallet_ids_seen),
                        self._max_wallets,
                        self._wallet_window,
                    )
                    return AbuseVerdict(
                        blocked=True,
                        reason="wallet_targeting_detected",
                        retry_after_seconds=self._block_duration,
                    )

        return AbuseVerdict(blocked=False)

    def reset(self, client_id: str) -> None:
        """Remove all state for a client (call on disconnect)."""
        with self._lock:
            self._states.pop(client_id, None)
