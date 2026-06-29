"""Redis-backed distributed token-bucket rate limiter for Horizon API calls.

Coordinates a single global requests-per-second budget across however many
ingestion worker processes are running, so the combined call rate never
exceeds Horizon's per-IP limit regardless of worker count.
"""

from __future__ import annotations

import time

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    import redis
except ImportError:  # pragma: no cover - redis is an optional runtime dependency
    redis = None


class TokenBucketLimiter:
    """Distributed token bucket shared across worker processes via Redis.

    Token state (`tokens`, `updated_at`) is stored in a single Redis hash
    and mutated through a WATCH/MULTI optimistic transaction, so concurrent
    callers across processes never grant the same token. If Redis is
    unreachable -- at construction time or on any later call -- the limiter
    logs a warning once and degrades to granting every request immediately,
    rather than blocking ingestion on a rate limiter outage.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        key: str = "ledgerlens:horizon_rate_limiter",
        capacity: int | None = None,
        refill_rate_per_sec: float | None = None,
        poll_interval_seconds: float = 0.02,
        client: "redis.Redis | None" = None,
    ):
        self._key = key
        self._capacity = float(capacity if capacity is not None else config.HORIZON_MAX_RPS)
        self._refill_rate = float(
            refill_rate_per_sec if refill_rate_per_sec is not None else self._capacity
        )
        self._poll_interval = poll_interval_seconds
        self._warned = False
        self._client = client if client is not None else self._connect(redis_url)

    def _connect(self, redis_url: str | None):
        if redis is None:
            self._warn("redis package not installed")
            return None
        try:
            client = redis.Redis.from_url(
                redis_url or config.REDIS_URL, socket_connect_timeout=1, socket_timeout=1
            )
            client.ping()
            return client
        except Exception as exc:
            self._warn(f"Redis unavailable ({exc})")
            return None

    def _warn(self, reason: str) -> None:
        if not self._warned:
            logger.warning(
                "%s — proceeding without a distributed Horizon rate limit", reason
            )
            self._warned = True

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Attempt to take `tokens` from the bucket without blocking.

        Returns True if granted (or if Redis is unavailable, in which case
        every call is granted). Returns False if the bucket is currently
        exhausted.
        """
        if self._client is None:
            return True

        try:
            with self._client.pipeline() as pipe:
                while True:
                    try:
                        pipe.watch(self._key)
                        raw = pipe.hmget(self._key, "tokens", "updated_at")
                        now = time.time()
                        current_tokens = float(raw[0]) if raw[0] is not None else self._capacity
                        updated_at = float(raw[1]) if raw[1] is not None else now

                        elapsed = max(0.0, now - updated_at)
                        current_tokens = min(
                            self._capacity, current_tokens + elapsed * self._refill_rate
                        )

                        granted = current_tokens >= tokens
                        if granted:
                            current_tokens -= tokens

                        pipe.multi()
                        pipe.hset(self._key, mapping={"tokens": current_tokens, "updated_at": now})
                        pipe.expire(self._key, 60)
                        pipe.execute()
                        return granted
                    except redis.WatchError:
                        continue
        except Exception as exc:
            self._client = None
            self._warn(f"Redis rate limiter call failed ({exc})")
            return True

    def acquire(self, timeout: float | None = None) -> bool:
        """Block (polling) until a token is granted, or `timeout` elapses.

        Returns True once granted. Returns False only if `timeout` is given
        and exceeded -- with no timeout this blocks until a token is free.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.try_acquire():
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(self._poll_interval)
