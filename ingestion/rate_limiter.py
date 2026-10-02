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
        client: redis.Redis | None = None,
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
        except Exception as exc:  # noqa: BLE001
            # Broad catch justified: Redis connection can fail for many reasons
            # (network, DNS, auth, etc.). Gracefully degrade to no-op rate limiting
            # so ingestion continues without a distributed rate limit.
            self._warn(f"Redis unavailable ({exc})")
            return None

    def _warn(self, reason: str) -> None:
        if not self._warned:
            logger.warning("%s — proceeding without a distributed Horizon rate limit", reason)
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


_REMAINING_HEADERS = ("x-ratelimit-remaining", "ratelimit-remaining")
_RESET_HEADERS = ("x-ratelimit-reset", "ratelimit-reset")


def _header_float(headers: dict, names: tuple[str, ...]) -> float | None:
    lowered = {str(k).lower(): v for k, v in (headers or {}).items()}
    for name in names:
        if name in lowered:
            try:
                return float(lowered[name])
            except (TypeError, ValueError):
                return None
    return None


class AdaptiveRateLimiter:
    """Per-source pacing driven by upstream rate-limit headers.

    Each source has a static ``max_rps`` ceiling. When a response carries
    ``X-RateLimit-Remaining`` / ``X-RateLimit-Reset`` (or the IETF
    ``RateLimit-*`` equivalents), the effective rate becomes
    ``remaining / seconds_until_reset``, clamped to ``(0, max_rps]`` so a
    generous header can never push pacing above the configured safety cap.
    A 429/503 with ``Retry-After`` pauses the source until that deadline.
    Sources that never send headers keep pacing at the static ``max_rps``.

    ``Reset`` values larger than ``epoch_threshold`` are treated as Unix
    timestamps, smaller ones as delta-seconds.
    """

    def __init__(
        self,
        max_rps: dict[str, float] | None = None,
        default_max_rps: float | None = None,
        clock=time.time,
        sleep=time.sleep,
        epoch_threshold: float = 1e9,
    ):
        self._ceilings = dict(max_rps or {})
        self._default = float(
            default_max_rps if default_max_rps is not None else config.HORIZON_MAX_RPS
        )
        self._clock = clock
        self._sleep = sleep
        self._epoch_threshold = epoch_threshold
        self._rates: dict[str, float] = {}
        self._next_allowed: dict[str, float] = {}

    def ceiling(self, source: str) -> float:
        return float(self._ceilings.get(source, self._default))

    def effective_rate(self, source: str) -> float:
        return self._rates.get(source, self.ceiling(source))

    def update_from_response(self, source: str, status_code: int, headers: dict) -> float:
        """Adjust ``source`` pacing from a response; returns the new effective rate."""
        now = self._clock()
        ceiling = self.ceiling(source)
        retry_after = _header_float(headers, ("retry-after",))
        if status_code in (429, 503) and retry_after is not None:
            self._next_allowed[source] = max(self._next_allowed.get(source, 0.0), now + retry_after)

        remaining = _header_float(headers, _REMAINING_HEADERS)
        reset = _header_float(headers, _RESET_HEADERS)
        if remaining is None:
            return self.effective_rate(source)  # header-less source: static fallback
        window = 1.0
        if reset is not None:
            window = reset - now if reset > self._epoch_threshold else reset
            window = max(window, 1e-3)
        if remaining <= 0:
            self._next_allowed[source] = max(self._next_allowed.get(source, 0.0), now + window)
            rate = ceiling
        else:
            rate = min(remaining / window, ceiling)
        self._rates[source] = rate
        return rate

    def delay_for(self, source: str) -> float:
        """Seconds to wait before the next request to ``source`` may be sent."""
        return max(0.0, self._next_allowed.get(source, 0.0) - self._clock())

    def acquire(self, source: str) -> float:
        """Blocks until ``source`` may be called; returns the seconds slept."""
        delay = self.delay_for(source)
        if delay > 0:
            self._sleep(delay)
        now = self._clock()
        self._next_allowed[source] = max(now, self._next_allowed.get(source, 0.0)) + (
            1.0 / self.effective_rate(source)
        )
        return delay
