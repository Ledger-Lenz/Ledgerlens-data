"""Idempotent trade ingestion using Redis deduplication cache.

Handles duplicate trades from Stellar Horizon SSE stream by caching trade IDs
in Redis with a 24-hour TTL. Gracefully degrades when Redis is unavailable.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter

    ledgerlens_duplicate_events_total = Counter(
        "ledgerlens_duplicate_events_total",
        "Total number of duplicate trade events discarded",
        ["asset_pair"],
    )
    ledgerlens_dedup_cache_hits_total = Counter(
        "ledgerlens_dedup_cache_hits_total",
        "Total cache hits in trade deduplication",
    )
except ImportError:
    ledgerlens_duplicate_events_total = None
    ledgerlens_dedup_cache_hits_total = None

# Optional Redis imports
try:
    import redis
    from redis.exceptions import RedisError

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    RedisError = Exception  # type: ignore


class SeenEventCache:
    """Redis-backed cache for deduplicating Horizon trade events.

    Uses a Redis sorted set (ZSET) with:
    - Score: timestamp (for TTL + expiration)
    - Member: SHA-256 hash of trade paging token

    TTL is enforced by Redis EXPIREAT, surviving process restarts.
    Gracefully degrades when Redis unavailable: logs warning and allows
    duplicate through (do not stop ingestion).
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
        key_prefix: str | None = None,
    ):
        """Initialize the deduplication cache.

        Args:
            redis_url: Redis connection URL (default config.REDIS_URL).
            ttl_seconds: Expiration time in seconds (default 24h).
            key_prefix: Key prefix for cache entries (default ledgerlens:trades:).
        """
        self.redis_url = redis_url or config.REDIS_URL
        self.ttl_seconds = ttl_seconds or config.TRADE_DEDUP_TTL_SECONDS
        self.key_prefix = key_prefix or config.TRADE_DEDUP_CACHE_KEY_PREFIX

        self._redis: Optional[redis.Redis] = None
        self._redis_available = False

        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection with fallback to no-op mode."""
        if not _REDIS_AVAILABLE:
            logger.warning("Redis library not installed; trade deduplication disabled")
            return

        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=True, socket_timeout=5)
            # Test connection
            self._redis.ping()
            self._redis_available = True
            logger.info("Connected to Redis for trade deduplication")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis ({self.redis_url}): {e} — proceeding without deduplication")
            self._redis = None
            self._redis_available = False

    def is_duplicate(
        self,
        trade_id: str,
        paging_token: str | None = None,
        asset_pair: str = "unknown",
    ) -> bool:
        """Check if trade has been seen before and cache it.

        Args:
            trade_id: Horizon trade ID.
            paging_token: Horizon paging token (used for hashing).
            asset_pair: Asset pair for metric labeling (e.g., "USDC/XLM").

        Returns:
            True if trade is a duplicate (already cached), False if new.
        """
        if not self._redis_available:
            # No Redis: allow all trades through
            return False

        try:
            # Hash the trade ID or paging token
            hash_input = paging_token or trade_id
            trade_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            # Construct cache key
            cache_key = f"{self.key_prefix}{asset_pair}"

            # Current timestamp (score in sorted set)
            current_time = time.time()

            # Check if trade hash exists in sorted set (exists = duplicate)
            score = self._redis.zscore(cache_key, trade_hash)

            if score is not None:
                # Duplicate found
                logger.debug(f"Trade duplicate detected: {trade_id} ({trade_hash[:8]}…)")
                if ledgerlens_duplicate_events_total:
                    ledgerlens_duplicate_events_total.labels(asset_pair=asset_pair).inc()
                if ledgerlens_dedup_cache_hits_total:
                    ledgerlens_dedup_cache_hits_total.inc()
                return True

            # New trade: add to cache
            self._redis.zadd(cache_key, {trade_hash: current_time})

            # Set expiration: absolute timestamp when key should expire
            expiration_time = int(current_time + self.ttl_seconds)
            self._redis.expireat(cache_key, expiration_time)

            logger.debug(f"Trade cached: {trade_id} ({trade_hash[:8]}…), expires at {expiration_time}")
            return False

        except RedisError as e:
            logger.warning(f"Redis error during dedup check: {e} — allowing trade through")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in is_duplicate: {e}")
            return False

    def cache_trade(
        self,
        trade_id: str,
        paging_token: str | None = None,
        asset_pair: str = "unknown",
    ) -> None:
        """Explicitly add a trade to the cache (used when not checking duplicates).

        Args:
            trade_id: Horizon trade ID.
            paging_token: Horizon paging token.
            asset_pair: Asset pair for organization.
        """
        if not self._redis_available:
            return

        try:
            hash_input = paging_token or trade_id
            trade_hash = hashlib.sha256(hash_input.encode()).hexdigest()
            cache_key = f"{self.key_prefix}{asset_pair}"
            current_time = time.time()

            self._redis.zadd(cache_key, {trade_hash: current_time})
            expiration_time = int(current_time + self.ttl_seconds)
            self._redis.expireat(cache_key, expiration_time)

        except Exception as e:
            logger.warning(f"Failed to cache trade {trade_id}: {e}")

    def get_cache_size(self, asset_pair: str = "unknown") -> int:
        """Get the number of cached trade hashes for an asset pair.

        Args:
            asset_pair: Asset pair identifier.

        Returns:
            Number of cached trades, or -1 if Redis unavailable.
        """
        if not self._redis_available:
            return -1

        try:
            cache_key = f"{self.key_prefix}{asset_pair}"
            return self._redis.zcard(cache_key) or 0
        except Exception as e:
            logger.warning(f"Failed to get cache size: {e}")
            return -1

    def clear_cache(self, asset_pair: str | None = None) -> bool:
        """Clear cached trades (for testing).

        Args:
            asset_pair: Specific pair to clear, or None to clear all.

        Returns:
            True if successful, False otherwise.
        """
        if not self._redis_available:
            return False

        try:
            if asset_pair:
                cache_key = f"{self.key_prefix}{asset_pair}"
                self._redis.delete(cache_key)
            else:
                # Clear all keys matching prefix
                pattern = f"{self.key_prefix}*"
                for key in self._redis.scan_iter(match=pattern):
                    self._redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return False

    def health_check(self) -> bool:
        """Check if Redis connection is healthy.

        Returns:
            True if Redis is available and responsive, False otherwise.
        """
        if not self._redis_available or not self._redis:
            return False

        try:
            self._redis.ping()
            return True
        except Exception:
            return False


# Global singleton cache instance
_cache_instance: Optional[SeenEventCache] = None


def get_trade_dedup_cache() -> SeenEventCache:
    """Get or create the global trade deduplication cache.

    Returns:
        SeenEventCache instance.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SeenEventCache()
    return _cache_instance


def is_duplicate_trade(
    trade_id: str,
    paging_token: str | None = None,
    asset_pair: str = "unknown",
) -> bool:
    """Convenience function: check if trade is a duplicate using the global cache.

    Args:
        trade_id: Horizon trade ID.
        paging_token: Horizon paging token.
        asset_pair: Asset pair for metric labeling.

    Returns:
        True if trade is a duplicate, False if new.
    """
    cache = get_trade_dedup_cache()
    return cache.is_duplicate(trade_id, paging_token, asset_pair)
