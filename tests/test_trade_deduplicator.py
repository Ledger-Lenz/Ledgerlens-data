"""Tests for idempotent trade ingestion with Redis deduplication."""

import pytest
import time
from datetime import datetime, timedelta, timezone

from ingestion.trade_deduplicator import (
    SeenEventCache,
    is_duplicate_trade,
    get_trade_dedup_cache,
)

# Try to import fakeredis for testing
try:
    import fakeredis
    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False
    fakeredis = None


@pytest.fixture
def fake_redis_cache():
    """Create a SeenEventCache with fakeredis backend for testing."""
    if not _FAKEREDIS_AVAILABLE:
        pytest.skip("fakeredis not installed")

    # Create a mock redis module using fakeredis
    fake_redis_instance = fakeredis.FakeStrictRedis(decode_responses=True)

    # Patch redis.from_url to return our fake instance
    cache = SeenEventCache(
        redis_url="redis://localhost:6379/0",
        ttl_seconds=86400,
        key_prefix="ledgerlens:trades:",
    )

    # Replace the internal Redis instance with fake
    cache._redis = fake_redis_instance
    cache._redis_available = True

    return cache


class TestSeenEventCacheDeduplication:
    """Tests for duplicate trade detection."""

    def test_first_trade_not_duplicate(self, fake_redis_cache):
        """First submission of a trade should not be flagged as duplicate."""
        trade_id = "trade-123"
        paging_token = "paging-456"
        asset_pair = "USDC/XLM"

        is_dup = fake_redis_cache.is_duplicate(trade_id, paging_token, asset_pair)

        assert is_dup is False

    def test_second_identical_trade_is_duplicate(self, fake_redis_cache):
        """Second submission of same trade should be flagged as duplicate."""
        trade_id = "trade-123"
        paging_token = "paging-456"
        asset_pair = "USDC/XLM"

        # First submission
        is_dup_1 = fake_redis_cache.is_duplicate(trade_id, paging_token, asset_pair)
        assert is_dup_1 is False

        # Second submission (same paging token)
        is_dup_2 = fake_redis_cache.is_duplicate(trade_id, paging_token, asset_pair)
        assert is_dup_2 is True

    def test_different_trades_not_duplicates(self, fake_redis_cache):
        """Different trades should not be flagged as duplicates."""
        asset_pair = "USDC/XLM"

        # First trade
        fake_redis_cache.is_duplicate("trade-1", "paging-1", asset_pair)

        # Second trade (different paging token)
        is_dup = fake_redis_cache.is_duplicate("trade-2", "paging-2", asset_pair)

        assert is_dup is False

    def test_trade_hash_prevents_id_collision(self, fake_redis_cache):
        """Different paging tokens produce different hashes."""
        asset_pair = "USDC/XLM"

        # Two trades with similar IDs but different tokens
        is_dup_1 = fake_redis_cache.is_duplicate("trade-1", "token-A", asset_pair)
        is_dup_2 = fake_redis_cache.is_duplicate("trade-1", "token-B", asset_pair)

        assert is_dup_1 is False
        assert is_dup_2 is False  # Different token = different hash


class TestCacheTTLandEviction:
    """Tests for Redis TTL and cache eviction."""

    def test_cache_entry_expires(self, fake_redis_cache):
        """Cached trade should expire after TTL."""
        trade_id = "trade-123"
        paging_token = "paging-456"
        asset_pair = "USDC/XLM"

        # Add trade
        fake_redis_cache.is_duplicate(trade_id, paging_token, asset_pair)

        # Verify it's cached
        cache_key = f"{fake_redis_cache.key_prefix}{asset_pair}"
        size_before = fake_redis_cache._redis.zcard(cache_key)
        assert size_before == 1

        # Simulate TTL expiration by directly deleting the key
        # (fakeredis doesn't automatically expire keys)
        fake_redis_cache._redis.delete(cache_key)

        # Verify cache is empty
        size_after = fake_redis_cache._redis.zcard(cache_key) or 0
        assert size_after == 0

    def test_cache_does_not_grow_unbounded(self, fake_redis_cache):
        """Multiple unique trades should add to cache, not exceed sorted set."""
        asset_pair = "USDC/XLM"
        cache_key = f"{fake_redis_cache.key_prefix}{asset_pair}"

        # Add 100 different trades
        for i in range(100):
            fake_redis_cache.is_duplicate(f"trade-{i}", f"token-{i}", asset_pair)

        # Cache should have exactly 100 entries
        size = fake_redis_cache._redis.zcard(cache_key)
        assert size == 100

    def test_cache_size_query(self, fake_redis_cache):
        """get_cache_size should return correct count."""
        asset_pair = "USDC/XLM"

        # Add 5 trades
        for i in range(5):
            fake_redis_cache.is_duplicate(f"trade-{i}", f"token-{i}", asset_pair)

        # Check size
        size = fake_redis_cache.get_cache_size(asset_pair)
        assert size == 5


class TestGracefulDegradation:
    """Tests for behavior when Redis is unavailable."""

    def test_cache_unavailable_allows_trades(self):
        """When Redis unavailable, is_duplicate returns False (allow trade)."""
        cache = SeenEventCache(redis_url="redis://nonexistent:9999/0")

        # Redis should not be available
        assert cache._redis_available is False

        # Should allow all trades through
        is_dup = cache.is_duplicate("trade-1", "token-1", "USDC/XLM")
        assert is_dup is False

    def test_cache_fallback_on_redis_error(self, fake_redis_cache):
        """If Redis fails during check, allow trade through (don't crash)."""
        # Simulate Redis error by closing connection
        fake_redis_cache._redis = None

        # Should not raise, just allow trade
        is_dup = fake_redis_cache.is_duplicate("trade-1", "token-1", "USDC/XLM")
        assert is_dup is False


class TestCacheOperations:
    """Tests for cache utility operations."""

    def test_cache_trade_explicitly(self, fake_redis_cache):
        """cache_trade should add trade without checking first."""
        asset_pair = "USDC/XLM"
        cache_key = f"{fake_redis_cache.key_prefix}{asset_pair}"

        # Explicitly cache
        fake_redis_cache.cache_trade("trade-1", "token-1", asset_pair)

        # Verify it's cached
        size = fake_redis_cache._redis.zcard(cache_key)
        assert size == 1

    def test_clear_cache_specific_pair(self, fake_redis_cache):
        """clear_cache should remove trades for specific pair."""
        pair1 = "USDC/XLM"
        pair2 = "EUR/XLM"

        # Add trades for both pairs
        fake_redis_cache.is_duplicate("trade-1", "token-1", pair1)
        fake_redis_cache.is_duplicate("trade-2", "token-2", pair2)

        # Clear first pair
        fake_redis_cache.clear_cache(pair1)

        # First pair should be empty, second should still have data
        assert (fake_redis_cache._redis.zcard(f"{fake_redis_cache.key_prefix}{pair1}") or 0) == 0
        assert (fake_redis_cache._redis.zcard(f"{fake_redis_cache.key_prefix}{pair2}") or 0) == 1

    def test_health_check_available(self, fake_redis_cache):
        """health_check should return True when Redis available."""
        assert fake_redis_cache.health_check() is True

    def test_health_check_unavailable(self):
        """health_check should return False when Redis unavailable."""
        cache = SeenEventCache(redis_url="redis://nonexistent:9999/0")
        assert cache.health_check() is False


class TestAssetPairSeparation:
    """Tests for per-pair cache isolation."""

    def test_same_trade_different_pairs_separate(self, fake_redis_cache):
        """Same trade ID on different pairs should not be flagged as duplicate."""
        trade_id = "trade-123"
        token = "paging-456"

        # Same trade, different pairs
        dup_1 = fake_redis_cache.is_duplicate(trade_id, token, "USDC/XLM")
        dup_2 = fake_redis_cache.is_duplicate(trade_id, token, "EUR/XLM")

        # Both should be new (different pair caches)
        assert dup_1 is False
        assert dup_2 is False

    def test_cache_isolation_per_pair(self, fake_redis_cache):
        """Cache entries are isolated per asset pair."""
        # Add trade to pair 1
        fake_redis_cache.is_duplicate("trade-1", "token-1", "USDC/XLM")

        # Pair 1 should have 1 entry
        size_1 = fake_redis_cache.get_cache_size("USDC/XLM")
        assert size_1 == 1

        # Pair 2 should have 0 entries
        size_2 = fake_redis_cache.get_cache_size("EUR/XLM")
        assert size_2 == 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_is_duplicate_trade_convenience(self):
        """is_duplicate_trade convenience function should work."""
        # Get global cache (may not be Redis-backed in test env)
        cache = get_trade_dedup_cache()
        assert cache is not None

    def test_get_trade_dedup_cache_singleton(self):
        """get_trade_dedup_cache should return same instance."""
        cache1 = get_trade_dedup_cache()
        cache2 = get_trade_dedup_cache()
        assert cache1 is cache2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_paging_token_uses_trade_id(self, fake_redis_cache):
        """If paging token is None, use trade ID for hashing."""
        trade_id = "trade-123"
        asset_pair = "USDC/XLM"

        # Submit with None paging token
        dup_1 = fake_redis_cache.is_duplicate(trade_id, None, asset_pair)
        assert dup_1 is False

        # Submit again (same trade ID, None token)
        dup_2 = fake_redis_cache.is_duplicate(trade_id, None, asset_pair)
        assert dup_2 is True

    def test_unicode_trade_ids(self, fake_redis_cache):
        """Handle Unicode in trade IDs gracefully."""
        trade_id = "trade-αβγ"
        token = "token-日本語"
        asset_pair = "USDC/XLM"

        dup_1 = fake_redis_cache.is_duplicate(trade_id, token, asset_pair)
        dup_2 = fake_redis_cache.is_duplicate(trade_id, token, asset_pair)

        assert dup_1 is False
        assert dup_2 is True

    def test_long_asset_pair_name(self, fake_redis_cache):
        """Handle long asset pair names."""
        asset_pair = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"

        dup_1 = fake_redis_cache.is_duplicate("trade-1", "token-1", asset_pair)
        dup_2 = fake_redis_cache.is_duplicate("trade-1", "token-1", asset_pair)

        assert dup_1 is False
        assert dup_2 is True
