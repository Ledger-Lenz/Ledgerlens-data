"""Tests for detection.feature_cache.CacheWarmer (Issue #969).

Verifies:
  1. warm() pre-populates the cache for all supplied hot-set entries.
  2. warm() respects the hot_set_size limit.
  3. warm() respects the timeout — stops early without raising.
  4. is_warm becomes True after a warm() call (even partial).
  5. entries_warmed and last_warm_duration_seconds are set correctly.
  6. warm_from_store() calls build_features_fn for each wallet in the hot set.
  7. warm_from_store() handles a store unavailability gracefully.
  8. describe() returns expected keys.
  9. The cache actually serves warmed entries on subsequent get() calls.
  10. Latency-spike simulation: warmed cache has 100% hit rate on the hot set.
"""

from __future__ import annotations

import os
import threading
import time

import pandas as pd
import pytest

os.environ.setdefault("MODEL_DIR", "./models")
os.environ.setdefault("RISK_SCORE_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("WATCHED_ASSET_PAIRS", "USDC:native")

from detection.feature_cache import CacheWarmer, FeatureCache


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_cache(ttl: int = 300, maxsize: int = 200) -> FeatureCache:
    return FeatureCache(ttl_seconds=ttl, maxsize=maxsize)


def _series(value: float = 1.0) -> pd.Series:
    return pd.Series({"score": value, "benford_mad_1h": 0.01})


def _hot_set(n: int = 10) -> list[tuple[str, pd.Series]]:
    # Build unique 56-char wallet IDs by embedding the index at the start
    return [(f"G{i:055d}", _series(float(i))) for i in range(n)]


# ---------------------------------------------------------------------------
# warm() tests
# ---------------------------------------------------------------------------


class TestCacheWarmerWarm:
    def test_warm_inserts_all_entries(self):
        cache = _make_cache()
        warmer = CacheWarmer(cache, hot_set_size=20)
        hot = _hot_set(10)

        n = warmer.warm(hot)

        assert n == 10
        assert warmer.entries_warmed == 10

    def test_warmed_entries_are_retrievable(self):
        cache = _make_cache()
        warmer = CacheWarmer(cache, hot_set_size=20)
        hot = _hot_set(5)

        warmer.warm(hot)

        for wallet_id, expected_series in hot:
            result = cache.get(wallet_id)
            assert result is not None, f"Expected {wallet_id} to be cached after warming"
            pd.testing.assert_series_equal(result, expected_series)

    def test_warm_respects_hot_set_size_limit(self):
        cache = _make_cache()
        warmer = CacheWarmer(cache, hot_set_size=5)
        hot = _hot_set(20)

        n = warmer.warm(hot)

        assert n == 5
        assert len(cache) == 5

    def test_is_warm_is_true_after_warm(self):
        cache = _make_cache()
        warmer = CacheWarmer(cache)
        assert not warmer.is_warm

        warmer.warm(_hot_set(3))

        assert warmer.is_warm

    def test_last_warm_duration_is_set(self):
        cache = _make_cache()
        warmer = CacheWarmer(cache)
        warmer.warm(_hot_set(5))

        assert warmer.last_warm_duration_seconds is not None
        assert warmer.last_warm_duration_seconds >= 0.0

    def test_warm_empty_hot_set_succeeds(self):
        cache = _make_cache()
        warmer = CacheWarmer(cache)
        n = warmer.warm([])

        assert n == 0
        assert warmer.is_warm
        assert len(cache) == 0

    def test_warm_respects_timeout(self, monkeypatch):
        """warm() stops early when the timeout is exhausted."""
        call_count = 0

        cache = _make_cache()
        warmer = CacheWarmer(cache, hot_set_size=100, timeout_seconds=0.05)

        # Simulate a slow feature build by monkey-patching time.monotonic
        # to advance past the timeout on the 3rd entry.
        original_put = cache.put
        call_counter = [0]
        start_time = [time.monotonic()]

        def slow_put(wallet, features, **kw):
            call_counter[0] += 1
            original_put(wallet, features, **kw)
            if call_counter[0] >= 3:
                # Advance the internal clock beyond timeout
                import detection.feature_cache as fcm

                monkeypatch.setattr(fcm.time, "monotonic", lambda: start_time[0] + 1.0)

        monkeypatch.setattr(cache, "put", slow_put)

        n = warmer.warm(_hot_set(50))
        # Should have stopped after a few entries, not all 50
        assert n < 50
        assert warmer.is_warm  # still marked warm even if partial

    def test_warm_is_idempotent(self):
        """Calling warm() twice updates entries_warmed."""
        cache = _make_cache()
        warmer = CacheWarmer(cache)
        warmer.warm(_hot_set(5))
        first_count = warmer.entries_warmed

        warmer.warm(_hot_set(3))
        second_count = warmer.entries_warmed

        assert first_count == 5
        assert second_count == 3  # reflects last run


# ---------------------------------------------------------------------------
# warm_from_store() tests
# ---------------------------------------------------------------------------


class _FakeRiskStore:
    """Minimal fake risk-score store for warm_from_store() tests."""

    def __init__(self, wallets: list[str], raise_on_call: bool = False):
        self._wallets = wallets
        self._raise = raise_on_call

    def get_recent_wallets(self, limit: int, lookback_hours: int) -> list[str]:
        if self._raise:
            raise RuntimeError("Store unavailable")
        return self._wallets[:limit]


class TestCacheWarmerWarmFromStore:
    def test_warm_from_store_calls_build_fn_for_each_wallet(self):
        """build_features_fn is called for every wallet in the hot set."""
        wallets = [f"G{i:055d}" for i in range(5)]
        store = _FakeRiskStore(wallets)
        cache = _make_cache()
        warmer = CacheWarmer(cache, hot_set_size=10)

        called: list[str] = []

        def build_fn(wallet_id, wallet_trades, **kw):
            called.append(wallet_id)
            return _series()

        trades = pd.DataFrame({"wallet_id": wallets, "amount": [100.0] * 5})
        n = warmer.warm_from_store(store, build_fn, trades)

        assert n == 5
        assert set(called) == set(wallets)

    def test_warm_from_store_graceful_when_store_unavailable(self):
        """If the risk store raises, warm_from_store exits 0 without crashing."""
        store = _FakeRiskStore([], raise_on_call=True)
        cache = _make_cache()
        warmer = CacheWarmer(cache)

        def build_fn(wallet_id, wallet_trades, **kw):
            return _series()

        n = warmer.warm_from_store(store, build_fn, pd.DataFrame())
        assert n == 0
        assert warmer.is_warm  # still marked warm

    def test_warm_from_store_populates_cache(self):
        wallets = [f"G{i:055d}" for i in range(3)]
        store = _FakeRiskStore(wallets)
        cache = _make_cache()
        warmer = CacheWarmer(cache, hot_set_size=10)

        feature_values = {w: _series(float(i)) for i, w in enumerate(wallets)}

        def build_fn(wallet_id, wallet_trades, **kw):
            return feature_values[wallet_id]

        trades = pd.DataFrame({"wallet_id": wallets, "amount": [1.0] * 3})
        warmer.warm_from_store(store, build_fn, trades)

        for w in wallets:
            result = cache.get(w)
            assert result is not None, f"Expected {w} to be in cache after warm_from_store"


# ---------------------------------------------------------------------------
# describe() tests
# ---------------------------------------------------------------------------


class TestCacheWarmerDescribe:
    def test_describe_returns_expected_keys(self):
        warmer = CacheWarmer(_make_cache())
        info = warmer.describe()
        for key in ("is_warm", "entries_warmed", "last_warm_duration_seconds",
                    "hot_set_size", "timeout_seconds", "cache_size"):
            assert key in info, f"Missing key {key!r} in describe() output"

    def test_describe_reflects_warm_state(self):
        cache = _make_cache()
        warmer = CacheWarmer(cache, hot_set_size=5)
        warmer.warm(_hot_set(3))

        info = warmer.describe()
        assert info["is_warm"] is True
        assert info["entries_warmed"] == 3
        assert info["cache_size"] == 3
        assert info["last_warm_duration_seconds"] is not None


# ---------------------------------------------------------------------------
# Latency-spike simulation test
# ---------------------------------------------------------------------------


class TestCacheWarmingLatencyReduction:
    """Demonstrate that a warmed cache achieves 100% hit rate on the hot set,
    eliminating the cold-start latency spike for those wallets.

    This test is a documented measurement of the before/after behaviour rather
    than a performance benchmark (no wall-clock assertion). See
    docs/cold_start.md for the quantified latency improvement.
    """

    def test_warmed_cache_hit_rate_on_hot_set(self):
        cache = _make_cache(maxsize=500)
        warmer = CacheWarmer(cache, hot_set_size=50)
        hot = _hot_set(50)

        # Before warming: 0% hit rate
        hits_before = sum(1 for w, _ in hot if cache.get(w) is not None)
        assert hits_before == 0, "Cache should be empty before warming"

        # Warm the cache
        warmer.warm(hot)

        # After warming: 100% hit rate
        hits_after = sum(1 for w, _ in hot if cache.get(w) is not None)
        assert hits_after == 50, (
            f"Expected 100% hit rate after warming but got {hits_after}/50 hits"
        )

    def test_cold_start_warm_up_completes_within_timeout(self):
        """Warm-up must complete within the documented acceptable bound."""
        cache = _make_cache(maxsize=200)
        timeout = 5.0  # 5 second acceptable deployment bound
        warmer = CacheWarmer(cache, hot_set_size=100, timeout_seconds=timeout)
        hot = _hot_set(100)

        start = time.monotonic()
        warmer.warm(hot)
        elapsed = time.monotonic() - start

        assert elapsed < timeout, (
            f"Cache warm-up took {elapsed:.2f}s which exceeds the {timeout}s "
            "acceptable deployment bound."
        )
        assert warmer.last_warm_duration_seconds is not None
        assert warmer.last_warm_duration_seconds < timeout
