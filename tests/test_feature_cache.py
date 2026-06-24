"""Tests for detection.feature_cache.FeatureCache (Issue #95)."""

import threading

import pandas as pd
import pytest

from detection import feature_cache as feature_cache_module
from detection.feature_cache import FeatureCache

WALLET_A = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
WALLET_B = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBWHF"


def _series(score: float = 1.0) -> pd.Series:
    return pd.Series({"score": score})


class _FakeClock:
    """Deterministic stand-in for time.monotonic()."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


def test_miss_when_wallet_never_cached():
    cache = FeatureCache(ttl_seconds=300, maxsize=10)
    assert cache.get(WALLET_A) is None


def test_hit_returns_cached_series_without_recompute():
    cache = FeatureCache(ttl_seconds=300, maxsize=10)
    features = _series(42.0)
    cache.put(WALLET_A, features)

    result = cache.get(WALLET_A)
    assert result is not None
    pd.testing.assert_series_equal(result, features)


def test_put_overwrites_existing_entry():
    cache = FeatureCache(ttl_seconds=300, maxsize=10)
    cache.put(WALLET_A, _series(1.0))
    cache.put(WALLET_A, _series(2.0))

    assert len(cache) == 1
    pd.testing.assert_series_equal(cache.get(WALLET_A), _series(2.0))


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


def test_entry_expires_after_ttl(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(feature_cache_module.time, "monotonic", clock)

    cache = FeatureCache(ttl_seconds=300, maxsize=10)
    cache.put(WALLET_A, _series())

    clock.advance(299)
    assert cache.get(WALLET_A) is not None

    clock.advance(2)  # total elapsed: 301s, past the 300s TTL
    assert cache.get(WALLET_A) is None


def test_expired_entry_is_evicted_on_access(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(feature_cache_module.time, "monotonic", clock)

    cache = FeatureCache(ttl_seconds=10, maxsize=10)
    cache.put(WALLET_A, _series())
    clock.advance(11)
    assert cache.get(WALLET_A) is None
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# maxsize / LRU eviction
# ---------------------------------------------------------------------------


def test_cache_never_exceeds_maxsize():
    cache = FeatureCache(ttl_seconds=300, maxsize=2)
    cache.put("wallet_1", _series())
    cache.put("wallet_2", _series())
    cache.put("wallet_3", _series())

    assert len(cache) == 2


def test_oldest_entry_evicted_when_full():
    cache = FeatureCache(ttl_seconds=300, maxsize=2)
    cache.put("wallet_1", _series())
    cache.put("wallet_2", _series())
    cache.put("wallet_3", _series())  # evicts wallet_1 (least-recently-used)

    assert cache.get("wallet_1") is None
    assert cache.get("wallet_2") is not None
    assert cache.get("wallet_3") is not None


def test_get_refreshes_lru_order():
    cache = FeatureCache(ttl_seconds=300, maxsize=2)
    cache.put("wallet_1", _series())
    cache.put("wallet_2", _series())
    cache.get("wallet_1")  # wallet_1 becomes most-recently-used
    cache.put("wallet_3", _series())  # should evict wallet_2, not wallet_1

    assert cache.get("wallet_1") is not None
    assert cache.get("wallet_2") is None
    assert cache.get("wallet_3") is not None


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_concurrent_put_and_get_does_not_raise_or_corrupt():
    cache = FeatureCache(ttl_seconds=300, maxsize=50)
    wallets = [f"wallet_{i}" for i in range(20)]
    errors: list[Exception] = []

    def worker(wallet: str) -> None:
        try:
            for i in range(50):
                cache.put(wallet, _series(float(i)))
                cache.get(wallet)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in wallets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(cache) <= 50


# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------


def test_hit_and_miss_increment_prometheus_counters():
    pytest.importorskip("prometheus_client")
    assert feature_cache_module.feature_cache_hits_total is not None
    assert feature_cache_module.feature_cache_misses_total is not None

    hits_before = feature_cache_module.feature_cache_hits_total._value.get()
    misses_before = feature_cache_module.feature_cache_misses_total._value.get()

    cache = FeatureCache(ttl_seconds=300, maxsize=10)
    cache.get(WALLET_B)  # miss
    cache.put(WALLET_B, _series())
    cache.get(WALLET_B)  # hit

    assert feature_cache_module.feature_cache_misses_total._value.get() == misses_before + 1
    assert feature_cache_module.feature_cache_hits_total._value.get() == hits_before + 1
