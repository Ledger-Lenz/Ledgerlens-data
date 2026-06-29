"""Tests for ingestion.rate_limiter.TokenBucketLimiter (Issue #284)."""

import threading

import fakeredis
import pytest

from ingestion.rate_limiter import TokenBucketLimiter


def _limiter(capacity=5, refill_rate=0.0):
    client = fakeredis.FakeRedis()
    return TokenBucketLimiter(client=client, capacity=capacity, refill_rate_per_sec=refill_rate)


def test_concurrent_acquisitions_grant_exactly_capacity():
    limiter = _limiter(capacity=5, refill_rate=0.0)
    results = [None] * 10

    def worker(i):
        results[i] = limiter.try_acquire()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 5
    assert results.count(False) == 5


def test_redis_unavailable_does_not_raise_and_grants(caplog):
    class _BrokenClient:
        def ping(self):
            raise ConnectionError("no redis")

        def pipeline(self):
            raise ConnectionError("no redis")

    limiter = TokenBucketLimiter(client=_BrokenClient(), capacity=5, refill_rate_per_sec=5)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True  # still granted, never raises


def test_no_redis_package_degrades_gracefully(monkeypatch):
    import ingestion.rate_limiter as rl_module

    monkeypatch.setattr(rl_module, "redis", None)
    limiter = TokenBucketLimiter(capacity=5, refill_rate_per_sec=5)
    assert limiter.try_acquire() is True
