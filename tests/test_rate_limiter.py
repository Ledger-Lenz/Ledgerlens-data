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


class _FakeClock:
    """Controllable stand-in for the stdlib `time` module: `sleep` advances
    the clock instead of actually waiting, so polling tests run instantly."""

    def __init__(self):
        self._now = 0.0
        self.sleep_calls = []

    def time(self):
        return self._now

    def monotonic(self):
        return self._now

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)
        self._now += seconds


def test_burst_within_budget_grants_all_without_waiting(monkeypatch):
    import ingestion.rate_limiter as rl_module

    clock = _FakeClock()
    monkeypatch.setattr(rl_module, "time", clock)
    limiter = _limiter(capacity=5, refill_rate=5.0)

    results = [limiter.try_acquire() for _ in range(5)]

    assert all(results)
    assert clock.sleep_calls == []


def test_burst_over_budget_spaces_calls_by_refill_interval(monkeypatch):
    import ingestion.rate_limiter as rl_module

    clock = _FakeClock()
    monkeypatch.setattr(rl_module, "time", clock)
    capacity = 3
    refill_rate = 3.0  # 1 token every ~0.333s
    poll_interval = 0.05
    limiter = TokenBucketLimiter(
        client=fakeredis.FakeRedis(),
        capacity=capacity,
        refill_rate_per_sec=refill_rate,
        poll_interval_seconds=poll_interval,
    )

    grant_times = []
    for _ in range(6):
        assert limiter.acquire(timeout=5) is True
        grant_times.append(clock.time())

    # The first `capacity` calls are granted immediately from the full bucket.
    assert grant_times[:capacity] == [0.0] * capacity

    # Calls beyond capacity must wait for tokens to refill, so each is spaced
    # roughly 1/refill_rate apart (polling grain is `poll_interval`).
    expected_gap = 1.0 / refill_rate
    for prev, nxt in zip(grant_times[capacity - 1 :], grant_times[capacity:], strict=True):
        assert nxt - prev == pytest.approx(expected_gap, abs=poll_interval)

    assert clock.sleep_calls  # confirms the limiter actually polled/waited
