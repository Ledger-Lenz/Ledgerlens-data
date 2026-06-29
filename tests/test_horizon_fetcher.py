"""Tests for ingestion.horizon_fetcher.fetch (Issue #284)."""

import pytest

from ingestion.horizon_fetcher import HorizonRateLimitExceeded, fetch


class _HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


class _NoopLimiter:
    def acquire(self, timeout=None):
        return True


def _no_jitter(low, high):
    return 0.0


def test_429_triggers_exponential_backoff_then_succeeds(monkeypatch):
    import config as config_module

    monkeypatch.setattr(config_module.config, "HORIZON_MAX_RETRIES", 4)

    attempts = {"n": 0}
    sleeps = []

    def call():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _HttpError(429)
        return "ok"

    result = fetch(
        call,
        limiter=_NoopLimiter(),
        sleep_fn=lambda d: sleeps.append(d),
        jitter_fn=_no_jitter,
    )

    assert result == "ok"
    assert attempts["n"] == 3
    assert sleeps == [1.0, 2.0]  # base 1s, doubling, no jitter


def test_429_exhausts_retries_and_raises(monkeypatch):
    import config as config_module

    monkeypatch.setattr(config_module.config, "HORIZON_MAX_RETRIES", 3)

    def call():
        raise _HttpError(429)

    with pytest.raises(HorizonRateLimitExceeded):
        fetch(call, limiter=_NoopLimiter(), sleep_fn=lambda d: None, jitter_fn=_no_jitter)


def test_non_429_4xx_is_not_retried(monkeypatch):
    import config as config_module

    monkeypatch.setattr(config_module.config, "HORIZON_MAX_RETRIES", 5)

    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        raise _HttpError(403)

    with pytest.raises(_HttpError):
        fetch(call, limiter=_NoopLimiter(), sleep_fn=lambda d: None, jitter_fn=_no_jitter)

    assert attempts["n"] == 1
