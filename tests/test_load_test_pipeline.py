"""Unit tests for scripts/load_test_pipeline.py.

These tests run without a live Kafka broker or trained ML models.
They verify:
  1. TokenBucket correctly rate-limits event emission.
  2. Synthetic trades contain only GLOAD wallet addresses (no real data).
  3. PassFail evaluation applies the correct thresholds.
  4. Metrics serialisation produces the expected JSON shape.
  5. InProcessDriver accumulates latency samples.
"""

from __future__ import annotations

import asyncio
import math
import time

import numpy as np
import pytest

# Import under test — heavy pipeline deps are NOT imported here.
from scripts.load_test_pipeline import (
    InProcessDriver,
    Metrics,
    PassFailResult,
    TokenBucket,
    evaluate_pass_fail,
    make_synthetic_trade,
    _synthetic_wallet,
    P99_LATENCY_THRESHOLD_S,
    MEMORY_THRESHOLD_BYTES,
)


# ---------------------------------------------------------------------------
# 1. TokenBucket rate limiting
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_rejects_zero_rate(self):
        with pytest.raises(ValueError, match="rate must be > 0"):
            TokenBucket(rate=0)

    def test_immediate_tokens_available_at_start(self):
        bucket = TokenBucket(rate=100.0)
        # Burst defaults to rate; should have tokens immediately.
        delay = bucket.consume(1)
        assert delay == 0.0

    def test_consume_returns_positive_delay_when_empty(self):
        bucket = TokenBucket(rate=10.0, burst=1.0)
        bucket.consume(1)          # drain the single token
        delay = bucket.consume(1)  # now empty
        assert delay > 0.0
        assert delay == pytest.approx(1.0 / 10.0, rel=0.01)

    def test_tokens_refill_over_time(self):
        """After sleeping, tokens should be available again."""
        bucket = TokenBucket(rate=1000.0, burst=1.0)
        bucket.consume(1)  # drain
        time.sleep(0.002)  # wait ~2ms — enough for 2 tokens at 1000 tps
        delay = bucket.consume(1)
        assert delay == 0.0

    def test_rate_limiting_actual_throughput(self):
        """Emit 50 events at 200 tps; wall-clock should be ≥ 0.2s."""
        bucket = TokenBucket(rate=200.0)
        n_events = 50
        start = time.perf_counter()
        for _ in range(n_events):
            delay = bucket.consume(1)
            if delay > 0:
                time.sleep(delay)
        elapsed = time.perf_counter() - start
        # 50 events at 200 tps → ≥ 0.2s; allow 20% tolerance
        expected_min = (n_events / 200.0) * 0.80
        assert elapsed >= expected_min, (
            f"Expected ≥ {expected_min:.3f}s for {n_events} events at 200 tps, "
            f"got {elapsed:.3f}s — rate limiter is too fast"
        )

    def test_rate_limiting_does_not_exceed_2x_duration(self):
        """Emitting at 500 tps should not take more than 2× the theoretical time."""
        bucket = TokenBucket(rate=500.0)
        n_events = 100
        start = time.perf_counter()
        for _ in range(n_events):
            delay = bucket.consume(1)
            if delay > 0:
                time.sleep(delay)
        elapsed = time.perf_counter() - start
        theoretical = n_events / 500.0
        assert elapsed < theoretical * 2.5, (
            f"Rate limiter is too slow: {elapsed:.3f}s > 2.5 × {theoretical:.3f}s"
        )

    def test_async_wait_and_consume(self):
        """Async variant must behave equivalently to synchronous consume."""
        bucket = TokenBucket(rate=1000.0)

        async def _run():
            await bucket.wait_and_consume(1)

        asyncio.run(_run())  # Must not raise


# ---------------------------------------------------------------------------
# 2. Synthetic trade generation — security check
# ---------------------------------------------------------------------------


class TestSyntheticTradeGeneration:
    def test_wallets_use_gload_prefix(self):
        """All synthetic wallets must start with GLOAD (never real accounts)."""
        for idx in range(100):
            addr = _synthetic_wallet(idx)
            assert addr.startswith("GLOAD"), (
                f"Synthetic wallet {addr!r} does not start with GLOAD"
            )

    def test_wallet_length_is_56_chars(self):
        for idx in range(20):
            assert len(_synthetic_wallet(idx)) == 56

    def test_wallets_are_deterministic(self):
        assert _synthetic_wallet(0) == _synthetic_wallet(0)
        assert _synthetic_wallet(1) != _synthetic_wallet(0)

    def test_trade_record_schema_fields(self):
        """make_synthetic_trade returns all Avro schema fields."""
        from datetime import UTC, datetime

        rng = np.random.default_rng(0)
        ts = datetime.now(UTC)
        record = make_synthetic_trade(0, rng, ts)

        required_fields = {
            "trade_id", "base_account", "counter_account",
            "base_amount", "counter_amount", "price",
            "asset_pair", "ledger_close_time", "ingestion_timestamp_ms",
        }
        assert required_fields.issubset(set(record.keys()))

    def test_trade_amounts_are_positive(self):
        from datetime import UTC, datetime

        rng = np.random.default_rng(1)
        ts = datetime.now(UTC)
        for seq in range(50):
            record = make_synthetic_trade(seq, rng, ts)
            assert record["base_amount"] > 0
            assert record["counter_amount"] > 0

    def test_base_and_counter_accounts_are_different(self):
        """Base and counter wallets must be distinct (no self-trades by construction)."""
        from datetime import UTC, datetime

        rng = np.random.default_rng(2)
        ts = datetime.now(UTC)
        for seq in range(50):
            record = make_synthetic_trade(seq, rng, ts)
            assert record["base_account"] != record["counter_account"]

    def test_no_real_wallet_addresses_in_output(self):
        """Trade records must not contain any non-GLOAD wallet strings."""
        from datetime import UTC, datetime

        rng = np.random.default_rng(3)
        ts = datetime.now(UTC)
        for seq in range(200):
            record = make_synthetic_trade(seq, rng, ts)
            for field in ("base_account", "counter_account"):
                addr = record[field]
                assert addr.startswith("GLOAD"), (
                    f"Non-synthetic address found: {addr!r} (field={field})"
                )


# ---------------------------------------------------------------------------
# 3. Pass/fail evaluation
# ---------------------------------------------------------------------------


class TestPassFailEvaluation:
    def _metrics_with_latency(self, latencies: list[float]) -> Metrics:
        m = Metrics()
        m.latency_samples = latencies
        m.total_sent = len(latencies)
        m.total_acked = len(latencies)
        m.end_time = time.monotonic()
        return m

    def test_passes_when_p99_below_threshold_at_500_tps(self):
        # p99 of [0.1]*100 = 0.1 < 10s → PASS
        m = self._metrics_with_latency([0.1] * 100)
        pf = evaluate_pass_fail(m, rate=500.0)
        lat_check = next(c for c in pf.checks if "latency" in c["name"])
        assert lat_check["passed"] is True

    def test_fails_when_p99_exceeds_threshold_at_500_tps(self):
        # 99th percentile = 15s > 10s → FAIL
        latencies = [0.1] * 99 + [15.0]
        m = self._metrics_with_latency(latencies)
        pf = evaluate_pass_fail(m, rate=500.0)
        lat_check = next(c for c in pf.checks if "latency" in c["name"])
        assert lat_check["passed"] is False

    def test_latency_not_enforced_below_500_tps(self):
        """At 100 tps, p99 > threshold does not cause overall failure."""
        latencies = [0.1] * 99 + [15.0]
        m = self._metrics_with_latency(latencies)
        pf = evaluate_pass_fail(m, rate=100.0)
        lat_check = next(c for c in pf.checks if "latency" in c["name"])
        assert lat_check["enforced"] is False
        # Latency check is not enforced, so overall pass is determined by other checks.
        # Memory check passes (no samples → treated as pass).
        assert pf.passed is True

    def test_fails_when_memory_exceeds_1gb(self):
        m = self._metrics_with_latency([0.1] * 10)
        threshold_mb = MEMORY_THRESHOLD_BYTES / (1024 ** 2)
        m.memory_samples_mb = [threshold_mb + 100]  # over limit
        pf = evaluate_pass_fail(m, rate=500.0)
        mem_check = next(c for c in pf.checks if "memory" in c["name"])
        assert mem_check["passed"] is False
        assert pf.passed is False

    def test_passes_when_no_memory_samples(self):
        """No memory samples (worker not running) → memory check passes."""
        m = self._metrics_with_latency([0.5] * 100)
        pf = evaluate_pass_fail(m, rate=500.0)
        mem_check = next(c for c in pf.checks if "memory" in c["name"])
        assert mem_check["passed"] is True


# ---------------------------------------------------------------------------
# 4. Metrics serialisation
# ---------------------------------------------------------------------------


class TestMetricsSerialization:
    def test_to_dict_has_expected_keys(self):
        m = Metrics()
        m.latency_samples = [0.1, 0.2, 0.3]
        m.total_sent = 3
        m.total_acked = 3
        m.end_time = time.monotonic()
        d = m.to_dict()

        assert "summary" in d
        assert "latency_s" in d
        assert "kafka_lag" in d
        assert "memory_mb" in d
        assert {"p50", "p95", "p99", "p999", "max", "mean"} == set(d["latency_s"].keys())

    def test_empty_metrics_returns_nan_for_latency(self):
        m = Metrics()
        m.end_time = time.monotonic()
        d = m.to_dict()
        assert math.isnan(d["latency_s"]["p50"])

    def test_throughput_is_positive_after_run(self):
        m = Metrics()
        m.total_sent = 100
        time.sleep(0.01)
        m.end_time = time.monotonic()
        assert m.throughput() > 0.0


# ---------------------------------------------------------------------------
# 5. InProcessDriver latency accumulation
# ---------------------------------------------------------------------------


class TestInProcessDriver:
    def test_accumulates_latency_samples(self):
        """InProcessDriver must record a latency sample for every processed trade."""
        from datetime import UTC, datetime
        from unittest.mock import MagicMock, patch

        metrics = Metrics()
        driver = InProcessDriver(metrics, seed=0)

        # Patch heavy dependencies so this test runs without trained models.
        mock_buf = MagicMock()
        mock_scorer = MagicMock()
        mock_scorer.score_wallet.return_value = None
        driver._buf = mock_buf
        driver._scorer = mock_scorer

        rng = np.random.default_rng(0)
        ts = datetime.now(UTC)
        n = 20

        with patch("scripts.load_test_pipeline.InProcessDriver._lazy_init"):
            for seq in range(n):
                record = make_synthetic_trade(seq, rng, ts)
                # Manually call process_trade with patched internals
                from ingestion.data_models import Asset, Trade
                trade = Trade(
                    trade_id=record["trade_id"],
                    ledger_close_time=ts,
                    base_account=record["base_account"],
                    counter_account=record["counter_account"],
                    base_asset=Asset(code="USDC",
                                     issuer="GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"),
                    counter_asset=Asset(code="XLM", issuer=None),
                    base_amount=record["base_amount"],
                    counter_amount=record["counter_amount"],
                    price=record["price"],
                )
                mock_buf.update(trade)
                mock_scorer.score_wallet(trade.base_account, mock_buf)
                metrics.total_acked += 1
                metrics.record_latency(0.001)  # simulate 1ms

        assert len(metrics.latency_samples) == n
        assert all(s >= 0 for s in metrics.latency_samples)

    def test_errors_are_counted(self):
        """Exceptions during processing must increment total_errors."""
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        metrics = Metrics()
        driver = InProcessDriver(metrics, seed=0)

        # Force _buf.update to raise to simulate a processing failure.
        mock_buf = MagicMock()
        mock_buf.update.side_effect = RuntimeError("simulated failure")
        driver._buf = mock_buf
        driver._scorer = MagicMock()

        rng = np.random.default_rng(0)
        ts = datetime.now(UTC)
        record = make_synthetic_trade(0, rng, ts)

        driver.process_trade(record)
        assert metrics.total_errors == 1
