"""Tests for bot detection fingerprinting from Horizon event patterns."""

import pytest
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from detection.bot_fingerprinter import (
    extract_bot_fingerprint,
    _compute_trust_line_latency,
    _compute_inter_trade_interval_cv,
    _compute_account_management_entropy,
    _is_plausible_timestamp,
    is_likely_bot,
)
from ingestion.data_models import BotFingerprint


def sample_bot_trades() -> pd.DataFrame:
    """Create synthetic bot trades with perfectly regular 5-second intervals."""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    bot_account = "GBOT" + "B" * 52
    counterparty = "GCTR" + "C" * 52

    trades = []
    for i in range(10):
        timestamp = base_time + timedelta(seconds=5 * i)
        trades.append({
            "trade_id": f"bot-trade-{i}",
            "ledger_close_time": timestamp.isoformat(),
            "base_account": bot_account if i % 2 == 0 else counterparty,
            "counter_account": counterparty if i % 2 == 0 else bot_account,
            "base_asset": "USDC",
            "counter_asset": "XLM",
            "amount": 100.0,
            "price": 0.1,
        })

    return pd.DataFrame(trades)


def sample_human_trades() -> pd.DataFrame:
    """Create synthetic human trades with irregular intervals."""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    human_account = "GHUM" + "H" * 51
    counterparty = "GCTR" + "C" * 52

    trades = []
    # Irregular intervals: 5s, 45s, 120s, 3s, 600s, etc.
    intervals = [5, 45, 120, 3, 600, 30, 90, 10, 200, 5]
    current_time = base_time

    for i, interval in enumerate(intervals):
        current_time += timedelta(seconds=interval)
        trades.append({
            "trade_id": f"human-trade-{i}",
            "ledger_close_time": current_time.isoformat(),
            "base_account": human_account if i % 2 == 0 else counterparty,
            "counter_account": counterparty if i % 2 == 0 else human_account,
            "base_asset": "USDC",
            "counter_asset": "XLM",
            "amount": 100.0 + np.random.randn() * 30,
            "price": 0.1,
        })

    return pd.DataFrame(trades)


def sample_effects_fast_trust_line() -> list[dict]:
    """Create Horizon effects with fast trust line creation (5 seconds)."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "type": "account_created",
            "created_at": base_time.isoformat(),
            "account": "GBOT" + "B" * 52,
        },
        {
            "type": "trust_line_created",
            "created_at": (base_time + timedelta(seconds=5)).isoformat(),
            "account": "GBOT" + "B" * 52,
        },
        {
            "type": "trust_line_created",
            "created_at": (base_time + timedelta(seconds=10)).isoformat(),
            "account": "GBOT" + "B" * 52,
        },
    ]


def sample_effects_slow_trust_line() -> list[dict]:
    """Create Horizon effects with slow trust line creation (5 hours)."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "type": "account_created",
            "created_at": base_time.isoformat(),
            "account": "GHUM" + "H" * 51,
        },
        {
            "type": "trust_line_created",
            "created_at": (base_time + timedelta(hours=5)).isoformat(),
            "account": "GHUM" + "H" * 51,
        },
    ]


class TestTrustLineLatency:
    """Tests for trust line creation latency computation."""

    def test_fast_trust_line_bot(self):
        """Bot account creates trust line within 5 seconds."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        effects = sample_effects_fast_trust_line()

        latency = _compute_trust_line_latency(base_time, effects)

        assert latency is not None
        assert latency == 5.0, f"Expected 5.0s, got {latency}s"

    def test_slow_trust_line_human(self):
        """Human account creates trust line after 5 hours."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        effects = sample_effects_slow_trust_line()

        latency = _compute_trust_line_latency(base_time, effects)

        assert latency is not None
        assert latency == 5 * 3600, f"Expected {5*3600}s, got {latency}s"

    def test_no_trust_line(self):
        """Account with no trust line returns None."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        effects = [
            {
                "type": "account_created",
                "created_at": base_time.isoformat(),
            }
        ]

        latency = _compute_trust_line_latency(base_time, effects)
        assert latency is None

    def test_rejects_future_timestamps(self):
        """Future timestamps are rejected as invalid."""
        base_time = datetime.now(timezone.utc) + timedelta(hours=1)
        effects = [
            {
                "type": "trust_line_created",
                "created_at": base_time.isoformat(),
            }
        ]

        latency = _compute_trust_line_latency(base_time, effects)
        assert latency is None

    def test_rejects_pre_genesis_timestamps(self):
        """Pre-Stellar-genesis timestamps are rejected."""
        base_time = datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # Before genesis

        latency = _compute_trust_line_latency(base_time, [])
        assert latency is None


class TestInterTradeIntervalCV:
    """Tests for inter-trade interval coefficient of variation."""

    def test_bot_account_cv_near_zero(self):
        """Bot with perfectly regular 5-second intervals has CV < 0.05."""
        trades_df = sample_bot_trades()
        bot_account = "GBOT" + "B" * 52

        cv = _compute_inter_trade_interval_cv(bot_account, trades_df)

        assert cv is not None
        assert cv < 0.05, f"Bot CV too high: {cv} (expected < 0.05)"

    def test_human_account_cv_high(self):
        """Human with irregular intervals has CV > 0.3."""
        trades_df = sample_human_trades()
        human_account = "GHUM" + "H" * 51

        cv = _compute_inter_trade_interval_cv(human_account, trades_df)

        assert cv is not None
        assert cv > 0.3, f"Human CV too low: {cv} (expected > 0.3)"

    def test_insufficient_trades_returns_none(self):
        """Accounts with fewer than 5 trades return None."""
        trades_df = pd.DataFrame([
            {
                "trade_id": "1",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "GACC" + "A" * 52,
                "counter_account": "GXXX" + "X" * 52,
                "amount": 100.0,
            },
            {
                "trade_id": "2",
                "ledger_close_time": "2024-01-01T00:05:00Z",
                "base_account": "GXXX" + "X" * 52,
                "counter_account": "GACC" + "A" * 52,
                "amount": 100.0,
            },
        ])

        cv = _compute_inter_trade_interval_cv("GACC" + "A" * 52, trades_df)
        assert cv is None

    def test_zero_intervals_returns_none(self):
        """All zero-interval trades return None."""
        trades_df = pd.DataFrame([
            {
                "trade_id": f"{i}",
                "ledger_close_time": "2024-01-01T00:00:00Z",  # All same time
                "base_account": "GACC" + "A" * 52,
                "counter_account": "GXXX" + "X" * 52,
                "amount": 100.0,
            }
            for i in range(6)
        ])

        cv = _compute_inter_trade_interval_cv("GACC" + "A" * 52, trades_df)
        assert cv is None

    def test_uses_population_std(self):
        """CV uses population standard deviation (ddof=0)."""
        trades_df = pd.DataFrame([
            {
                "trade_id": f"{i}",
                "ledger_close_time": (datetime(2024, 1, 1, 0, 0, 0) + timedelta(seconds=10*i)).isoformat(),
                "base_account": "GACC" + "A" * 52,
                "counter_account": "GXXX" + "X" * 52,
                "amount": 100.0,
            }
            for i in range(5)
        ])

        cv = _compute_inter_trade_interval_cv("GACC" + "A" * 52, trades_df)

        # All intervals are exactly 10 seconds
        # Population std = 0, sample std > 0
        # We should get 0 (population), not NaN or inf (sample)
        assert cv == 0.0


class TestAccountManagementEntropy:
    """Tests for operation type entropy."""

    def test_clustered_operations_low_entropy(self):
        """Few operation types = low entropy (bot-like)."""
        effects = [
            {"type": "manage_offer"} for _ in range(8)
        ] + [
            {"type": "trust_line_created"} for _ in range(2)
        ]

        entropy = _compute_account_management_entropy(effects)

        assert entropy is not None
        # 80% manage_offer, 20% trust_line_created
        # H = -0.8*log2(0.8) - 0.2*log2(0.2) ≈ 0.72 bits (low)
        assert entropy < 1.5

    def test_diverse_operations_high_entropy(self):
        """Many operation types = high entropy (human-like)."""
        effects = [
            {"type": "manage_offer"},
            {"type": "trust_line_created"},
            {"type": "account_created"},
            {"type": "trade"},
            {"type": "path_payment_strict_send"},
            {"type": "manage_data"},
            {"type": "set_options"},
            {"type": "payment"},
        ]

        entropy = _compute_account_management_entropy(effects)

        # 8 equally distributed types: H = log2(8) = 3 bits (high)
        assert entropy == 3.0

    def test_empty_effects_zero_entropy(self):
        """Empty effects list returns 0 entropy."""
        entropy = _compute_account_management_entropy([])
        assert entropy == 0.0


class TestPlausibleTimestamp:
    """Tests for timestamp validation."""

    def test_valid_current_timestamp(self):
        """Current timestamp is plausible."""
        ts = datetime.now(timezone.utc)
        assert _is_plausible_timestamp(ts) is True

    def test_valid_past_timestamp(self):
        """Past timestamp (after genesis) is plausible."""
        ts = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert _is_plausible_timestamp(ts) is True

    def test_rejects_pre_genesis(self):
        """Pre-Stellar-genesis timestamp is rejected."""
        ts = datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert _is_plausible_timestamp(ts) is False

    def test_rejects_far_future(self):
        """Timestamp far in the future is rejected."""
        ts = datetime.now(timezone.utc) + timedelta(days=1)
        assert _is_plausible_timestamp(ts) is False

    def test_allows_clock_skew(self):
        """Timestamp up to 60 seconds in future is allowed (clock skew)."""
        ts = datetime.now(timezone.utc) + timedelta(seconds=30)
        assert _is_plausible_timestamp(ts) is True


class TestExtractBotFingerprint:
    """Tests for complete fingerprint extraction."""

    def test_bot_fingerprint_bot_account(self):
        """Bot account produces expected fingerprint."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        effects = sample_effects_fast_trust_line()
        trades_df = sample_bot_trades()
        bot_account = "GBOT" + "B" * 52

        fingerprint = extract_bot_fingerprint(
            bot_account,
            effects=effects,
            trades_df=trades_df,
            account_created_at=base_time,
        )

        assert fingerprint.account_id == bot_account
        assert fingerprint.trust_line_creation_latency_seconds == 5.0
        assert fingerprint.inter_trade_interval_cv is not None
        assert fingerprint.inter_trade_interval_cv < 0.05
        assert fingerprint.is_valid is True

    def test_human_fingerprint_human_account(self):
        """Human account produces expected fingerprint."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        effects = sample_effects_slow_trust_line()
        trades_df = sample_human_trades()
        human_account = "GHUM" + "H" * 51

        fingerprint = extract_bot_fingerprint(
            human_account,
            effects=effects,
            trades_df=trades_df,
            account_created_at=base_time,
        )

        assert fingerprint.account_id == human_account
        assert fingerprint.trust_line_creation_latency_seconds == 5 * 3600
        assert fingerprint.inter_trade_interval_cv is not None
        assert fingerprint.inter_trade_interval_cv > 0.3
        assert fingerprint.is_valid is True

    def test_insufficient_data_confidence_low(self):
        """Account with insufficient data has low confidence."""
        fingerprint = extract_bot_fingerprint(
            "GACC" + "A" * 52,
            effects=None,
            trades_df=pd.DataFrame(),
            account_created_at=None,
        )

        assert fingerprint.confidence < 0.5


class TestIsLikelyBot:
    """Tests for bot classification heuristic."""

    def test_fast_trust_line_signals_bot(self):
        """Fast trust line creation signals bot."""
        fingerprint = BotFingerprint(
            account_id="GBOT" + "B" * 52,
            trust_line_creation_latency_seconds=5.0,
            inter_trade_interval_cv=None,
            account_management_cluster_score=0.0,
        )

        assert is_likely_bot(fingerprint, threshold=0.5) is True

    def test_low_interval_cv_signals_bot(self):
        """Low interval CV signals bot."""
        fingerprint = BotFingerprint(
            account_id="GBOT" + "B" * 52,
            trust_line_creation_latency_seconds=None,
            inter_trade_interval_cv=0.05,
            account_management_cluster_score=0.0,
        )

        assert is_likely_bot(fingerprint, threshold=0.5) is True

    def test_low_entropy_signals_bot(self):
        """Low operation entropy signals bot."""
        fingerprint = BotFingerprint(
            account_id="GBOT" + "B" * 52,
            trust_line_creation_latency_seconds=None,
            inter_trade_interval_cv=None,
            account_management_cluster_score=1.0,
        )

        assert is_likely_bot(fingerprint, threshold=0.5) is True

    def test_human_account_not_bot(self):
        """Human account does not signal bot."""
        fingerprint = BotFingerprint(
            account_id="GHUM" + "H" * 51,
            trust_line_creation_latency_seconds=5 * 3600,
            inter_trade_interval_cv=0.8,
            account_management_cluster_score=3.0,
        )

        assert is_likely_bot(fingerprint, threshold=0.5) is False
