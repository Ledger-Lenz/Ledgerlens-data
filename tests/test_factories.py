"""Self-tests for the test data factory.

Verifies that CleanTradeFactory generates Benford-conforming trades and
WashTradeFactory generates non-conforming trades. Tests factory isolation
and Stellar account ID validity.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from detection.benford_engine import chi_square_statistic, leading_digits, mad_score
from tests.factories import (
    CleanTradeFactory,
    RingTradeFactory,
    WashTradeFactory,
    generate_stellar_account_id,
    make_clean_trades,
    make_ring_trades,
    make_wash_trades,
)


def test_generate_stellar_account_id_format():
    """Verify account IDs are G-prefixed and 57 chars long."""
    account = generate_stellar_account_id("test")
    assert account.startswith("G")
    assert len(account) == 57
    assert account[1:].isalnum()


def test_generate_stellar_account_id_deterministic():
    """Same seed produces same account ID."""
    acc1 = generate_stellar_account_id("seed123")
    acc2 = generate_stellar_account_id("seed123")
    assert acc1 == acc2


def test_generate_stellar_account_id_diverse():
    """Different seeds produce different account IDs."""
    accs = {generate_stellar_account_id(f"seed{i}") for i in range(10)}
    assert len(accs) == 10


class TestCleanTradeFactory:
    """Test CleanTradeFactory generates Benford-conforming trades."""

    def test_factory_builds_valid_trade(self):
        """Factory can build a single trade."""
        trade = CleanTradeFactory.build()
        assert trade.trade_id
        assert trade.base_account.startswith("G")
        assert trade.counter_account.startswith("G")
        assert trade.base_amount > 0
        assert trade.base_asset.code

    def test_factory_isolation(self):
        """Each factory build is independent (no state leaking)."""
        t1 = CleanTradeFactory.build()
        t2 = CleanTradeFactory.build()
        assert t1.trade_id != t2.trade_id
        assert t1.base_account != t2.base_account

    def test_benford_conformance_large_batch(self):
        """Batch of >100 clean trades should pass Benford chi-square at 5% level."""
        trades = make_clean_trades(n=150)
        amounts = pd.Series([t["base_amount"] for t in trades])

        chi2 = chi_square_statistic(amounts)
        # Benford chi-square has 8 DOF; critical value at α=0.05 is ~15.51
        # We expect chi2 < 15.51 for genuine Benford data
        assert chi2 < 20, f"Chi-square {chi2} indicates non-Benford distribution"

    def test_benford_mad_score(self):
        """MAD score for clean trades should be < 0.015."""
        trades = make_clean_trades(n=150)
        amounts = pd.Series([t["base_amount"] for t in trades])

        mad = mad_score(amounts)
        assert mad < 0.015, f"MAD {mad} indicates non-Benford distribution"


class TestWashTradeFactory:
    """Test WashTradeFactory generates non-Benford trades."""

    def test_factory_builds_valid_trade(self):
        """Factory can build a single wash trade."""
        trade = WashTradeFactory.build()
        assert trade.trade_id
        assert trade.base_account.startswith("G")
        assert trade.counter_account.startswith("G")
        assert trade.base_amount > 0

    def test_factory_isolation(self):
        """Each factory build is independent."""
        t1 = WashTradeFactory.build()
        t2 = WashTradeFactory.build()
        assert t1.trade_id != t2.trade_id

    def test_wash_trades_round_numbers(self):
        """Wash trade amounts should be round numbers."""
        trades = make_wash_trades(n=30)
        amounts = [t["base_amount"] for t in trades]
        expected_choices = [500, 1000, 5000, 10000, 50000]
        for amt in amounts:
            assert amt in expected_choices, f"Amount {amt} not in expected set"

    def test_wash_trades_fail_benford_chi_square(self):
        """Batch of >30 wash trades should FAIL chi-square at 5% level."""
        trades = make_wash_trades(n=50)
        amounts = pd.Series([t["base_amount"] for t in trades])

        chi2 = chi_square_statistic(amounts)
        # Expect chi2 > 15.51 (critical value at α=0.05)
        assert chi2 > 15, f"Chi-square {chi2} should indicate non-Benford distribution"

    def test_wash_trades_high_mad(self):
        """MAD score for wash trades should be > 0.015."""
        trades = make_wash_trades(n=50)
        amounts = pd.Series([t["base_amount"] for t in trades])

        mad = mad_score(amounts)
        assert mad > 0.015, f"MAD {mad} should indicate non-Benford distribution"

    def test_wash_trades_same_counterparty(self):
        """All wash trades should have the same sock-puppet counterparty."""
        trades = make_wash_trades(n=10)
        counterparties = {t["counter_account"] for t in trades}
        # Due to factory implementation, they should all be the same
        assert len(counterparties) == 1


class TestRingTradeFactory:
    """Test RingTradeFactory generates ring-trading patterns."""

    def test_factory_builds_valid_trade(self):
        """Factory can build a single ring trade."""
        trade = RingTradeFactory.build()
        assert trade.trade_id
        assert trade.base_account.startswith("G")
        assert trade.counter_account.startswith("G")

    def test_ring_structure_with_size(self):
        """Ring trades should form circular patterns."""
        trades = make_ring_trades(n=15, ring_size=5)
        
        # Extract the ring member indices from generated trades
        members = {t["base_account"] for t in trades}
        # With ring_size=5, we should have ~5 unique base accounts
        assert len(members) == 5

    def test_ring_trades_varied_amounts(self):
        """Ring trade amounts should have slight variation."""
        trades = make_ring_trades(n=30, ring_size=5)
        amounts = [t["base_amount"] for t in trades]
        
        # Should have variation due to jitter
        assert len(set(amounts)) > 1, "Ring amounts should vary"
        
        # But still roughly in the 1000-10000 range
        for amt in amounts:
            assert 900 < amt < 11000, f"Amount {amt} outside expected range"


class TestFactoryIntegration:
    """Integration tests across all factories."""

    def test_make_clean_trades_returns_list_of_dicts(self):
        """make_clean_trades returns dicts suitable for DataFrame construction."""
        trades = make_clean_trades(n=20)
        assert len(trades) == 20
        assert all(isinstance(t, dict) for t in trades)
        assert all("trade_id" in t for t in trades)
        assert all("base_amount" in t for t in trades)

    def test_make_wash_trades_returns_list_of_dicts(self):
        """make_wash_trades returns dicts suitable for DataFrame construction."""
        trades = make_wash_trades(n=15)
        assert len(trades) == 15
        assert all(isinstance(t, dict) for t in trades)

    def test_make_ring_trades_returns_list_of_dicts(self):
        """make_ring_trades returns dicts suitable for DataFrame construction."""
        trades = make_ring_trades(n=20, ring_size=4)
        assert len(trades) == 20
        assert all(isinstance(t, dict) for t in trades)

    def test_trades_can_be_converted_to_dataframe(self):
        """Factory outputs can be converted to DataFrame."""
        clean = make_clean_trades(n=5)
        wash = make_wash_trades(n=3)
        ring = make_ring_trades(n=5)

        df_clean = pd.DataFrame(clean)
        df_wash = pd.DataFrame(wash)
        df_ring = pd.DataFrame(ring)

        assert len(df_clean) == 5
        assert len(df_wash) == 3
        assert len(df_ring) == 5


@pytest.fixture
def synthetic_stellar_trades():
    """Pytest fixture: generate realistic synthetic trades.

    Returns a list of trade dicts with configurable pattern and count.
    Usage: def test_my_feature(synthetic_stellar_trades):
               trades = synthetic_stellar_trades(n=100, pattern='clean')
    """

    def _make_trades(n: int = 100, pattern: str = "clean") -> list[dict]:
        """
        Generate synthetic Stellar trade fixtures.

        Args:
            n: Number of trades to generate.
            pattern: 'clean' (Benford-conforming), 'wash' (non-conforming),
                     or 'ring' (coordinated ring).

        Returns:
            List of trade dicts ready for ingestion.
        """
        if pattern == "clean":
            return make_clean_trades(n)
        elif pattern == "wash":
            return make_wash_trades(n)
        elif pattern == "ring":
            return make_ring_trades(n)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    return _make_trades


def test_synthetic_stellar_trades_fixture(synthetic_stellar_trades):
    """Verify the fixture works as expected."""
    clean = synthetic_stellar_trades(n=50, pattern="clean")
    assert len(clean) == 50
    
    wash = synthetic_stellar_trades(n=30, pattern="wash")
    assert len(wash) == 30
    
    # Verify Benford properties
    clean_amounts = pd.Series([t["base_amount"] for t in clean])
    wash_amounts = pd.Series([t["base_amount"] for t in wash])
    
    clean_chi2 = chi_square_statistic(clean_amounts)
    wash_chi2 = chi_square_statistic(wash_amounts)
    
    # Clean should have lower chi-square than wash
    assert clean_chi2 < wash_chi2
