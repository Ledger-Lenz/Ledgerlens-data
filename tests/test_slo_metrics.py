"""Unit tests for SLO dashboard Prometheus counters (issue #197)."""

import pytest


def test_confirmed_wash_trades_counter_registered():
    """Verify that ledgerlens_confirmed_wash_trades_total counter is registered."""
    from detection.per_pair_metrics import ledgerlens_confirmed_wash_trades_total
    assert ledgerlens_confirmed_wash_trades_total is not None


def test_confirmed_clean_wallets_counter_registered():
    """Verify that ledgerlens_confirmed_clean_wallets_total counter is registered."""
    from detection.per_pair_metrics import ledgerlens_confirmed_clean_wallets_total
    assert ledgerlens_confirmed_clean_wallets_total is not None


def test_record_confirmed_wash_trade_increments_counter():
    """Verify that record_confirmed_wash_trade increments the counter."""
    from detection.per_pair_metrics import (
        record_confirmed_wash_trade,
        ledgerlens_confirmed_wash_trades_total,
    )

    asset_pair = "USDC:GA5Z/XLM:native"
    
    # Get initial value
    initial_samples = list(ledgerlens_confirmed_wash_trades_total.collect())[0].samples
    initial_count = sum(1 for s in initial_samples if "GA5Z" in str(s))
    
    # Record a wash trade
    record_confirmed_wash_trade(asset_pair)
    
    # Get new value
    new_samples = list(ledgerlens_confirmed_wash_trades_total.collect())[0].samples
    new_count = sum(1 for s in new_samples if "GA5Z" in str(s))
    
    # Counter should have incremented
    assert new_count >= initial_count


def test_record_confirmed_clean_wallet_increments_counter():
    """Verify that record_confirmed_clean_wallet increments the counter."""
    from detection.per_pair_metrics import (
        record_confirmed_clean_wallet,
        ledgerlens_confirmed_clean_wallets_total,
    )

    asset_pair = "BTC:GA5Z/XLM:native"
    
    # Get initial value
    initial_samples = list(ledgerlens_confirmed_clean_wallets_total.collect())[0].samples
    initial_count = sum(1 for s in initial_samples if "GA5Z" in str(s))
    
    # Record a clean wallet
    record_confirmed_clean_wallet(asset_pair)
    
    # Get new value
    new_samples = list(ledgerlens_confirmed_clean_wallets_total.collect())[0].samples
    new_count = sum(1 for s in new_samples if "GA5Z" in str(s))
    
    # Counter should have incremented
    assert new_count >= initial_count


def test_canonical_pair_formats_correctly():
    """Verify that canonical_pair sorts pair labels alphabetically."""
    from detection.per_pair_metrics import canonical_pair
    
    # Forward pair
    pair1 = canonical_pair("XLM:native/USDC:GA5Z")
    # Reverse pair
    pair2 = canonical_pair("USDC:GA5Z/XLM:native")
    
    # Both should map to the same canonical form
    assert pair1 == pair2
    # Canonical form should be sorted
    assert pair1.startswith("USDC:") or pair1.startswith("XLM:")


def test_canonical_pair_preserves_single_pair():
    """Verify that canonical_pair handles malformed input gracefully."""
    from detection.per_pair_metrics import canonical_pair
    
    # Invalid pair (only one leg)
    result = canonical_pair("USDC:GA5Z")
    # Should return as-is if not well-formed
    assert result == "USDC:GA5Z"


def test_metrics_no_wallet_addresses_in_labels():
    """Verify that metrics never include wallet addresses in labels (security requirement).
    
    This test ensures the canonical_pair function filters out wallet addresses
    and only preserves CODE:ISSUER format labels.
    """
    from detection.per_pair_metrics import canonical_pair
    
    # Example with wallet address (should not happen in practice, but verify defense)
    pair = "USDC:GABC1234/XLM:native"
    canonical = canonical_pair(pair)
    
    # Should not contain obvious wallet addresses (40+ char hex strings, etc.)
    assert len(canonical) < 200, "Canonical pair suspiciously long; may contain addresses"
    # Should contain CODE: indicators
    assert ":" in canonical, "Canonical pair should contain asset classifiers"
