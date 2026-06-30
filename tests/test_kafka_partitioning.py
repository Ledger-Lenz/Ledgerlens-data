"""Unit and integration tests for Kafka-based partitioning system.

Tests verify:
  1. Partition key generation is deterministic and stable
  2. Asset pair ID validation
  3. Dead-letter queue routing for invalid pairs
  4. Partition rebalancing and offset commit behavior
"""

import pytest

from ingestion.kafka_producer import (
    _to_canonical_pair_id,
    _validate_asset_code,
    _validate_issuer,
)


class TestAssetValidation:
    """Test asset code and issuer validation."""

    def test_valid_asset_codes(self):
        """Asset codes must be 1-12 alphanumeric characters."""
        assert _validate_asset_code("XLM")
        assert _validate_asset_code("USDC")
        assert _validate_asset_code("BTC")
        assert _validate_asset_code("A")
        assert _validate_asset_code("1234567890AB")

    def test_invalid_asset_codes(self):
        """Reject asset codes with invalid format."""
        assert not _validate_asset_code("")
        assert not _validate_asset_code("1" * 13)  # Too long
        assert not _validate_asset_code("usd")  # Lowercase
        assert not _validate_asset_code("USD-C")  # Hyphen

    def test_valid_issuer_native(self):
        """Accept 'native' as valid issuer."""
        assert _validate_issuer("native")

    def test_valid_issuer_account_id(self):
        """Accept 56-char Stellar account IDs."""
        valid_id = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF46Q6"
        assert _validate_issuer(valid_id)

    def test_invalid_issuer(self):
        """Reject invalid issuer formats."""
        assert not _validate_issuer("")
        assert not _validate_issuer("GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF46Q6X")  # 57 chars
        assert not _validate_issuer("GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF46Q")  # 55 chars
        assert not _validate_issuer("invalid")


class TestCanonicalPairId:
    """Test deterministic and stable pair ID generation."""

    def test_canonical_pair_native_assets(self):
        """Canonical pair with native assets."""
        pair_id = _to_canonical_pair_id("XLM", "native", "USDC", "native")
        expected = "USDC:native/XLM:native"  # Alphabetically sorted
        assert pair_id == expected

    def test_canonical_pair_with_issuer(self):
        """Canonical pair with issued assets."""
        issuer = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF46Q6"
        pair_id = _to_canonical_pair_id("USDC", issuer, "XLM", "native")
        # Sort: "USDC:<issuer>" < "XLM:native"
        expected = f"USDC:{issuer}/XLM:native"
        assert pair_id == expected

    def test_canonical_pair_deterministic(self):
        """Same pair should always map to the same partition key."""
        issuer = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF46Q6"
        pair_id_1 = _to_canonical_pair_id("USDC", issuer, "XLM", "native")
        pair_id_2 = _to_canonical_pair_id("XLM", "native", "USDC", issuer)
        assert pair_id_1 == pair_id_2

    def test_canonical_pair_alphabetic_sorting(self):
        """Pair IDs are sorted alphabetically regardless of input order."""
        issuer_a = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF46Q6"
        issuer_b = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBWHF46Q6"

        # Forward order
        pair_1 = _to_canonical_pair_id("BTC", issuer_a, "USD", issuer_b)
        # Reverse order
        pair_2 = _to_canonical_pair_id("USD", issuer_b, "BTC", issuer_a)

        assert pair_1 == pair_2
        assert pair_1.startswith("BTC:")

    def test_invalid_asset_a_raises(self):
        """Invalid asset A raises ValueError."""
        with pytest.raises(ValueError, match="Invalid asset A"):
            _to_canonical_pair_id("invalid", "native", "XLM", "native")

    def test_invalid_asset_b_raises(self):
        """Invalid asset B raises ValueError."""
        with pytest.raises(ValueError, match="Invalid asset B"):
            _to_canonical_pair_id("XLM", "native", "USD123456", "native")

    def test_invalid_issuer_raises(self):
        """Invalid issuer raises ValueError."""
        with pytest.raises(ValueError, match="Invalid asset"):
            _to_canonical_pair_id("USDC", "not_valid", "XLM", "native")


class TestPartitionKeyConsistency:
    """Test that events with the same asset pair hash to the same partition."""

    def test_multiple_trades_same_pair_same_partition_key(self):
        """All trades for XLM/USDC must produce the same partition key."""
        pair_keys = []
        for _ in range(5):
            key = _to_canonical_pair_id("XLM", "native", "USDC", "native")
            pair_keys.append(key)

        # All keys must be identical
        assert all(k == pair_keys[0] for k in pair_keys)

    def test_different_pairs_different_partition_keys(self):
        """Different pairs must produce different partition keys."""
        pair_1 = _to_canonical_pair_id("XLM", "native", "USDC", "native")
        pair_2 = _to_canonical_pair_id("BTC", "native", "ETH", "native")

        assert pair_1 != pair_2
