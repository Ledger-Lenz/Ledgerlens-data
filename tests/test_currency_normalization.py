"""Comprehensive test suite for currency normalization.

Tests cover:
- Exchange rate providers (Mock, Cached)
- Asset classification and metadata
- Normalization functions (single, aggregate)
- Normalization strategies (XLM, USD, MultiHop)
- Edge cases (missing rates, stale rates, multi-hop)
- Integration scenarios (realistic trades)
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from ingestion.data_models import Asset
from utils.currency_normalization import (
    AssetClassifier,
    AssetType,
    CachedRateProvider,
    CurrencyPair,
    MockExchangeRateProvider,
    MultiHopNormalization,
    NormalizationStatus,
    NormalizedAmount,
    StablecoinType,
    USDNormalization,
    XLMNormalization,
    aggregate_normalized,
    create_default_provider,
    create_usd_strategy,
    create_xlm_strategy,
    format_normalized_amount,
    normalize_amount,
)
from utils.decimal_guards import DecimalAmount

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def xlm_asset():
    """Native XLM asset."""
    return Asset(code="XLM", issuer=None)


@pytest.fixture
def usdc_asset():
    """USDC stablecoin asset."""
    return Asset(
        code="USDC",
        issuer="GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
    )


@pytest.fixture
def usdt_asset():
    """USDT stablecoin asset."""
    return Asset(
        code="USDT",
        issuer="GCQTGZQQ5G4PTM2GL7CDIFKUBIPEC52BROAQIAPW53XBRJVN6ZJVTG6V",
    )


@pytest.fixture
def unknown_asset():
    """Unknown token asset."""
    return Asset(code="UNKNOWN", issuer="GTEST123456789")


@pytest.fixture
def mock_provider():
    """Mock exchange rate provider."""
    return MockExchangeRateProvider()


@pytest.fixture
def cached_provider(mock_provider):
    """Cached exchange rate provider."""
    return CachedRateProvider(mock_provider, ttl=timedelta(minutes=5))


# ---------------------------------------------------------------------------
# CurrencyPair Tests
# ---------------------------------------------------------------------------


class TestCurrencyPair:
    """Test CurrencyPair data structure."""

    def test_create_pair(self, usdc_asset, xlm_asset):
        """Create currency pair."""
        pair = CurrencyPair(
            from_asset=usdc_asset,
            to_asset=xlm_asset,
            rate=Decimal("8.5"),
            timestamp=datetime.now(),
            source="test",
            confidence=Decimal("0.95"),
        )

        assert pair.rate == Decimal("8.5")
        assert pair.confidence == Decimal("0.95")

    def test_inverse_pair(self, usdc_asset, xlm_asset):
        """Inverse pair calculation."""
        pair = CurrencyPair(
            from_asset=usdc_asset,
            to_asset=xlm_asset,
            rate=Decimal("8.5"),
            timestamp=datetime.now(),
            source="test",
        )

        inv = pair.inverse()

        assert inv.from_asset == xlm_asset
        assert inv.to_asset == usdc_asset
        assert abs(inv.rate - Decimal("1") / Decimal("8.5")) < Decimal("0.0001")

    def test_pair_key(self, usdc_asset, xlm_asset):
        """Pair key generation."""
        pair = CurrencyPair(
            from_asset=usdc_asset,
            to_asset=xlm_asset,
            rate=Decimal("8.5"),
            timestamp=datetime.now(),
            source="test",
        )

        key = pair.pair_key()

        assert key[0].startswith("USDC:")
        assert key[1] == "XLM:native"

    def test_is_stale(self, usdc_asset, xlm_asset):
        """Stale rate detection."""
        old_time = datetime.now() - timedelta(minutes=10)
        pair = CurrencyPair(
            from_asset=usdc_asset,
            to_asset=xlm_asset,
            rate=Decimal("8.5"),
            timestamp=old_time,
            source="test",
        )

        assert pair.is_stale(threshold=timedelta(minutes=5))

    def test_not_stale(self, usdc_asset, xlm_asset):
        """Recent rate not stale."""
        recent_time = datetime.now() - timedelta(seconds=30)
        pair = CurrencyPair(
            from_asset=usdc_asset,
            to_asset=xlm_asset,
            rate=Decimal("8.5"),
            timestamp=recent_time,
            source="test",
        )

        assert not pair.is_stale(threshold=timedelta(minutes=5))

    def test_invalid_rate(self, usdc_asset, xlm_asset):
        """Negative rate rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            CurrencyPair(
                from_asset=usdc_asset,
                to_asset=xlm_asset,
                rate=Decimal("-1.0"),
                timestamp=datetime.now(),
                source="test",
            )

    def test_invalid_confidence(self, usdc_asset, xlm_asset):
        """Invalid confidence rejected."""
        with pytest.raises(ValueError, match="Confidence must be 0-1"):
            CurrencyPair(
                from_asset=usdc_asset,
                to_asset=xlm_asset,
                rate=Decimal("8.5"),
                timestamp=datetime.now(),
                source="test",
                confidence=Decimal("1.5"),
            )


# ---------------------------------------------------------------------------
# MockExchangeRateProvider Tests
# ---------------------------------------------------------------------------


class TestMockExchangeRateProvider:
    """Test mock exchange rate provider."""

    def test_default_rates(self, mock_provider, usdc_asset, xlm_asset):
        """Default rates available."""
        rate = mock_provider.get_rate(usdc_asset, xlm_asset)

        assert rate is not None
        assert rate.rate == Decimal("8.5")

    def test_set_custom_rate(self, mock_provider, unknown_asset, xlm_asset):
        """Set custom rate."""
        mock_provider.set_rate(unknown_asset, xlm_asset, Decimal("100.0"))

        rate = mock_provider.get_rate(unknown_asset, xlm_asset)

        assert rate is not None
        assert rate.rate == Decimal("100.0")

    def test_inverse_rate_auto_set(self, mock_provider, unknown_asset, xlm_asset):
        """Setting rate auto-sets inverse."""
        mock_provider.set_rate(unknown_asset, xlm_asset, Decimal("100.0"))

        # Check inverse
        inv_rate = mock_provider.get_rate(xlm_asset, unknown_asset)

        assert inv_rate is not None
        assert abs(inv_rate.rate - Decimal("0.01")) < Decimal("0.0001")

    def test_same_asset_identity(self, mock_provider, xlm_asset):
        """Same asset returns identity rate."""
        rate = mock_provider.get_rate(xlm_asset, xlm_asset)

        assert rate is not None
        assert rate.rate == Decimal("1.0")
        assert rate.source == "identity"

    def test_unavailable_rate(self, mock_provider, unknown_asset, xlm_asset):
        """Unavailable rate returns None."""
        # Don't set rate for unknown asset
        rate = mock_provider.get_rate(unknown_asset, xlm_asset)

        # Default mock provider doesn't have rate for unknown asset
        assert rate is None

    def test_is_available(self, mock_provider, usdc_asset, unknown_asset):
        """Check if asset has rates."""
        assert mock_provider.is_available(usdc_asset)
        # Unknown asset not set up, so not available
        assert not mock_provider.is_available(unknown_asset)

    def test_batch_rates(self, mock_provider, usdc_asset, usdt_asset, xlm_asset):
        """Batch rate fetching."""
        pairs = [
            (usdc_asset, xlm_asset),
            (usdt_asset, xlm_asset),
        ]

        rates = mock_provider.get_rates_batch(pairs)

        assert len(rates) == 2


# ---------------------------------------------------------------------------
# CachedRateProvider Tests
# ---------------------------------------------------------------------------


class TestCachedRateProvider:
    """Test cached rate provider."""

    def test_cache_hit(self, cached_provider, usdc_asset, xlm_asset):
        """Second call uses cache."""
        # First call
        rate1 = cached_provider.get_rate(usdc_asset, xlm_asset)

        # Second call (should be cached)
        rate2 = cached_provider.get_rate(usdc_asset, xlm_asset)

        assert rate1 == rate2

    def test_cache_expiry(self, mock_provider, usdc_asset, xlm_asset):
        """Expired cache re-fetches."""
        # Create provider with very short TTL
        cached = CachedRateProvider(mock_provider, ttl=timedelta(milliseconds=1))

        # First call
        rate1 = cached.get_rate(usdc_asset, xlm_asset)

        # Wait for expiry
        import time

        time.sleep(0.002)

        # Second call (should re-fetch)
        rate2 = cached.get_rate(usdc_asset, xlm_asset)

        # Should still work, but not be same object
        assert rate1.rate == rate2.rate

    def test_cache_stats(self, cached_provider, usdc_asset, xlm_asset):
        """Cache statistics."""
        # Add some entries
        cached_provider.get_rate(usdc_asset, xlm_asset)

        stats = cached_provider.get_cache_stats()

        assert stats["cache_size"] >= 1
        assert stats["newest_entry"] is not None

    def test_clear_cache(self, cached_provider, usdc_asset, xlm_asset):
        """Clear cache."""
        # Add entry
        cached_provider.get_rate(usdc_asset, xlm_asset)

        # Clear
        cached_provider.clear_cache()

        stats = cached_provider.get_cache_stats()
        assert stats["cache_size"] == 0


# ---------------------------------------------------------------------------
# AssetClassifier Tests
# ---------------------------------------------------------------------------


class TestAssetClassifier:
    """Test asset classifier."""

    def test_classify_native(self, xlm_asset):
        """Classify native XLM."""
        classifier = AssetClassifier()
        metadata = classifier.classify(xlm_asset)

        assert metadata.asset_type == AssetType.NATIVE
        assert metadata.liquidity_score == Decimal("1.0")

    def test_classify_known_stablecoin(self, usdc_asset):
        """Classify known stablecoin."""
        classifier = AssetClassifier()
        metadata = classifier.classify(usdc_asset)

        assert metadata.asset_type == AssetType.STABLECOIN
        assert metadata.stablecoin_type == StablecoinType.FIAT_BACKED
        assert metadata.liquidity_score >= Decimal("0.9")

    def test_classify_unknown_token(self, unknown_asset):
        """Classify unknown token."""
        classifier = AssetClassifier()
        metadata = classifier.classify(unknown_asset)

        assert metadata.asset_type == AssetType.TOKEN
        assert metadata.liquidity_score == Decimal("0.5")

    def test_is_stablecoin(self, usdc_asset, xlm_asset):
        """Check if stablecoin."""
        classifier = AssetClassifier()

        assert classifier.is_stablecoin(usdc_asset)
        assert not classifier.is_stablecoin(xlm_asset)

    def test_is_native(self, xlm_asset, usdc_asset):
        """Check if native."""
        classifier = AssetClassifier()

        assert classifier.is_native(xlm_asset)
        assert not classifier.is_native(usdc_asset)

    def test_get_preferred_base(self, usdc_asset):
        """Get preferred base currency."""
        classifier = AssetClassifier()
        base = classifier.get_preferred_base(usdc_asset)

        assert base.code == "XLM"
        assert base.issuer is None


# ---------------------------------------------------------------------------
# Normalization Function Tests
# ---------------------------------------------------------------------------


class TestNormalizationFunctions:
    """Test core normalization functions."""

    def test_normalize_same_currency(self, mock_provider, xlm_asset):
        """Normalize to same currency."""
        amount = DecimalAmount("100")

        normalized = normalize_amount(amount, xlm_asset, xlm_asset, mock_provider)

        assert normalized.value == Decimal("100")
        assert normalized.is_same_currency()
        assert normalized.is_successful()

    def test_normalize_with_rate(self, mock_provider, usdc_asset, xlm_asset):
        """Normalize with exchange rate."""
        amount = DecimalAmount("100")

        normalized = normalize_amount(amount, usdc_asset, xlm_asset, mock_provider)

        # 100 USDC * 8.5 = 850 XLM
        assert normalized.value == Decimal("850.0")
        assert normalized.original_value == Decimal("100")
        assert normalized.base_asset == xlm_asset
        assert normalized.is_successful()

    def test_normalize_no_rate(self, mock_provider, unknown_asset, xlm_asset):
        """Normalize with no rate available."""
        amount = DecimalAmount("100")

        normalized = normalize_amount(amount, unknown_asset, xlm_asset, mock_provider)

        assert normalized.status == NormalizationStatus.NO_RATE
        assert not normalized.is_successful()

    def test_normalize_stale_rate(self, usdc_asset, xlm_asset):
        """Normalize with stale rate."""
        # Create provider with old rate
        provider = MockExchangeRateProvider()

        # Get rate and modify timestamp to be old
        old_time = datetime.now() - timedelta(minutes=10)

        amount = DecimalAmount("100")
        normalized = normalize_amount(amount, usdc_asset, xlm_asset, provider, timestamp=old_time)

        # Should still work but flag as stale
        assert normalized.value == Decimal("850.0")

    def test_aggregate_empty(self, mock_provider, xlm_asset):
        """Aggregate empty list."""
        amounts = []

        total = aggregate_normalized(amounts, xlm_asset, mock_provider)

        assert total.value == Decimal("0")
        assert total.is_successful()

    def test_aggregate_single_currency(self, mock_provider, xlm_asset):
        """Aggregate same currency."""
        amounts = [
            (DecimalAmount("100"), xlm_asset),
            (DecimalAmount("200"), xlm_asset),
            (DecimalAmount("300"), xlm_asset),
        ]

        total = aggregate_normalized(amounts, xlm_asset, mock_provider)

        assert total.value == Decimal("600")
        assert total.is_successful()

    def test_aggregate_multiple_currencies(self, mock_provider, usdc_asset, usdt_asset, xlm_asset):
        """Aggregate multiple currencies."""
        amounts = [
            (DecimalAmount("100"), usdc_asset),  # 100 * 8.5 = 850
            (DecimalAmount("100"), usdt_asset),  # 100 * 8.4 = 840
            (DecimalAmount("100"), xlm_asset),  # 100 * 1 = 100
        ]

        total = aggregate_normalized(amounts, xlm_asset, mock_provider)

        # 850 + 840 + 100 = 1790
        assert total.value == Decimal("1790.0")
        assert total.is_successful()

    def test_aggregate_with_failures(self, mock_provider, unknown_asset, xlm_asset):
        """Aggregate with some failed conversions."""
        amounts = [
            (DecimalAmount("100"), xlm_asset),  # Success
            (DecimalAmount("100"), unknown_asset),  # Fail (no rate)
        ]

        total = aggregate_normalized(amounts, xlm_asset, mock_provider)

        # Should only include successful conversion
        assert total.value == Decimal("100")
        assert total.status == NormalizationStatus.ERROR  # Flag that some failed


# ---------------------------------------------------------------------------
# Strategy Tests
# ---------------------------------------------------------------------------


class TestXLMNormalization:
    """Test XLM normalization strategy."""

    def test_get_base_asset(self):
        """Base asset is XLM."""
        provider = MockExchangeRateProvider()
        strategy = XLMNormalization(provider)

        base = strategy.get_base_asset()

        assert base.code == "XLM"
        assert base.issuer is None

    def test_normalize_usdc_to_xlm(self, mock_provider, usdc_asset):
        """Normalize USDC to XLM."""
        strategy = XLMNormalization(mock_provider)
        amount = DecimalAmount("100")

        normalized = strategy.normalize(amount, usdc_asset)

        assert normalized.value == Decimal("850.0")
        assert normalized.base_asset.code == "XLM"

    def test_normalize_xlm_to_xlm(self, mock_provider, xlm_asset):
        """Normalize XLM to XLM (identity)."""
        strategy = XLMNormalization(mock_provider)
        amount = DecimalAmount("100")

        normalized = strategy.normalize(amount, xlm_asset)

        assert normalized.value == Decimal("100")
        assert normalized.is_same_currency()


class TestUSDNormalization:
    """Test USD normalization strategy."""

    def test_get_base_asset(self):
        """Base asset is USDC."""
        provider = MockExchangeRateProvider()
        strategy = USDNormalization(provider)

        base = strategy.get_base_asset()

        assert base.code == "USDC"

    def test_normalize_stablecoin_to_usd(self, mock_provider, usdc_asset):
        """Stablecoins treated as 1:1 with USD."""
        strategy = USDNormalization(mock_provider)
        amount = DecimalAmount("100")

        normalized = strategy.normalize(amount, usdc_asset)

        # Should be 1:1 (or very close)
        assert normalized.value == Decimal("100")
        assert normalized.confidence >= Decimal("0.99")

    def test_normalize_xlm_to_usd(self, mock_provider, xlm_asset):
        """Convert XLM to USD."""
        strategy = USDNormalization(mock_provider)
        amount = DecimalAmount("850")

        normalized = strategy.normalize(amount, xlm_asset)

        # 850 XLM / 8.5 = 100 USDC
        assert normalized.value == Decimal("100.0")
        assert normalized.base_asset.code == "USDC"


class TestMultiHopNormalization:
    """Test multi-hop normalization strategy."""

    def test_direct_conversion(self, mock_provider, usdc_asset, xlm_asset):
        """Direct conversion when available."""
        strategy = MultiHopNormalization(mock_provider, base_asset=xlm_asset)
        amount = DecimalAmount("100")

        normalized = strategy.normalize(amount, usdc_asset)

        assert normalized.value == Decimal("850.0")
        assert normalized.status == NormalizationStatus.SUCCESS
        assert len(normalized.conversion_path) == 2  # USDC -> XLM

    def test_multi_hop_conversion(self, mock_provider, unknown_asset, usdc_asset, xlm_asset):
        """Multi-hop when direct not available."""
        # Set up rates: UNKNOWN -> XLM, XLM -> USDC
        mock_provider.set_rate(unknown_asset, xlm_asset, Decimal("10.0"))

        strategy = MultiHopNormalization(mock_provider, base_asset=usdc_asset)
        amount = DecimalAmount("100")

        normalized = strategy.normalize(amount, unknown_asset)

        # 100 UNKNOWN * 10 = 1000 XLM
        # 1000 XLM / 8.5 = ~117.65 USDC
        assert normalized.status == NormalizationStatus.MULTI_HOP
        assert len(normalized.conversion_path) == 3  # UNKNOWN -> XLM -> USDC
        assert normalized.confidence < Decimal("1.0")  # Penalty for multi-hop

    def test_no_path_available(self, mock_provider, unknown_asset, xlm_asset):
        """No conversion path available."""
        strategy = MultiHopNormalization(mock_provider, base_asset=xlm_asset)
        amount = DecimalAmount("100")

        normalized = strategy.normalize(amount, unknown_asset)

        assert normalized.status == NormalizationStatus.NO_RATE


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_default_provider(self):
        """Create default provider."""
        provider = create_default_provider()

        assert provider is not None

    def test_create_xlm_strategy(self):
        """Create XLM strategy."""
        strategy = create_xlm_strategy()

        assert isinstance(strategy, XLMNormalization)
        assert strategy.get_base_asset().code == "XLM"

    def test_create_usd_strategy(self):
        """Create USD strategy."""
        strategy = create_usd_strategy()

        assert isinstance(strategy, USDNormalization)
        assert strategy.get_base_asset().code == "USDC"

    def test_format_normalized_amount_same_currency(self, xlm_asset):
        """Format same currency."""
        normalized = NormalizedAmount(
            value=Decimal("100"),
            base_asset=xlm_asset,
            original_value=Decimal("100"),
            original_asset=xlm_asset,
        )

        formatted = format_normalized_amount(normalized)

        assert "100.00 XLM" in formatted

    def test_format_normalized_amount_conversion(self, usdc_asset, xlm_asset):
        """Format with conversion."""
        pair = CurrencyPair(
            from_asset=usdc_asset,
            to_asset=xlm_asset,
            rate=Decimal("8.5"),
            timestamp=datetime.now(),
            source="test",
            confidence=Decimal("0.95"),
        )

        normalized = NormalizedAmount(
            value=Decimal("850.0"),
            base_asset=xlm_asset,
            original_value=Decimal("100"),
            original_asset=usdc_asset,
            exchange_rate=pair,
            confidence=Decimal("0.95"),
            conversion_path=[usdc_asset, xlm_asset],
        )

        formatted = format_normalized_amount(normalized)

        assert "100.00 USDC" in formatted
        assert "850.00 XLM" in formatted
        assert "95%" in formatted  # Confidence


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_trade_normalization(self, mock_provider, usdc_asset, xlm_asset):
        """Normalize trade amounts."""
        strategy = XLMNormalization(mock_provider)

        # Trade: 100 USDC for 850 XLM
        base_amount = DecimalAmount("100")
        counter_amount = DecimalAmount("850")

        # Normalize both to XLM
        norm_base = strategy.normalize(base_amount, usdc_asset)
        norm_counter = strategy.normalize(counter_amount, xlm_asset)

        # Both should be in XLM now
        assert norm_base.base_asset.code == "XLM"
        assert norm_counter.base_asset.code == "XLM"

        # Values should match (within rounding)
        assert abs(norm_base.value - norm_counter.value) < Decimal("0.01")

    def test_multi_asset_portfolio(self, mock_provider, usdc_asset, usdt_asset, xlm_asset):
        """Calculate total portfolio value."""
        XLMNormalization(mock_provider)

        # Portfolio holdings
        holdings = [
            (DecimalAmount("1000"), usdc_asset),
            (DecimalAmount("500"), usdt_asset),
            (DecimalAmount("5000"), xlm_asset),
        ]

        # Aggregate to XLM
        total = aggregate_normalized(holdings, xlm_asset, mock_provider)

        # 1000*8.5 + 500*8.4 + 5000 = 8500 + 4200 + 5000 = 17700
        assert total.value == Decimal("17700.0")
        assert total.is_successful()

    def test_confidence_weighting(self, mock_provider, usdc_asset, xlm_asset):
        """Confidence affects normalization."""
        # Set rate with low confidence
        mock_provider.set_rate(usdc_asset, xlm_asset, Decimal("8.5"), confidence=Decimal("0.5"))

        strategy = XLMNormalization(mock_provider)
        amount = DecimalAmount("100")

        normalized = strategy.normalize(amount, usdc_asset)

        # Value should be correct
        assert normalized.value == Decimal("850.0")

        # But confidence should be low
        assert normalized.confidence == Decimal("0.5")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestUnsupportedCurrencyCode:
    """Tests for unsupported/malformed currency codes (#678)."""

    def test_garbage_currency_code_returns_no_rate(self, mock_provider):
        """A completely unrecognized asset code returns NO_RATE, not an exception."""
        from decimal import Decimal

        garbage_asset = Asset(code="GARBAGE123", issuer="GINVALID000000000000000000000000000000000000000000")
        amount = DecimalAmount("100")
        xlm = Asset(code="XLM", issuer=None)

        normalized = normalize_amount(amount, garbage_asset, xlm, mock_provider)

        assert normalized.status == NormalizationStatus.NO_RATE
        assert not normalized.is_successful()
        assert normalized.confidence == Decimal("0.0")
        assert normalized.original_value == Decimal("100")

    def test_empty_string_currency_code_returns_no_rate(self, mock_provider):
        """An empty-string asset code returns NO_RATE without raising."""
        empty_asset = Asset(code="", issuer=None)
        amount = DecimalAmount("100")
        xlm = Asset(code="XLM", issuer=None)

        normalized = normalize_amount(amount, empty_asset, xlm, mock_provider)

        assert normalized.status == NormalizationStatus.NO_RATE
        assert normalized.confidence == Decimal("0.0")

    def test_fully_malformed_asset_passes_through_unchanged(self, mock_provider):
        """A malformed asset with no rate returns the original amount unchanged."""
        malformed = Asset(code="@@@INVALID@@@", issuer="GBADBADBADBADBADBADBADBADBADBADBADBADBADBAD")
        amount = DecimalAmount("250")
        xlm = Asset(code="XLM", issuer=None)

        normalized = normalize_amount(amount, malformed, xlm, mock_provider)

        assert normalized.status == NormalizationStatus.NO_RATE
        assert normalized.value == Decimal("250")
        assert normalized.original_asset == malformed
        assert normalized.base_asset == malformed  # Keeps original when no rate
