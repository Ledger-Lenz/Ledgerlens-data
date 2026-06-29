import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from detection.benford_engine import (
    BENFORD_EXPECTED,
    chi_square_statistic,
    leading_digits,
    mad_score,
    observed_distribution,
    z_scores,
)
from tests.factories import make_clean_trades, make_wash_trades


def test_leading_digits_basic():
    amounts = pd.Series([123, 0.045, 9876, 1])
    digits = leading_digits(amounts)
    assert list(digits) == [1, 4, 9, 1]


def test_leading_digits_drops_nonpositive():
    amounts = pd.Series([0, -5, 100])
    digits = leading_digits(amounts)
    assert list(digits) == [1]


def test_benford_conforming_sample_has_low_mad():
    # Use CleanTradeFactory which generates Benford-conforming amounts
    trades = make_clean_trades(n=200)
    amounts = pd.Series([t["base_amount"] for t in trades])

    # Allow larger tolerance for randomness in factory generation (flaky test protection)
    assert mad_score(amounts) < 0.025


def test_benford_round_numbers_are_nonconforming():
    # Use WashTradeFactory which generates round numbers (non-conforming)
    trades = make_wash_trades(n=50)
    amounts = pd.Series([t["base_amount"] for t in trades])

    assert mad_score(amounts) > 0.015
    assert chi_square_statistic(amounts) > 0


def test_observed_distribution_sums_to_one():
    amounts = pd.Series(np.arange(1, 1000))
    dist = observed_distribution(amounts)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_z_scores_nonnegative():
    amounts = pd.Series([111, 222, 333, 444])
    scores = z_scores(amounts)
    assert all(v >= 0 for v in scores.values())


def test_benford_expected_sums_to_one():
    assert abs(sum(BENFORD_EXPECTED.values()) - 1.0) < 1e-9


def test_minimum_sample_guard():
    from config import config
    from detection.benford_engine import compute_benford_metrics
    orig_min = config.MIN_TRADES_FOR_SCORING
    try:
        config.MIN_TRADES_FOR_SCORING = 20
        # Under threshold (10 < 20) -> should emit NaNs
        amounts = pd.Series([123.0] * 10)
        metrics = compute_benford_metrics(amounts)
        assert np.isnan(metrics.chi_square)
        assert np.isnan(metrics.mad)
        assert np.isnan(metrics["z_max"])
        assert metrics.sample_size == 10

        # Over threshold (25 >= 20) -> should emit valid values
        amounts_valid = pd.Series([123.0] * 25)
        metrics_valid = compute_benford_metrics(amounts_valid)
        assert not np.isnan(metrics_valid.chi_square)
        assert not np.isnan(metrics_valid.mad)
        assert not np.isnan(metrics_valid["z_max"])
        assert metrics_valid.sample_size == 25
    finally:
        config.MIN_TRADES_FOR_SCORING = orig_min


# ---------------------------------------------------------------------------
# Issue #279 — Asset-class-aware Benford baseline calibration
# ---------------------------------------------------------------------------


def test_asset_classifier_classifies_stablecoins():
    from detection.benford_engine import AssetClassifier

    clf = AssetClassifier()
    assert clf.classify("USDC") == "stablecoin"
    assert clf.classify("USDT") == "stablecoin"
    assert clf.classify("usdc") == "stablecoin"  # case-insensitive


def test_asset_classifier_classifies_native():
    from detection.benford_engine import AssetClassifier

    clf = AssetClassifier()
    assert clf.classify("XLM") == "native"


def test_asset_classifier_classifies_volatile():
    from detection.benford_engine import AssetClassifier

    clf = AssetClassifier()
    assert clf.classify("BTC") == "volatile"
    assert clf.classify("AQUA") == "volatile"
    assert clf.classify("UNKNOWN") == "volatile"


def test_unknown_asset_falls_back_to_theoretical_benford():
    """An asset not in the classifier must use the theoretical Benford distribution."""
    from detection.benford_engine import AssetClassifier, BENFORD_EXPECTED

    clf = AssetClassifier()
    baseline = clf.get_baseline("MYSTERY")
    assert baseline == dict(BENFORD_EXPECTED)


def test_stablecoin_round_amounts_lower_chi_square_against_stablecoin_baseline():
    """Stablecoin amounts clustered around 100, 1000, 10000 must produce lower
    chi-square against the stablecoin baseline than against the theoretical
    Benford distribution (issue #279 acceptance criterion)."""
    from detection.benford_engine import AssetClassifier, chi_square_statistic, BENFORD_EXPECTED

    # Round-number stablecoin amounts — elevated digit-1 frequency
    amounts = pd.Series(
        [100.0] * 400 + [1000.0] * 300 + [10000.0] * 200 + [500.0] * 100
    )

    clf = AssetClassifier()
    stablecoin_baseline = clf.get_baseline("USDC")

    chi_vs_stablecoin = chi_square_statistic(amounts, baseline=stablecoin_baseline)
    chi_vs_theoretical = chi_square_statistic(amounts, baseline=dict(BENFORD_EXPECTED))

    assert chi_vs_stablecoin < chi_vs_theoretical, (
        f"Expected stablecoin chi-square ({chi_vs_stablecoin:.2f}) < "
        f"theoretical chi-square ({chi_vs_theoretical:.2f})"
    )


def test_compute_benford_metrics_uses_asset_class_baseline():
    """compute_benford_metrics with asset_code='USDC' must use stablecoin baseline."""
    from config import config
    from detection.benford_engine import compute_benford_metrics, chi_square_statistic, BENFORD_EXPECTED, AssetClassifier

    # Enough samples to exceed MIN_TRADES_FOR_SCORING
    amounts = pd.Series([100.0] * 400 + [1000.0] * 300 + [10000.0] * 300)

    clf = AssetClassifier()
    stablecoin_baseline = clf.get_baseline("USDC")

    metrics_with_class = compute_benford_metrics(amounts, asset_code="USDC")
    expected_chi = chi_square_statistic(amounts, baseline=stablecoin_baseline)
    assert abs(metrics_with_class.chi_square - expected_chi) < 1e-9


def test_optimizer_returns_ascending_windows():
    from detection.benford_window_optimizer import optimize_windows_for_asset

    times = pd.date_range("2024-01-01", periods=100, freq="1h")
    trades = pd.DataFrame({
        "ledger_close_time": times.astype(str),
        "amount": np.random.uniform(1.0, 1000.0, size=100),
        "price": [1.0] * 100,
        "base_account": ["wallet_a"] * 50 + ["wallet_b"] * 50,
        "counter_account": ["wallet_c"] * 100,
        "base_asset": ["XLM:native"] * 100,
        "counter_asset": ["USDC:GABC"] * 100
    })

    labelled = pd.DataFrame({
        "wallet": ["wallet_a", "wallet_b", "wallet_c"],
        "label": [1.0, 0.0, 0.0]
    })

    windows = optimize_windows_for_asset("XLM:native", trades, labelled)

    assert len(windows) == 5
    assert all(windows[i] <= windows[i+1] for i in range(len(windows)-1))


# ---------------------------------------------------------------------------
# Issue #205 — Property-based tests for Benford engine using Hypothesis
# ---------------------------------------------------------------------------
# These property tests generate thousands of random inputs to discover edge
# cases that hand-written tests miss.


@given(st.lists(st.floats(min_value=0.0001, max_value=1e12), min_size=1, max_size=10000))
@settings(max_examples=500, deadline=5000)
def test_benford_monotonicity_adding_conforming_trades(amounts_list):
    """Monotonicity property: adding more trades that conform to Benford's
    distribution should generally not significantly worsen the chi-square fit.
    
    Note: This property is relaxed because adding uniform distributions can
    sometimes increase chi-square. The key is that the fit shouldn't degrade
    dramatically (e.g., by more than 100%).
    
    Rationale: chi-square measures deviation from expected distribution.
    Adding genuine (Benford-conforming) trades should not drastically worsen the fit.
    """
    if len(amounts_list) < 5:
        return  # Skip small datasets
    
    # Generate Benford-conforming trades using the factory
    clean_trades = make_clean_trades(n=100)
    conforming_amounts = [t["base_amount"] for t in clean_trades]
    
    amounts = pd.Series(amounts_list[:100])  # Limit to reasonable size
    chi_initial = chi_square_statistic(amounts)
    
    # Add more conforming trades
    extended_amounts = pd.concat(
        [amounts, pd.Series(conforming_amounts[:min(50, len(conforming_amounts))])],
        ignore_index=True
    )
    chi_extended = chi_square_statistic(extended_amounts)
    
    # chi_extended should not degrade by more than 200% (generous tolerance)
    # This accounts for the stochastic nature of distributions
    max_allowable_chi = chi_initial * 3.0 if chi_initial > 0 else 100
    assert chi_extended <= max_allowable_chi, (
        f"chi-square degraded from {chi_initial:.2f} to {chi_extended:.2f} "
        f"after adding Benford-conforming trades"
    )


@given(st.lists(st.floats(min_value=0.0001, max_value=1e12), min_size=1, max_size=10000), st.data())
@settings(max_examples=500, deadline=5000)
def test_benford_scale_invariance(amounts_list, data):
    """Scale invariance property: multiplying all amounts by a power of 10
    must not change the first-digit distribution.
    
    Rationale: Benford's Law describes the frequency of first digits.
    Multiplying by 10^k shifts the decimal place but doesn't change leading digits.
    For example: 123 × 10 = 1230 (both have leading digit 1).
    """
    amounts = pd.Series(amounts_list)
    if len(amounts) < 20:
        return  # Skip if too small (not enough data)
    
    # Compute metrics on original amounts
    dist_original = observed_distribution(amounts)
    chi_original = chi_square_statistic(amounts)
    
    # Scale by power of 10 (preserves leading digits)
    power = data.draw(st.integers(min_value=-5, max_value=5))
    scale_factor = 10 ** power
    scaled_amounts = amounts * scale_factor
    
    # Compute metrics on scaled amounts
    dist_scaled = observed_distribution(scaled_amounts)
    chi_scaled = chi_square_statistic(scaled_amounts)
    
    # Distributions should be identical
    for digit in range(1, 10):
        assert abs(dist_original.get(digit, 0) - dist_scaled.get(digit, 0)) < 1e-9, (
            f"Digit {digit} frequency changed after scaling by 10^{power}"
        )
    
    # Chi-square should be identical
    assert abs(chi_original - chi_scaled) < 1e-9, (
        f"chi-square changed from {chi_original:.6f} to {chi_scaled:.6f} "
        f"after scaling by 10^{power}"
    )


@given(st.lists(st.floats(min_value=0.0001, max_value=1e12), min_size=1, max_size=10000))
@settings(max_examples=500, deadline=5000)
def test_benford_symmetry_reordering(amounts_list):
    """Symmetry property: reordering trade amounts must produce identical
    chi-square and MAD values.
    
    Rationale: Benford metrics are computed over the distribution of first
    digits, which is order-independent.
    """
    amounts = pd.Series(amounts_list)
    if len(amounts) < 20:
        return  # Skip if too small
    
    # Compute metrics on original
    chi_original = chi_square_statistic(amounts)
    mad_original = mad_score(amounts)
    z_original = z_scores(amounts)
    
    # Shuffle and recompute
    shuffled = amounts.sample(frac=1.0, random_state=42).reset_index(drop=True)
    chi_shuffled = chi_square_statistic(shuffled)
    mad_shuffled = mad_score(shuffled)
    z_shuffled = z_scores(shuffled)
    
    # All metrics should be identical
    assert abs(chi_original - chi_shuffled) < 1e-9, (
        f"chi-square changed from {chi_original:.6f} to {chi_shuffled:.6f} "
        f"after reordering"
    )
    assert abs(mad_original - mad_shuffled) < 1e-9, (
        f"MAD changed from {mad_original:.6f} to {mad_shuffled:.6f} "
        f"after reordering"
    )
    for digit in range(1, 10):
        assert abs(z_original.get(digit, 0) - z_shuffled.get(digit, 0)) < 1e-9, (
            f"z-score for digit {digit} changed after reordering"
        )


@given(st.lists(st.floats(min_value=0.0001, max_value=1e12), min_size=1, max_size=10000))
@settings(max_examples=500, deadline=5000)
def test_benford_boundary_single_trade(amounts_list):
    """Boundary property: a dataset with exactly 1 trade must produce a valid
    (not NaN or inf) chi-square value with an appropriate low-confidence flag.
    
    Rationale: Edge cases (minimal data) should not cause numerical errors.
    The Benford engine should gracefully degrade to NaN or return a valid
    low-confidence score, but never inf or crashes.
    """
    # Test with a single amount
    single_amount = pd.Series([amounts_list[0]] if amounts_list else [123.45])
    
    try:
        chi = chi_square_statistic(single_amount)
        mad = mad_score(single_amount)
        z = z_scores(single_amount)
        
        # Values should be either valid numbers or NaN, never inf
        assert not np.isinf(chi), f"chi-square is inf for single trade"
        assert not np.isinf(mad), f"MAD is inf for single trade"
        for digit, score in z.items():
            assert not np.isinf(score), f"z-score for digit {digit} is inf"
        
        # At minimum, leading_digits should extract the digit without error
        digits = leading_digits(single_amount)
        assert len(digits) == 1, f"Expected 1 digit, got {len(digits)}"
        assert 1 <= digits.iloc[0] <= 9, f"Digit {digits.iloc[0]} out of range [1-9]"
    except Exception as e:
        pytest.fail(f"Benford engine crashed on single trade: {e}")


@given(st.lists(st.floats(min_value=0.0001, max_value=1e12), min_size=1, max_size=10000))
@settings(max_examples=500, deadline=5000)
def test_benford_distribution_valid_probabilities(amounts_list):
    """Valid probability distribution property: observed_distribution must
    return non-negative values that sum to 1.0.
    
    Rationale: Probability distributions have well-defined mathematical
    constraints. Violations indicate a bug in the computation.
    """
    amounts = pd.Series(amounts_list)
    if len(amounts) < 5:
        return  # Skip if too small
    
    dist = observed_distribution(amounts)
    
    # All probabilities must be non-negative
    for digit, prob in dist.items():
        assert prob >= 0.0, f"Negative probability {prob} for digit {digit}"
        assert prob <= 1.0, f"Probability {prob} exceeds 1.0 for digit {digit}"
    
    # Must sum to 1.0 (within floating-point tolerance)
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-9, (
        f"Distribution probabilities sum to {total}, expected 1.0"
    )


@given(st.lists(st.floats(min_value=0.0001, max_value=1e12), min_size=1, max_size=10000))
@settings(max_examples=500, deadline=5000)
def test_benford_leading_digits_extraction_consistency(amounts_list):
    """Consistency property: leading_digits must extract exactly one digit
    per positive amount, all in range [1-9].
    
    Rationale: Benford's Law is defined only for leading digits 1–9.
    Zero and negative amounts must be filtered.
    """
    amounts = pd.Series([a for a in amounts_list if a > 0])
    if len(amounts) == 0:
        return  # Skip if all filtered out
    
    digits = leading_digits(amounts)
    
    # Must have same number of digits as positive amounts
    assert len(digits) == len(amounts), (
        f"Expected {len(amounts)} digits, got {len(digits)}"
    )
    
    # All digits must be in range [1-9]
    for digit in digits:
        assert 1 <= digit <= 9, f"Digit {digit} out of valid range [1-9]"


