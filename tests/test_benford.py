import numpy as np
import pandas as pd

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

    assert mad_score(amounts) < 0.015


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

