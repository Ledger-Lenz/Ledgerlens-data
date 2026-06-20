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


def test_leading_digits_basic():
    amounts = pd.Series([123, 0.045, 9876, 1])
    digits = leading_digits(amounts)
    assert list(digits) == [1, 4, 9, 1]


def test_leading_digits_drops_nonpositive():
    amounts = pd.Series([0, -5, 100])
    digits = leading_digits(amounts)
    assert list(digits) == [1]


def test_benford_conforming_sample_has_low_mad():
    # Generate a large sample from a log-uniform distribution, which
    # conforms closely to Benford's Law.
    rng = np.random.default_rng(42)
    amounts = pd.Series(10 ** rng.uniform(0, 4, size=20000))

    assert mad_score(amounts) < 0.015


def test_benford_round_numbers_are_nonconforming():
    # Wash-trading-style fixed lot sizes concentrated on digit 5.
    amounts = pd.Series([500] * 100 + [5000] * 100 + [50000] * 100)

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
    from detection.benford_engine import compute_benford_metrics
    # Create a small series with only 5 trades
    amounts = pd.Series([100.0] * 5)

    # Run compute_benford_metrics
    metrics = compute_benford_metrics(amounts)

    # Assert features are NaNs
    assert pd.isna(metrics["chi_square"])
    assert pd.isna(metrics["mad"])
    assert metrics["mad_nonconforming"] is False
    assert all(pd.isna(v) for v in metrics["z_scores"].values())
    assert metrics["sample_size"] == 5


def test_window_candidates():
    from detection.benford_window_optimizer import get_window_candidates
    # Test high velocity
    cands = get_window_candidates(trades_per_hour=100.0, min_trades=20, max_window=720, num_candidates=10)
    assert len(cands) > 1
    assert cands[0] >= 1
    assert cands[-1] <= 720

    # Test low velocity
    cands_low = get_window_candidates(trades_per_hour=0.1, min_trades=20, max_window=720, num_candidates=10)
    assert cands_low[0] >= 200
    assert cands_low[-1] == 720


def test_optimize_windows_for_asset():
    from detection.benford_window_optimizer import optimize_windows_for_asset

    # Create mock trades
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    trades_df = pd.DataFrame({
        "trade_id": [f"t{i}" for i in range(100)],
        "ledger_close_time": dates,
        "base_account": ["WALLET_A" if i % 2 == 0 else "WALLET_B" for i in range(100)],
        "counter_account": ["WALLET_B" if i % 2 == 0 else "WALLET_A" for i in range(100)],
        "base_asset": ["USDC:issuer"] * 100,
        "counter_asset": ["XLM:native"] * 100,
        "amount": rng.uniform(10, 100, size=100),
        "price": [1.0] * 100,
    })

    # Create labelled dataset
    labelled_df = pd.DataFrame([
        {"wallet": "WALLET_A", "label": 1},
        {"wallet": "WALLET_B", "label": 0},
    ])

    # Run optimizer
    windows = optimize_windows_for_asset(
        asset_code="USDC:issuer",
        trades_df=trades_df,
        labelled_df=labelled_df,
        num_windows=5,
        n_calls=6,  # keep it small and fast for testing
    )

    # Assert they are sorted in ascending order and have length 5
    assert len(windows) == 5
    assert windows == sorted(windows)
