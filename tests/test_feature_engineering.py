"""Tests for detection/feature_engineering.py, including hardening features.

Covers existing feature tests (previously in test_features.py) and new tests
for the hardening functions added in the adversarial robustness work:
  - entropy_of_amounts returns 0.0 for a single repeated value
  - inter_arrival_cv returns 0.0 for perfectly uniform spacing
  - cross_wallet_volume_corr is bounded in [-1, 1]
"""

import numpy as np
import pandas as pd

from detection.feature_engineering import (
    build_feature_matrix,
    compute_hardening_features,
    compute_trade_pattern_features,
    compute_volume_timing_features,
)
from tests.factories import make_clean_trades

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_trades() -> pd.DataFrame:
    """Use factory to generate realistic sample trades."""
    trades = make_clean_trades(n=2)
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Pre-existing feature tests (keep passing)
# ---------------------------------------------------------------------------


def test_compute_trade_pattern_features_empty():
    features = compute_trade_pattern_features("A", pd.DataFrame())
    assert features["counterparty_concentration_ratio"] == 0.0
    assert features["round_trip_frequency"] == 0.0


def test_compute_trade_pattern_features_concentration():
    df = _sample_trades()
    features = compute_trade_pattern_features("A", df)
    assert features["counterparty_concentration_ratio"] == 1.0


def test_compute_volume_timing_features_empty():
    features = compute_volume_timing_features(pd.DataFrame())
    assert features["volume_per_counterparty_ratio"] == 0.0


def test_build_feature_matrix_returns_row_per_wallet():
    df = _sample_trades()
    matrix = build_feature_matrix(df)
    assert set(matrix["wallet"]) == {"A", "B"}
    assert "benford_chi_square_1h" in matrix.columns
    assert "counterparty_concentration_ratio" in matrix.columns


def test_build_feature_matrix_empty_input():
    matrix = build_feature_matrix(pd.DataFrame())
    assert matrix.empty


# ---------------------------------------------------------------------------
# Hardening feature tests
# ---------------------------------------------------------------------------


def _uniform_trades(n: int = 20, interval_seconds: int = 60) -> pd.DataFrame:
    """Trades with perfectly uniform inter-arrival times."""
    times = pd.date_range("2024-01-01", periods=n, freq=f"{interval_seconds}s", tz="UTC")
    return pd.DataFrame(
        {
            "trade_id": [f"t{i}" for i in range(n)],
            "ledger_close_time": times,
            "base_account": "W",
            "counter_account": [f"CP{i}" for i in range(n)],
            "base_asset": "A:B",
            "counter_asset": "C:D",
            "amount": 100.0,
            "price": 1.0,
        }
    )


def _single_amount_trades(n: int = 20, amount: float = 999.0) -> pd.DataFrame:
    """All trades with the exact same amount — zero entropy."""
    times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "trade_id": [f"t{i}" for i in range(n)],
            "ledger_close_time": times,
            "base_account": "W",
            "counter_account": "CP",
            "base_asset": "A:B",
            "counter_asset": "C:D",
            "amount": amount,
            "price": 1.0,
        }
    )


def test_entropy_of_amounts_zero_for_single_value():
    """entropy_of_amounts must return 0.0 when all amounts are identical."""
    df = _single_amount_trades()
    features = compute_hardening_features(df)
    assert features["entropy_of_amounts"] == 0.0


def test_inter_arrival_cv_zero_for_uniform_spacing():
    """inter_arrival_cv must return 0.0 for perfectly uniform inter-arrivals."""
    df = _uniform_trades()
    features = compute_hardening_features(df)
    assert features["inter_arrival_cv"] == pytest.approx(0.0, abs=1e-6)


def test_inter_arrival_cv_nonzero_for_random_spacing():
    """inter_arrival_cv must be > 0 when arrival times are irregular."""
    rng = np.random.default_rng(0)
    n = 30
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    random_offsets = rng.exponential(scale=60, size=n).cumsum()
    times = [t0 + pd.Timedelta(seconds=float(s)) for s in random_offsets]
    df = pd.DataFrame(
        {
            "trade_id": [f"t{i}" for i in range(n)],
            "ledger_close_time": times,
            "base_account": "W",
            "counter_account": "CP",
            "base_asset": "A:B",
            "counter_asset": "C:D",
            "amount": 100.0,
            "price": 1.0,
        }
    )
    features = compute_hardening_features(df)
    assert features["inter_arrival_cv"] > 0.0


def test_cross_wallet_volume_corr_bounded():
    """cross_wallet_volume_corr must be in [-1, 1]."""
    rng = np.random.default_rng(5)
    n = 40
    times = pd.date_range("2024-01-01", periods=n, freq="30s", tz="UTC")
    counterparties = ["CP_A"] * (n // 2) + ["CP_B"] * (n // 2)
    df = pd.DataFrame(
        {
            "trade_id": [f"t{i}" for i in range(n)],
            "ledger_close_time": times,
            "base_account": "W",
            "counter_account": counterparties,
            "base_asset": "A:B",
            "counter_asset": "C:D",
            "amount": rng.uniform(10, 1000, size=n),
            "price": 1.0,
        }
    )
    features = compute_hardening_features(df)
    corr = features["cross_wallet_volume_corr"]
    assert -1.0 - 1e-9 <= corr <= 1.0 + 1e-9


def test_hardening_features_empty_dataframe():
    features = compute_hardening_features(pd.DataFrame())
    assert features["inter_arrival_cv"] == 0.0
    assert features["entropy_of_amounts"] == 0.0
    assert features["cross_wallet_volume_corr"] == 0.0


def test_build_feature_matrix_includes_hardening_features():
    """build_feature_matrix must include the three hardening feature columns."""
    df = _sample_trades()
    matrix = build_feature_matrix(df)
    assert "inter_arrival_cv" in matrix.columns
    assert "entropy_of_amounts" in matrix.columns
    assert "cross_wallet_volume_corr" in matrix.columns


def test_build_feature_matrix_accepts_gnn_embedding_features():
    embeddings = {
        "A": {f"gnn_embedding_{i}": float(i) for i in range(64)},
        "B": {f"gnn_embedding_{i}": float(i + 1) for i in range(64)},
    }
    matrix = build_feature_matrix(_sample_trades(), gnn_embeddings=embeddings)
    assert all(f"gnn_embedding_{i}" in matrix.columns for i in range(64))
    assert matrix.loc[matrix["wallet"] == "A", "gnn_embedding_63"].iloc[0] == 63.0


# Needed for approx assertions
import pytest  # noqa: E402 (placed after test functions intentionally for clarity)


# ---------------------------------------------------------------------------
# counterparty_variance — worked example from docs/contributor_feature_guide.md
# ---------------------------------------------------------------------------


def _trades_with_counterparty_volumes(vol_map: dict) -> pd.DataFrame:
    """Build a minimal trades DataFrame with exact per-counterparty volumes."""
    rows = []
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    for i, (cp, vol) in enumerate(vol_map.items()):
        rows.append(
            {
                "ledger_close_time": t0 + pd.Timedelta(minutes=i),
                "base_account": "W",
                "counter_account": cp,
                "amount": vol,
                "base_asset": "USDC:GA5Z",
                "counter_asset": "XLM:native",
            }
        )
    return pd.DataFrame(rows)


def test_counterparty_variance_empty_returns_zero():
    """Empty DataFrame must return 0.0 without raising."""
    features = compute_trade_pattern_features("W", pd.DataFrame())
    assert features["counterparty_variance"] == 0.0


def test_counterparty_variance_equal_volumes_returns_zero():
    """Equal volume per counterparty means zero variance."""
    df = _trades_with_counterparty_volumes({"CP_A": 500.0, "CP_B": 500.0, "CP_C": 500.0})
    features = compute_trade_pattern_features("W", df)
    assert features["counterparty_variance"] == pytest.approx(0.0, abs=1e-9)


def test_counterparty_variance_single_counterparty_returns_zero():
    """Single counterparty — variance is undefined, should return 0.0."""
    df = _trades_with_counterparty_volumes({"CP_A": 1000.0})
    features = compute_trade_pattern_features("W", df)
    assert features["counterparty_variance"] == 0.0


def test_counterparty_variance_known_value():
    """Verify the normalised variance against a hand-computed value.

    vol = [900, 100], mean = 500, var (population) = 160000
    counterparty_variance = 160000 / 500^2 = 0.64
    """
    df = _trades_with_counterparty_volumes({"CP_A": 900.0, "CP_B": 100.0})
    features = compute_trade_pattern_features("W", df)
    assert features["counterparty_variance"] == pytest.approx(0.64, abs=1e-6)


def test_counterparty_variance_always_in_range_property():
    """Hypothesis: counterparty_variance is always in [0.0, 1.0] and finite."""
    import math

    from hypothesis import given, settings
    from hypothesis import strategies as st

    @given(
        n=st.integers(min_value=0, max_value=200),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=200)
    def _property(n, seed):
        rng = np.random.default_rng(seed)
        if n == 0:
            df = pd.DataFrame()
        else:
            times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
            cps = [f"CP_{rng.integers(0, 5)}" for _ in range(n)]
            df = pd.DataFrame(
                {
                    "ledger_close_time": times,
                    "base_account": "W",
                    "counter_account": cps,
                    "amount": rng.uniform(1.0, 10_000.0, n),
                    "base_asset": "USDC:GA5Z",
                    "counter_asset": "XLM:native",
                }
            )
        features = compute_trade_pattern_features("W", df)
        val = features["counterparty_variance"]
        assert 0.0 <= val <= 1.0, f"Out of range [{val}] for n={n} seed={seed}"
        assert math.isfinite(val), f"Non-finite [{val}] for n={n} seed={seed}"

    _property()


def test_build_feature_matrix_includes_counterparty_variance():
    """counterparty_variance must appear in every row of build_feature_matrix."""
    # Build a minimal concrete DataFrame to avoid the factory's Asset hashability issue
    times = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "ledger_close_time": times,
            "base_account": ["W"] * 5 + ["CP_A"] * 5,
            "counter_account": ["CP_A"] * 5 + ["W"] * 5,
            "amount": [100.0, 200.0, 300.0, 400.0, 500.0] * 2,
            "base_asset": "USDC:GA5Z",
            "counter_asset": "XLM:native",
        }
    )
    matrix = build_feature_matrix(df)
    assert "counterparty_variance" in matrix.columns
    assert matrix["counterparty_variance"].notna().all()
    assert (matrix["counterparty_variance"].between(0.0, 1.0)).all()
