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
