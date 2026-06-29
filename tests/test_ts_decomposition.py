"""Tests for detection/ts_decomposition.py (issue #61).

Verifies that STL decomposition strips seasonal components from trade
amount series before Benford scoring:
  - to_amount_time_series: bins trades into 1-minute sums
  - detect_dominant_period: FFT periodogram detects 4h and 24h cycles
  - decompose_amounts: STL succeeds on synthetic data within 500 ms
  - decompose_trade_amounts: full pipeline returns residuals or None
  - compute_benford_features: residual features added when decompose=True,
    raw features preserved alongside them
"""

import time

import numpy as np
import pandas as pd
import pytest

from detection.benford_engine import mad_score
from detection.feature_engineering import compute_benford_features
from detection.ts_decomposition import (
    decompose_amounts,
    decompose_trade_amounts,
    detect_dominant_period,
    to_amount_time_series,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(n: int, start: str = "2024-01-01", freq: str = "1min") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    times = pd.date_range(start, periods=n, freq=freq)
    amounts = 10 ** rng.uniform(0, 4, size=n)
    return pd.DataFrame(
        {
            "ledger_close_time": times.astype(str),
            "amount": amounts,
            "base_account": "GA",
            "counter_account": "GB",
        }
    )


def _make_sinusoidal_trades(n: int, period_bins: int = 24, freq: str = "1min") -> pd.DataFrame:
    """Trades with a strong sinusoidal seasonal component in amounts."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=n, freq=freq)
    t = np.arange(n)
    seasonal = 5000 * np.sin(2 * np.pi * t / period_bins) + 5001
    noise = rng.uniform(1, 10, size=n)
    return pd.DataFrame(
        {
            "ledger_close_time": times.astype(str),
            "amount": seasonal + noise,
            "base_account": "GA",
            "counter_account": "GB",
        }
    )


# ---------------------------------------------------------------------------
# to_amount_time_series
# ---------------------------------------------------------------------------


def test_to_amount_time_series_bins_correctly():
    trades = _make_trades(10)
    series = to_amount_time_series(trades)
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    assert (series >= 0).all()


def test_to_amount_time_series_sums_within_bin():
    times = pd.date_range("2024-01-01", periods=4, freq="30s")
    df = pd.DataFrame(
        {
            "ledger_close_time": times.astype(str),
            "amount": [10.0, 20.0, 30.0, 40.0],
            "base_account": "GA",
            "counter_account": "GB",
        }
    )
    series = to_amount_time_series(df, freq="1min")
    assert series.iloc[0] == pytest.approx(30.0)  # 10+20 in first minute
    assert series.iloc[1] == pytest.approx(70.0)  # 30+40 in second minute


def test_to_amount_time_series_empty():
    assert to_amount_time_series(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# detect_dominant_period
# ---------------------------------------------------------------------------


def test_detect_dominant_period_finds_4h_cycle():
    n = 1440  # 24 hours of 1-min bins
    period = 240  # 4 hours
    t = np.arange(n)
    series = pd.Series(np.sin(2 * np.pi * t / period))
    detected = detect_dominant_period(series)
    assert detected is not None
    assert abs(detected - period) <= 5


def test_detect_dominant_period_finds_24h_cycle():
    n = 4320  # 3 days of 1-min bins
    period = 1440  # 24 hours
    t = np.arange(n)
    series = pd.Series(np.sin(2 * np.pi * t / period))
    detected = detect_dominant_period(series)
    assert detected is not None
    assert abs(detected - period) <= 20


def test_detect_dominant_period_returns_none_for_flat():
    assert detect_dominant_period(pd.Series([1.0] * 100)) is None


def test_detect_dominant_period_returns_none_for_short():
    assert detect_dominant_period(pd.Series([1.0, 2.0, 3.0])) is None


# ---------------------------------------------------------------------------
# decompose_amounts
# ---------------------------------------------------------------------------


def test_decompose_amounts_returns_trend_seasonal_resid():
    n = 200
    t = np.arange(n)
    rng = np.random.default_rng(1)
    series = pd.Series(np.sin(2 * np.pi * t / 24) + 10 + rng.normal(0, 0.1, n))
    result = decompose_amounts(series, period=24)
    assert hasattr(result, "resid")
    assert hasattr(result, "seasonal")
    assert hasattr(result, "trend")
    assert len(result.resid) == n


def test_decompose_amounts_raises_for_insufficient_data():
    with pytest.raises(ValueError, match="observations"):
        decompose_amounts(pd.Series([1.0, 2.0, 3.0]), period=24)


def test_stl_performance_720h():
    """STL on a 720-point series (proxy for 720h at 1h freq) must finish < 500 ms."""
    n = 720
    t = np.arange(n)
    rng = np.random.default_rng(2)
    series = pd.Series(np.sin(2 * np.pi * t / 24) * 1000 + 5000 + rng.normal(0, 10, n))
    start = time.perf_counter()
    result = decompose_amounts(series, period=24)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"STL took {elapsed:.3f}s, expected < 0.5s"
    assert len(result.resid) == n


# ---------------------------------------------------------------------------
# decompose_trade_amounts
# ---------------------------------------------------------------------------


def test_decompose_trade_amounts_none_for_short_series():
    assert decompose_trade_amounts(_make_trades(10)) is None


def test_decompose_trade_amounts_returns_series_or_none():
    trades = _make_trades(200, freq="1min")
    result = decompose_trade_amounts(trades)
    assert result is None or isinstance(result, pd.Series)


def test_decompose_trade_amounts_sinusoidal_returns_residuals():
    """Sinusoidal trades with sufficient obs should produce non-None residuals."""
    period_bins = 24
    trades = _make_sinusoidal_trades(n=300, period_bins=period_bins)
    result = decompose_trade_amounts(trades)
    assert result is not None
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_sinusoidal_residuals_better_benford_conformity():
    """After STL, residuals of a sinusoidal series should conform better to Benford."""
    period_bins = 24
    trades = _make_sinusoidal_trades(n=300, period_bins=period_bins)
    residuals = decompose_trade_amounts(trades)
    if residuals is None:
        pytest.skip("Insufficient data for STL in this configuration")

    raw_mad = mad_score(trades["amount"])

    pos_residuals = residuals.abs()
    pos_residuals = pos_residuals[pos_residuals > 0]
    residual_mad = mad_score(pos_residuals)

    assert (
        residual_mad <= raw_mad
    ), f"Residual MAD ({residual_mad:.4f}) should be <= raw MAD ({raw_mad:.4f})"


# ---------------------------------------------------------------------------
# compute_benford_features integration
# ---------------------------------------------------------------------------


def test_compute_benford_features_has_residual_keys_when_decompose_true():
    trades = _make_trades(200)
    features = compute_benford_features(trades, decompose=True)
    assert any("benford_chi_square" in k for k in features)
    assert any("benford_mad" in k for k in features)
    assert any("benford_residual_chi_square" in k for k in features)
    assert any("benford_residual_mad" in k for k in features)


def test_compute_benford_features_preserves_raw_keys_alongside_residual():
    trades = _make_trades(200)
    raw_keys = set(compute_benford_features(trades, decompose=False))
    decomposed_keys = set(compute_benford_features(trades, decompose=True))
    assert raw_keys.issubset(decomposed_keys), f"Missing raw keys: {raw_keys - decomposed_keys}"


def test_compute_benford_features_no_residual_keys_when_decompose_false():
    trades = _make_trades(200)
    features = compute_benford_features(trades, decompose=False)
    assert not any("residual" in k for k in features)


def test_compute_benford_features_nan_residuals_for_short_data():
    """Short trade window → STL impossible → residual features are NaN."""
    trades = _make_trades(5)
    features = compute_benford_features(trades, decompose=True)
    residual_keys = [k for k in features if "residual" in k]
    assert len(residual_keys) > 0
    for k in residual_keys:
        assert (
            features[k] != features[k] or features[k] is None or np.isnan(features[k])
        ), f"Expected NaN for {k} with insufficient data, got {features[k]}"
