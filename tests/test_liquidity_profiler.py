"""Unit tests for AssetLiquidityProfiler (issue #57).

Acceptance criteria covered:
  - Single-asset regime assignment (AC-3, AC-4)
  - Calibrated chi-square returns 0 for a distribution that exactly matches
    the baseline (AC-6)
  - fit() is reproducible: same input → same cluster assignments (AC-3)
  - Calibrated features emitted alongside raw features; old columns preserved (AC-4)
  - Fallback to theoretical Benford for unknown assets
"""

import numpy as np
import pandas as pd
import pytest

from detection.feature_engineering import compute_benford_features
from detection.liquidity_profiler import AssetLiquidityProfiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(n: int, seed: int = 0, price: float = 0.1) -> pd.DataFrame:
    """Return a minimal trade DataFrame with Benford-ish amounts."""
    rng = np.random.default_rng(seed)
    amounts = 10 ** rng.uniform(0, 4, size=n)
    times = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "ledger_close_time": times.astype(str),
            "amount": amounts,
            "price": price + rng.normal(0, 0.001, size=n),
            "base_account": "A",
            "counter_account": "B",
        }
    )


def _make_wash_trades(n: int) -> pd.DataFrame:
    """Trades with fixed lot sizes — non-Benford distribution."""
    amounts = [500.0] * (n // 3) + [5000.0] * (n // 3) + [50000.0] * (n - 2 * (n // 3))
    times = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "ledger_close_time": times.astype(str),
            "amount": amounts,
            "price": 0.1,
            "base_account": "A",
            "counter_account": "B",
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_asset_regime_assignment():
    """A profiler fit on one asset should assign that asset to a valid regime."""
    profiler = AssetLiquidityProfiler(n_clusters=2)
    histories = {"XLM:native": _make_trades(200)}
    profiler.fit(histories)

    regime = profiler.get_regime_id("XLM:native")
    assert regime in (0, 1)


def test_unknown_asset_returns_minus_one():
    profiler = AssetLiquidityProfiler(n_clusters=2)
    profiler.fit({"XLM:native": _make_trades(100)})
    assert profiler.get_regime_id("MYSTERY:GABC") == -1


def test_calibrated_chi_square_zero_for_matching_distribution():
    """When a sample's digit distribution exactly matches the regime baseline,
    calibrated chi-square must return 0.

    Strategy: fit on amounts all starting with digit 1, so the regime baseline
    is {1: 1.0, 2-9: 0.0}.  A test series also starting entirely with 1 then
    produces observed == expected for the only non-zero baseline bucket (d=1),
    so chi-square = 0.
    """
    # All amounts in [1.0, 2.0) → leading digit is always 1
    all_digit_one = pd.Series([1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9] * 200)

    profiler = AssetLiquidityProfiler(n_clusters=1)
    profiler.fit({"ASSET:native": pd.DataFrame({"amount": all_digit_one})})

    # Test series: same property (all leading digit 1)
    test_series = pd.Series([1.0, 1.1, 1.5, 1.9] * 100)
    chi = profiler.calibrated_chi_square(test_series, "ASSET:native")
    assert chi == pytest.approx(0.0, abs=1e-6)


def test_fit_is_reproducible():
    """Same input must produce the same cluster assignments on every call."""
    histories = {f"ASSET{i}": _make_trades(100, seed=i) for i in range(6)}
    profiler_a = AssetLiquidityProfiler(n_clusters=3)
    profiler_b = AssetLiquidityProfiler(n_clusters=3)
    profiler_a.fit(histories)
    profiler_b.fit(histories)

    for asset in histories:
        assert profiler_a.get_regime_id(asset) == profiler_b.get_regime_id(asset)


def test_calibrated_mad_lower_than_raw_for_market_maker():
    """A market-maker with fixed lot sizes should have lower calibrated MAD
    (vs. its regime baseline) than raw MAD (vs. theoretical Benford)."""
    mm_trades = _make_wash_trades(300)
    # Fit profiler on market-maker trades as the 'clean' regime
    profiler = AssetLiquidityProfiler(n_clusters=1)
    profiler.fit({"MM:issuer": mm_trades})

    # New batch of market-maker trades
    new_trades = _make_wash_trades(300)
    from detection.benford_engine import mad_score

    raw = mad_score(new_trades["amount"])
    cal = profiler.calibrated_mad(new_trades["amount"], "MM:issuer")
    assert cal < raw


def test_fallback_to_theoretical_benford_for_unknown_asset():
    """calibrated_chi_square on an unknown asset must behave like the standard
    chi-square (both use the theoretical Benford distribution as baseline)."""
    from detection.benford_engine import chi_square_statistic

    profiler = AssetLiquidityProfiler(n_clusters=2)
    profiler.fit({"XLM:native": _make_trades(200)})

    rng = np.random.default_rng(7)
    amounts = pd.Series(10 ** rng.uniform(0, 3, size=500))

    cal = profiler.calibrated_chi_square(amounts, "UNKNOWN:GABC")
    raw = chi_square_statistic(amounts)
    assert cal == pytest.approx(raw, rel=1e-6)


def test_compute_benford_features_emits_calibrated_columns():
    """compute_benford_features with a profiler must emit calibrated columns
    alongside the raw columns (backward compatibility preserved)."""
    trades = _make_trades(200)
    profiler = AssetLiquidityProfiler(n_clusters=1)
    profiler.fit({"XLM:native": trades})

    features = compute_benford_features(
        trades, decompose=False, liquidity_profiler=profiler, asset="XLM:native"
    )

    from config import config

    for h in config.BENFORD_WINDOWS_HOURS:
        assert f"benford_chi_square_{h}h" in features, f"missing raw chi {h}h"
        assert f"benford_mad_{h}h" in features, f"missing raw mad {h}h"
        assert f"benford_calibrated_chi_{h}h" in features, f"missing cal chi {h}h"
        assert f"benford_calibrated_mad_{h}h" in features, f"missing cal mad {h}h"

    assert "benford_regime_id" in features
    assert "benford_regime_baseline_mad" in features
    assert "benford_deviation_from_regime" in features


def test_compute_benford_features_without_profiler_unchanged():
    """Without a profiler, compute_benford_features must not emit calibrated keys."""
    trades = _make_trades(100)
    features = compute_benford_features(trades, decompose=False)
    assert not any("calibrated" in k for k in features)
    assert "benford_regime_id" not in features


def test_empty_trades_calibrated_features_default():
    """Empty trade frame must not raise and calibrated chi/mad must be 0."""
    profiler = AssetLiquidityProfiler(n_clusters=1)
    profiler.fit({"XLM:native": _make_trades(100)})

    features = compute_benford_features(
        pd.DataFrame(), decompose=False, liquidity_profiler=profiler, asset="XLM:native"
    )
    from config import config

    for h in config.BENFORD_WINDOWS_HOURS:
        assert features.get(f"benford_calibrated_chi_{h}h", None) == pytest.approx(0.0)
        assert features.get(f"benford_calibrated_mad_{h}h", None) == pytest.approx(0.0)
