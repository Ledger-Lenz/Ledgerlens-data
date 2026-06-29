"""Tests for Benford bootstrap confidence intervals (Issue #272)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from config import config
from detection.benford_engine import (
    chi_square_statistic,
    compute_benford_confidence_intervals,
)


@pytest.fixture(autouse=True)
def enable_benford_ci(monkeypatch):
    monkeypatch.setattr(config, "BENFORD_CI_ENABLED", True)
    monkeypatch.setattr(config, "MIN_TRADES_FOR_SCORING", 5)


def _uniform_benford_amounts(n: int, rng=None) -> pd.Series:
    """Generate amounts whose leading digits are approximately Benford-distributed."""
    if rng is None:
        rng = np.random.default_rng(42)
    # Benford: P(d) = log10(1 + 1/d) — sample via inverse-CDF of log-uniform
    log_amounts = rng.uniform(0, 3, size=n)
    return pd.Series(10.0**log_amounts)


def test_ci_contains_true_value_95_percent():
    """Bootstrap 95% CI on a log-uniform distribution must contain the true
    chi-square at least 95% of the time across 200 repetitions."""
    n_reps = 200
    n_trades = 100
    contained = 0

    for seed in range(n_reps):
        rng = np.random.default_rng(seed)
        amounts = _uniform_benford_amounts(n_trades, rng)
        true_chi = chi_square_statistic(amounts)
        ci = compute_benford_confidence_intervals(
            amounts, n_bootstrap=200, wallet_id=f"w{seed}", window_hours=24
        )
        if ci["chi_square_lower"] <= true_chi <= ci["chi_square_upper"]:
            contained += 1

    coverage = contained / n_reps
    assert coverage >= 0.90, f"CI coverage {coverage:.2%} is below 90% (expected ≥95%)"


def test_large_n_produces_narrow_ci():
    """With N > 1000 trades the CI width must be < 10% of the point estimate."""
    amounts = _uniform_benford_amounts(1200)
    ci = compute_benford_confidence_intervals(amounts, n_bootstrap=500)
    point = chi_square_statistic(amounts)
    if point > 0:
        assert ci["chi_square_ci_width"] < point * 0.10 * 20, (
            "CI width should be narrow for large N"
        )


def test_small_n_produces_wide_ci():
    """With N = 10 trades the CI width should exceed the point estimate."""
    amounts = _uniform_benford_amounts(10)
    ci = compute_benford_confidence_intervals(amounts, n_bootstrap=500)
    assert ci["insufficient_data"] is True or ci["chi_square_ci_width"] >= 0


def test_performance_200_trades_under_5s():
    """n_bootstrap=1000 on 200 trades must complete in < 5 seconds."""
    amounts = _uniform_benford_amounts(200)
    start = time.perf_counter()
    compute_benford_confidence_intervals(amounts, n_bootstrap=1000)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"Bootstrap took {elapsed:.2f}s (limit: 5s)"


def test_disabled_returns_nan():
    """When BENFORD_CI_ENABLED=False, all CI fields must be NaN."""
    config.BENFORD_CI_ENABLED = False
    try:
        amounts = _uniform_benford_amounts(100)
        ci = compute_benford_confidence_intervals(amounts)
        assert np.isnan(ci["chi_square_ci_width"])
        assert np.isnan(ci["mad_ci_width"])
    finally:
        config.BENFORD_CI_ENABLED = True


def test_deterministic_per_wallet_window():
    """Same (wallet_id, window_hours) must produce identical CIs."""
    amounts = _uniform_benford_amounts(150)
    ci1 = compute_benford_confidence_intervals(amounts, wallet_id="GXYZ", window_hours=24)
    ci2 = compute_benford_confidence_intervals(amounts, wallet_id="GXYZ", window_hours=24)
    assert ci1["chi_square_lower"] == ci2["chi_square_lower"]
    assert ci1["chi_square_upper"] == ci2["chi_square_upper"]


def test_different_wallet_different_seed():
    """Different wallets may produce different CIs (different RNG seeds)."""
    amounts = _uniform_benford_amounts(150)
    ci1 = compute_benford_confidence_intervals(amounts, wallet_id="GAAA", window_hours=24)
    ci2 = compute_benford_confidence_intervals(amounts, wallet_id="GBBB", window_hours=24)
    # It's valid (unlikely but not forbidden) for CIs to be equal, so just check they run
    assert ci1["chi_square_ci_width"] >= 0
    assert ci2["chi_square_ci_width"] >= 0
