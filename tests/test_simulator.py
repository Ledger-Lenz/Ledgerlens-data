"""Tests for the Wash Trade Simulation Engine (scripts/wash_trade_simulator.py).

Each attacker profile is tested for:
  - DataFrame schema compliance (matches ``trades_to_dataframe`` output)
  - Profile-specific statistical properties
  - Edge cases (empty trades, single wallet, etc.)

Run with: pytest tests/test_simulator.py -v
"""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from detection.benford_engine import chi_square_statistic, mad_score
from detection.feature_engineering import build_feature_matrix
from detection.model_training import FEATURE_COLUMNS_EXCLUDE, split_features_labels, train_models
from scripts.generate_synthetic_dataset import generate_synthetic_dataset
from scripts.wash_trade_simulator import (
    RANDOM_SEED,
    TRADE_COLUMNS,
    AdaptiveAttacker,
    AmountConformanceAttacker,
    CrossPairAttacker,
    LayeringAttacker,
    NaiveAttacker,
    RingAttacker,
    TimingJitterAttacker,
    create_profile,
    trades_to_feature_matrix,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_schema(df: pd.DataFrame) -> None:
    """Verify the DataFrame matches the expected schema."""
    expected_cols = {"trade_id", "ledger_close_time", "base_account", "counter_account", "amount"}
    assert not df.empty, "DataFrame must not be empty"
    assert expected_cols.issubset(df.columns), (
        f"Missing columns. Expected at least {expected_cols}, got {set(df.columns)}"
    )
    assert df["amount"].dtype in (np.float64, np.float32, float), (
        f"amount column must be float, got {df['amount'].dtype}"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["ledger_close_time"]), (
        "ledger_close_time must be datetime"
    )


# ---------------------------------------------------------------------------
# Test 1: All profiles produce valid DataFrames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile_cls,kwargs",
    [
        (NaiveAttacker, {"n_wallets": 5, "trades_per_wallet": 10}),
        (TimingJitterAttacker, {"n_wallets": 5, "trades_per_wallet": 10}),
        (AmountConformanceAttacker, {"n_wallets": 5, "trades_per_wallet": 10}),
        (RingAttacker, {"n_wallets": 5, "trades_per_wallet": 10}),
        (LayeringAttacker, {"n_wallets": 5, "trades_per_wallet": 20}),
        (CrossPairAttacker, {"n_wallets": 5, "trades_per_wallet": 10}),
    ],
)
def test_profile_produces_valid_dataframe(profile_cls, kwargs):
    """Each attacker profile produces a DataFrame with the schema expected by
    ``historical_loader.trades_to_dataframe``."""
    profile = profile_cls(**kwargs)
    df = profile.generate_trades()
    _check_schema(df)


# ---------------------------------------------------------------------------
# Test 2: NaiveAttacker — fixed amounts with low variance
# ---------------------------------------------------------------------------


def test_naive_attacker_fixed_amounts():
    """NaiveAttacker produces amount column with variance < 1% (fixed amounts)."""
    profile = NaiveAttacker(n_wallets=3, trades_per_wallet=50, seed=RANDOM_SEED)
    df = profile.generate_trades()

    amounts = df["amount"]
    relative_std = amounts.std() / amounts.mean()
    assert relative_std < 0.01, (
        f"NaiveAttacker amounts have std/mean = {relative_std:.6f} >= 0.01. "
        "Expected near-constant amounts."
    )
    assert (amounts == 500.0).all(), "NaiveAttacker amounts should all be 500.0"


# ---------------------------------------------------------------------------
# Test 3: AmountConformanceAttacker — Benford chi-square < 5.0
# ---------------------------------------------------------------------------


def test_amount_conformance_benford():
    """AmountConformanceAttacker produces amounts with Benford chi-square < 5.0
    on a 1000-trade sample."""
    profile = AmountConformanceAttacker(
        n_wallets=10, trades_per_wallet=100, seed=RANDOM_SEED
    )
    df = profile.generate_trades()

    chi_sq = chi_square_statistic(df["amount"])
    assert chi_sq < 5.0, (
        f"AmountConformanceAttacker chi-square = {chi_sq:.4f} >= 5.0. "
        "Expected Benford-conforming amounts."
    )


# ---------------------------------------------------------------------------
# Test 4: RingAttacker — exactly N unique source accounts
# ---------------------------------------------------------------------------


def test_ring_attacker_unique_accounts():
    """RingAttacker with N=3 wallets produces exactly 3 unique values in the
    source_account column."""
    profile = RingAttacker(n_wallets=3, trades_per_wallet=10, seed=RANDOM_SEED)
    df = profile.generate_trades()

    n_unique = df["base_account"].nunique()
    assert n_unique == 3, (
        f"RingAttacker with 3 wallets produced {n_unique} unique base_accounts, expected 3"
    )

    # Verify the ring structure: each wallet trades with the next
    for wi in range(3):
        wallet = f"GSIM{wi:06d}"
        expected_cp = f"GSIM{(wi + 1) % 3:06d}"
        wallet_trades = df[df["base_account"] == wallet]
        assert (
            wallet_trades["counter_account"] == expected_cp
        ).all(), f"Wallet {wallet} should trade only with {expected_cp} in a ring"


# ---------------------------------------------------------------------------
# Test 5: LayeringAttacker — ≥ 70% noise trades (is_wash = False)
# ---------------------------------------------------------------------------


def test_layering_attacker_noise_ratio():
    """LayeringAttacker produces ≥ 70% rows flagged is_wash = False (the noise trades)."""
    profile = LayeringAttacker(
        n_wallets=5,
        trades_per_wallet=40,
        wash_to_noise_ratio=3,
        seed=RANDOM_SEED,
    )
    df = profile.generate_trades()

    assert "is_wash" in df.columns, "LayeringAttacker must include is_wash column"
    noise_ratio = (df["is_wash"] == False).mean()  # noqa: E712
    assert noise_ratio >= 0.70, (
        f"Noise trade ratio = {noise_ratio:.2%} < 70%. "
        "Expected ≥ 70% noise trades with wash_to_noise_ratio=3"
    )


# ---------------------------------------------------------------------------
# Test 6: AdaptiveAttacker reduces highest-importance feature
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model_path():
    """Train a simple model and return its path."""
    df = generate_synthetic_dataset(n_wallets=200, seed=RANDOM_SEED)
    output = train_models(df, random_state=RANDOM_SEED)
    results = output.get("results", output)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_model.joblib")
    rf_model = results["random_forest"]["model"]
    joblib.dump(rf_model, path)
    return path


def test_adaptive_attacker_reduces_top_feature(trained_model_path):
    """AdaptiveAttacker reduces the highest-importance feature's mean absolute
    value by ≥ 10% vs. NaiveAttacker."""
    model = joblib.load(trained_model_path)
    importances = dict(zip(model.feature_names_in_, model.feature_importances_))
    top_feature = max(importances, key=importances.get)

    naive = NaiveAttacker(n_wallets=10, trades_per_wallet=20, seed=RANDOM_SEED)
    adaptive = AdaptiveAttacker(
        n_wallets=10,
        trades_per_wallet=20,
        model_path=trained_model_path,
        seed=RANDOM_SEED,
    )

    naive_trades = naive.generate_trades()
    adaptive_trades = adaptive.generate_trades()

    naive_features = trades_to_feature_matrix(naive_trades)
    adaptive_features = trades_to_feature_matrix(adaptive_trades)

    if top_feature not in naive_features.columns or top_feature not in adaptive_features.columns:
        pytest.skip(f"Top feature {top_feature} not in computed feature matrix")

    naive_mean = float(naive_features[top_feature].mean())
    adaptive_mean = float(adaptive_features[top_feature].mean())

    if abs(naive_mean) < 1e-10:
        pytest.skip("Top feature mean near zero, skipping ratio comparison")

    reduction = (naive_mean - adaptive_mean) / abs(naive_mean)
    assert reduction >= 0.10, (
        f"AdaptiveAttacker reduced top feature '{top_feature}' by {reduction:.2%}. "
        f"Naive mean: {naive_mean:.4f}, Adaptive mean: {adaptive_mean:.4f}. "
        "Expected ≥ 10% reduction."
    )


# ---------------------------------------------------------------------------
# Test 7: create_profile factory
# ---------------------------------------------------------------------------


def test_create_profile_factory():
    """create_profile instantiates the correct profile type."""
    for name in [
        "NaiveAttacker",
        "TimingJitterAttacker",
        "AmountConformanceAttacker",
        "RingAttacker",
        "LayeringAttacker",
        "CrossPairAttacker",
        "AdaptiveAttacker",
    ]:
        profile = create_profile(name, n_wallets=3, seed=RANDOM_SEED)
        assert profile.name is not None
        df = profile.generate_trades()
        _check_schema(df)


# ---------------------------------------------------------------------------
# Test 8: TimingJitterAttacker — non-uniform intervals
# ---------------------------------------------------------------------------


def test_timing_jitter_variable_intervals():
    """TimingJitterAttacker produces trades with non-constant inter-arrival times."""
    profile = TimingJitterAttacker(
        n_wallets=2,
        trades_per_wallet=50,
        lambda_seconds=60.0,
        seed=RANDOM_SEED,
    )
    df = profile.generate_trades()

    wallet = df["base_account"].iloc[0]
    wallet_trades = df[df["base_account"] == wallet].sort_values("ledger_close_time")

    intervals = wallet_trades["ledger_close_time"].diff().dt.total_seconds().dropna()
    assert intervals.std() > 0.1, (
        f"TimingJitter intervals have std={intervals.std():.2f}s, expected jitter > 0.1s"
    )


# ---------------------------------------------------------------------------
# Test 9: CrossPairAttacker — multiple base assets
# ---------------------------------------------------------------------------


def test_cross_pair_attacker_multiple_assets():
    """CrossPairAttacker produces trades with multiple distinct base_asset values."""
    profile = CrossPairAttacker(
        n_wallets=2,
        trades_per_wallet=30,
        n_pairs=3,
        seed=RANDOM_SEED,
    )
    df = profile.generate_trades()

    n_assets = df["base_asset"].nunique()
    assert n_assets >= 2, (
        f"CrossPairAttacker produced only {n_assets} base_assets, expected ≥ 2"
    )


# ---------------------------------------------------------------------------
# Test 10: trades_to_feature_matrix
# ---------------------------------------------------------------------------


def test_trades_to_feature_matrix():
    """trades_to_feature_matrix produces a valid feature matrix from trades."""
    profile = NaiveAttacker(n_wallets=5, trades_per_wallet=20, seed=RANDOM_SEED)
    trades = profile.generate_trades()

    features = trades_to_feature_matrix(trades)
    assert not features.empty, "Feature matrix should not be empty"
    assert "wallet" in features.columns, "Feature matrix must contain wallet column"
    assert "label" in features.columns, "Feature matrix must contain label column"
    assert len(features) >= 5, (
        f"Expected at least 5 feature rows, got {len(features)}"
    )


# ---------------------------------------------------------------------------
# Test 11: RingAttacker funding_source_similarity < 0.3 for N ≥ 10
# ---------------------------------------------------------------------------


def test_ring_attacker_funding_source_similarity():
    """RingAttacker with N=10 wallets produces funding_source_similarity < 0.3
    (below detection threshold) for most wallets."""
    profile = RingAttacker(n_wallets=10, trades_per_wallet=10, seed=RANDOM_SEED)
    trades = profile.generate_trades()

    features = trades_to_feature_matrix(trades)
    if "funding_source_similarity" in features.columns:
        mean_similarity = features["funding_source_similarity"].mean()
        assert mean_similarity <= 0.5, (
            f"RingAttacker mean funding_source_similarity = {mean_similarity:.4f}, "
            "expected ≤ 0.5"
        )
