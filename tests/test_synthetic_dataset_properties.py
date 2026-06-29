"""Property-based tests for scripts.generate_synthetic_dataset using Hypothesis."""

from hypothesis import given, settings
from hypothesis import strategies as st

from config import config
from scripts.generate_synthetic_dataset import _generate_feature_level

# Build the full expected feature column list (mirrors _generate_feature_level)
_BENFORD_COLS = [
    f"benford_{metric}_{h}h"
    for h in config.BENFORD_WINDOWS_HOURS
    for metric in ("chi_square", "mad", "z_max")
] + [
    f"benford_residual_{metric}_{h}h"
    for h in config.BENFORD_WINDOWS_HOURS
    for metric in ("chi_square", "mad")
]
_SCALAR_COLS = [
    "counterparty_concentration_ratio",
    "round_trip_frequency",
    "net_roundtrip_ratio",
    "self_matching_rate",
    "order_cancellation_rate",
    "volume_per_counterparty_ratio",
    "intra_minute_clustering",
    "off_hours_activity_ratio",
    "volume_spike_frequency",
    "funding_source_similarity",
    "network_centrality",
    "account_age_days",
    "cross_pair_trade_synchrony",
    "net_asset_flow_deviation",
    "cross_pair_counterparty_overlap",
    "cross_pair_volume_correlation",
    "pair_diversity_score",
    "cross_pair_mad_std",
    "inter_arrival_cv",
    "entropy_of_amounts",
    "cross_wallet_volume_corr",
]
REQUIRED_FEATURE_COLS = _BENFORD_COLS + _SCALAR_COLS


# ── strategies ───────────────────────────────────────────────────────────────

n_wallets_st = st.integers(min_value=2, max_value=100)
seed_st = st.integers(min_value=0, max_value=2**31 - 1)
wash_noise_st = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
wash_offset_st = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make(n_wallets=10, seed=42, wash_noise=1.0, wash_offset=0.0):
    return _generate_feature_level(
        n_wallets=n_wallets, seed=seed, wash_noise=wash_noise, wash_offset=wash_offset
    )


# ── property tests ────────────────────────────────────────────────────────────


@given(n_wallets=n_wallets_st, seed=seed_st)
@settings(max_examples=50)
def test_row_count_equals_n_wallets(n_wallets, seed):
    """Output always has exactly n_wallets rows."""
    df = _make(n_wallets=n_wallets, seed=seed)
    assert len(df) == n_wallets


@given(n_wallets=n_wallets_st, seed=seed_st)
@settings(max_examples=50)
def test_label_is_binary(n_wallets, seed):
    """Label column contains only 0 or 1."""
    df = _make(n_wallets=n_wallets, seed=seed)
    assert set(df["label"].unique()).issubset({0, 1})


@given(n_wallets=st.integers(min_value=20, max_value=200), seed=seed_st)
@settings(max_examples=50)
def test_naive_attacker_roughly_half_labelled_wash(n_wallets, seed):
    """~50 % of rows are label=1 (±5 %) for the NaiveAttacker path."""
    df = _make(n_wallets=n_wallets, seed=seed)
    ratio = df["label"].mean()
    assert 0.45 <= ratio <= 0.55


@given(n_wallets=n_wallets_st, seed=seed_st, wash_noise=wash_noise_st, wash_offset=wash_offset_st)
@settings(max_examples=50)
def test_no_nans_in_numeric_columns(n_wallets, seed, wash_noise, wash_offset):
    """No NaN values in any numeric feature column."""
    df = _make(n_wallets=n_wallets, seed=seed, wash_noise=wash_noise, wash_offset=wash_offset)
    numeric = df.select_dtypes(include="number")
    assert not numeric.isnull().any().any()


@given(n_wallets=n_wallets_st, seed=seed_st)
@settings(max_examples=30)
def test_wash_noise_zero_produces_zero_wash_features(n_wallets, seed):
    """wash_noise=0.0 sets all wash-trading feature values to 0."""
    df = _make(n_wallets=n_wallets, seed=seed, wash_noise=0.0)
    wash_rows = df[df["label"] == 1]
    # Benford features for wash rows are wash_noise * rand + wash_offset;
    # with wash_noise=0 and wash_offset=0 the result is 0.
    for col in _BENFORD_COLS:
        assert (wash_rows[col] == 0.0).all(), f"{col} should be 0 when wash_noise=0"


@given(n_wallets=n_wallets_st, seed=seed_st, wash_noise=wash_noise_st, wash_offset=wash_offset_st)
@settings(max_examples=50)
def test_required_feature_columns_present(n_wallets, seed, wash_noise, wash_offset):
    """All required feature columns are present regardless of parameters."""
    df = _make(n_wallets=n_wallets, seed=seed, wash_noise=wash_noise, wash_offset=wash_offset)
    missing = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
    assert missing == [], f"Missing columns: {missing}"


@given(n_wallets=st.integers(min_value=10, max_value=100), seed=seed_st)
@settings(max_examples=50)
def test_concentration_features_higher_for_wash_rows(n_wallets, seed):
    """Wash rows have higher average concentration ratio than legit rows."""
    df = _make(n_wallets=n_wallets, seed=seed)
    wash = df[df["label"] == 1]["counterparty_concentration_ratio"].mean()
    legit = df[df["label"] == 0]["counterparty_concentration_ratio"].mean()
    assert wash > legit


@given(n_wallets=st.integers(min_value=10, max_value=100), seed=seed_st)
@settings(max_examples=50)
def test_benford_mad_higher_for_wash_rows(n_wallets, seed):
    """Wash rows have higher average Benford MAD than legit rows (first window)."""
    first_window = config.BENFORD_WINDOWS_HOURS[0]
    col = f"benford_mad_{first_window}h"
    df = _make(n_wallets=n_wallets, seed=seed)
    assert df[df["label"] == 1][col].mean() > df[df["label"] == 0][col].mean()
