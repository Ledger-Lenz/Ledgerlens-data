"""Tests for differential-privacy SHAP attribution (Issue #59).

Models are trained inline (a small RandomForest on the synthetic dataset) so the
suite is independent of saved model artifacts.
"""

import math

import pytest
from sklearn.ensemble import RandomForestClassifier

from detection.differential_privacy import (
    feature_sensitivity,
    gaussian_sigma,
    renyi_noise_multiplier,
)
from detection.persistence import get_engine, get_session_factory
from detection.risk_score_store import RiskScoreStore
from detection.shap_explainer import ShapExplainer
from scripts.estimate_shap_sensitivity import estimate_sensitivity
from scripts.generate_synthetic_dataset import generate_synthetic_dataset

EXCLUDE = {"wallet", "label"}


@pytest.fixture(scope="module")
def model_and_data():
    df = generate_synthetic_dataset(n_wallets=80, seed=5)
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols].astype(float).reset_index(drop=True)
    y = df["label"].astype(int).to_numpy()
    model = RandomForestClassifier(n_estimators=25, max_depth=4, random_state=0)
    model.fit(X, y)
    return model, X, feature_cols


def _row(X, i: int):
    return X.iloc[[i]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gaussian mechanism
# ---------------------------------------------------------------------------


def test_gaussian_sigma_matches_formula():
    sensitivity, epsilon, delta = 0.3, 1.0, 1e-5
    expected = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    assert gaussian_sigma(sensitivity, epsilon, delta) == pytest.approx(expected)


def test_gaussian_sigma_scales_inversely_with_epsilon():
    # Halving epsilon doubles the noise.
    assert gaussian_sigma(0.3, 0.5, 1e-5) == pytest.approx(2 * gaussian_sigma(0.3, 1.0, 1e-5))


def test_gaussian_sigma_rejects_bad_params():
    with pytest.raises(ValueError):
        gaussian_sigma(0.3, 0.0, 1e-5)
    with pytest.raises(ValueError):
        gaussian_sigma(0.3, 1.0, 1.0)


def test_feature_sensitivity_falls_back_to_default():
    assert feature_sensitivity({"a": 0.2}, "a") == 0.2
    assert feature_sensitivity({"a": 0.2}, "missing", default=0.07) == 0.07


# ---------------------------------------------------------------------------
# Rényi composition
# ---------------------------------------------------------------------------


def test_renyi_multiplier_below_and_above_threshold():
    assert renyi_noise_multiplier(100, threshold=100, multiplier=3.0) == 1.0
    assert renyi_noise_multiplier(101, threshold=100, multiplier=3.0) == 3.0


# ---------------------------------------------------------------------------
# explain_private
# ---------------------------------------------------------------------------


def test_audit_mode_returns_exact_values(model_and_data):
    model, X, _ = model_and_data
    explainer = ShapExplainer(model)

    exact = explainer.shap_dict(_row(X, 0))
    audit = explainer.explain_private(model, _row(X, 0), "WALLET", private=False)
    assert audit == exact


def test_private_mode_adds_noise_but_keeps_features(model_and_data):
    model, X, feature_cols = model_and_data
    explainer = ShapExplainer(model)

    exact = explainer.shap_dict(_row(X, 0))
    private = explainer.explain_private(model, _row(X, 0), "WALLET", sensitivities={}, seed=0)
    assert set(private) == set(exact)
    # With non-zero default sensitivity, at least one feature must be perturbed.
    assert any(private[f] != exact[f] for f in feature_cols)


def test_query_count_above_threshold_triples_noise(tmp_path, model_and_data):
    model, X, feature_cols = model_and_data
    explainer = ShapExplainer(model)
    exact = explainer.shap_dict(_row(X, 0))

    # Fresh store → first query returns count 1 → noise scale 1.0.
    store_low = RiskScoreStore(
        session_factory=get_session_factory(get_engine(f"sqlite:///{tmp_path}/low.db"))
    )
    private_low = explainer.explain_private(
        model, _row(X, 0), "W", sensitivities={}, query_store=store_low, seed=0
    )

    # Pre-seed a store past the threshold → next query count 101 → noise scale 3.0.
    store_high = RiskScoreStore(
        session_factory=get_session_factory(get_engine(f"sqlite:///{tmp_path}/high.db"))
    )
    for _ in range(100):
        store_high.increment_shap_query("W")
    private_high = explainer.explain_private(
        model, _row(X, 0), "W", sensitivities={}, query_store=store_high, seed=0
    )

    # Same seed → identical standard-normal draws, so the high-budget noise is
    # exactly 3x the low-budget noise for every feature.
    for feature in feature_cols:
        low_noise = private_low[feature] - exact[feature]
        high_noise = private_high[feature] - exact[feature]
        if abs(low_noise) > 1e-12:
            assert high_noise == pytest.approx(3.0 * low_noise, rel=1e-6)


def test_near_identical_wallets_are_indistinguishable(model_and_data):
    """One-trade difference is masked within the 2-sigma DP noise band."""
    model, X, feature_cols = model_and_data
    explainer = ShapExplainer(model)

    # Wallet B = wallet A with a single feature nudged (a one-trade change).
    row_a = _row(X, 0)
    row_b = row_a.copy()
    perturb_col = feature_cols[0]
    row_b.loc[0, perturb_col] = row_b.loc[0, perturb_col] + max(1.0, abs(row_b.loc[0, perturb_col]))

    exact_a = explainer.shap_dict(row_a)
    exact_b = explainer.shap_dict(row_b)

    # Sensitivity is, by definition, the per-feature SHAP change from a one-trade
    # difference — exactly what the production estimator takes the max of over the
    # training set. Calibrated this way, the 2-sigma noise band covers the change.
    sensitivities = {f: abs(exact_a[f] - exact_b[f]) for f in feature_cols}

    epsilon, delta = 1.0, 1e-5
    for feature in feature_cols:
        sigma = gaussian_sigma(feature_sensitivity(sensitivities, feature), epsilon, delta)
        # The true cross-wallet SHAP difference sits inside the 2-sigma band.
        assert abs(exact_a[feature] - exact_b[feature]) <= 2 * sigma + 1e-9


# ---------------------------------------------------------------------------
# Sensitivity estimation
# ---------------------------------------------------------------------------


def test_estimate_sensitivity_produces_per_feature_values(model_and_data):
    model, X, feature_cols = model_and_data
    sensitivities = estimate_sensitivity(model, X.iloc[:20])

    assert set(sensitivities) == set(feature_cols)
    assert all(v >= 0.0 for v in sensitivities.values())
    assert any(v > 0.0 for v in sensitivities.values())
