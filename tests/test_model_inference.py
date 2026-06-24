"""Tests for detection/model_inference.py — BFT voting and RiskScorer."""

import pytest

from detection.model_inference import (
    RiskScorer,
    _has_consensus,
    bft_trimmed_mean,
)
from detection.model_training import save_models, train_models
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    df = generate_synthetic_dataset(n_wallets=60, seed=2)
    output = train_models(df, test_size=0.3, random_state=2)
    model_dir = str(tmp_path_factory.mktemp("models"))
    save_models(output["results"], model_dir)
    return output, model_dir, df


# ---------------------------------------------------------------------------
# BFT trimmed mean
# ---------------------------------------------------------------------------


def test_bft_trimmed_mean_median_of_three():
    score, diverged = bft_trimmed_mean([20.0, 50.0, 80.0])
    assert score == 50.0
    assert diverged is True  # |80-20| = 60 > default threshold 30


def test_bft_trimmed_mean_no_divergence():
    score, diverged = bft_trimmed_mean([40.0, 45.0, 50.0])
    # span = 10 < 30; median = 45
    assert score == 45.0
    assert diverged is False


def test_bft_divergence_flag_raised_when_span_exceeds_threshold():
    _, diverged = bft_trimmed_mean([0.0, 50.0, 100.0])
    assert diverged is True


def test_bft_trimmed_mean_single_value():
    score, diverged = bft_trimmed_mean([77.0])
    assert score == 77.0
    assert diverged is False


# ---------------------------------------------------------------------------
# Consensus check
# ---------------------------------------------------------------------------


def test_consensus_failure_when_no_two_models_agree():
    # Scores spread 40 points apart — no two within 10
    assert _has_consensus([0.0, 40.0, 80.0]) is False


def test_consensus_passes_when_two_models_agree():
    assert _has_consensus([45.0, 50.0, 90.0]) is True


# ---------------------------------------------------------------------------
# RiskScorer integration
# ---------------------------------------------------------------------------


def test_risk_scorer_score_returns_contract_shape(trained_models):
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)
    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)

    assert "score" in result
    assert "benford_flag" in result
    assert "ml_flag" in result
    assert "confidence" in result
    assert 0 <= result["score"] <= 100
    assert 0 <= result["confidence"] <= 100


def test_risk_scorer_score_matrix(trained_models):
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)
    features = df.drop(columns=["label"])
    scored = scorer.score_matrix(features)

    assert "wallet" in scored.columns
    assert "score" in scored.columns
    assert len(scored) == len(features)


def test_risk_scorer_raises_without_models(tmp_path):
    scorer = RiskScorer(model_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        scorer.score(
            generate_synthetic_dataset(n_wallets=2, seed=3).drop(columns=["label"]).iloc[0]
        )


def test_bft_divergence_key_present_when_flagged(trained_models, monkeypatch):
    """Patch model outputs to force divergence and verify bft_divergence=True."""
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)

    # Monkey-patch models to return known divergent probabilities
    class FakeModel:
        def __init__(self, prob):
            self.prob = prob

        def predict_proba(self, X):
            return [[1 - self.prob, self.prob]]

    scorer.models = {
        "random_forest": FakeModel(0.1),  # score=10
        "xgboost": FakeModel(0.5),  # score=50
        "lightgbm": FakeModel(0.9),  # score=90  — span=80>30
    }

    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)
    assert result.get("bft_divergence") is True


def test_bft_prometheus_counter_incremented_on_divergence(trained_models, monkeypatch):
    import detection.model_inference as mi

    counter_calls = []

    monkeypatch.setattr(mi, "_increment_bft_counter", lambda: counter_calls.append(1))

    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)

    class FakeModel:
        def __init__(self, prob):
            self.prob = prob

        def predict_proba(self, X):
            return [[1 - self.prob, self.prob]]

    scorer.models = {
        "random_forest": FakeModel(0.1),
        "xgboost": FakeModel(0.5),
        "lightgbm": FakeModel(0.9),
    }

    row = df.drop(columns=["label"]).iloc[0]
    scorer.score(row)
    assert len(counter_calls) == 1


def test_risk_scorer_default_weights_none_preserves_bft_behavior(trained_models):
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)
    assert scorer.weights is None

    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)
    assert "calibrated" not in result


def test_risk_scorer_weighted_mode_returns_calibrated_score(trained_models):
    _, model_dir, df = trained_models
    scorer = RiskScorer(
        model_dir=model_dir,
        weights={"random_forest": 0.5, "xgboost": 0.3, "lightgbm": 0.2},
    )
    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)

    assert result["calibrated"] is True
    assert 0 <= result["score"] <= 100
    assert "consensus_failure" not in result


def test_risk_scorer_weights_must_sum_to_one(trained_models):
    with pytest.raises(ValueError):
        RiskScorer(weights={"random_forest": 0.5, "xgboost": 0.5, "lightgbm": 0.5})


def test_risk_scorer_weighted_mode_rejects_unknown_model_names(trained_models):
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir, weights={"random_forest": 1.0})
    scorer.weights = {"not_a_real_model": 1.0}

    row = df.drop(columns=["label"]).iloc[0]
    with pytest.raises(ValueError):
        scorer.score(row)


def test_consensus_failure_score(trained_models, monkeypatch):
    """When no two models agree, score must be 100 and consensus_failure=True."""
    _, model_dir, df = trained_models
    scorer = RiskScorer(model_dir=model_dir)

    class FakeModel:
        def __init__(self, prob):
            self.prob = prob

        def predict_proba(self, X):
            return [[1 - self.prob, self.prob]]

    scorer.models = {
        "random_forest": FakeModel(0.0),  # 0
        "xgboost": FakeModel(0.4),  # 40
        "lightgbm": FakeModel(0.85),  # 85
    }

    row = df.drop(columns=["label"]).iloc[0]
    result = scorer.score(row)
    assert result["consensus_failure"] is True
    assert result["score"] == 100
    assert result["confidence"] == 0
