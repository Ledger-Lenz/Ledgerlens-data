"""Tests for detection/active_learning/query_strategies.py."""

import numpy as np
import pandas as pd
import pytest

from detection.active_learning.query_strategies import (
    BADGE,
    STRATEGY_REGISTRY,
    CommitteeDisagreement,
    CoreSet,
    Entropy,
    LeastConfidence,
    MarginSampling,
    get_strategy,
)

# ---------------------------------------------------------------------------
# Fake model helpers
# ---------------------------------------------------------------------------


class FakeModel:
    """Model that returns fixed class-1 probabilities."""

    def __init__(self, probs: list[float]):
        self._probs = np.array(probs)

    def predict_proba(self, X):
        n = len(X)
        p = self._probs[:n] if len(self._probs) >= n else np.tile(self._probs, n)[:n]
        return np.column_stack([1 - p, p])


def _pool(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, 4)), columns=["f1", "f2", "f3", "f4"])
    df.insert(0, "wallet", [f"W{i:03d}" for i in range(n)])
    return df


# ---------------------------------------------------------------------------
# LeastConfidence
# ---------------------------------------------------------------------------


def test_least_confidence_selects_most_uncertain():
    pool = _pool(5)
    # Assign known probabilities: wallet W002 is most uncertain (closest to 0.5)
    probs = [0.95, 0.9, 0.51, 0.8, 0.85]
    model = FakeModel(probs)
    result = LeastConfidence().select(pool, n_query=1, model=model)
    assert result == ["W002"]


def test_least_confidence_returns_exactly_n_query():
    pool = _pool(10)
    model = FakeModel([0.6] * 10)
    result = LeastConfidence().select(pool, n_query=3, model=model)
    assert len(result) == 3


def test_least_confidence_pool_smaller_than_n_query():
    pool = _pool(3)
    model = FakeModel([0.7, 0.6, 0.5])
    result = LeastConfidence().select(pool, n_query=10, model=model)
    assert len(result) == 3


def test_least_confidence_requires_model():
    with pytest.raises(ValueError):
        LeastConfidence().select(_pool(3), n_query=2)


# ---------------------------------------------------------------------------
# MarginSampling
# ---------------------------------------------------------------------------


def test_margin_returns_n_query():
    pool = _pool(8)
    model = FakeModel([0.55, 0.9, 0.51, 0.8, 0.85, 0.6, 0.7, 0.75])
    result = MarginSampling().select(pool, n_query=3, model=model)
    assert len(result) == 3


def test_margin_pool_smaller_than_n_query():
    pool = _pool(2)
    model = FakeModel([0.6, 0.7])
    assert len(MarginSampling().select(pool, n_query=10, model=model)) == 2


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def test_entropy_selects_n_query():
    pool = _pool(6)
    model = FakeModel([0.5, 0.9, 0.1, 0.55, 0.8, 0.6])
    result = Entropy().select(pool, n_query=2, model=model)
    assert len(result) == 2


def test_entropy_most_uncertain_selected():
    """Wallet with prob 0.5 has highest entropy."""
    pool = _pool(3)
    probs = [0.5, 0.9, 0.1]
    model = FakeModel(probs)
    result = Entropy().select(pool, n_query=1, model=model)
    assert result == ["W000"]


def test_entropy_pool_smaller_than_n_query():
    pool = _pool(2)
    model = FakeModel([0.5, 0.6])
    assert len(Entropy().select(pool, n_query=100, model=model)) == 2


# ---------------------------------------------------------------------------
# CoreSet
# ---------------------------------------------------------------------------


def test_coreset_returns_n_query():
    pool = _pool(8, seed=1)
    result = CoreSet().select(pool, n_query=3)
    assert len(result) == 3


def test_coreset_all_unique():
    pool = _pool(10)
    result = CoreSet().select(pool, n_query=5)
    assert len(result) == len(set(result))


def test_coreset_pool_smaller_than_n_query():
    pool = _pool(3)
    result = CoreSet().select(pool, n_query=10)
    assert len(result) == 3


def test_coreset_maximises_min_distance():
    """In a 1D space, coreset should pick spread-out points."""
    df = pd.DataFrame(
        {
            "wallet": ["A", "B", "C", "D"],
            "f1": [0.0, 0.01, 0.5, 1.0],
        }
    )
    labelled = pd.DataFrame({"wallet": ["A"], "f1": [0.0]})
    result = CoreSet().select(df, n_query=2, labelled_pool=labelled)
    # Furthest from A=0 is D=1.0; then furthest from {A, D} is C=0.5
    assert "D" in result


# ---------------------------------------------------------------------------
# BADGE
# ---------------------------------------------------------------------------


def test_badge_returns_n_query():
    pool = _pool(8, seed=2)
    model = FakeModel([0.5] * 8)
    result = BADGE().select(pool, n_query=3, model=model)
    assert len(result) == 3


def test_badge_pool_smaller_than_n_query():
    pool = _pool(2)
    model = FakeModel([0.5, 0.6])
    assert len(BADGE().select(pool, n_query=10, model=model)) == 2


# ---------------------------------------------------------------------------
# CommitteeDisagreement
# ---------------------------------------------------------------------------


def test_committee_disagreement_returns_n_query():
    pool = _pool(6)
    models = {
        "rf": FakeModel([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        "xgb": FakeModel([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "lgbm": FakeModel([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    }
    result = CommitteeDisagreement().select(pool, n_query=3, models=models)
    assert len(result) == 3


def test_committee_selects_high_variance_wallets():
    """Constructed test: RF=0.9, XGB=0.1, LGBM=0.5 → high variance for wallet W000."""
    pool = _pool(3)
    models = {
        "rf": FakeModel([0.9, 0.55, 0.52]),
        "xgb": FakeModel([0.1, 0.53, 0.51]),
        "lgbm": FakeModel([0.5, 0.54, 0.50]),
    }
    result = CommitteeDisagreement().select(pool, n_query=1, models=models)
    assert result == ["W000"]


def test_committee_disagreement_pool_smaller_than_n_query():
    pool = _pool(2)
    models = {"rf": FakeModel([0.9, 0.1]), "xgb": FakeModel([0.1, 0.9])}
    result = CommitteeDisagreement().select(pool, n_query=10, models=models)
    assert len(result) == 2


def test_committee_requires_model_or_models():
    with pytest.raises(ValueError):
        CommitteeDisagreement().select(_pool(3), n_query=2)


# ---------------------------------------------------------------------------
# Registry / get_strategy
# ---------------------------------------------------------------------------


def test_strategy_registry_has_all_six():
    expected = {
        "least_confidence",
        "margin",
        "entropy",
        "coreset",
        "badge",
        "committee_disagreement",
        "coreset_hybrid",
    }
    assert expected == set(STRATEGY_REGISTRY.keys())


def test_get_strategy_returns_correct_instance():
    assert isinstance(get_strategy("entropy"), Entropy)
    assert isinstance(get_strategy("committee_disagreement"), CommitteeDisagreement)


def test_get_strategy_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown query strategy"):
        get_strategy("nonexistent")
