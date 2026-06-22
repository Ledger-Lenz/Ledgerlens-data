"""Tests for the adversarial robustness framework (`detection/adversarial`)."""

import math

import numpy as np
import pytest

from detection.adversarial.attack import (
    EnsembleScoreFunction,
    FGSMAttack,
    PGDAttack,
    feature_scale_from_matrix,
)
from detection.adversarial.augmentation import adversarial_augmentation_gain
from detection.adversarial.evaluate import (
    AdversarialEvaluator,
    evaluate_attack,
    most_vulnerable_features,
)
from detection.model_training import FEATURE_COLUMNS_EXCLUDE, train_models
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


@pytest.fixture(scope="module")
def trained():
    df = generate_synthetic_dataset(n_wallets=120, seed=7)
    results = train_models(df, test_size=0.3, random_state=7)
    models = {name: result["model"] for name, result in results.items()}
    feature_columns = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    score_fn = EnsembleScoreFunction(models, feature_columns)
    scale = feature_scale_from_matrix(df[feature_columns])
    return df, models, feature_columns, score_fn, scale


def _high_score_wash_rows(df, feature_columns, score_fn, threshold=80.0):
    wash = df[df["label"] == 1]
    X = wash[feature_columns].to_numpy(dtype=float)
    scores = score_fn.score_batch(X)
    return wash.loc[scores >= threshold]


def test_score_function_range_and_batch(trained):
    df, _, feature_columns, score_fn, _ = trained
    X = df[feature_columns].to_numpy(dtype=float)

    scores = score_fn.score_batch(X)
    assert scores.shape == (len(df),)
    assert scores.min() >= 0.0 and scores.max() <= 100.0

    single = score_fn(X[0])
    assert isinstance(single, float)
    assert single == pytest.approx(scores[0])


def test_score_function_requires_models(trained):
    _, _, feature_columns, _, _ = trained
    with pytest.raises(ValueError):
        EnsembleScoreFunction({}, feature_columns)


def test_gradient_shape_and_immutable_mask(trained):
    _, _, feature_columns, score_fn, scale = trained
    n = len(feature_columns)
    mask = np.ones(n)
    mask[0] = 0.0
    attack = PGDAttack(score_fn, epsilon=2.0, feature_scale=scale, mutable_mask=mask)

    x0 = np.zeros(n)
    grad = attack.gradient(x0)
    assert grad.shape == (n,)
    # Immutable feature never receives gradient signal.
    assert grad[0] == 0.0


def test_pgd_reduces_score_within_budget(trained):
    df, _, feature_columns, score_fn, scale = trained
    rows = _high_score_wash_rows(df, feature_columns, score_fn)
    assert len(rows) > 0, "expected some strongly-flagged wash wallets"

    attack = PGDAttack(score_fn, epsilon=4.0, steps=40, step_size=0.5, feature_scale=scale)
    x0 = rows[feature_columns].to_numpy(dtype=float)[0]
    clean = score_fn(x0)
    adv = attack.perturb(x0, target_score=40.0)
    adv_score = score_fn(adv)

    # Attack lowers the score and stays inside the L-infinity epsilon ball.
    assert adv_score <= clean
    linf = np.max(np.abs((adv - x0) / scale))
    assert linf <= attack.epsilon + 1e-6


def test_fgsm_respects_epsilon_ball(trained):
    df, _, feature_columns, score_fn, scale = trained
    rows = _high_score_wash_rows(df, feature_columns, score_fn)
    attack = FGSMAttack(score_fn, epsilon=1.5, feature_scale=scale)

    x0 = rows[feature_columns].to_numpy(dtype=float)[0]
    adv = attack.perturb(x0)
    linf = np.max(np.abs((adv - x0) / scale))
    assert linf <= attack.epsilon + 1e-6


def test_evaluate_attack_summary(trained):
    df, _, feature_columns, score_fn, scale = trained
    rows = _high_score_wash_rows(df, feature_columns, score_fn).head(10)
    attack = PGDAttack(score_fn, epsilon=5.0, steps=40, step_size=0.5, feature_scale=scale)

    summary = evaluate_attack(attack, rows, target_score=40.0)
    assert summary.n_attacked == len(rows)
    assert 0.0 <= summary.success_rate <= 1.0
    assert len(summary.outcomes) == len(rows)
    # On well-separated synthetic data a generous budget should fool most wallets.
    assert summary.success_rate >= 0.5
    assert summary.mean_adversarial_score <= summary.mean_clean_score


def test_evaluator_steps_to_success_bounded(trained):
    df, _, feature_columns, score_fn, scale = trained
    rows = _high_score_wash_rows(df, feature_columns, score_fn).head(8)
    attack = PGDAttack(score_fn, epsilon=5.0, steps=40, step_size=0.5, feature_scale=scale)

    summary = evaluate_attack(attack, rows, target_score=40.0)
    for outcome in summary.outcomes:
        if outcome.success:
            assert outcome.steps is not None
            assert 0 <= outcome.steps <= 40


def test_feature_vulnerability_and_ranking(trained):
    df, _, feature_columns, score_fn, scale = trained
    rows = _high_score_wash_rows(df, feature_columns, score_fn).head(6)
    evaluator = AdversarialEvaluator(score_fn, feature_scale=scale, target_score=40.0)

    per_feature, vulnerable = evaluator.feature_vulnerability(
        rows, max_epsilon=5.0, n_iter=16, top_n=5
    )
    assert set(per_feature) == set(feature_columns)
    for value in per_feature.values():
        assert value >= 0.0  # epsilon is non-negative (may be inf)

    assert len(vulnerable) == 5
    epsilons = [v["mean_min_epsilon"] for v in vulnerable]
    assert epsilons == sorted(epsilons)  # ascending = most vulnerable first


def test_most_vulnerable_features_sorts_inf_last():
    ranked = most_vulnerable_features({"a": 2.0, "b": math.inf, "c": 0.5}, top_n=3)
    assert [r["feature"] for r in ranked] == ["c", "a", "b"]


def test_adversarial_augmentation_returns_aucs(trained):
    df, _, _, _, _ = trained
    result = adversarial_augmentation_gain(
        df,
        lambda fn, sc: PGDAttack(fn, epsilon=3.0, steps=20, step_size=0.5, feature_scale=sc),
        target_score=40.0,
        test_size=0.3,
        random_state=7,
    )
    assert set(result) >= {"baseline_auc", "augmented_auc", "gain", "n_adversarial_train"}
    assert 0.0 <= result["baseline_auc"] <= 1.0
    assert 0.0 <= result["augmented_auc"] <= 1.0
    assert result["gain"] == pytest.approx(result["augmented_auc"] - result["baseline_auc"])
    assert result["n_adversarial_train"] > 0
