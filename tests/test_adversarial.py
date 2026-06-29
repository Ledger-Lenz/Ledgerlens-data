import pytest

from detection.adversarial.attack import FGSMAttack, PGDAttack
from detection.adversarial.augmentation import (
    evaluate_augmentation,
    generate_adversarial_examples,
)
from detection.adversarial.robustness import (
    attack_success_rate,
    evaluate_robustness,
    feature_scale_from_matrix,
    high_scoring_wallets,
    minimum_epsilon_per_feature,
    most_vulnerable_features,
)
from detection.model_inference import RiskScorer
from detection.model_training import save_models, train_models
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


@pytest.fixture(scope="module")
def scorer_and_data(tmp_path_factory):
    df = generate_synthetic_dataset(n_wallets=80, seed=7)
    results = train_models(df, test_size=0.3, random_state=7)
    model_dir = str(tmp_path_factory.mktemp("adv_models"))
    save_models(results, model_dir)
    return RiskScorer(model_dir=model_dir), df


def _wash_row(df):
    return df[df["label"] == 1].drop(columns=["label"]).iloc[0]


def test_score_continuous_is_unrounded_and_in_range(scorer_and_data):
    scorer, df = scorer_and_data
    row = df.drop(columns=["label"]).iloc[0]
    value = scorer.score_continuous(row)
    assert 0.0 <= value <= 100.0
    # Continuous score should agree with the rounded contract score.
    assert round(value) == scorer.score(row)["score"]


def test_pgd_reduces_score_of_wash_wallet(scorer_and_data):
    scorer, df = scorer_and_data
    row = _wash_row(df)
    scale = feature_scale_from_matrix(df.drop(columns=["label"]))

    before = scorer.score_continuous(row)
    attack = PGDAttack(scorer, epsilon=1.0, steps=40, step_size=0.1, feature_scale=scale)
    perturbed = attack.perturb(row, target_score=40)
    after = scorer.score_continuous(perturbed)

    assert after < before


def test_fgsm_does_not_exceed_epsilon_budget(scorer_and_data):
    scorer, df = scorer_and_data
    row = _wash_row(df)
    scale = feature_scale_from_matrix(df.drop(columns=["label"]))

    epsilon = 0.5
    attack = FGSMAttack(scorer, epsilon=epsilon, feature_scale=scale)
    perturbed = attack.perturb(row)

    feature_cols = [c for c in row.index if c not in {"wallet", "label"}]
    for col in feature_cols:
        budget = epsilon * scale[col]
        assert abs(perturbed[col] - row[col]) <= budget + 1e-9


def test_pgd_preserves_non_feature_columns(scorer_and_data):
    scorer, df = scorer_and_data
    row = _wash_row(df)
    attack = PGDAttack(scorer, epsilon=0.5, steps=5, step_size=0.1)
    perturbed = attack.perturb(row)
    assert perturbed["wallet"] == row["wallet"]


def test_attack_success_rate_shape(scorer_and_data):
    scorer, df = scorer_and_data
    wash = df[df["label"] == 1].drop(columns=["label"]).head(5)
    scale = feature_scale_from_matrix(df.drop(columns=["label"]))
    attack = PGDAttack(scorer, epsilon=1.0, steps=20, step_size=0.1, feature_scale=scale)

    result = attack_success_rate(scorer, wash, attack, target_score=40)
    assert set(result) == {"total", "successes", "success_rate", "target_score", "rows"}
    assert result["total"] == len(wash)
    assert 0.0 <= result["success_rate"] <= 1.0


def test_high_scoring_wallets_filters_by_score(scorer_and_data):
    scorer, df = scorer_and_data
    features = df.drop(columns=["label"])
    cohort = high_scoring_wallets(scorer, features, high_score=80)
    assert len(cohort) <= len(features)
    for _, row in cohort.iterrows():
        assert scorer.score_continuous(row) >= 80


def test_minimum_epsilon_per_feature_keys(scorer_and_data):
    scorer, df = scorer_and_data
    row = _wash_row(df)
    scale = feature_scale_from_matrix(df.drop(columns=["label"]))
    attack = PGDAttack(scorer, epsilon=3.0, steps=40, step_size=0.3, feature_scale=scale)
    result = minimum_epsilon_per_feature(scorer, row, scale, attack, target_score=40)
    feature_cols = [c for c in row.index if c not in {"wallet", "label"}]
    assert set(result) == set(feature_cols)
    for eps in result.values():
        assert eps is None or eps >= 0


def test_most_vulnerable_features_ranked(scorer_and_data):
    scorer, df = scorer_and_data
    wash = df[df["label"] == 1].drop(columns=["label"]).head(4)
    scale = feature_scale_from_matrix(df.drop(columns=["label"]))
    attack = PGDAttack(scorer, epsilon=3.0, steps=40, step_size=0.3, feature_scale=scale)
    ranked = most_vulnerable_features(scorer, wash, scale, attack, target_score=40, top_n=3)
    assert len(ranked) <= 3
    for entry in ranked:
        assert set(entry) == {"feature", "perturbation_rate", "mean_epsilon", "min_epsilon"}
    # Sorted by perturbation rate descending.
    rates = [e["perturbation_rate"] for e in ranked]
    assert rates == sorted(rates, reverse=True)


def test_evaluate_robustness_report_shape(scorer_and_data):
    scorer, df = scorer_and_data
    wash = df[df["label"] == 1].drop(columns=["label"])
    # epsilon=3 reliably evades, so the vulnerable-feature path is populated.
    report = evaluate_robustness(scorer, wash, epsilon=3.0, steps=40, top_n=3, vuln_sample=6)
    assert set(report) == {
        "config",
        "cohort_size",
        "total_wallets",
        "fgsm",
        "pgd",
        "most_vulnerable_features",
    }
    assert 0.0 <= report["pgd"]["success_rate"] <= 1.0
    assert report["pgd"]["success_rate"] >= 0.8  # acceptance criterion: >= 80%
    assert len(report["most_vulnerable_features"]) >= 1
    for entry in report["most_vulnerable_features"]:
        assert set(entry) == {"feature", "perturbation_rate", "mean_epsilon", "min_epsilon"}


def test_generate_adversarial_examples_preserves_labels(scorer_and_data):
    scorer, df = scorer_and_data
    adv = generate_adversarial_examples(scorer, df, epsilon=0.5, steps=10, only_label=1)
    assert (adv["label"] == 1).all()
    assert list(adv.columns) == list(df.columns)


def test_evaluate_augmentation_reports_auc(tmp_path):
    df = generate_synthetic_dataset(n_wallets=120, seed=11)
    report = evaluate_augmentation(
        df, epsilon=1.0, steps=15, test_size=0.3, random_state=11, model_dir=str(tmp_path)
    )
    assert set(report) == {
        "feature_count",
        "baseline_adversarial_auc",
        "augmented_adversarial_auc",
        "absolute_improvement",
        "relative_improvement",
        "adversarial_examples_added",
    }
    assert report["adversarial_examples_added"] > 0
