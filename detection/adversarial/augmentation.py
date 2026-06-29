"""Adversarial training augmentation for the detection ensemble.

Implements the "adversarial augmentation improves robustness" acceptance
criterion: generate PGD-perturbed copies of the wash wallets, append them to
the training set with their *true* label kept (still wash trading), retrain
the ensemble, and measure the AUC-ROC gain on a held-out *adversarial* test
set.

This complements `scripts/adversarial_training_loop.py` (referenced in the
issue) by providing the reusable, testable building blocks: dataset
augmentation, adversarial-test-set construction, and the before/after
AUC-ROC comparison.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

from detection.adversarial.attack import PGDAttack
from detection.adversarial.robustness import feature_scale_from_matrix
from detection.model_inference import RiskScorer
from detection.model_training import (
    FEATURE_COLUMNS_EXCLUDE,
    save_models,
    split_features_labels,
    train_models,
)
from utils.logging import get_logger

logger = get_logger(__name__)


def generate_adversarial_examples(
    scorer: RiskScorer,
    df: pd.DataFrame,
    *,
    epsilon: float = 3.0,
    steps: int = 40,
    target_score: float = 40,
    only_label: int | None = 1,
) -> pd.DataFrame:
    """Return PGD-perturbed copies of `df`'s rows, labels preserved.

    By default only label-1 (wash) rows are perturbed (`only_label=1`), since
    those are the examples an adversary would try to disguise. Pass
    `only_label=None` to perturb every row.
    """
    feature_scale = feature_scale_from_matrix(df)
    attack = PGDAttack(
        scorer, epsilon=epsilon, steps=steps, step_size=epsilon / 10, feature_scale=feature_scale
    )

    source = df if only_label is None else df[df["label"] == only_label]
    perturbed_rows = []
    for _, row in source.iterrows():
        perturbed = attack.perturb(row, target_score=target_score)
        perturbed["label"] = row["label"]
        if "wallet" in row.index:
            perturbed["wallet"] = f"{row['wallet']}_adv"
        perturbed_rows.append(perturbed)

    if not perturbed_rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(perturbed_rows)[df.columns]


def _auc_on_adversarial_set(scorer: RiskScorer, adv_df: pd.DataFrame) -> float:
    """AUC-ROC of `scorer` on an adversarial set (mixed labels required)."""
    _, y = split_features_labels(adv_df)
    if y.nunique() < 2:
        # AUC is undefined with a single class; signal with NaN.
        return float("nan")
    scores = adv_df.apply(scorer.score_continuous, axis=1).to_numpy()
    return float(roc_auc_score(y, scores))


def evaluate_augmentation(
    df: pd.DataFrame,
    *,
    epsilon: float = 3.0,
    steps: int = 40,
    target_score: float = 40,
    test_size: float = 0.3,
    random_state: int = 42,
    model_dir: str | None = None,
) -> dict:
    """Train baseline vs. adversarially-augmented ensembles and compare their
    AUC-ROC on a held-out adversarial test set.

    Steps:
      1. Split `df` into train/test by wallet rows.
      2. Train a baseline ensemble on the clean training split.
      3. Build an adversarial test set from the clean test split (PGD on the
         baseline scorer) and measure baseline AUC-ROC on it.
      4. Augment the training split with adversarial examples, retrain, and
         measure the augmented AUC-ROC on the *same* adversarial test set.

    Returns a report dict with both AUC-ROC values and the absolute /
    relative improvement.
    """
    from sklearn.model_selection import train_test_split

    feature_cols = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    # 1. Baseline ensemble on clean data.
    baseline_results = train_models(train_df, test_size=test_size, random_state=random_state)
    baseline_dir = _persist(baseline_results, model_dir, suffix="baseline")
    baseline_scorer = RiskScorer(model_dir=baseline_dir)

    # 2. Adversarial test set built against the baseline scorer.
    adv_test = generate_adversarial_examples(
        baseline_scorer,
        test_df,
        epsilon=epsilon,
        steps=steps,
        target_score=target_score,
        only_label=1,
    )
    # Keep the clean negatives so AUC is well-defined (two classes present).
    adv_eval_set = pd.concat([test_df[test_df["label"] == 0], adv_test], ignore_index=True)
    baseline_auc = _auc_on_adversarial_set(baseline_scorer, adv_eval_set)

    # 3. Augment training data with adversarial examples and retrain.
    adv_train = generate_adversarial_examples(
        baseline_scorer,
        train_df,
        epsilon=epsilon,
        steps=steps,
        target_score=target_score,
        only_label=1,
    )
    augmented_train = pd.concat([train_df, adv_train], ignore_index=True)
    augmented_results = train_models(
        augmented_train, test_size=test_size, random_state=random_state
    )
    augmented_dir = _persist(augmented_results, model_dir, suffix="augmented")
    augmented_scorer = RiskScorer(model_dir=augmented_dir)

    augmented_auc = _auc_on_adversarial_set(augmented_scorer, adv_eval_set)

    improvement = augmented_auc - baseline_auc
    rel = (improvement / baseline_auc) if baseline_auc else float("nan")
    logger.info(
        "Adversarial AUC-ROC: baseline=%.4f augmented=%.4f (+%.4f, %.1f%%)",
        baseline_auc,
        augmented_auc,
        improvement,
        100 * rel,
    )

    return {
        "feature_count": len(feature_cols),
        "baseline_adversarial_auc": baseline_auc,
        "augmented_adversarial_auc": augmented_auc,
        "absolute_improvement": improvement,
        "relative_improvement": rel,
        "adversarial_examples_added": len(adv_train),
    }


def _persist(results: dict, model_dir: str | None, suffix: str) -> str:
    """Persist trained models under a per-variant subdirectory and return it.

    Falls back to a system temp dir when `model_dir` is not supplied so the
    helper stays side-effect free relative to the configured `MODEL_DIR`.
    """
    import os
    import tempfile

    base = model_dir or tempfile.mkdtemp(prefix="ledgerlens_adv_")
    target = os.path.join(base, suffix)
    save_models(results, target)
    return target
