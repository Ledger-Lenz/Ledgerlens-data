"""Adversarial training (data augmentation) for ensemble robustness.

Implements the adversarial-augmentation half of the robustness study: train
a baseline ensemble, measure its AUC-ROC on an *attacked* test set, then
retrain with adversarial examples mixed into the training data and measure
the AUC-ROC gain on the same attacked test set.

Acceptance target: augmentation lifts AUC-ROC on the perturbed test set by
>= 5 percentage points (Madry et al., 2018).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from detection.adversarial.attack import (
    EnsembleScoreFunction,
    GradientAttack,
    feature_scale_from_matrix,
)
from detection.model_training import FEATURE_COLUMNS_EXCLUDE, MODEL_REGISTRY

# A factory turns a score function + per-feature scale into a configured attack,
# so the same attack settings can be rebuilt against the retrained ensemble.
AttackFactory = Callable[[EnsembleScoreFunction, np.ndarray], GradientAttack]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]


def _fit_ensemble(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict:
    """Fit every model in `MODEL_REGISTRY` on SMOTE-balanced data.

    Mirrors `model_training.train_models` but without the internal split, so
    the caller controls train/test partitioning and adversarial augmentation.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    models = {}
    for name, model_cls in MODEL_REGISTRY.items():
        model = model_cls(random_state=random_state)
        model.fit(X_res, y_res)
        models[name] = model
    return models


def _ensemble_proba(models: dict, X: pd.DataFrame) -> np.ndarray:
    probs = np.zeros(len(X), dtype=float)
    for model in models.values():
        probs += model.predict_proba(X)[:, 1]
    return probs / len(models)


def generate_adversarial_examples(
    attack: GradientAttack,
    feature_rows: pd.DataFrame,
    *,
    target_score: float = 40.0,
) -> pd.DataFrame:
    """Perturb each row in `feature_rows`, returning a matching feature matrix.

    Column order/labels follow the attack's `feature_columns`. Used both to
    build the attacked test set and the augmented training rows.
    """
    feature_columns = attack.feature_columns
    X = feature_rows.reindex(columns=feature_columns).to_numpy(dtype=float)
    perturbed = np.vstack([attack.perturb(x0, target_score=target_score) for x0 in X])
    return pd.DataFrame(perturbed, columns=feature_columns, index=feature_rows.index)


def adversarial_augmentation_gain(
    df: pd.DataFrame,
    attack_factory: AttackFactory,
    *,
    target_score: float = 40.0,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """Measure the AUC-ROC robustness gain from adversarial augmentation.

    Steps:
      1. Split `df` (a labelled feature matrix) into train/test, stratified.
      2. Train a baseline ensemble on the clean training split.
      3. Attack the positive (wash, label==1) rows of *both* splits using an
         attack built against the baseline ensemble.
      4. Build a perturbed test set (positives replaced with their adversarial
         versions) and record the baseline ensemble's AUC-ROC on it.
      5. Retrain the ensemble on the training split augmented with the attacked
         positive rows (kept as label==1), and record its AUC-ROC on the *same*
         perturbed test set.

    Returns ``{baseline_auc, augmented_auc, gain, n_adversarial_train, ...}``.
    """
    feature_columns = _feature_columns(df)
    X = df[feature_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    baseline_models = _fit_ensemble(X_train, y_train, random_state=random_state)

    scale = feature_scale_from_matrix(X_train)
    score_fn = EnsembleScoreFunction(baseline_models, feature_columns)
    attack = attack_factory(score_fn, scale)

    # Perturbed test set: positives swapped for their adversarial versions.
    test_pos_mask = (y_test == 1).to_numpy()
    X_test_pos = X_test.loc[test_pos_mask]
    X_test_adv = X_test.copy()
    if len(X_test_pos):
        adv_test = generate_adversarial_examples(attack, X_test_pos, target_score=target_score)
        X_test_adv.loc[test_pos_mask] = adv_test.to_numpy()

    baseline_auc = float(roc_auc_score(y_test, _ensemble_proba(baseline_models, X_test_adv)))

    # Augment training data with attacked positives (still wash -> label 1).
    X_train_pos = X_train.loc[(y_train == 1).to_numpy()]
    adv_train = generate_adversarial_examples(attack, X_train_pos, target_score=target_score)
    X_train_aug = pd.concat([X_train, adv_train], ignore_index=True)
    y_train_aug = pd.concat(
        [y_train, pd.Series(np.ones(len(adv_train), dtype=int))], ignore_index=True
    )

    augmented_models = _fit_ensemble(X_train_aug, y_train_aug, random_state=random_state)
    augmented_auc = float(roc_auc_score(y_test, _ensemble_proba(augmented_models, X_test_adv)))

    return {
        "baseline_auc": baseline_auc,
        "augmented_auc": augmented_auc,
        "gain": augmented_auc - baseline_auc,
        "n_adversarial_train": int(len(adv_train)),
        "n_test_positives_attacked": int(len(X_test_pos)),
    }
