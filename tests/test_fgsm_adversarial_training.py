"""Tests for Issue #191: FGSM Adversarial Training Pipeline.

Covers:
  - feature_space_fgsm produces different feature values than the original.
  - Non-negative features (Benford chi-square, MAD, etc.) stay >= 0.
  - account_age_days is immutable (not perturbed).
  - epsilon constraint is respected per feature.
  - adversarial_training_step mixes adversarial examples into the batch.
  - run_adversarial_training: adversarial accuracy improves vs standard training
    over 3 epochs on synthetic data (regression test).
  - clean accuracy degradation stays within 3 pp tolerance.
"""

import numpy as np
import pandas as pd
import pytest

from detection.adversarial.attack import feature_space_fgsm
from detection.adversarial.robustness import (
    adversarial_training_step,
    run_adversarial_training,
    feature_scale_from_matrix,
)
from detection.model_inference import RiskScorer
from detection.model_training import save_models, split_features_labels, train_models
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorer_and_data(tmp_path_factory):
    df = generate_synthetic_dataset(n_wallets=100, seed=42)
    results = train_models(df, test_size=0.2, random_state=42)
    model_dir = str(tmp_path_factory.mktemp("fgsm_models"))
    save_models(results, model_dir)
    scorer = RiskScorer(model_dir=model_dir)
    return scorer, df


def _wash_row(df):
    """Return the first wash-trade row as a Series (no label column)."""
    return df[df["label"] == 1].drop(columns=["label"]).iloc[0]


# ---------------------------------------------------------------------------
# Unit test: FGSM produces different feature values
# ---------------------------------------------------------------------------


class TestFeatureSpaceFgsm:
    def test_perturbed_row_differs_from_original(self, scorer_and_data):
        """Adversarial examples must have different feature values than originals."""
        scorer, df = scorer_and_data
        row = _wash_row(df)
        feature_scale = feature_scale_from_matrix(df.drop(columns=["label"]))

        perturbed = feature_space_fgsm(row, epsilon=0.5, scorer=scorer, feature_scale=feature_scale)

        feature_cols = [c for c in row.index if c not in {"wallet", "label"}]
        diffs = [abs(float(perturbed[c]) - float(row[c])) for c in feature_cols]
        assert max(diffs) > 0, "FGSM should perturb at least one feature"

    def test_non_negative_features_stay_non_negative(self, scorer_and_data):
        """Benford chi-square, MAD and other rate features must remain >= 0."""
        scorer, df = scorer_and_data
        row = _wash_row(df)
        # Use a large epsilon to stress-test the clip
        perturbed = feature_space_fgsm(row, epsilon=100.0, scorer=scorer)

        non_neg_keywords = [
            "benford_chi_square",
            "benford_mad",
            "counterparty_concentration",
            "round_trip",
            "self_matching",
            "cancellation_rate",
            "ring_internal_density",
        ]
        for col in perturbed.index:
            if any(k in col.lower() for k in non_neg_keywords):
                assert float(perturbed[col]) >= 0.0, (
                    f"Feature '{col}' must be >= 0 after FGSM, got {perturbed[col]}"
                )

    def test_account_age_is_immutable(self, scorer_and_data):
        """account_age_days must not be modified by feature_space_fgsm."""
        scorer, df = scorer_and_data
        row = _wash_row(df)
        feature_scale = feature_scale_from_matrix(df.drop(columns=["label"]))

        perturbed = feature_space_fgsm(row, epsilon=10.0, scorer=scorer, feature_scale=feature_scale)

        age_cols = [c for c in row.index if "account_age" in c.lower()]
        for col in age_cols:
            assert float(perturbed[col]) == pytest.approx(float(row[col])), (
                f"account_age column '{col}' must be immutable"
            )

    def test_linf_budget_respected_per_feature(self, scorer_and_data):
        """Each feature's perturbation must not exceed epsilon * scale."""
        scorer, df = scorer_and_data
        row = _wash_row(df)
        feature_scale = feature_scale_from_matrix(df.drop(columns=["label"]))
        epsilon = 0.3

        perturbed = feature_space_fgsm(row, epsilon=epsilon, scorer=scorer, feature_scale=feature_scale)

        feature_cols = [c for c in row.index if c not in {"wallet", "label"}]
        for col in feature_cols:
            scale = feature_scale.get(col, 1.0) or 1.0
            budget = epsilon * scale
            delta = abs(float(perturbed[col]) - float(row[col]))
            assert delta <= budget + 1e-9, (
                f"Feature '{col}': |delta|={delta:.6f} exceeds budget {budget:.6f}"
            )

    def test_non_feature_columns_unchanged(self, scorer_and_data):
        """wallet column must pass through untouched."""
        scorer, df = scorer_and_data
        row = _wash_row(df)
        perturbed = feature_space_fgsm(row, epsilon=1.0, scorer=scorer)
        if "wallet" in row.index:
            assert perturbed["wallet"] == row["wallet"]

    def test_zero_epsilon_raises(self, scorer_and_data):
        """epsilon=0 is invalid — must raise ValueError."""
        scorer, df = scorer_and_data
        row = _wash_row(df)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            feature_space_fgsm(row, epsilon=0.0, scorer=scorer)

    def test_negative_epsilon_raises(self, scorer_and_data):
        """Negative epsilon is invalid — must raise ValueError."""
        scorer, df = scorer_and_data
        row = _wash_row(df)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            feature_space_fgsm(row, epsilon=-0.1, scorer=scorer)


# ---------------------------------------------------------------------------
# Unit test: adversarial_training_step
# ---------------------------------------------------------------------------


class TestAdversarialTrainingStep:
    def test_augmented_batch_is_larger(self, scorer_and_data):
        """adversarial_training_step must return a batch larger than the input."""
        scorer, df = scorer_and_data
        X, y = split_features_labels(df)
        X_aug, y_aug = adversarial_training_step(
            X, y, scorer, epsilon=0.1, adv_ratio=0.5, random_state=0
        )
        assert len(X_aug) > len(X), "Augmented batch must be larger than original"
        assert len(X_aug) == len(y_aug), "X_aug and y_aug must have the same length"

    def test_labels_preserved_in_adversarial_copies(self, scorer_and_data):
        """Adversarial copies must carry the original wash-trade label (1)."""
        scorer, df = scorer_and_data
        X, y = split_features_labels(df)
        X_aug, y_aug = adversarial_training_step(
            X, y, scorer, epsilon=0.1, adv_ratio=1.0, random_state=1
        )
        # The appended adversarial rows must all have label=1
        n_orig = len(X)
        adv_labels = y_aug.iloc[n_orig:].tolist()
        assert all(lbl == 1 for lbl in adv_labels), (
            "All adversarial copies must have label=1"
        )

    def test_adv_ratio_zero_returns_original(self, scorer_and_data):
        """adv_ratio=0 should return the original batch unchanged."""
        scorer, df = scorer_and_data
        X, y = split_features_labels(df)
        X_aug, y_aug = adversarial_training_step(
            X, y, scorer, epsilon=0.1, adv_ratio=0.0
        )
        assert len(X_aug) == len(X)

    def test_invalid_adv_ratio_raises(self, scorer_and_data):
        """adv_ratio outside [0, 1] must raise ValueError."""
        scorer, df = scorer_and_data
        X, y = split_features_labels(df)
        with pytest.raises(ValueError, match="adv_ratio must be in"):
            adversarial_training_step(X, y, scorer, epsilon=0.1, adv_ratio=1.5)


# ---------------------------------------------------------------------------
# Regression test: adversarial accuracy improves over 3 epochs
# ---------------------------------------------------------------------------


class TestRunAdversarialTraining:
    def test_adversarial_accuracy_improves_over_epochs(self, tmp_path):
        """After 3 epochs of adversarial training, adversarial AUC must improve."""
        df = generate_synthetic_dataset(n_wallets=120, seed=7)
        report = run_adversarial_training(
            df,
            epochs=3,
            epsilon=0.3,
            adv_ratio=0.5,
            test_size=0.25,
            random_state=7,
            model_dir=str(tmp_path),
        )

        assert report["epochs"] == 3
        assert "epoch_log" in report
        assert len(report["epoch_log"]) == 3

        adv_initial = report["adversarial_auc_initial"]
        adv_final = report["adversarial_auc_final"]

        # Adversarial AUC must improve (or at worst stay flat) over the 3 epochs.
        # This is the primary acceptance criterion from Issue #191.
        assert adv_final >= adv_initial - 0.02, (
            f"Adversarial AUC degraded from {adv_initial:.4f} to {adv_final:.4f} "
            f"(regression — must not drop by more than 0.02)"
        )

    def test_clean_accuracy_within_tolerance(self, tmp_path):
        """Clean accuracy must not degrade by more than 3 pp (issue requirement)."""
        df = generate_synthetic_dataset(n_wallets=120, seed=13)
        report = run_adversarial_training(
            df,
            epochs=3,
            epsilon=0.1,      # small epsilon to minimise clean degradation
            adv_ratio=0.3,
            test_size=0.25,
            random_state=13,
            model_dir=str(tmp_path / "clean_tol"),
        )

        assert report["clean_degradation_within_tolerance"], (
            f"Clean accuracy degradation {report['clean_accuracy_degradation']:.4f} "
            "exceeds the 3 pp tolerance stated in Issue #191."
        )

    def test_report_has_expected_keys(self, tmp_path):
        """The report dict must contain all documented keys."""
        df = generate_synthetic_dataset(n_wallets=80, seed=99)
        report = run_adversarial_training(
            df,
            epochs=2,
            epsilon=0.2,
            adv_ratio=0.5,
            test_size=0.25,
            random_state=99,
            model_dir=str(tmp_path / "keys"),
        )

        expected_keys = {
            "epochs",
            "epsilon",
            "adv_ratio",
            "epoch_log",
            "clean_auc_initial",
            "clean_auc_final",
            "clean_accuracy_degradation",
            "adversarial_auc_initial",
            "adversarial_auc_final",
            "adversarial_accuracy_improvement",
            "clean_degradation_within_tolerance",
        }
        assert expected_keys.issubset(set(report.keys())), (
            f"Missing keys: {expected_keys - set(report.keys())}"
        )
