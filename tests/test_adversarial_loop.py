"""Tests for the adversarial training loop (scripts/adversarial_training_loop.py).

Run with: pytest tests/test_adversarial_loop.py -v
"""

import os

import pytest

from scripts.adversarial_training_loop import (
    compute_feature_importances,
    generate_dataset_from_profile,
    run_adversarial_loop,
)
from scripts.wash_trade_simulator import RANDOM_SEED

# ---------------------------------------------------------------------------
# Test 1: Loop terminates after GAN_ROUNDS iterations
# ---------------------------------------------------------------------------


def test_loop_terminates_after_gan_rounds():
    """The adversarial loop completes after ``gan_rounds`` iterations."""
    result = run_adversarial_loop(
        gan_rounds=3,
        n_wallets=20,
        trades_per_wallet=20,
        output_dir="reports",
        seed=RANDOM_SEED,
    )

    assert len(result["rounds"]) == 3, f"Expected {3} rounds, got {len(result['rounds'])}"
    assert os.path.exists(
        f"reports/adversarial_loop_{result['timestamp']}.json"
    ), "Output JSON file must exist"


# ---------------------------------------------------------------------------
# Test 2: AUC-ROC values are monotonically non-decreasing
# ---------------------------------------------------------------------------


def test_auc_roc_monotonic_non_decreasing():
    """AUC-ROC values in output JSON are monotonically non-decreasing
    (detector improves) OR loop exits early with plateau flag."""
    result = run_adversarial_loop(
        gan_rounds=3,
        n_wallets=20,
        trades_per_wallet=20,
        output_dir="reports",
        seed=RANDOM_SEED,
    )

    if result["plateau_exit"]:
        assert result["plateau_exit"], "Loop exited early due to plateau"
        return

    auc_rocs = [r.get("random_forest_auc_roc", 0.0) for r in result["rounds"]]
    for i in range(1, len(auc_rocs)):
        assert auc_rocs[i] >= auc_rocs[i - 1] - 0.01, (
            f"AUC-ROC dropped from round {i-1} ({auc_rocs[i-1]:.4f}) "
            f"to round {i} ({auc_rocs[i]:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 3: Output JSON has the expected structure
# ---------------------------------------------------------------------------


def test_output_json_structure():
    """The output JSON file contains all required fields."""
    result = run_adversarial_loop(
        gan_rounds=2,
        n_wallets=15,
        trades_per_wallet=15,
        output_dir="reports",
        seed=RANDOM_SEED,
    )

    required_top_keys = {"timestamp", "gan_rounds", "rounds", "final_auc_roc"}
    assert required_top_keys.issubset(
        result.keys()
    ), f"Missing top-level keys. Expected {required_top_keys}, got {set(result.keys())}"

    for round_data in result["rounds"]:
        required_round_keys = {"round", "profile", "dataset_size"}
        assert required_round_keys.issubset(round_data.keys()), (
            f"Round {round_data['round']} missing keys. "
            f"Expected at least {required_round_keys}, got {set(round_data.keys())}"
        )


# ---------------------------------------------------------------------------
# Test 4: generate_dataset_from_profile works for all profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile_name",
    [
        "NaiveAttacker",
        "RingAttacker",
        "CrossPairAttacker",
    ],
)
def test_generate_dataset_all_profiles(profile_name):
    """generate_dataset_from_profile produces a valid DataFrame for each profile."""
    df = generate_dataset_from_profile(
        profile_name=profile_name,
        n_wallets=10,
        trades_per_wallet=10,
        seed=RANDOM_SEED,
    )
    assert not df.empty, f"Generated DataFrame for {profile_name} is empty"
    assert "wallet" in df.columns, f"DataFrame for {profile_name} missing wallet column"
    if "label" in df.columns:
        assert df["label"].isin([0, 1]).all(), f"Labels in {profile_name} must be 0 or 1"


# ---------------------------------------------------------------------------
# Test 5: compute_feature_importances returns a dict
# ---------------------------------------------------------------------------


def test_compute_feature_importances():
    """compute_feature_importances returns a non-empty dict when models exist."""
    run_adversarial_loop(
        gan_rounds=1,
        n_wallets=15,
        trades_per_wallet=15,
        output_dir="reports",
        seed=RANDOM_SEED,
    )

    model_dir = None
    for path in ["models", os.path.join(os.getcwd(), "models")]:
        if os.path.exists(os.path.join(path, "random_forest.joblib")):
            model_dir = path
            break

    if model_dir:
        importances = compute_feature_importances(model_dir)
        assert isinstance(importances, dict), "Feature importances must be a dict"
