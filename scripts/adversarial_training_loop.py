"""Generative Adversarial Training Loop (GAN-style) for the LedgerLens
Wash Trade Simulation Engine.

Round 0:  train the detector on a NaiveAttacker-generated dataset
Round N:  generate a new dataset using AdaptiveAttacker (which reads
          Round N-1 model's feature importances), retrain detector
Repeat for ``config.GAN_ROUNDS`` iterations or until detector AUC-ROC
plateaus (< 0.005 improvement).

Per-round metrics are written to ``reports/adversarial_loop_{timestamp}.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from config import config
from detection.model_training import (
    FEATURE_COLUMNS_EXCLUDE,
    MODEL_REGISTRY,
    load_training_data,
    save_models,
    split_features_labels,
    train_models,
)
from scripts.wash_trade_simulator import (
    AdaptiveAttacker,
    NaiveAttacker,
    trades_to_feature_matrix,
)

RANDOM_SEED = 42


def compute_feature_importances(model_dir: str) -> dict[str, float]:
    """Aggregate feature importances across all models in *model_dir*.

    Returns a dict mapping feature name -> mean importance across all
    trained ensemble members.
    """
    all_importances: list[dict[str, float]] = []
    for name in MODEL_REGISTRY:
        path = os.path.join(model_dir, f"{name}.joblib")
        if not os.path.exists(path):
            continue
        try:
            model = joblib.load(path)
            if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
                names = model.feature_names_in_
                vals = model.feature_importances_
                all_importances.append(dict(zip(names, vals)))
        except Exception:
            continue

    if not all_importances:
        return {}

    keys = set().union(*all_importances)
    aggregated = {}
    for k in keys:
        vals = [imp[k] for imp in all_importances if k in imp]
        aggregated[k] = float(np.mean(vals)) if vals else 0.0
    return aggregated


def generate_dataset_from_profile(
    profile_name: str,
    n_wallets: int | None = None,
    trades_per_wallet: int | None = None,
    model_path: str | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a labelled feature matrix using the specified attacker profile.

    If ``profile_name`` is ``"NaiveAttacker"``, half the wallets are wash
    and half are legitimate (using a legitimacy generator).  For other
    profiles, all wallets are wash traders.
    """
    from scripts.generate_synthetic_dataset import generate_synthetic_dataset

    if profile_name == "NaiveAttacker":
        return generate_synthetic_dataset(
            n_wallets=n_wallets or config.SIMULATOR_N_WALLETS * 2,
            seed=seed,
        )

    from scripts.wash_trade_simulator import AdaptiveAttacker, BaseAttackerProfile, create_profile

    profile: BaseAttackerProfile
    if profile_name == "AdaptiveAttacker" and model_path:
        profile = AdaptiveAttacker(
            n_wallets=n_wallets or config.SIMULATOR_N_WALLETS,
            trades_per_wallet=trades_per_wallet or config.SIMULATOR_TRADES_PER_WALLET,
            model_path=model_path,
            seed=seed,
        )
    else:
        profile = create_profile(
            profile_name,
            n_wallets=n_wallets or config.SIMULATOR_N_WALLETS,
            trades_per_wallet=trades_per_wallet or config.SIMULATOR_TRADES_PER_WALLET,
            seed=seed,
        )

    trades = profile.generate_trades()
    df = trades_to_feature_matrix(trades)

    if df.empty:
        df = generate_synthetic_dataset(
            n_wallets=n_wallets or config.SIMULATOR_N_WALLETS,
            seed=seed,
        )
        df["profile"] = profile_name

    if "profile" not in df.columns:
        df["profile"] = profile_name

    return df


def run_adversarial_loop(
    gan_rounds: int = 5,
    n_wallets: int = 50,
    trades_per_wallet: int = 100,
    plateau_threshold: float = 0.005,
    output_dir: str = "reports",
    seed: int = RANDOM_SEED,
) -> dict:
    """Run the GAN-style adversarial training loop.

    Returns a dict with per-round metrics.
    """
    round_metrics = []

    model_dir = tempfile.mkdtemp(prefix="adversarial_models_")

    for round_idx in range(gan_rounds):
        print(f"\n{'='*60}")
        print(f"Adversarial Loop — Round {round_idx}")
        print(f"{'='*60}")

        if round_idx == 0:
            profile_name = "NaiveAttacker"
            model_path = None
        else:
            profile_name = "AdaptiveAttacker"
            model_path = os.path.join(model_dir, "random_forest.joblib")
            if not os.path.exists(model_path):
                model_path = None

        df = generate_dataset_from_profile(
            profile_name=profile_name,
            n_wallets=n_wallets,
            trades_per_wallet=trades_per_wallet,
            model_path=model_path,
            seed=seed + round_idx,
        )

        print(f"  Dataset: {len(df)} rows, profile={profile_name}")
        if "label" in df.columns:
            label_dist = df["label"].value_counts().to_dict()
            print(f"  Label distribution: {label_dist}")

        training_output = train_models(df, random_state=seed + round_idx)
        results = training_output["results"]

        save_models(results, model_dir)

        round_data: dict[str, float | int | str] = {
            "round": round_idx,
            "profile": profile_name,
            "dataset_size": len(df),
        }
        for model_name, result in results.items():
            round_data[f"{model_name}_auc_roc"] = float(result["metrics"]["auc_roc"])
            round_data[f"{model_name}_pr_auc"] = float(result["metrics"]["pr_auc"])
            round_data[f"{model_name}_f1"] = float(result["metrics"]["f1"])

        round_metrics.append(round_data)

        metric_keys = ("random_forest_auc_roc", "xgboost_auc_roc", "lightgbm_auc_roc")
        print(
            f"  Metrics: { {k: round(float(round_data[k]), 4) for k in metric_keys if k in round_data} }"
        )

        if round_idx > 0:
            prev_auc = float(round_metrics[round_idx - 1].get("random_forest_auc_roc", 0.0))
            curr_auc = float(round_data.get("random_forest_auc_roc", 0.0))
            improvement = curr_auc - prev_auc
            print(f"  AUC-ROC improvement: {improvement:.4f}")
            if 0 < improvement < plateau_threshold:
                print(f"  Plateau detected (improvement {improvement:.4f} < {plateau_threshold}). Stopping.")
                break

    result = {
        "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%S"),
        "gan_rounds": len(round_metrics),
        "plateau_threshold": plateau_threshold,
        "rounds": round_metrics,
        "final_auc_roc": float(round_metrics[-1].get("random_forest_auc_roc", 0.0)) if round_metrics else 0.0,
        "monotonic_non_decreasing": all(
            float(round_metrics[i].get("random_forest_auc_roc", 0.0))
            >= float(round_metrics[i - 1].get("random_forest_auc_roc", 0.0))
            for i in range(1, len(round_metrics))
        ),
        "plateau_exit": len(round_metrics) < gan_rounds,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"adversarial_loop_{result['timestamp']}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {out_path}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gan-rounds", type=int, default=config.GAN_ROUNDS)
    parser.add_argument("--n-wallets", type=int, default=config.SIMULATOR_N_WALLETS)
    parser.add_argument("--trades-per-wallet", type=int, default=config.SIMULATOR_TRADES_PER_WALLET)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_adversarial_loop(
        gan_rounds=args.gan_rounds,
        n_wallets=args.n_wallets,
        trades_per_wallet=args.trades_per_wallet,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
