"""Realism metrics for the Wash Trade Simulation Engine.

Computes:
  1. **Fréchet Feature Distance (FFD)**: distance between the feature
     distributions of simulated and real (Testnet) wash-trade wallets.
  2. **Discriminator accuracy**: train a held-out classifier to distinguish
     simulated from real — a good simulator achieves <= 55% accuracy
     (near-chance).

Usage:
    python -m scripts.evaluate_simulator_realism \
        --simulated data/synthetic_dataset.parquet \
        --real data/labelled_dataset.parquet \
        --output reports/simulator_realism_TIMESTAMP.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from scripts.wash_trade_simulator import RANDOM_SEED

FEATURE_COLUMNS_EXCLUDE = {
    "wallet",
    "label",
    "profile",
    "labelling_signal",
    "review_notes",
    "data_window_start",
    "data_window_end",
    "n_trades",
    "is_wash",
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in FEATURE_COLUMNS_EXCLUDE and not c.startswith("benford_residual_")
    ]


def compute_frechet_feature_distance(
    real_features: pd.DataFrame,
    sim_features: pd.DataFrame,
) -> float:
    """Compute Fréchet Feature Distance between real and simulated feature distributions.

    FFD = ||mu_real - mu_sim||^2 + Tr(Sigma_real + Sigma_sim - 2 * sqrt(Sigma_real * Sigma_sim))

    Lower values indicate more similar distributions.
    """
    common_cols = [c for c in real_features.columns if c in sim_features.columns]
    real_data = real_features[common_cols].select_dtypes(include=[np.number]).dropna()
    sim_data = sim_features[common_cols].select_dtypes(include=[np.number]).dropna()

    if len(real_data) < 2 or len(sim_data) < 2:
        return float("inf")

    mu_real = real_data.mean(axis=0).values
    mu_sim = sim_data.mean(axis=0).values

    sigma_real = np.cov(real_data.values, rowvar=False)
    sigma_sim = np.cov(sim_data.values, rowvar=False)

    diff = mu_real - mu_sim
    mean_diff = np.dot(diff, diff)

    sigma_sum = sigma_real + sigma_sim
    sigma_prod = sigma_real @ sigma_sim
    try:
        sqrt_sigma_prod = np.real(sqrtm(sigma_prod))
        trace_term = np.trace(sigma_sum - 2 * sqrt_sigma_prod)
    except (np.linalg.LinAlgError, ValueError):
        trace_term = 0.0

    return float(mean_diff + trace_term)


def compute_discriminator_accuracy(
    real_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train a held-out classifier to distinguish simulated from real data.

    A good simulator achieves <= 60% discriminator accuracy (near-chance).

    Returns:
        dict with accuracy, auc_roc, and cross_val_mean metrics.
    """
    feature_cols = _get_feature_cols(pd.concat([real_df, sim_df], ignore_index=True))

    real_features = real_df[feature_cols].select_dtypes(include=[np.number]).dropna()
    sim_features = sim_df[feature_cols].select_dtypes(include=[np.number]).dropna()

    if len(real_features) < 5 or len(sim_features) < 5:
        return {"error": "Insufficient samples", "accuracy": 1.0, "auc_roc": 1.0}

    n_min = min(len(real_features), len(sim_features))
    X_real = real_features.iloc[:n_min].values
    X_sim = sim_features.iloc[:n_min].values

    X = np.vstack([X_real, X_sim])
    y = np.hstack([np.zeros(n_min), np.ones(n_min)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    cv_scores = cross_val_score(clf, X, y, cv=min(5, len(np.unique(y))), scoring="accuracy")

    return {
        "accuracy": round(float(accuracy), 4),
        "auc_roc": round(float(auc_roc), 4),
        "cross_val_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cross_val_std": round(float(cv_scores.std()), 4),
        "n_real_samples": n_min,
        "n_sim_samples": n_min,
    }


def evaluate_realism(
    sim_path: str,
    real_path: str,
    output_dir: str = "reports",
    seed: int = RANDOM_SEED,
) -> dict:
    """Run the full simulator realism evaluation."""
    print(f"Loading simulated data from {sim_path}")
    sim_df = pd.read_parquet(sim_path)

    print(f"Loading real data from {real_path}")
    real_df = pd.read_parquet(real_path)

    common_feature_cols = _get_feature_cols(pd.concat([real_df, sim_df], ignore_index=True))

    sim_features = sim_df[common_feature_cols].select_dtypes(include=[np.number]).dropna()
    real_features = real_df[common_feature_cols].select_dtypes(include=[np.number]).dropna()

    print(f"  Real samples: {len(real_features)}, Sim samples: {len(sim_features)}")

    print("Computing Fréchet Feature Distance...")
    ffd = compute_frechet_feature_distance(real_features, sim_features)

    print("Computing discriminator accuracy...")
    disc = compute_discriminator_accuracy(real_df, sim_df, seed=seed)

    result = {
        "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%S"),
        "simulated_data": sim_path,
        "real_data": real_path,
        "frechet_feature_distance": round(ffd, 4) if ffd != float("inf") else None,
        "discriminator_accuracy": disc,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"simulator_realism_{result['timestamp']}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults written to {out_path}")
    if disc.get("error"):
        print(f"  Discriminator error: {disc['error']}")
    else:
        print(f"  FFD: {result['frechet_feature_distance']}")
        print(f"  Discriminator accuracy: {disc.get('accuracy', 'N/A')}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulated", default="data/synthetic_dataset.parquet")
    parser.add_argument("--real", default="data/labelled_dataset.parquet")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_realism(
        sim_path=args.simulated,
        real_path=args.real,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
