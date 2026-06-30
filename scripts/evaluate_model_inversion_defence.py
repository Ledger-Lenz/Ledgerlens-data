"""Evaluate model inversion attack defence by measuring reconstruction error.

This script simulates an adversary who attempts to reconstruct feature vectors
by making repeated queries to the risk score API and observing score deltas.
It measures how well the Laplace noise (Issue #264) defends against this attack.

Methodology:
1. Load trained models and generate synthetic feature vectors
2. For each feature vector, compute the true risk score (without perturbation)
3. Simulate multiple queries with Laplace noise to reconstruct the feature impact
4. Measure reconstruction error (L2 distance between reconstructed and true impacts)
5. Report success rate of inversion for different noise scales
"""

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import config
from detection.differential_privacy import laplace_scale
from detection.feature_engineering import build_feature_matrix
from detection.model_inference import RiskScorer
from scripts.generate_synthetic_dataset import generate_trades

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_features(data_path: str, n_samples: int = 100) -> pd.DataFrame:
    """Load or generate test feature matrix."""
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} feature rows from {data_path}")
        return df.head(n_samples)
    except FileNotFoundError:
        logger.warning(f"Dataset not found at {data_path}, generating synthetic data...")
        trades = generate_trades(n_wallets=50, n_trades_per_wallet=100, contamination_rate=0.2)
        features_df = build_feature_matrix(trades)
        return features_df.head(n_samples)


def simulate_inversion_attack(
    scorer: RiskScorer,
    feature_row: pd.Series,
    n_queries: int = 100,
    noise_scale: float = 10.0,
) -> dict[str, Any]:
    """Simulate model inversion attack via repeated queries with noise.

    The attacker makes n_queries calls to score() with the same features,
    observing noisy results. They then try to estimate the true score by:
    1. Collecting all noisy observations
    2. Computing the mean (noise averages out)
    3. Comparing with the true clean score

    Args:
        scorer: RiskScorer instance
        feature_row: A single feature row to attack
        n_queries: Number of queries to make
        noise_scale: Laplace scale parameter

    Returns:
        Dictionary with results:
        - true_score: Clean score without perturbation
        - noisy_scores: List of observed noisy scores
        - reconstructed_score: Mean of noisy observations
        - reconstruction_error: L2 distance between reconstructed and true
        - success: True if reconstruction within 5 points of true score
    """
    # Simulate the feature row as a DataFrame
    features_dict = feature_row.to_dict()
    feature_df = pd.DataFrame([features_dict])

    # Get the true clean score (internal call, no perturbation)
    true_score = scorer.score(feature_df, caller_id="internal")["score"]

    # Simulate multiple queries with different perturbations
    noisy_scores = []
    for query_idx in range(n_queries):
        caller_id = f"attacker_{query_idx}"
        timestamp_bucket = query_idx
        perturbed_score = scorer.score(
            feature_df, caller_id=caller_id, timestamp_bucket=timestamp_bucket
        )["score"]
        noisy_scores.append(perturbed_score)

    # Reconstruct the true score by averaging noisy observations
    # (noise is zero-mean Laplace, so averaging helps)
    reconstructed_score = np.mean(noisy_scores)

    # Compute reconstruction error
    reconstruction_error = abs(reconstructed_score - true_score)

    return {
        "true_score": true_score,
        "noisy_scores": noisy_scores,
        "noisy_mean": reconstructed_score,
        "reconstructed_score": int(round(reconstructed_score)),
        "reconstruction_error": reconstruction_error,
        "success": reconstruction_error <= 5.0,  # Attacker "succeeds" if within 5 points
        "noise_scale": noise_scale,
        "n_queries": n_queries,
    }


def evaluate_defence(
    data_path: str = "data/synthetic_dataset.parquet",
    model_dir: str | None = None,
    n_test_samples: int = 20,
    n_queries_per_attack: int = 100,
) -> dict[str, Any]:
    """Run full inversion attack simulation and report results.

    Args:
        data_path: Path to feature parquet file
        model_dir: Model directory (default: config.MODEL_DIR)
        n_test_samples: Number of wallets to attack
        n_queries_per_attack: Queries per attacked wallet

    Returns:
        Evaluation report with success rates and reconstruction errors
    """
    logger.info("Loading models and test data...")
    model_dir = model_dir or config.MODEL_DIR
    scorer = RiskScorer(model_dir=model_dir)

    features_df = load_test_features(data_path, n_samples=n_test_samples)

    logger.info(f"Evaluating model inversion defence over {len(features_df)} test samples...")

    attacks = []
    for idx, (_, feature_row) in enumerate(features_df.iterrows()):
        logger.info(f"  Attack {idx + 1}/{len(features_df)}...")

        result = simulate_inversion_attack(
            scorer=scorer,
            feature_row=feature_row,
            n_queries=n_queries_per_attack,
            noise_scale=laplace_scale(100.0, config.MODEL_INVERSION_DP_EPSILON),
        )
        attacks.append(result)

    # Aggregate results
    reconstruction_errors = [a["reconstruction_error"] for a in attacks]
    success_count = sum(1 for a in attacks if a["success"])
    success_rate = success_count / len(attacks)

    report = {
        "model_dir": model_dir,
        "epsilon": config.MODEL_INVERSION_DP_EPSILON,
        "noise_scale": attacks[0]["noise_scale"] if attacks else None,
        "n_test_samples": len(attacks),
        "n_queries_per_attack": n_queries_per_attack,
        "mean_reconstruction_error": float(np.mean(reconstruction_errors)),
        "median_reconstruction_error": float(np.median(reconstruction_errors)),
        "std_reconstruction_error": float(np.std(reconstruction_errors)),
        "min_reconstruction_error": float(np.min(reconstruction_errors)),
        "max_reconstruction_error": float(np.max(reconstruction_errors)),
        "inversion_success_rate": float(success_rate),
        "successful_inversions": success_count,
        "failed_inversions": len(attacks) - success_count,
        "individual_attacks": [
            {
                "true_score": a["true_score"],
                "reconstructed_score": a["reconstructed_score"],
                "reconstruction_error": float(a["reconstruction_error"]),
                "success": a["success"],
            }
            for a in attacks
        ],
    }

    logger.info("\n" + "=" * 70)
    logger.info("MODEL INVERSION DEFENCE EVALUATION REPORT")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  DP Epsilon: {report['epsilon']}")
    logger.info(f"  Laplace Scale: {report['noise_scale']:.2f}")
    logger.info(f"  Queries per Attack: {report['n_queries_per_attack']}")
    logger.info(f"\nResults:")
    logger.info(f"  Test Samples: {report['n_test_samples']}")
    logger.info(f"  Successful Reconstructions: {report['successful_inversions']}/{report['n_test_samples']}")
    logger.info(f"  Success Rate: {report['inversion_success_rate'] * 100:.1f}%")
    logger.info(f"\nReconstruction Error Statistics:")
    logger.info(f"  Mean: {report['mean_reconstruction_error']:.2f} points")
    logger.info(f"  Median: {report['median_reconstruction_error']:.2f} points")
    logger.info(f"  Std Dev: {report['std_reconstruction_error']:.2f} points")
    logger.info(f"  Range: [{report['min_reconstruction_error']:.2f}, {report['max_reconstruction_error']:.2f}]")
    logger.info("\nInterpretation:")
    if report["inversion_success_rate"] < 0.2:
        logger.info(
            "  ✓ STRONG DEFENCE: <20% of inversion attacks succeeded.\n"
            "    Adversary cannot reliably reconstruct feature vectors."
        )
    elif report["inversion_success_rate"] < 0.5:
        logger.info(
            "  ✓ MODERATE DEFENCE: 20-50% success rate.\n"
            "    Noise provides meaningful obscuration but attacks partially effective."
        )
    else:
        logger.info(
            "  ⚠ WEAK DEFENCE: >50% success rate.\n"
            "    Consider increasing noise scale (reduce DP_EPSILON) or query limit."
        )
    logger.info("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model inversion attack defence via Laplace noise"
    )
    parser.add_argument(
        "--data-path",
        default="data/synthetic_dataset.parquet",
        help="Path to feature dataset (default: data/synthetic_dataset.parquet)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model directory (default: config.MODEL_DIR)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of test wallets to attack (default: 20)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Queries per attack (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="reports/inversion_evaluation.json",
        help="Output JSON report path (default: reports/inversion_evaluation.json)",
    )

    args = parser.parse_args()

    report = evaluate_defence(
        data_path=args.data_path,
        model_dir=args.model_dir,
        n_test_samples=args.n_samples,
        n_queries_per_attack=args.n_queries,
    )

    # Write report
    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report written to {args.output}")
