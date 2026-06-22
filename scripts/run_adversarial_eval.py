"""Generate an adversarial robustness report for the wash-trade ensemble.

Attacks high-scoring wash wallets with PGD (or FGSM), then reports:
  * attack success rate and mean steps-to-success (target: 80+ -> <40),
  * the minimum L-infinity epsilon needed per feature,
  * the most vulnerable (cheapest-to-perturb) features,
  * the AUC-ROC gain from adversarial augmentation (target: >= +5pp).

Models are loaded from `--model-dir` if present, otherwise an ensemble is
trained on `--data-path` so the report is reproducible end-to-end without
pre-trained artifacts.

Usage:
    python -m scripts.run_adversarial_eval \\
        --data-path data/synthetic_dataset.parquet \\
        --output reports/adversarial_robustness.json
"""

import argparse
import json
import os

import pandas as pd

from detection.adversarial.attack import (
    EnsembleScoreFunction,
    FGSMAttack,
    PGDAttack,
    feature_scale_from_matrix,
)
from detection.adversarial.augmentation import adversarial_augmentation_gain
from detection.adversarial.evaluate import AdversarialEvaluator
from detection.model_training import FEATURE_COLUMNS_EXCLUDE, MODEL_REGISTRY, train_models
from utils.logging import get_logger

logger = get_logger(__name__)

# Acceptance thresholds from the issue.
SUCCESS_RATE_TARGET = 0.80
AUGMENTATION_GAIN_TARGET = 0.05


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]


def load_models(data_path: str, model_dir: str | None, random_state: int) -> dict:
    """Load trained models from `model_dir`, else train an ensemble on the data."""
    if model_dir:
        from detection.model_inference import RiskScorer

        scorer = RiskScorer(model_dir=model_dir)
        if scorer.models:
            logger.info("Loaded %d models from %s", len(scorer.models), model_dir)
            return scorer.models
        logger.warning("No models in %s; training a fresh ensemble", model_dir)

    df = pd.read_parquet(data_path)
    logger.info("Training ensemble on %s (%d rows)", data_path, len(df))
    results = train_models(df, random_state=random_state)
    return {name: result["model"] for name, result in results.items()}


def build_attack(kind: str, score_fn, scale, args):
    if kind == "fgsm":
        return FGSMAttack(score_fn, epsilon=args.epsilon, feature_scale=scale)
    return PGDAttack(
        score_fn,
        epsilon=args.epsilon,
        steps=args.steps,
        step_size=args.step_size,
        feature_scale=scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        required=True,
        help="Labelled feature matrix (parquet) with a 'label' column",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory of trained model artifacts (default: train on --data-path)",
    )
    parser.add_argument("--attack", choices=["pgd", "fgsm"], default="pgd")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="L-infinity budget in per-feature scale (std) units",
    )
    parser.add_argument("--steps", type=int, default=40, help="PGD iterations")
    parser.add_argument("--step-size", type=float, default=0.2, help="PGD step size (scale units)")
    parser.add_argument("--target-score", type=float, default=40.0)
    parser.add_argument("--clean-threshold", type=float, default=80.0)
    parser.add_argument(
        "--max-wallets",
        type=int,
        default=50,
        help="Cap on high-scoring wash wallets to attack (finite differences are costly)",
    )
    parser.add_argument(
        "--max-epsilon", type=float, default=5.0, help="Per-feature epsilon search cap"
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-augmentation", action="store_true")
    parser.add_argument(
        "--output",
        default="reports/adversarial_robustness.json",
        help="Path to write the JSON robustness report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.data_path)
    feature_columns = _feature_columns(df)

    models = load_models(args.data_path, args.model_dir, args.random_state)
    if set(models) != set(MODEL_REGISTRY):
        logger.warning("Loaded models %s differ from registry %s", set(models), set(MODEL_REGISTRY))

    score_fn = EnsembleScoreFunction(models, feature_columns)
    scale = feature_scale_from_matrix(df[feature_columns])
    attack = build_attack(args.attack, score_fn, scale, args)

    evaluator = AdversarialEvaluator(
        score_fn,
        feature_scale=scale,
        target_score=args.target_score,
        clean_threshold=args.clean_threshold,
    )

    # Attack the wash wallets the model already flags strongly.
    wash = df[df["label"] == 1] if "label" in df.columns else df
    high_score = evaluator.select_high_score_rows(wash)
    if args.max_wallets and len(high_score) > args.max_wallets:
        high_score = high_score.head(args.max_wallets)
    logger.info(
        "Attacking %d wash wallets with clean score >= %.0f",
        len(high_score),
        args.clean_threshold,
    )

    summary = evaluator.evaluate(attack, high_score)
    per_feature_eps, vulnerable = evaluator.feature_vulnerability(
        high_score, max_epsilon=args.max_epsilon
    )

    report = {
        "config": {
            "attack": args.attack,
            "epsilon": args.epsilon,
            "steps": args.steps,
            "step_size": args.step_size,
            "target_score": args.target_score,
            "clean_threshold": args.clean_threshold,
            "data_path": args.data_path,
            "n_wallets_attacked": summary.n_attacked,
        },
        "attack_success": {
            "success_rate": summary.success_rate,
            "mean_clean_score": summary.mean_clean_score,
            "mean_adversarial_score": summary.mean_adversarial_score,
            "mean_steps_to_success": summary.mean_steps_to_success,
            "meets_target": summary.success_rate >= SUCCESS_RATE_TARGET,
        },
        "min_epsilon_per_feature": {
            k: (None if v == float("inf") else v) for k, v in per_feature_eps.items()
        },
        "most_vulnerable_features": [
            {
                **v,
                "mean_min_epsilon": (
                    None if v["mean_min_epsilon"] == float("inf") else v["mean_min_epsilon"]
                ),
            }
            for v in vulnerable
        ],
    }

    if not args.skip_augmentation:
        logger.info("Measuring adversarial augmentation gain (retraining ensemble)")
        gain = adversarial_augmentation_gain(
            df,
            lambda fn, sc: build_attack(args.attack, fn, sc, args),
            target_score=args.target_score,
            random_state=args.random_state,
        )
        gain["meets_target"] = gain["gain"] >= AUGMENTATION_GAIN_TARGET
        report["adversarial_augmentation"] = gain

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Attack success rate: %.1f%%", summary.success_rate * 100)
    logger.info("Mean score %.1f -> %.1f", summary.mean_clean_score, summary.mean_adversarial_score)
    if vulnerable:
        top = vulnerable[0]
        logger.info(
            "Most vulnerable feature: %s (mean min epsilon %.3f)",
            top["feature"],
            top["mean_min_epsilon"],
        )
    if "adversarial_augmentation" in report:
        aug = report["adversarial_augmentation"]
        logger.info(
            "Adversarial augmentation AUC: %.3f -> %.3f (gain %+.3f)",
            aug["baseline_auc"],
            aug["augmented_auc"],
            aug["gain"],
        )
    logger.info("Wrote robustness report to %s", args.output)


if __name__ == "__main__":
    main()
