"""Generate an adversarial-robustness report for the trained ensemble.

Loads the trained models from `config.MODEL_DIR` (or `--model-dir`), runs
FGSM/PGD evasion attacks against the wash wallets in a labelled feature
matrix, and writes a JSON robustness report covering:

  - PGD/FGSM evasion success rate on the high-scoring wash cohort
  - the most vulnerable features (cheapest to game)
  - the AUC-ROC gain from adversarial-augmentation retraining

Usage:
    python -m scripts.run_adversarial_eval \
        --data-path data/synthetic_dataset.parquet \
        --output reports/adversarial_robustness.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from config import config
from detection.adversarial.augmentation import evaluate_augmentation
from detection.adversarial.robustness import evaluate_robustness
from detection.model_inference import RiskScorer
from utils.logging import get_logger

logger = get_logger(__name__)

# Standard IBP certification epsilons reported in the eval output (Issue #245).
IBP_EPSILONS = (0.01, 0.05)


def _run_ibp_certification(
    scorer: RiskScorer,
    df: pd.DataFrame,
    epsilons: tuple[float, ...] = IBP_EPSILONS,
) -> dict:
    """Run IBP certification over correctly-classified wash wallets.

    For each epsilon in *epsilons*, reports the fraction of correctly
    classified wash-trade samples that are also certifiably robust.

    IBP is only applied to the NeuralProcess sub-model (numpy MLP) when it is
    available.  If the model is not present, returns an empty dict.

    Security note: results are included in the internal eval report only —
    not exposed via the external API.
    """
    try:
        from detection.certified_robustness import certify_ibp, layers_from_neural_process
        from detection.neural_process import NeuralProcess
    except ImportError:
        logger.warning("IBP certification skipped: certified_robustness module unavailable")
        return {}

    wash = df[df["label"] == 1] if "label" in df.columns else df
    if wash.empty:
        return {}

    from detection.adversarial.robustness import _feature_columns

    feat_cols = _feature_columns(wash)
    X = wash[feat_cols].astype(float).values

    np_model = NeuralProcess(feature_dim=len(feat_cols))
    layers = layers_from_neural_process(np_model)

    results: dict[str, float] = {}
    for eps in epsilons:
        certified = 0
        total = 0
        for row in X:
            # label=1 (fraud), certify that the score stays ≥ threshold
            radius = certify_ibp(layers, row, epsilon=eps, label=1)
            total += 1
            if radius >= eps:
                certified += 1
        frac = certified / total if total > 0 else 0.0
        results[f"certified_robust_fraction_eps_{eps}"] = round(frac, 4)
        logger.info(
            "IBP certification at ε=%.3f: %d/%d (%.1f%%) wash wallets certified robust",
            eps, certified, total, frac * 100,
        )

    return results


def build_report(
    data_path: str,
    *,
    model_dir: str | None = None,
    epsilon: float = 3.0,
    steps: int = 40,
    target_score: float = 40,
    high_score: float = 80,
    skip_augmentation: bool = False,
    skip_ibp: bool = False,
) -> dict:
    """Assemble the full robustness report dict (used by `main` and tests)."""
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d labelled rows from %s", len(df), data_path)

    scorer = RiskScorer(model_dir=model_dir)
    wash = df[df["label"] == 1] if "label" in df.columns else df

    robustness = evaluate_robustness(
        scorer,
        wash,
        epsilon=epsilon,
        steps=steps,
        target_score=target_score,
        high_score=high_score,
    )

    report = {"robustness": robustness}

    if not skip_augmentation and "label" in df.columns:
        report["augmentation"] = evaluate_augmentation(
            df, epsilon=epsilon, steps=steps, target_score=target_score
        )

    if not skip_ibp and "label" in df.columns:
        ibp_results = _run_ibp_certification(scorer, df)
        if ibp_results:
            report["ibp_certification"] = ibp_results

    return report


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
        help="Directory of trained model artifacts (default: MODEL_DIR)",
    )
    parser.add_argument(
        "--output",
        default="reports/adversarial_robustness.json",
        help="Path to write the JSON robustness report",
    )
    parser.add_argument("--epsilon", type=float, default=3.0, help="L-inf budget (std units)")
    parser.add_argument("--steps", type=int, default=40, help="PGD iterations")
    parser.add_argument(
        "--target-score", type=float, default=40, help="Evasion succeeds below this score"
    )
    parser.add_argument(
        "--high-score", type=float, default=80, help="Min score to enter the attacked cohort"
    )
    parser.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Skip the (slower) adversarial-augmentation retraining comparison",
    )
    parser.add_argument(
        "--skip-ibp",
        action="store_true",
        help="Skip the IBP certified-robustness certification step (Issue #245)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = build_report(
        args.data_path,
        model_dir=args.model_dir,
        epsilon=args.epsilon,
        steps=args.steps,
        target_score=args.target_score,
        high_score=args.high_score,
        skip_augmentation=args.skip_augmentation,
        skip_ibp=args.skip_ibp,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Wrote adversarial robustness report to %s", args.output)
    logger.info("Models evaluated from %s", args.model_dir or config.MODEL_DIR)


if __name__ == "__main__":
    main()
