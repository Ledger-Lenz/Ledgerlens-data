"""Robustness metrics for the risk-scoring ensemble under evasion attacks.

Given a trained `RiskScorer` and a feature matrix of known wash wallets,
these helpers quantify the attack surface described in the adversarial
robustness issue:

  - `attack_success_rate` — fraction of high-scoring wash wallets pushed
    below the alert threshold by an attack within its `epsilon` budget.
  - `minimum_epsilon_per_feature` — for each feature, the smallest L-inf
    budget (in scaled units) at which a single-feature FGSM step alone
    evades, i.e. how "cheap" that feature is to game.
  - `most_vulnerable_features` — features ranked by how often / how cheaply
    they enable evasion, logged for feature hardening.
  - `evaluate_robustness` — assembles the full report dict consumed by
    `scripts/run_adversarial_eval.py`.

All gradients flow through `RiskScorer.score_continuous` (see
`detection.adversarial.attack`).
"""

import numpy as np
import pandas as pd

from detection.adversarial.attack import DEFAULT_TARGET_SCORE, FGSMAttack, PGDAttack
from detection.model_training import FEATURE_COLUMNS_EXCLUDE
from utils.logging import get_logger

logger = get_logger(__name__)

# Wallets must score at least this high before an attack to count as an
# "alerting" wash wallet worth attacking (matches the issue's "80+" cohort).
DEFAULT_HIGH_SCORE = 80.0


def _feature_columns(feature_matrix: pd.DataFrame) -> list[str]:
    return [c for c in feature_matrix.columns if c not in FEATURE_COLUMNS_EXCLUDE]


def feature_scale_from_matrix(feature_matrix: pd.DataFrame) -> dict:
    """Per-feature standard deviation, used as the default L-inf scale so a
    single `epsilon` is comparable across heterogeneous feature magnitudes.

    Columns with zero/degenerate variance map to `1.0` (no rescaling).
    """
    cols = _feature_columns(feature_matrix)
    stds = feature_matrix[cols].astype(float).std(ddof=0)
    return {c: (float(stds[c]) if stds[c] > 0 else 1.0) for c in cols}


def high_scoring_wallets(
    scorer, feature_matrix: pd.DataFrame, high_score: float = DEFAULT_HIGH_SCORE
) -> pd.DataFrame:
    """Rows whose continuous score is `>= high_score` (the cohort an attacker
    would actually bother evading)."""
    if feature_matrix.empty:
        return feature_matrix
    scores = feature_matrix.apply(scorer.score_continuous, axis=1)
    return feature_matrix[scores >= high_score]


def attack_success_rate(
    scorer,
    feature_matrix: pd.DataFrame,
    attack,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> dict:
    """Run `attack` on every row and report the evasion success rate.

    A row is a "success" when its post-attack continuous score drops below
    `target_score`. Returns counts plus per-row before/after scores.
    """
    rows = []
    successes = 0
    for _, feature_row in feature_matrix.iterrows():
        before = scorer.score_continuous(feature_row)
        perturbed = attack.perturb(feature_row, target_score=target_score)
        after = scorer.score_continuous(perturbed)
        evaded = after < target_score
        successes += int(evaded)
        rows.append(
            {
                "wallet": feature_row.get("wallet"),
                "score_before": float(before),
                "score_after": float(after),
                "evaded": bool(evaded),
            }
        )

    total = len(rows)
    return {
        "total": total,
        "successes": successes,
        "success_rate": (successes / total) if total else 0.0,
        "target_score": float(target_score),
        "rows": rows,
    }


# A per-feature L-inf component below this (in scaled units) is treated as
# "not perturbed" when attributing which features an attack relied on.
_PERTURBED_EPS = 1e-6


def minimum_epsilon_per_feature(
    scorer,
    feature_row: pd.Series,
    feature_scale: dict,
    attack,
    target_score: float = DEFAULT_TARGET_SCORE,
) -> dict:
    """Per-feature L-inf epsilon that `attack` spent to evade on this row.

    Runs `attack.perturb` once and, if the perturbed row evades (continuous
    score below `target_score`), reports `|delta| / scale` for each feature —
    the per-feature L-inf budget the successful attack actually used. If the
    attack failed to evade, every feature maps to `None`.
    """
    feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
    perturbed = attack.perturb(feature_row, target_score=target_score)
    evaded = scorer.score_continuous(perturbed) < target_score

    result: dict[str, float | None] = {}
    for col in feature_cols:
        if not evaded:
            result[col] = None
            continue
        scale = feature_scale.get(col, 1.0) or 1.0
        result[col] = abs(float(perturbed[col]) - float(feature_row[col])) / scale
    return result


def most_vulnerable_features(
    scorer,
    feature_matrix: pd.DataFrame,
    feature_scale: dict,
    attack,
    target_score: float = DEFAULT_TARGET_SCORE,
    top_n: int = 5,
    sample: int | None = 25,
) -> list[dict]:
    """Rank features by how much a successful attack relies on perturbing them.

    For each (sampled) wash wallet that the attack evades, attributes the
    per-feature L-inf epsilon it spent (`minimum_epsilon_per_feature`). A
    feature is more vulnerable the more often it is perturbed and the larger
    the budget spent on it. Returns the `top_n` most vulnerable, each with the
    perturbation rate and the mean / minimum epsilon spent on that feature.

    `sample` bounds the (per-row attack) cost to at most that many rows
    (deterministic head; pass `None` to use every row).
    """
    feature_cols = _feature_columns(feature_matrix)
    spent: dict[str, list[float]] = {c: [] for c in feature_cols}

    analysed = feature_matrix if sample is None else feature_matrix.head(sample)
    successes = 0
    for _, feature_row in analysed.iterrows():
        per_feature = minimum_epsilon_per_feature(
            scorer, feature_row, feature_scale, attack, target_score=target_score
        )
        if any(v is not None for v in per_feature.values()):
            successes += 1
        for col, eps in per_feature.items():
            if eps is not None and eps > _PERTURBED_EPS:
                spent[col].append(eps)

    ranked = []
    for col in feature_cols:
        hits = spent[col]
        if not hits:
            continue
        ranked.append(
            {
                "feature": col,
                "perturbation_rate": len(hits) / successes if successes else 0.0,
                "mean_epsilon": float(np.mean(hits)),
                "min_epsilon": float(np.min(hits)),
            }
        )

    # Most vulnerable: perturbed in the most successful attacks, most heavily.
    ranked.sort(key=lambda r: (-r["perturbation_rate"], -r["mean_epsilon"]))
    return ranked[:top_n]


def evaluate_robustness(
    scorer,
    feature_matrix: pd.DataFrame,
    *,
    epsilon: float = 3.0,
    steps: int = 40,
    target_score: float = DEFAULT_TARGET_SCORE,
    high_score: float = DEFAULT_HIGH_SCORE,
    top_n: int = 5,
    vuln_sample: int | None = 25,
) -> dict:
    """Full robustness report for `scorer` over the wash wallets in
    `feature_matrix` (rows are expected to be label-1 / wash wallets).

    Restricts to the high-scoring cohort, runs FGSM and PGD success-rate
    sweeps, computes per-feature minimum epsilons, and ranks the most
    vulnerable features. The (expensive) per-feature analysis uses at most
    `vuln_sample` cohort rows. The returned dict is JSON-serialisable.
    """
    feature_scale = feature_scale_from_matrix(feature_matrix)

    cohort = high_scoring_wallets(scorer, feature_matrix, high_score=high_score)
    logger.info(
        "Adversarial cohort: %d/%d wallets score >= %.0f",
        len(cohort),
        len(feature_matrix),
        high_score,
    )

    fgsm = FGSMAttack(scorer, epsilon=epsilon, feature_scale=feature_scale)
    pgd = PGDAttack(
        scorer, epsilon=epsilon, steps=steps, step_size=epsilon / 10, feature_scale=feature_scale
    )

    fgsm_result = attack_success_rate(scorer, cohort, fgsm, target_score=target_score)
    pgd_result = attack_success_rate(scorer, cohort, pgd, target_score=target_score)
    logger.info(
        "FGSM success rate: %.1f%% | PGD success rate: %.1f%%",
        100 * fgsm_result["success_rate"],
        100 * pgd_result["success_rate"],
    )

    vulnerable = most_vulnerable_features(
        scorer,
        cohort,
        feature_scale,
        pgd,
        target_score=target_score,
        top_n=top_n,
        sample=vuln_sample,
    )
    for entry in vulnerable:
        logger.info(
            "Vulnerable feature %s: perturbation_rate=%.2f mean_epsilon=%.3f min_epsilon=%.3f",
            entry["feature"],
            entry["perturbation_rate"],
            entry["mean_epsilon"],
            entry["min_epsilon"],
        )

    return {
        "config": {
            "epsilon": epsilon,
            "steps": steps,
            "target_score": target_score,
            "high_score": high_score,
        },
        "cohort_size": len(cohort),
        "total_wallets": len(feature_matrix),
        "fgsm": {k: v for k, v in fgsm_result.items() if k != "rows"},
        "pgd": {k: v for k, v in pgd_result.items() if k != "rows"},
        "most_vulnerable_features": vulnerable,
    }
