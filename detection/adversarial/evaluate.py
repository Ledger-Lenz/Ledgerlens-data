"""Adversarial robustness evaluation for the wash-trade ensemble.

Quantifies how easily an adversary can perturb a flagged wallet's features
to slip below the alert threshold. Produces the metrics required by the
robustness report:

* attack success rate and steps-to-success (PGD: 80+ -> <40 in <=40 steps),
* the minimum L-infinity epsilon needed to fool the model *per feature*,
* a ranking of the most vulnerable (cheapest-to-perturb) features.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from detection.adversarial.attack import EnsembleScoreFunction, PGDAttack


@dataclass
class AttackOutcome:
    """Result of attacking a single wallet's feature row."""

    clean_score: float
    adversarial_score: float
    success: bool
    steps: int | None
    # Per-feature L-infinity perturbation actually applied, in scale units.
    linf_perturbation: float


@dataclass
class RobustnessSummary:
    """Aggregate adversarial-robustness metrics over a set of wallets."""

    n_attacked: int
    success_rate: float
    mean_clean_score: float
    mean_adversarial_score: float
    mean_steps_to_success: float | None
    outcomes: list[AttackOutcome] = field(default_factory=list)


def _to_matrix(feature_rows: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Select `feature_columns` from a feature matrix as a float ndarray."""
    return feature_rows.reindex(columns=feature_columns).to_numpy(dtype=float)


def evaluate_attack(
    attack,
    feature_rows: pd.DataFrame,
    *,
    target_score: float = 40.0,
) -> RobustnessSummary:
    """Run `attack` against every row of `feature_rows` and summarise success.

    `feature_rows` is a feature matrix (rows already filtered to the wallets
    of interest, e.g. high-scoring wash wallets). Columns are aligned to the
    attack's `feature_columns`; any extras (`wallet`, `label`) are ignored.
    """
    score_fn: EnsembleScoreFunction = attack.score_fn
    X = _to_matrix(feature_rows, attack.feature_columns)

    outcomes: list[AttackOutcome] = []
    for x0 in X:
        clean = score_fn(x0)
        adv = attack.perturb(x0, target_score=target_score)
        adv_score = score_fn(adv)
        steps = (
            attack.steps_to_target(x0, target_score=target_score)
            if isinstance(attack, PGDAttack)
            else (1 if adv_score < target_score else None)
        )
        linf = float(np.max(np.abs((adv - x0) / attack.scale))) if len(x0) else 0.0
        outcomes.append(
            AttackOutcome(
                clean_score=clean,
                adversarial_score=adv_score,
                success=bool(adv_score < target_score),
                steps=steps,
                linf_perturbation=linf,
            )
        )

    n = len(outcomes)
    successes = [o for o in outcomes if o.success]
    steps_vals = [o.steps for o in successes if o.steps is not None]
    return RobustnessSummary(
        n_attacked=n,
        success_rate=(len(successes) / n) if n else 0.0,
        mean_clean_score=float(np.mean([o.clean_score for o in outcomes])) if n else 0.0,
        mean_adversarial_score=(
            float(np.mean([o.adversarial_score for o in outcomes])) if n else 0.0
        ),
        mean_steps_to_success=(float(np.mean(steps_vals)) if steps_vals else None),
        outcomes=outcomes,
    )


def min_epsilon_for_feature(
    score_fn: EnsembleScoreFunction,
    x0: np.ndarray,
    idx: int,
    *,
    scale: np.ndarray,
    target_score: float = 40.0,
    max_epsilon: float = 5.0,
    n_iter: int = 24,
) -> float:
    """Smallest single-feature L-infinity epsilon that pushes the score below target.

    Perturbs only feature `idx` (in whichever direction lowers the score) and
    bisects on the magnitude. Returns the epsilon in scale units, or ``inf`` if
    no perturbation within `max_epsilon` succeeds. The tree score is only
    approximately monotonic in a single feature, so bisection yields an estimate
    rather than an exact threshold.
    """
    best = math.inf
    for direction in (-1.0, 1.0):
        x_hi = x0.copy()
        x_hi[idx] = x0[idx] + direction * max_epsilon * scale[idx]
        if score_fn(x_hi) >= target_score:
            continue  # not achievable in this direction within the budget

        lo, hi = 0.0, max_epsilon
        probe = x0.copy()
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            probe[idx] = x0[idx] + direction * mid * scale[idx]
            if score_fn(probe) < target_score:
                hi = mid
            else:
                lo = mid
        best = min(best, hi)
    return best


def min_epsilon_per_feature(
    score_fn: EnsembleScoreFunction,
    feature_rows: pd.DataFrame,
    *,
    feature_scale: np.ndarray,
    target_score: float = 40.0,
    max_epsilon: float = 5.0,
    n_iter: int = 24,
) -> dict[str, float]:
    """Mean minimum single-feature epsilon across `feature_rows`, per feature.

    For each feature, averages `min_epsilon_for_feature` over the supplied
    wallets (ignoring wallets where that feature alone cannot fool the model).
    Features with a smaller mean epsilon are cheaper to perturb — i.e. more
    vulnerable.
    """
    feature_columns = score_fn.feature_columns
    X = _to_matrix(feature_rows, feature_columns)

    per_feature: dict[str, float] = {}
    for idx, name in enumerate(feature_columns):
        eps_values = [
            min_epsilon_for_feature(
                score_fn,
                x0,
                idx,
                scale=feature_scale,
                target_score=target_score,
                max_epsilon=max_epsilon,
                n_iter=n_iter,
            )
            for x0 in X
        ]
        finite = [e for e in eps_values if math.isfinite(e)]
        per_feature[name] = float(np.mean(finite)) if finite else math.inf
    return per_feature


def most_vulnerable_features(per_feature_epsilon: dict[str, float], top_n: int = 10) -> list[dict]:
    """Rank features by ascending mean min-epsilon (cheapest to perturb first).

    Each entry: ``{"feature": str, "mean_min_epsilon": float}``. Features that
    could never fool the model on their own (``inf``) sort last.
    """
    ranked = sorted(per_feature_epsilon.items(), key=lambda kv: kv[1])
    return [{"feature": name, "mean_min_epsilon": eps} for name, eps in ranked[:top_n]]


class AdversarialEvaluator:
    """High-level entry point tying attacks and feature analysis together."""

    def __init__(
        self,
        score_fn: EnsembleScoreFunction,
        *,
        feature_scale: np.ndarray,
        target_score: float = 40.0,
        clean_threshold: float = 80.0,
    ):
        self.score_fn = score_fn
        self.feature_scale = np.asarray(feature_scale, dtype=float)
        self.target_score = target_score
        self.clean_threshold = clean_threshold

    def select_high_score_rows(self, feature_rows: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose clean score is at or above `clean_threshold`.

        These are the wallets the model already flags — the population an
        adversary would actually want to disguise.
        """
        X = _to_matrix(feature_rows, self.score_fn.feature_columns)
        if len(X) == 0:
            return feature_rows.iloc[:0]
        scores = self.score_fn.score_batch(X)
        return feature_rows.loc[scores >= self.clean_threshold]

    def evaluate(self, attack, feature_rows: pd.DataFrame) -> RobustnessSummary:
        return evaluate_attack(attack, feature_rows, target_score=self.target_score)

    def feature_vulnerability(
        self,
        feature_rows: pd.DataFrame,
        *,
        max_epsilon: float = 5.0,
        n_iter: int = 24,
        top_n: int = 10,
    ) -> tuple[dict[str, float], list[dict]]:
        """Return ``(min_epsilon_per_feature, most_vulnerable_features)``."""
        per_feature = min_epsilon_per_feature(
            self.score_fn,
            feature_rows,
            feature_scale=self.feature_scale,
            target_score=self.target_score,
            max_epsilon=max_epsilon,
            n_iter=n_iter,
        )
        return per_feature, most_vulnerable_features(per_feature, top_n=top_n)
