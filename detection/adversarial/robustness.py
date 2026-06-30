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

from detection.adversarial.attack import DEFAULT_TARGET_SCORE, FGSMAttack, PGDAttack, feature_space_fgsm
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


# ---------------------------------------------------------------------------
# Adversarial training integration (Issue #191)
# ---------------------------------------------------------------------------

ADV_TRAINING_RATIO_DEFAULT = 0.5


def adversarial_training_step(
    X_batch: pd.DataFrame,
    y_batch: "pd.Series",
    scorer,
    *,
    epsilon: float = 0.1,
    adv_ratio: float = ADV_TRAINING_RATIO_DEFAULT,
    feature_scale: dict | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, "pd.Series"]:
    """Generate adversarial examples for wash-trade rows and mix them into the batch.

    For each label-1 (wash trade) row in the batch, `feature_space_fgsm` is
    applied to generate a perturbed copy that increases the model's score (i.e.
    sits near the decision boundary from the hard side). A random subset of
    these adversarial copies (controlled by ``adv_ratio``) is appended to the
    batch and returned alongside the original batch.

    This function is called once per training epoch from the adversarial
    training loop in ``detection.model_training`` when
    ``ADV_TRAINING_ENABLED=true``.

    Args:
        X_batch:       Feature DataFrame (no label column).
        y_batch:       Corresponding label Series.
        scorer:        Trained ``RiskScorer`` used for finite-difference
                       gradient estimation.  Must expose
                       ``score_continuous_batch``.
        epsilon:       Feature-space FGSM perturbation budget (standardised
                       units when ``feature_scale`` is provided).
        adv_ratio:     Fraction of wash-trade rows to augment (0–1).  E.g.
                       0.5 means 50% of label-1 rows get an adversarial copy
                       appended.
        feature_scale: Per-feature standard deviation dict.  Defaults to
                       all-ones (raw units).
        random_state:  RNG seed for reproducible row sampling.

    Returns:
        ``(X_augmented, y_augmented)`` — the original batch plus adversarial
        copies, with integer label indices reset.

    Raises:
        ValueError: if ``adv_ratio`` is not in ``[0, 1]``.
    """
    if not (0.0 <= adv_ratio <= 1.0):
        raise ValueError(f"adv_ratio must be in [0, 1], got {adv_ratio}")

    rng = np.random.default_rng(random_state)

    # Identify wash-trade rows
    wash_idx = y_batch[y_batch == 1].index.tolist()
    if not wash_idx or adv_ratio == 0:
        return X_batch, y_batch

    # Sample the requested fraction
    n_adv = max(1, int(len(wash_idx) * adv_ratio))
    sampled_idx = rng.choice(wash_idx, size=min(n_adv, len(wash_idx)), replace=False).tolist()

    scale = feature_scale or {}
    adv_rows = []
    adv_labels = []
    for idx in sampled_idx:
        row = X_batch.loc[idx].copy()
        # Attach a temporary label column so feature_space_fgsm can identify
        # non-feature columns via FEATURE_COLUMNS_EXCLUDE — then strip it.
        perturbed = feature_space_fgsm(row, epsilon, scorer, feature_scale=scale)
        adv_rows.append(perturbed)
        adv_labels.append(int(y_batch.loc[idx]))

    if not adv_rows:
        return X_batch, y_batch

    X_adv = pd.DataFrame(adv_rows, columns=X_batch.columns)
    y_adv = pd.Series(adv_labels, name=y_batch.name)

    X_out = pd.concat([X_batch, X_adv], ignore_index=True)
    y_out = pd.concat([y_batch.reset_index(drop=True), y_adv], ignore_index=True)
    return X_out, y_out


def run_adversarial_training(
    df: pd.DataFrame,
    *,
    epochs: int = 3,
    epsilon: float = 0.1,
    adv_ratio: float = ADV_TRAINING_RATIO_DEFAULT,
    test_size: float = 0.2,
    random_state: int = 42,
    model_dir: str | None = None,
) -> dict:
    """Train the ensemble with FGSM adversarial augmentation for ``epochs`` epochs.

    Each epoch:
      1. Trains an ensemble on the (optionally augmented) training set.
      2. Generates adversarial examples for the wash-trade rows using the
         freshly trained scorer.
      3. Mixes adversarial examples at ``adv_ratio`` into the training set
         for the *next* epoch.

    Benchmarks clean accuracy (AUC-ROC on unperturbed test set) and
    adversarial accuracy (AUC-ROC on FGSM-perturbed test set) after the
    final epoch.

    Returns a report dict that is JSON-serialisable and is logged at INFO
    level by ``detection.model_training.main`` when
    ``ADV_TRAINING_ENABLED=true``.

    Raises:
        ValueError: if ``epochs < 1``.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    from detection.model_training import split_features_labels, save_models, train_models
    from detection.model_inference import RiskScorer

    import tempfile, os

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    X_test, y_test = split_features_labels(test_df)
    feature_scale = feature_scale_from_matrix(df.drop(columns=["label"], errors="ignore"))

    current_train = train_df.copy()
    epoch_log = []

    for epoch in range(epochs):
        results = train_models(current_train, test_size=0.0 if len(current_train) < 5 else 0.1, random_state=random_state)
        tmp_dir = model_dir or tempfile.mkdtemp(prefix="ledgerlens_adv_train_")
        epoch_dir = os.path.join(tmp_dir, f"epoch_{epoch}")
        save_models(results, epoch_dir)
        scorer = RiskScorer(model_dir=epoch_dir)

        # Clean accuracy on test set
        X_feat, _ = split_features_labels(current_train)
        probs_clean = np.array([scorer.score_continuous(row) / 100.0 for _, row in X_test.iterrows()])
        try:
            clean_auc = float(roc_auc_score(y_test, probs_clean))
        except Exception:
            clean_auc = float("nan")

        # Adversarial accuracy: FGSM-perturb the test set
        X_test_adv = X_test.copy()
        for idx in X_test_adv.index:
            row = X_test_adv.loc[idx]
            X_test_adv.loc[idx] = feature_space_fgsm(
                row, epsilon, scorer, feature_scale=feature_scale
            )
        probs_adv = np.array([scorer.score_continuous(row) / 100.0 for _, row in X_test_adv.iterrows()])
        try:
            adv_auc = float(roc_auc_score(y_test, probs_adv))
        except Exception:
            adv_auc = float("nan")

        epoch_log.append({"epoch": epoch, "clean_auc": clean_auc, "adversarial_auc": adv_auc})
        logger.info(
            "Adversarial training epoch %d/%d — clean_auc=%.4f  adversarial_auc=%.4f",
            epoch + 1,
            epochs,
            clean_auc,
            adv_auc,
        )

        # Augment training set for the next epoch (not needed after last epoch)
        if epoch < epochs - 1:
            X_train, y_train = split_features_labels(current_train)
            X_aug, y_aug = adversarial_training_step(
                X_train,
                y_train,
                scorer,
                epsilon=epsilon,
                adv_ratio=adv_ratio,
                feature_scale=feature_scale,
                random_state=random_state + epoch,
            )
            aug_df = X_aug.copy()
            aug_df["label"] = y_aug.values
            # Preserve wallet column if present
            if "wallet" in current_train.columns:
                n_orig = len(X_train)
                aug_wallets = list(current_train["wallet"])
                aug_wallets += [f"{w}_adv_e{epoch}" for w in current_train["wallet"].iloc[:len(X_aug) - n_orig]]
                aug_df["wallet"] = aug_wallets
            current_train = aug_df

    # Final metrics from last epoch
    first_clean = epoch_log[0]["clean_auc"] if epoch_log else float("nan")
    last_clean = epoch_log[-1]["clean_auc"] if epoch_log else float("nan")
    first_adv = epoch_log[0]["adversarial_auc"] if epoch_log else float("nan")
    last_adv = epoch_log[-1]["adversarial_auc"] if epoch_log else float("nan")

    clean_degradation = first_clean - last_clean
    adv_improvement = last_adv - first_adv

    report = {
        "epochs": epochs,
        "epsilon": epsilon,
        "adv_ratio": adv_ratio,
        "epoch_log": epoch_log,
        "clean_auc_initial": first_clean,
        "clean_auc_final": last_clean,
        "clean_accuracy_degradation": clean_degradation,
        "adversarial_auc_initial": first_adv,
        "adversarial_auc_final": last_adv,
        "adversarial_accuracy_improvement": adv_improvement,
        "clean_degradation_within_tolerance": bool(clean_degradation <= 0.03),
    }
    logger.info(
        "Adversarial training complete — clean_auc: %.4f→%.4f (Δ=%.4f, within_3pt_tol=%s) "
        "adversarial_auc: %.4f→%.4f (Δ=%.4f)",
        first_clean, last_clean, clean_degradation,
        report["clean_degradation_within_tolerance"],
        first_adv, last_adv, adv_improvement,
    )
    return report
