"""Adversarial robustness evaluation for the LedgerLens ensemble ML models.

Implements three adversarial attacks against the wash-trading detection ensemble:
  1. Gradient feature attack (white-box PGD)
  2. Benford-conforming amounts generation
  3. Counterparty diversification simulation

Also implements two hardening measures:
  - Option B: Temporal Benford divergence feature
  - Option C: Ensemble disagreement flag

Usage:
    python -m scripts.adversarial_eval --model-dir ./models --output reports/adversarial_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from config import config
from detection.benford_engine import mad_score
from detection.model_inference import RiskScorer
from detection.model_training import FEATURE_COLUMNS_EXCLUDE
from scripts.generate_synthetic_dataset import generate_synthetic_dataset

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Feature bounds: (min, max) for each feature column used in gradient attack
# ---------------------------------------------------------------------------


def _build_feature_bounds(feature_cols: list[str]) -> dict[str, tuple[float, float]]:
    """Return (lower, upper) bounds for each feature column."""
    bounds = {}
    for col in feature_cols:
        if any(
            col.startswith(p)
            for p in [
                "counterparty_concentration_ratio",
                "round_trip_frequency",
                "self_matching_rate",
                "order_cancellation_rate",
                "off_hours_activity_ratio",
                "volume_spike_frequency",
                "funding_source_similarity",
                "network_centrality",
                "intra_minute_clustering",
            ]
        ):
            bounds[col] = (0.0, 1.0)
        elif col.startswith("benford_mad_"):
            bounds[col] = (0.0, 1.0)
        elif col.startswith("benford_chi_square_"):
            bounds[col] = (0.0, float("inf"))
        elif col.startswith("benford_z_max_"):
            bounds[col] = (0.0, float("inf"))
        elif col == "account_age_days":
            bounds[col] = (0.0, float("inf"))
        elif col == "volume_per_counterparty_ratio":
            bounds[col] = (0.0, float("inf"))
        else:
            bounds[col] = (0.0, float("inf"))
    return bounds


def _ensemble_prob(feature_row: pd.Series, models: dict) -> float:
    """Compute ensemble average probability of wash trading (label=1)."""
    feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
    X = feature_row[feature_cols].to_frame().T.astype(float)
    probs = [model.predict_proba(X)[0, 1] for model in models.values()]
    return float(np.mean(probs))


def gradient_feature_attack(
    feature_row: pd.Series,
    models: dict,
    max_iterations: int = 100,
    step_size: float = 0.01,
    target_prob: float = 0.45,
) -> tuple[pd.Series, float]:
    """Perturb feature_row until ensemble P(wash) < target_prob using PGD.

    Uses finite-difference gradient estimation to minimally perturb the
    feature vector (L1-norm) until the ensemble probability drops below
    target_prob. Box constraints enforce valid feature ranges throughout.

    Returns:
        (perturbed_row, l1_distance_from_original)
    """
    feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
    bounds = _build_feature_bounds(feature_cols)

    x = feature_row.copy()
    original = feature_row.copy()

    for _ in range(max_iterations):
        prob = _ensemble_prob(x, models)
        if prob < target_prob:
            break

        # Finite-difference gradient estimation
        grad = np.zeros(len(feature_cols))
        eps = 1e-4
        for i, col in enumerate(feature_cols):
            x_plus = x.copy()
            x_plus[col] = x[col] + eps
            # clip to bounds
            lo, hi = bounds[col]
            x_plus[col] = float(
                np.clip(x_plus[col], lo, hi if hi != float("inf") else x_plus[col] + 1)
            )
            prob_plus = _ensemble_prob(x_plus, models)
            grad[i] = (prob_plus - prob) / eps

        # Gradient descent step (minimise probability → negative gradient direction)
        for i, col in enumerate(feature_cols):
            lo, hi = bounds[col]
            new_val = x[col] - step_size * grad[i]
            if hi == float("inf"):
                x[col] = float(max(new_val, lo))
            else:
                x[col] = float(np.clip(new_val, lo, hi))

    l1_dist = float(sum(abs(x[col] - original[col]) for col in feature_cols))
    return x, l1_dist


def benford_conforming_amounts(
    n_trades: int,
    base_amount: float,
    seed: int = 42,
) -> pd.Series:
    """Generate wash-trade amounts that conform to Benford's Law.

    Scales base_amount by log-uniform noise so that the leading digit
    distribution matches Benford's expected frequencies.

    Returns:
        pd.Series of trade amounts conforming to Benford's Law.
    """
    rng = np.random.default_rng(seed)
    # Log-uniform noise over [base/10, base*10] produces Benford-conforming digits
    log_min = np.log10(base_amount / 10.0) if base_amount > 0 else 0.0
    log_max = np.log10(base_amount * 10.0) if base_amount > 0 else 1.0
    log_amounts = rng.uniform(log_min, log_max, size=n_trades)
    amounts = 10.0**log_amounts
    return pd.Series(amounts, name="amount")


def diversified_counterparty_simulation(
    n_counterparties: int,
    trades_per_counterparty: int,
    wallet: str,
) -> pd.DataFrame:
    """Simulate wash trades spread across n_counterparties.

    Returns a DataFrame with columns: base_account, counter_account, amount,
    ledger_close_time — matching the schema expected by feature_engineering.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    base_time = pd.Timestamp("2024-01-01", tz="UTC")

    for cp_idx in range(n_counterparties):
        counterparty = f"GCOUNTERPARTY{cp_idx:04d}"
        for t_idx in range(trades_per_counterparty):
            rows.append(
                {
                    "base_account": wallet,
                    "counter_account": counterparty,
                    "amount": float(rng.uniform(100, 1000)),
                    "ledger_close_time": base_time
                    + pd.Timedelta(minutes=cp_idx * trades_per_counterparty + t_idx),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hardening measures
# ---------------------------------------------------------------------------


def compute_benford_temporal_divergence(
    wallet_trades: pd.DataFrame,
    short_window_hours: int = 1,
    long_window_hours: int = 168,
    amount_col: str = "amount",
    time_col: str = "ledger_close_time",
) -> float:
    """Option B: Temporal Benford divergence feature.

    Computes the ratio of short-window MAD to long-window MAD.
    A sophisticated evader who cleans up recent behaviour will have
    anomalously low short-window MAD relative to long-window MAD,
    producing a divergence score < 1.0.

    Returns:
        benford_temporal_divergence: short_mad / (long_mad + epsilon)
    """
    if wallet_trades.empty:
        return 1.0

    timestamps = pd.to_datetime(wallet_trades[time_col])
    ref = timestamps.max()

    short_mask = timestamps > ref - pd.Timedelta(hours=short_window_hours)
    long_mask = timestamps > ref - pd.Timedelta(hours=long_window_hours)

    short_mad = mad_score(wallet_trades.loc[short_mask, amount_col])
    long_mad = mad_score(wallet_trades.loc[long_mask, amount_col])

    epsilon = 1e-9
    return float(short_mad / (long_mad + epsilon))


def compute_ensemble_disagreement(feature_row: pd.Series, models: dict) -> dict:
    """Option C: Ensemble disagreement flag.

    If any two models in the ensemble disagree by > 0.3 in probability,
    flag the wallet as requiring review regardless of average score.

    Returns:
        dict with 'high_disagreement_flag' (bool) and 'max_disagreement' (float)
    """
    feature_cols = [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]
    X = feature_row[feature_cols].to_frame().T.astype(float)
    probs = [model.predict_proba(X)[0, 1] for model in models.values()]

    max_disagreement = max(probs) - min(probs) if len(probs) > 1 else 0.0
    high_disagreement_flag = bool(max_disagreement > 0.3)

    return {
        "high_disagreement_flag": high_disagreement_flag,
        "max_disagreement": float(max_disagreement),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(model_dir: str | None = None) -> dict:
    """Run the full adversarial benchmark and return results dict."""
    scorer = RiskScorer(model_dir=model_dir)
    if not scorer.models:
        raise RuntimeError(
            f"No trained models found in {model_dir or config.MODEL_DIR}. Run model_training first."
        )

    # Generate synthetic test set and get wash-trading rows (label=1)
    df = generate_synthetic_dataset(n_wallets=200, seed=RANDOM_SEED)
    wash_df = df[df["label"] == 1].copy()

    feature_cols = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]

    # -----------------------------------------------------------------------
    # Attack 1: Gradient feature attack
    # -----------------------------------------------------------------------
    gradient_evasions = 0
    l1_costs = []

    for _, row in wash_df.iterrows():
        feature_row = row[feature_cols + ["wallet"]]
        original_score = scorer.score(feature_row)["score"]
        perturbed_row, l1_dist = gradient_feature_attack(feature_row, scorer.models)
        perturbed_score = scorer.score(perturbed_row)["score"]

        if perturbed_score < config.RISK_SCORE_FLAG_THRESHOLD:
            gradient_evasions += 1
        l1_costs.append(l1_dist)

    gradient_evasion_rate = gradient_evasions / len(wash_df)
    median_l1_cost = float(np.median(l1_costs))

    # -----------------------------------------------------------------------
    # Attack 2: Benford-conforming amounts
    # -----------------------------------------------------------------------
    benford_evasions = 0
    benford_scores_conforming = []
    benford_scores_nonconforming = []

    for _, row in wash_df.iterrows():
        feature_row = row[feature_cols + ["wallet"]].copy()

        # Generate conforming amounts and recompute benford_mad features
        amounts = benford_conforming_amounts(n_trades=200, base_amount=500.0, seed=RANDOM_SEED)
        conforming_mad = mad_score(amounts)

        # Override all benford_mad features with conforming MAD
        for col in feature_cols:
            if col.startswith("benford_mad_"):
                feature_row[col] = conforming_mad
            if col.startswith("benford_chi_square_"):
                feature_row[col] = 0.0
            if col.startswith("benford_z_max_"):
                feature_row[col] = 0.0

        score_val = scorer.score(feature_row)["score"]
        benford_scores_conforming.append(score_val)
        if score_val < config.RISK_SCORE_FLAG_THRESHOLD:
            benford_evasions += 1

        original_score = scorer.score(row[feature_cols + ["wallet"]])["score"]
        benford_scores_nonconforming.append(original_score)

    benford_evasion_rate = benford_evasions / len(wash_df)

    # -----------------------------------------------------------------------
    # Attack 3: Counterparty diversification
    # -----------------------------------------------------------------------
    cp_scores = {}
    for n_cp in [1, 2, 5, 10]:
        sim_df = diversified_counterparty_simulation(
            n_counterparties=n_cp,
            trades_per_counterparty=10,
            wallet="GWASHTEST0001",
        )
        from detection.feature_engineering import compute_trade_pattern_features

        features = compute_trade_pattern_features("GWASHTEST0001", sim_df)
        cp_scores[n_cp] = features["counterparty_concentration_ratio"]

    # -----------------------------------------------------------------------
    # Hardening: Option C — ensemble disagreement (before hardening)
    # -----------------------------------------------------------------------
    disagreement_flags_before = []
    for _, row in wash_df.iterrows():
        feature_row = row[feature_cols + ["wallet"]]
        perturbed_row, _ = gradient_feature_attack(feature_row, scorer.models)
        disagreement = compute_ensemble_disagreement(perturbed_row, scorer.models)
        disagreement_flags_before.append(disagreement["high_disagreement_flag"])

    # With Option C: any wallet with high_disagreement_flag is flagged for review
    # → counts as detected even if score < threshold
    hardened_evasions = sum(
        1
        for i, (_, row) in enumerate(wash_df.iterrows())
        # evaded AND not flagged by disagreement
        if not disagreement_flags_before[i]
        and scorer.score(gradient_feature_attack(row[feature_cols + ["wallet"]], scorer.models)[0])[
            "score"
        ]
        < config.RISK_SCORE_FLAG_THRESHOLD
    )
    hardened_evasion_rate = hardened_evasions / len(wash_df)

    # -----------------------------------------------------------------------
    # Assemble benchmark
    # -----------------------------------------------------------------------
    benchmark = {
        "seed": RANDOM_SEED,
        "n_test_positives": len(wash_df),
        "risk_score_flag_threshold": config.RISK_SCORE_FLAG_THRESHOLD,
        "evasion_rate": round(gradient_evasion_rate, 4),
        "median_l1_cost": round(median_l1_cost, 6),
        "gradient_attack": {
            "evasion_rate": round(gradient_evasion_rate, 4),
            "median_l1_cost": round(median_l1_cost, 6),
            "mean_l1_cost": round(float(np.mean(l1_costs)), 6),
        },
        "benford_evasion_attack": {
            "evasion_rate": round(benford_evasion_rate, 4),
            "mean_score_conforming": round(float(np.mean(benford_scores_conforming)), 2),
            "mean_score_nonconforming": round(float(np.mean(benford_scores_nonconforming)), 2),
        },
        "counterparty_diversification_attack": {
            "concentration_by_n_counterparties": {
                str(k): round(v, 4) for k, v in cp_scores.items()
            },
        },
        "hardening_results": {
            "option_c_ensemble_disagreement": {
                "baseline_evasion_rate": round(gradient_evasion_rate, 4),
                "hardened_evasion_rate": round(hardened_evasion_rate, 4),
                "reduction": round(gradient_evasion_rate - hardened_evasion_rate, 4),
            }
        },
    }

    return benchmark


def main() -> None:
    args = parse_args()
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True
    )

    print("Running adversarial benchmark...")
    benchmark = run_benchmark(model_dir=args.model_dir)

    with open(args.output, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"Benchmark written to {args.output}")
    print(f"  Gradient attack evasion rate: {benchmark['evasion_rate']:.1%}")
    print(f"  Median L1 cost:               {benchmark['median_l1_cost']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output", default="reports/adversarial_benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
