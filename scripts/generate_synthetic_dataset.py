"""Generate a synthetic labelled feature matrix for local training/demo/tests.

Until the real "Open dataset release" (see README roadmap), this script
produces a feature matrix with the same columns as
`detection.feature_engineering.build_feature_matrix`, plus a `label` column
(1 = wash trading, 0 = legitimate), so `detection.model_training.train_models`
can be exercised end-to-end without live Horizon data.

Usage:
    python -m scripts.generate_synthetic_dataset --n-wallets 500 --output data/synthetic.parquet
    python -m scripts.generate_synthetic_dataset --profile RingAttacker --n-wallets 10 --output data/ring.parquet
    python -m scripts.generate_synthetic_dataset --profile AdaptiveAttacker --gan-rounds 5
"""

import argparse

import numpy as np
import pandas as pd

from config import config

BENFORD_FEATURE_TEMPLATE = ["benford_chi_square_{h}h", "benford_mad_{h}h", "benford_z_max_{h}h"]


def generate_synthetic_dataset(
    n_wallets: int = 500,
    seed: int = 42,
    wash_offset: float = 0.0,
    wash_noise: float = 1.0,
    profile: str = "NaiveAttacker",
    model_path: str | None = None,
) -> pd.DataFrame:
    """Generate `n_wallets` rows, roughly half legitimate (label 0) and half
    wash-trading-like (label 1) with systematically different feature
    distributions.

    wash_offset and wash_noise allow varying the wash-trading pattern for meta-learning.
    When *profile* is ``"NaiveAttacker"`` (default), uses the original
    feature-level generation for backward compatibility.  Otherwise, uses
    the Wash Trade Simulation Engine to generate trade-level data and then
    computes features with ``build_feature_matrix``.
    """
    if profile == "NaiveAttacker":
        return _generate_feature_level(n_wallets=n_wallets, seed=seed, wash_offset=wash_offset, wash_noise=wash_noise)
    return _generate_from_simulator(
        profile=profile,
        n_wallets=n_wallets,
        seed=seed,
        model_path=model_path,
    )


def _generate_feature_level(n_wallets: int, seed: int, wash_offset: float = 0.0, wash_noise: float = 1.0) -> pd.DataFrame:
    """Original feature-level generation (backward-compatible path)."""
    rng = np.random.default_rng(seed)
    n_legit = n_wallets // 2

    rows = []
    for i in range(n_wallets):
        is_wash = i >= n_legit
        row = {"wallet": f"GSYNTH{i:06d}"}

        for hours in config.BENFORD_WINDOWS_HOURS:
            if is_wash:
                row[f"benford_chi_square_{hours}h"] = rng.uniform(20, 100) * wash_noise + wash_offset
                row[f"benford_mad_{hours}h"] = rng.uniform(0.02, 0.08) * wash_noise + (wash_offset / 1000)
                row[f"benford_z_max_{hours}h"] = rng.uniform(3, 10) * wash_noise + (wash_offset / 10)
            else:
                row[f"benford_chi_square_{hours}h"] = rng.uniform(0, 10)
                row[f"benford_mad_{hours}h"] = rng.uniform(0.0, 0.014)
                row[f"benford_z_max_{hours}h"] = rng.uniform(0, 2)

        for hours in config.BENFORD_WINDOWS_HOURS:
            if is_wash:
                row[f"benford_residual_chi_square_{hours}h"] = rng.uniform(15, 80) * wash_noise + wash_offset
                row[f"benford_residual_mad_{hours}h"] = rng.uniform(0.018, 0.07) * wash_noise + (wash_offset / 1000)
            else:
                row[f"benford_residual_chi_square_{hours}h"] = rng.uniform(0, 8)
                row[f"benford_residual_mad_{hours}h"] = rng.uniform(0.0, 0.012)

        if is_wash:
            row["counterparty_concentration_ratio"] = rng.uniform(0.7, 1.0) * wash_noise
            row["round_trip_frequency"] = rng.uniform(0.3, 1.0) * wash_noise
            row["net_roundtrip_ratio"] = rng.uniform(0.3, 1.0) * wash_noise
            row["self_matching_rate"] = rng.uniform(0.3, 1.0) * wash_noise
            row["order_cancellation_rate"] = rng.uniform(0.4, 0.9) * wash_noise
            row["volume_per_counterparty_ratio"] = rng.uniform(1000, 10000) * wash_noise
            row["intra_minute_clustering"] = rng.uniform(0.3, 1.0) * wash_noise
            row["off_hours_activity_ratio"] = rng.uniform(0.2, 0.8) * wash_noise
            row["volume_spike_frequency"] = rng.uniform(0.2, 0.6) * wash_noise
            row["funding_source_similarity"] = rng.uniform(0.5, 1.0) * wash_noise
            row["network_centrality"] = rng.uniform(0.3, 1.0) * wash_noise
            row["account_age_days"] = rng.uniform(0, 30) * wash_noise
            row["cross_pair_trade_synchrony"] = rng.uniform(0.4, 1.0) * wash_noise
            row["net_asset_flow_deviation"] = rng.uniform(0.0, 0.3) * wash_noise
            row["cross_pair_counterparty_overlap"] = rng.uniform(0.5, 1.0) * wash_noise
            row["cross_pair_volume_correlation"] = rng.uniform(0.4, 1.0) * wash_noise
            row["pair_diversity_score"] = rng.uniform(0.0, 0.3) * wash_noise
            row["cross_pair_mad_std"] = rng.uniform(0.01, 0.05) * wash_noise
            row["inter_arrival_cv"] = rng.uniform(0.0, 0.2) * wash_noise
            row["entropy_of_amounts"] = rng.uniform(0.0, 1.0) * wash_noise
            row["cross_wallet_volume_corr"] = rng.uniform(0.5, 1.0) * wash_noise
        else:
            row["counterparty_concentration_ratio"] = rng.uniform(0.0, 0.5)
            row["round_trip_frequency"] = rng.uniform(0.0, 0.1)
            row["net_roundtrip_ratio"] = rng.uniform(0.0, 0.1)
            row["self_matching_rate"] = rng.uniform(0.0, 0.1)
            row["order_cancellation_rate"] = rng.uniform(0.0, 0.3)
            row["volume_per_counterparty_ratio"] = rng.uniform(10, 1000)
            row["intra_minute_clustering"] = rng.uniform(0.0, 0.2)
            row["off_hours_activity_ratio"] = rng.uniform(0.0, 0.3)
            row["volume_spike_frequency"] = rng.uniform(0.0, 0.1)
            row["funding_source_similarity"] = rng.uniform(0.0, 0.3)
            row["network_centrality"] = rng.uniform(0.0, 0.2)
            row["account_age_days"] = rng.uniform(30, 1000)
            row["cross_pair_trade_synchrony"] = rng.uniform(0.0, 0.2)
            row["net_asset_flow_deviation"] = rng.uniform(0.5, 1.0)
            row["cross_pair_counterparty_overlap"] = rng.uniform(0.0, 0.3)
            row["cross_pair_volume_correlation"] = rng.uniform(-0.5, 0.3)
            row["pair_diversity_score"] = rng.uniform(0.5, 1.0)
            row["cross_pair_mad_std"] = rng.uniform(0.0, 0.01)
            row["inter_arrival_cv"] = rng.uniform(0.5, 2.0)
            row["entropy_of_amounts"] = rng.uniform(3.0, 6.0)
            row["cross_wallet_volume_corr"] = rng.uniform(-0.3, 0.3)

        row["label"] = int(is_wash)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _generate_from_simulator(
    profile: str,
    n_wallets: int,
    seed: int,
    model_path: str | None = None,
) -> pd.DataFrame:
    """Generate a feature matrix via the Wash Trade Simulation Engine."""
    from scripts.wash_trade_simulator import (
        AdaptiveAttacker,
        create_profile,
        trades_to_feature_matrix,
    )

    if profile == "AdaptiveAttacker" and model_path:
        attacker = AdaptiveAttacker(
            n_wallets=n_wallets,
            model_path=model_path,
            seed=seed,
        )
    else:
        attacker = create_profile(
            profile,
            n_wallets=n_wallets,
            seed=seed,
        )

    trades = attacker.generate_trades()
    df = trades_to_feature_matrix(trades)

    if df.empty:
        return _generate_feature_level(n_wallets=n_wallets, seed=seed)

    df["profile"] = profile
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wallets", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wash-offset", type=float, default=0.0)
    parser.add_argument("--wash-noise", type=float, default=1.0)
    parser.add_argument("--output", default="data/synthetic_dataset.parquet")
    parser.add_argument(
        "--profile",
        default="NaiveAttacker",
        choices=[
            "NaiveAttacker",
            "TimingJitterAttacker",
            "AmountConformanceAttacker",
            "RingAttacker",
            "LayeringAttacker",
            "CrossPairAttacker",
            "AdaptiveAttacker",
        ],
        help="Attacker profile for the Wash Trade Simulation Engine (default: NaiveAttacker)",
    )
    parser.add_argument(
        "--gan-rounds",
        type=int,
        default=0,
        help="Run N rounds of adversarial training loop (0 = skip). "
        "Requires --profile AdaptiveAttacker.",
    )
    parser.add_argument("--model-path", default=None, help="Path to trained model for AdaptiveAttacker")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gan_rounds > 0:
        from scripts.adversarial_training_loop import run_adversarial_loop

        run_adversarial_loop(
            gan_rounds=args.gan_rounds,
            n_wallets=args.n_wallets,
            seed=args.seed,
        )
        return

    df = generate_synthetic_dataset(
        n_wallets=args.n_wallets,
        seed=args.seed,
        wash_offset=args.wash_offset,
        wash_noise=args.wash_noise,
        profile=args.profile,
        model_path=args.model_path,
    )
    df.to_parquet(args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
