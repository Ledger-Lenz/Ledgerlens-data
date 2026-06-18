"""Generate a synthetic labelled feature matrix for local training/demo/tests.

Until the real "Open dataset release" (see README roadmap), this script
produces a feature matrix with the same columns as
`detection.feature_engineering.build_feature_matrix`, plus a `label` column
(1 = wash trading, 0 = legitimate), so `detection.model_training.train_models`
can be exercised end-to-end without live Horizon data.

Usage:
    python -m scripts.generate_synthetic_dataset --n-wallets 500 --output data/synthetic.parquet
"""

import argparse

import numpy as np
import pandas as pd

from config import config

BENFORD_FEATURE_TEMPLATE = ["benford_chi_square_{h}h", "benford_mad_{h}h", "benford_z_max_{h}h"]


def generate_synthetic_dataset(n_wallets: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate `n_wallets` rows, roughly half legitimate (label 0) and half
    wash-trading-like (label 1) with systematically different feature
    distributions.
    """
    rng = np.random.default_rng(seed)
    n_legit = n_wallets // 2

    rows = []
    for i in range(n_wallets):
        is_wash = i >= n_legit
        row = {"wallet": f"GSYNTH{i:06d}"}

        for hours in config.BENFORD_WINDOWS_HOURS:
            if is_wash:
                row[f"benford_chi_square_{hours}h"] = rng.uniform(20, 100)
                row[f"benford_mad_{hours}h"] = rng.uniform(0.02, 0.08)
                row[f"benford_z_max_{hours}h"] = rng.uniform(3, 10)
            else:
                row[f"benford_chi_square_{hours}h"] = rng.uniform(0, 10)
                row[f"benford_mad_{hours}h"] = rng.uniform(0.0, 0.014)
                row[f"benford_z_max_{hours}h"] = rng.uniform(0, 2)

        if is_wash:
            row["counterparty_concentration_ratio"] = rng.uniform(0.7, 1.0)
            row["round_trip_frequency"] = rng.uniform(0.3, 1.0)
            row["self_matching_rate"] = rng.uniform(0.3, 1.0)
            row["order_cancellation_rate"] = rng.uniform(0.4, 0.9)
            row["volume_per_counterparty_ratio"] = rng.uniform(1000, 10000)
            row["intra_minute_clustering"] = rng.uniform(0.3, 1.0)
            row["off_hours_activity_ratio"] = rng.uniform(0.2, 0.8)
            row["volume_spike_frequency"] = rng.uniform(0.2, 0.6)
            row["funding_source_similarity"] = rng.uniform(0.5, 1.0)
            row["network_centrality"] = rng.uniform(0.3, 1.0)
            row["account_age_days"] = rng.uniform(0, 30)
            # Cross-asset features for wash traders (coordinated across pairs)
            row["cross_pair_trade_synchrony"] = rng.uniform(0.4, 1.0)
            row["net_asset_flow_deviation"] = rng.uniform(0.0, 0.3)
            row["cross_pair_counterparty_overlap"] = rng.uniform(0.5, 1.0)
            row["cross_pair_volume_correlation"] = rng.uniform(0.4, 1.0)
            row["pair_diversity_score"] = rng.uniform(0.0, 0.3)
            row["cross_pair_mad_std"] = rng.uniform(0.01, 0.05)
        else:
            row["counterparty_concentration_ratio"] = rng.uniform(0.0, 0.5)
            row["round_trip_frequency"] = rng.uniform(0.0, 0.1)
            row["self_matching_rate"] = rng.uniform(0.0, 0.1)
            row["order_cancellation_rate"] = rng.uniform(0.0, 0.3)
            row["volume_per_counterparty_ratio"] = rng.uniform(10, 1000)
            row["intra_minute_clustering"] = rng.uniform(0.0, 0.2)
            row["off_hours_activity_ratio"] = rng.uniform(0.0, 0.3)
            row["volume_spike_frequency"] = rng.uniform(0.0, 0.1)
            row["funding_source_similarity"] = rng.uniform(0.0, 0.3)
            row["network_centrality"] = rng.uniform(0.0, 0.2)
            row["account_age_days"] = rng.uniform(30, 1000)
            # Cross-asset features for legitimate traders (diverse activity)
            row["cross_pair_trade_synchrony"] = rng.uniform(0.0, 0.2)
            row["net_asset_flow_deviation"] = rng.uniform(0.5, 1.0)
            row["cross_pair_counterparty_overlap"] = rng.uniform(0.0, 0.3)
            row["cross_pair_volume_correlation"] = rng.uniform(-0.5, 0.3)
            row["pair_diversity_score"] = rng.uniform(0.5, 1.0)
            row["cross_pair_mad_std"] = rng.uniform(0.0, 0.01)

        row["label"] = int(is_wash)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wallets", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic_dataset.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_synthetic_dataset(n_wallets=args.n_wallets, seed=args.seed)
    df.to_parquet(args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
