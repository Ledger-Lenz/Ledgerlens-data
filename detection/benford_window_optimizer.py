"""Adaptive Per-Asset Benford Window Selection via Bayesian Optimization."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from skopt import gp_minimize
from skopt.space import Integer

from config import config
from detection.benford_engine import mad_score


def estimate_trades_per_hour(asset_trades: pd.DataFrame) -> float:
    """Rolling median of trades per clock hour over the last 30 days."""
    if asset_trades.empty or "ledger_close_time" not in asset_trades.columns:
        return 0.0

    times = pd.to_datetime(asset_trades["ledger_close_time"], utc=True)
    max_time = times.max()
    start_time = max_time - pd.Timedelta(days=30)

    recent = asset_trades[times >= start_time]
    if recent.empty:
        return 0.0

    recent_times = pd.to_datetime(recent["ledger_close_time"], utc=True)
    s = pd.Series(1, index=recent_times)
    counts = s.resample("1h").sum()

    # Reindex to full 30-day period (720 hours)
    full_idx = pd.date_range(start=start_time, end=max_time, freq="1h", tz="UTC")
    counts = counts.reindex(full_idx, fill_value=0)

    return float(counts.median())


def get_window_candidates(
    trades_per_hour: float,
    min_trades: int = 20,
    max_window: int = 720,
    num_candidates: int = 10,
) -> list[int]:
    """Generate log-spaced candidate window sizes based on trade velocity."""
    if trades_per_hour <= 0:
        return [max_window]

    min_w = math.ceil(min_trades / trades_per_hour)
    min_w = max(1, min_w)

    if min_w >= max_window:
        return [max_window]

    candidates = np.logspace(np.log10(min_w), np.log10(max_window), num=num_candidates)
    rounded = np.round(candidates).astype(int)
    unique_sorted = sorted(list(set(rounded)))
    valid = [w for w in unique_sorted if min_w <= w <= max_window]
    if not valid:
        return [max_window]
    return valid


def compute_wallet_mad_for_window(
    wallet_trades: pd.DataFrame,
    w: int,
    min_trades: int = 20,
) -> float:
    """Compute the Benford MAD score for a single wallet in a window of size w."""
    if wallet_trades.empty or "ledger_close_time" not in wallet_trades.columns:
        return float("nan")

    times = pd.to_datetime(wallet_trades["ledger_close_time"], utc=True)
    ref = times.max()
    window_start = ref - pd.Timedelta(hours=w)
    window_df = wallet_trades[(times > window_start) & (times <= ref)]

    n = int((window_df["amount"] > 0).sum())
    if n < min_trades:
        return float("nan")

    return mad_score(window_df["amount"])


def optimize_windows_for_asset(
    asset_code: str,
    trades_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
    num_windows: int = 5,
    n_calls: int = 15,
) -> list[int]:
    """Optimize window sizes for a single asset via Bayesian Optimization."""
    # Filter trades to those involving the asset
    if ":" in asset_code:
        asset_trades = trades_df[
            (trades_df["base_asset"] == asset_code) | (trades_df["counter_asset"] == asset_code)
        ]
    else:
        # Code-only match
        base_codes = trades_df["base_asset"].str.split(":").str[0]
        counter_codes = trades_df["counter_asset"].str.split(":").str[0]
        asset_trades = trades_df[(base_codes == asset_code) | (counter_codes == asset_code)]

    if asset_trades.empty:
        return config.BENFORD_WINDOWS_HOURS

    # Get labels
    valid_labels = labelled_df[labelled_df["label"].notna()]
    if valid_labels.empty:
        return config.BENFORD_WINDOWS_HOURS

    # Find wallets that traded this asset and have a label
    asset_wallets = set(pd.unique(asset_trades[["base_account", "counter_account"]].values.ravel()))
    target_wallets = valid_labels[valid_labels["wallet"].isin(asset_wallets)]
    if target_wallets.empty:
        return config.BENFORD_WINDOWS_HOURS

    # Estimate trades per hour
    trades_per_hour = estimate_trades_per_hour(asset_trades)
    min_trades = config.MIN_TRADES_FOR_SCORING

    min_w = math.ceil(min_trades / trades_per_hour) if trades_per_hour > 0 else 720
    min_w = max(1, min_w)
    max_w = 720

    if min_w >= max_w:
        return config.BENFORD_WINDOWS_HOURS

    # Pre-group trades by wallet for performance
    wallet_to_trades = {}
    for wallet in target_wallets["wallet"]:
        mask = (asset_trades["base_account"] == wallet) | (asset_trades["counter_account"] == wallet)
        wallet_to_trades[wallet] = asset_trades[mask]

    # Search space
    space = [Integer(min_w, max_w, name="window_size")]

    history = {}

    def objective(params):
        w = params[0]
        if w in history:
            return -history[w]

        mads = []
        labels = []
        for _, row in target_wallets.iterrows():
            wallet = row["wallet"]
            label = row["label"]
            wt = wallet_to_trades[wallet]
            mad = compute_wallet_mad_for_window(wt, w, min_trades=min_trades)
            mads.append(mad)
            labels.append(label)

        mads_filled = np.nan_to_num(mads, nan=0.0)

        if len(set(labels)) < 2:
            pr_auc = 0.0
        else:
            pr_auc = average_precision_score(labels, mads_filled)

        history[w] = pr_auc
        return -pr_auc

    # Run gp_minimize
    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=min(5, n_calls),
        random_state=42,
    )

    best_w = int(res.x[0])

    # Construct window schedule log-spaced between min_w and best_w
    if min_w < best_w:
        windows = np.logspace(np.log10(min_w), np.log10(best_w), num=num_windows)
        windows = np.round(windows).astype(int)
        windows = sorted(list(set(windows)))
        while len(windows) < num_windows:
            inserted = False
            for val in range(min_w, best_w + 1):
                if val not in windows:
                    windows.append(val)
                    windows.sort()
                    inserted = True
                    break
            if not inserted:
                for val in range(best_w + 1, max_w + 1):
                    if val not in windows:
                        windows.append(val)
                        windows.sort()
                        inserted = True
                        break
                if not inserted:
                    break
    else:
        windows = np.logspace(np.log10(min_w), np.log10(max_w), num=num_windows)
        windows = np.round(windows).astype(int)
        windows = sorted(list(set(windows)))
        while len(windows) < num_windows:
            inserted = False
            for val in range(min_w, max_w + 1):
                if val not in windows:
                    windows.append(val)
                    windows.sort()
                    inserted = True
                    break
            if not inserted:
                break

    return windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Per-Asset Benford Window Selection via Bayesian Optimization"
    )
    parser.add_argument("--trades", required=True, help="Raw trades Parquet file")
    parser.add_argument("--labelled", required=True, help="Labelled dataset Parquet file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save JSON files (defaults to config.MODEL_DIR)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or config.MODEL_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading trades from {args.trades}...")
    trades_df = pd.read_parquet(args.trades)
    print(f"Loading labels from {args.labelled}...")
    labelled_df = pd.read_parquet(args.labelled)

    assets = set(trades_df["base_asset"].unique()).union(set(trades_df["counter_asset"].unique()))
    assets = {a for a in assets if a and a != "native" and not a.startswith("XLM:")}

    print(f"Found {len(assets)} unique assets to optimize.")

    for asset in sorted(assets):
        asset_code = asset.split(":")[0]
        print(f"Optimizing windows for asset {asset} (code: {asset_code})...")

        windows = optimize_windows_for_asset(asset, trades_df, labelled_df)
        print(f"Optimized windows for {asset}: {windows}")

        output_path = Path(output_dir) / f"{asset_code}_benford_windows.json"
        with open(output_path, "w") as f:
            json.dump(windows, f)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
