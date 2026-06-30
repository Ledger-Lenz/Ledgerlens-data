"""Adaptive Per-Asset Benford Window Selection via Bayesian Optimization.

Selects the optimal Benford window length for each asset pair at runtime based on
trade volume density. For low-liquidity pairs, short windows produce noisy statistics.
For high-frequency pairs, longer windows may mask recent manipulation.

This module provides:
  - `select_optimal_window()`: picks the shortest window whose trade count exceeds
    a configurable threshold, with fallback to the longest window if none qualify.
  - `optimize_windows_for_asset()`: offline Bayesian optimization of window schedules
    per asset using labelled wash-trade data.
"""

import logging
import math
import json
import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, auc, f1_score

from config import config
from detection.benford_engine import mad_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime Adaptive Window Selection (Issue #178)
# ---------------------------------------------------------------------------


def select_optimal_window(
    pair_id: str,
    trade_counts_per_window: dict[int, int],
    min_sample_size: int | None = None,
    candidate_windows: list[int] | None = None,
) -> int:
    """Select the optimal Benford window for a pair based on trade volume density.
    
    Algorithm:
      1. If provided, use the configured candidate windows; otherwise use all windows
         in trade_counts_per_window.
      2. Find the shortest window whose trade count >= min_sample_size.
      3. If no window meets the threshold, fall back to the longest window and emit a warning.
      4. Handle edge case of all-zero trade counts (new pair never seen) by returning
         the longest window.
    
    Parameters
    ----------
    pair_id:
        Asset pair identifier (e.g. 'USDC:GA.../XLM:native'), used only for logging.
    trade_counts_per_window:
        Dict mapping window size in hours -> count of trades in that window.
        Example: {1: 5, 4: 20, 24: 100, 168: 500, 720: 2000}
    min_sample_size:
        Minimum number of trades required for a window to be considered valid.
        Defaults to config.BENFORD_MIN_SAMPLE_SIZE (default 50). Must be >= 10.
    candidate_windows:
        List of window sizes to consider (in hours). If omitted, uses sorted keys
        of trade_counts_per_window.
    
    Returns
    -------
    int:
        The selected window size in hours. Guaranteed to be a key in either
        trade_counts_per_window or candidate_windows.
    
    Raises
    ------
    ValueError:
        If min_sample_size < 10 (prevents trivially small samples).
    """
    if min_sample_size is None:
        min_sample_size = getattr(config, "BENFORD_MIN_SAMPLE_SIZE", 50)
    
    if min_sample_size < 10:
        raise ValueError(
            f"BENFORD_MIN_SAMPLE_SIZE must be >= 10, got {min_sample_size}. "
            "Threshold below 10 produces trivially small samples."
        )
    
    # Determine candidate windows
    if candidate_windows is None:
        candidate_windows = sorted(trade_counts_per_window.keys())
    
    # Edge case: all windows have zero trades (new pair never seen before)
    if not candidate_windows or all(
        trade_counts_per_window.get(w, 0) == 0 for w in candidate_windows
    ):
        logger.warning(
            f"Pair {pair_id}: all candidate windows have zero trades. "
            f"Falling back to longest window {max(candidate_windows or [1])}."
        )
        return max(candidate_windows) if candidate_windows else 720
    
    # Find the shortest window that meets the minimum sample threshold
    for window_hours in candidate_windows:
        count = trade_counts_per_window.get(window_hours, 0)
        if count >= min_sample_size:
            logger.debug(
                f"Pair {pair_id}: selected window {window_hours}h "
                f"({count} trades >= {min_sample_size} threshold)."
            )
            return window_hours
    
    # No window meets the threshold: fall back to the longest window
    max_window = max(candidate_windows)
    max_count = trade_counts_per_window.get(max_window, 0)
    logger.warning(
        f"Pair {pair_id}: no window meets minimum sample threshold ({min_sample_size}). "
        f"Falling back to longest window {max_window}h ({max_count} trades). "
        "Consider increasing BENFORD_MIN_SAMPLE_SIZE or monitoring pair liquidity."
    )
    return max_window


def estimate_trades_per_hour(asset_trades: pd.DataFrame) -> float:
    """Rolling median of trades per clock hour over the last 30 days."""
    if asset_trades.empty:
        return 0.0

    # Ensure ledger_close_time is datetime
    time_col = "ledger_close_time"
    if time_col not in asset_trades.columns:
        for col in ["timestamp", "time", "date"]:
            if col in asset_trades.columns:
                time_col = col
                break

    if time_col not in asset_trades.columns:
        # Fall back to trade count divided by hours if no timestamp column
        return 0.0

    timestamps = pd.to_datetime(asset_trades[time_col])
    max_time = timestamps.max()
    start_time = max_time - pd.Timedelta(days=30)

    recent_trades = asset_trades[timestamps >= start_time]
    if recent_trades.empty:
        return 0.0

    recent_times = pd.to_datetime(recent_trades[time_col])
    # Group by clock hour
    hourly_counts = recent_times.dt.floor("h").value_counts()

    # Reindex to include hours with 0 trades
    full_range = pd.date_range(start=start_time.floor("h"), end=max_time.floor("h"), freq="h")
    if len(full_range) == 0:
        return 0.0
    hourly_counts = hourly_counts.reindex(full_range, fill_value=0)

    return float(hourly_counts.median())


def get_candidate_grid(trades_per_hour: float, min_trades: int = 20) -> list[int]:
    """Generate log-spaced candidate windows based on trade velocity."""
    max_window = 720
    if trades_per_hour <= 0:
        min_window = max_window
    else:
        min_window = int(math.ceil(min_trades / trades_per_hour))

    min_window = max(1, min(min_window, max_window))

    if min_window >= max_window:
        return [max_window]

    # Generate log-spaced candidates between min_window and max_window (e.g. 10 candidates)
    candidates = np.logspace(np.log10(min_window), np.log10(max_window), num=10)
    candidates_int = np.unique(np.round(candidates).astype(int))
    return sorted([int(c) for c in candidates_int])


def optimize_windows_for_asset(
    asset_code: str,
    asset_trades: pd.DataFrame,
    labelled_df: pd.DataFrame,
    n_iterations: int = 8,
) -> list[int]:
    """Optimizes the Benford window schedule for a given asset using Bayesian Optimization."""
    # 1. Estimate trades per hour
    tph = estimate_trades_per_hour(asset_trades)

    # 2. Get candidate grid
    min_trades = getattr(config, "MIN_TRADES_FOR_SCORING", 20)
    candidates = get_candidate_grid(tph, min_trades)

    if not candidates:
        return sorted(config.BENFORD_WINDOWS_HOURS)

    if len(candidates) <= 5:
        # Pad/backfill to return exactly 5 windows in ascending order
        res = set(candidates)
        for fallback in [1, 4, 24, 168, 720]:
            if len(res) >= 5:
                break
            res.add(fallback)
        return sorted(list(res))[:5]

    # 3. Evaluation function for window w
    def evaluate_window(w: int) -> float:
        mads = []
        labels = []
        for _, row in labelled_df.iterrows():
            wallet = row["wallet"]
            label = row["label"]
            if pd.isna(label):
                continue

            # Get trades involving wallet in this asset
            w_trades = asset_trades[(asset_trades["base_account"] == wallet) | (asset_trades["counter_account"] == wallet)]
            if w_trades.empty:
                continue

            # Parse timestamps
            time_col = "ledger_close_time"
            if time_col not in w_trades.columns:
                for col in ["timestamp", "time", "date"]:
                    if col in w_trades.columns:
                        time_col = col
                        break

            timestamps = pd.to_datetime(w_trades[time_col])
            ref = timestamps.max()
            window_start = ref - pd.Timedelta(hours=w)
            window_amounts = w_trades.loc[(timestamps > window_start) & (timestamps <= ref), "amount"]

            # Minimum sample guard
            if len(window_amounts[window_amounts > 0]) < min_trades:
                mads.append(np.nan)
            else:
                mads.append(mad_score(window_amounts))
            labels.append(label)

        labels = np.array(labels)
        mads = np.array(mads)

        valid = ~np.isnan(mads)
        if not any(valid) or len(np.unique(labels[valid])) < 2:
            return 0.0

        # PR-AUC metric
        precision, recall, _ = precision_recall_curve(labels[valid], mads[valid])
        pr_auc = float(auc(recall, precision))

        # F1 optimization backup
        best_f1 = 0.0
        thresholds = np.percentile(mads[valid], np.linspace(0, 100, 20))
        for thresh in thresholds:
            preds = (mads[valid] >= thresh).astype(int)
            f1 = f1_score(labels[valid], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        # Return F1 or PR-AUC depending on which signal we want to prioritize, or average them.
        # PR-AUC is standard for the objective, so let's use PR-AUC.
        return pr_auc

    # Initial evaluations (min, median, max of candidate grid)
    initial_indices = [0, len(candidates) // 2, len(candidates) - 1]
    initial_indices = sorted(list(set(initial_indices)))

    X_obs = []
    y_obs = []

    evaluated_indices = set()
    for idx in initial_indices:
        w = candidates[idx]
        score = evaluate_window(w)
        X_obs.append([float(w)])
        y_obs.append(score)
        evaluated_indices.add(idx)

    # Bayesian Optimization Loop with Gaussian Process (GP) surrogate
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=42
    )

    n_iters = min(n_iterations, len(candidates) - len(evaluated_indices))
    for _ in range(n_iters):
        if not X_obs:
            break
        gp.fit(np.array(X_obs), np.array(y_obs))

        remaining_indices = [i for i in range(len(candidates)) if i not in evaluated_indices]
        if not remaining_indices:
            break

        X_candidates = np.array([[float(candidates[i])] for i in remaining_indices])

        # Expected Improvement (EI) Acquisition Function
        y_best = np.max(y_obs)
        mu, sigma = gp.predict(X_candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        improvement = mu - y_best - 0.01  # xi = 0.01
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        best_cand_idx = np.argmax(ei)
        next_idx = remaining_indices[best_cand_idx]

        w_next = candidates[next_idx]
        score_next = evaluate_window(w_next)

        X_obs.append([float(w_next)])
        y_obs.append(score_next)
        evaluated_indices.add(next_idx)

    # Final GP Fit and prediction over all candidates
    gp.fit(np.array(X_obs), np.array(y_obs))
    X_all = np.array([[float(c)] for c in candidates])
    preds = gp.predict(X_all)

    # Select top 5 candidates with the highest GP predicted score
    cand_preds = list(zip(candidates, preds))
    cand_preds.sort(key=lambda x: x[1], reverse=True)

    top_5 = [cand for cand, score in cand_preds[:5]]
    final_windows = sorted(top_5)

    while len(final_windows) < 5:
        for fallback in candidates + [1, 4, 24, 168, 720]:
            if fallback not in final_windows:
                final_windows.append(fallback)
                final_windows = sorted(final_windows)
            if len(final_windows) >= 5:
                break

    return final_windows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Offline Benford window optimization per asset")
    parser.add_argument("--trades", required=True, help="Path to raw trades Parquet file")
    parser.add_argument("--labelled", required=True, help="Path to labelled dataset Parquet file")
    parser.add_argument("--output-dir", default="models", help="Directory to save JSON results")
    args = parser.parse_args()

    # Load data
    trades_df = pd.read_parquet(args.trades)
    labelled_df = pd.read_parquet(args.labelled)

    # Find unique assets in base_asset and counter_asset columns
    assets = set()
    if "base_asset" in trades_df.columns:
        assets.update(trades_df["base_asset"].dropna().unique())
    if "counter_asset" in trades_df.columns:
        assets.update(trades_df["counter_asset"].dropna().unique())

    os.makedirs(args.output_dir, exist_ok=True)

    for asset in sorted(list(assets)):
        # Filter trades specifically for this asset
        asset_mask = (trades_df["base_asset"] == asset) | (trades_df["counter_asset"] == asset)
        asset_trades = trades_df[asset_mask]

        # Labelled dataset subset for wallets trading this asset
        wallets_with_trades = set(pd.unique(asset_trades[["base_account", "counter_account"]].values.ravel()))
        asset_labelled_df = labelled_df[labelled_df["wallet"].isin(wallets_with_trades)]

        if len(asset_labelled_df) < 5:
            # Use velocity-based fallback if not enough labels
            tph = estimate_trades_per_hour(asset_trades)
            min_trades = getattr(config, "MIN_TRADES_FOR_SCORING", 20)
            candidates = get_candidate_grid(tph, min_trades)
            res = set(candidates)
            for fallback in [1, 4, 24, 168, 720]:
                if len(res) >= 5:
                    break
                res.add(fallback)
            final_windows = sorted(list(res))[:5]
        else:
            final_windows = optimize_windows_for_asset(asset, asset_trades, asset_labelled_df)

        # Clean asset name for valid filename
        clean_name = asset.replace(":", "_").replace("/", "_")
        output_path = os.path.join(args.output_dir, f"{clean_name}_benford_windows.json")

        with open(output_path, "w") as f:
            json.dump({
                "asset": asset,
                "windows": final_windows
            }, f, indent=2)

        print(f"Optimized window schedule for asset {asset}: {final_windows} -> Saved to {output_path}")


if __name__ == "__main__":
    main()
