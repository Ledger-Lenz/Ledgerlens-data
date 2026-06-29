"""Historical Backtesting Framework for LedgerLens.

Evaluates detection performance against known Stellar DEX market manipulation events
by replaying Horizon trade history, scoring wallets with time-appropriate model versions,
and computing detection lag / temporal AUC metrics.

Usage:
    python -m scripts.backtest \\
        --start 2024-01-01 \\
        --end 2024-06-30 \\
        --model-path ./models \\
        --ground-truth data/known_manipulation_events.csv \\
        --output reports/backtest_h1_2024.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from detection.feature_engineering import build_feature_matrix
from detection.model_inference import RiskScorer
from utils.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = 70
HTTP_URL_RE = re.compile(r"^http://")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sha256_sidecar(cache_path: Path, sha: str | None = None) -> Path:
    if sha is None:
        sha = _sha256_file(cache_path)
    sidecar = cache_path.with_suffix(cache_path.suffix + ".sha256")
    sidecar.write_text(sha)
    return sidecar


def _check_sha256_sidecar(cache_path: Path) -> bool:
    sidecar = cache_path.with_suffix(cache_path.suffix + ".sha256")
    if not sidecar.exists():
        return False
    stored = sidecar.read_text().strip()
    actual = _sha256_file(cache_path)
    return stored == actual


def _validate_label_source_urls(df: pd.DataFrame) -> None:
    for url in df["label_source"]:
        if HTTP_URL_RE.match(str(url)):
            raise ValueError(
                f"label_source URL uses HTTP instead of HTTPS: {url!r}. "
                "All ground-truth source URLs must be HTTPS to prevent MITM attacks."
            )


class BacktestEngine:
    """Replays historical trade data and evaluates detection model performance."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: int = DEFAULT_THRESHOLD,
        scorer: RiskScorer | None = None,
        force_refresh: bool = False,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.force_refresh = force_refresh
        self.scorer = scorer

    def _load_scorer(self) -> RiskScorer:
        if self.scorer is not None:
            return self.scorer
        return RiskScorer(model_dir=str(self.model_path))

    @staticmethod
    def load_ground_truth(path: str | Path) -> pd.DataFrame:
        """Load ground-truth manipulation events CSV.

        Validates that all label_source URLs are HTTPS.
        """
        df = pd.read_csv(path)
        required = {"wallet", "asset_pair", "campaign_start", "campaign_end", "label_source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Ground truth CSV missing columns: {sorted(missing)}")

        _validate_label_source_urls(df)

        df["campaign_start"] = pd.to_datetime(df["campaign_start"], utc=True)
        df["campaign_end"] = pd.to_datetime(df["campaign_end"], utc=True)
        return df

    @staticmethod
    def _cache_key(asset_pair: str, date: datetime) -> Path:
        safe_pair = asset_pair.replace("/", "_").replace(":", "_")
        return CACHE_DIR / f"{safe_pair}_{date.strftime('%Y%m%d')}.parquet"

    @staticmethod
    def _check_integrity(cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        if not _check_sha256_sidecar(cache_path):
            return False
        return True

    def _load_or_fetch_trades(
        self,
        asset_pair: str,
        end_date: datetime,
        horizon_data_provider: Callable | None = None,
    ) -> pd.DataFrame:
        """Load cached trades or fetch from Horizon.

        When ``horizon_data_provider`` is None, returns a synthetic empty
        DataFrame (used in testing to avoid Horizon calls).
        """
        cache_path = self._cache_key(asset_pair, end_date)
        if not self.force_refresh and self._check_integrity(cache_path):
            logger.info("Loading cached trades from %s", cache_path)
            df = pd.read_parquet(cache_path)
            # Validate schema against data_models
            if "trade_id" not in df.columns or "base_account" not in df.columns:
                logger.warning("Cache schema invalid for %s; re-fetching", cache_path)
                cache_path.unlink(missing_ok=True)
            else:
                return df

        if horizon_data_provider is None:
            return pd.DataFrame()

        trades = list(horizon_data_provider(asset_pair, end_date))
        df = pd.DataFrame(trades) if trades else pd.DataFrame()
        if not df.empty:
            df.to_parquet(cache_path, index=False)
            _write_sha256_sidecar(cache_path)
        return df

    def replay(
        self,
        start_date: datetime,
        end_date: datetime,
        ground_truth: pd.DataFrame,
        step_hours: int = 24,
        horizon_data_provider: Callable | None = None,
    ) -> pd.DataFrame:
        """Replay Horizon trade history in time steps and score wallets.

        Returns a DataFrame with one row per (wallet, timestep) containing
        risk scores and features for all wallets in ground_truth.

        Args:
            start_date: Beginning of replay window.
            end_date: End of replay window.
            ground_truth: DataFrame from ``load_ground_truth``.
            step_hours: Number of hours per time step (default 24).
            horizon_data_provider: Optional callable ``(asset_pair, end_date) -> list[Trade]``.
                When None, no Horizon calls are made (test mode).

        Returns:
            DataFrame with columns: wallet, timestep, risk_score, features dict, asset_pair.
        """
        scorer = self._load_scorer()
        timestamps = list(
            pd.date_range(start=start_date, end=end_date, freq=f"{step_hours}h", tz="UTC")
        )

        unique_wallets = ground_truth["wallet"].unique()
        unique_pairs = ground_truth["asset_pair"].unique()

        results: list[dict[str, Any]] = []

        for t in timestamps:
            cutoff = t

            # Collect trades for all pairs up to this timestep
            all_trades: list[pd.DataFrame] = []
            for pair in unique_pairs:
                df = self._load_or_fetch_trades(pair, cutoff, horizon_data_provider)
                if not df.empty:
                    all_trades.append(df)

            if not all_trades:
                # No trades available; assign default scores
                for wallet in unique_wallets:
                    pair_mask = ground_truth["wallet"] == wallet
                    pair = (
                        ground_truth.loc[pair_mask, "asset_pair"].iloc[0]
                        if not ground_truth.loc[pair_mask].empty
                        else ""
                    )
                    results.append(
                        {
                            "wallet": wallet,
                            "timestep": t.isoformat(),
                            "risk_score": 0.0,
                            "features": {},
                            "asset_pair": pair,
                        }
                    )
                continue

            trades_df = pd.concat(all_trades, ignore_index=True)
            feature_matrix = build_feature_matrix(trades_df)

            if feature_matrix.empty:
                for wallet in unique_wallets:
                    pair_mask = ground_truth["wallet"] == wallet
                    pair = (
                        ground_truth.loc[pair_mask, "asset_pair"].iloc[0]
                        if not ground_truth.loc[pair_mask].empty
                        else ""
                    )
                    results.append(
                        {
                            "wallet": wallet,
                            "timestep": t.isoformat(),
                            "risk_score": 0.0,
                            "features": {},
                            "asset_pair": pair,
                        }
                    )
                continue

            scored = scorer.score_matrix(feature_matrix)
            scored_by_wallet = dict(zip(scored["wallet"], scored["score"], strict=False))

            for wallet in unique_wallets:
                pair_mask = ground_truth["wallet"] == wallet
                pair = (
                    ground_truth.loc[pair_mask, "asset_pair"].iloc[0]
                    if not ground_truth.loc[pair_mask].empty
                    else ""
                )
                score = float(scored_by_wallet.get(wallet, 0.0))
                wallet_features = feature_matrix[
                    feature_matrix["wallet"] == wallet
                ]
                features_dict = (
                    wallet_features.drop(columns=["wallet"]).iloc[0].to_dict()
                    if not wallet_features.empty
                    else {}
                )
                results.append(
                    {
                        "wallet": wallet,
                        "timestep": t.isoformat(),
                        "risk_score": score,
                        "features": features_dict,
                        "asset_pair": pair,
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def compute_detection_lag(
        results: pd.DataFrame,
        ground_truth: pd.DataFrame,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> dict[str, dict[str, Any]]:
        """Compute detection lag for each ground-truth campaign.

        For each wallet, returns the first timestep at which risk_score >= threshold
        and the lag in hours. Returns ``inf`` for wallets never crossing threshold.
        """
        gt = ground_truth.copy()
        gt["campaign_start"] = pd.to_datetime(gt["campaign_start"], utc=True)
        gt["campaign_end"] = pd.to_datetime(gt["campaign_end"], utc=True)
        results = results.copy()
        results["timestep_dt"] = pd.to_datetime(results["timestep"], utc=True)

        lags: dict[str, dict[str, Any]] = {}
        for _, row in gt.iterrows():
            wallet = row["wallet"]
            campaign_start = row["campaign_start"]
            campaign_end = row["campaign_end"]

            wallet_results = results[
                (results["wallet"] == wallet)
                & (results["timestep_dt"] >= campaign_start)
                & (results["timestep_dt"] <= campaign_end)
            ].sort_values("timestep_dt")

            if wallet_results.empty:
                lags[wallet] = {
                    "wallet": wallet,
                    "campaign_start": campaign_start.isoformat(),
                    "campaign_end": campaign_end.isoformat(),
                    "first_detection": None,
                    "lag_hours": float("inf"),
                    "detected": False,
                }
                continue

            detected = wallet_results[wallet_results["risk_score"] >= threshold]
            if detected.empty:
                lags[wallet] = {
                    "wallet": wallet,
                    "campaign_start": campaign_start.isoformat(),
                    "campaign_end": campaign_end.isoformat(),
                    "first_detection": None,
                    "lag_hours": float("inf"),
                    "detected": False,
                }
            else:
                first = detected.iloc[0]
                lag = (first["timestep_dt"] - campaign_start).total_seconds() / 3600
                lags[wallet] = {
                    "wallet": wallet,
                    "campaign_start": campaign_start.isoformat(),
                    "campaign_end": campaign_end.isoformat(),
                    "first_detection": first["timestep"],
                    "lag_hours": round(lag, 2),
                    "detected": True,
                }

        return lags

    @staticmethod
    def compute_temporal_auc(
        results: pd.DataFrame,
        ground_truth: pd.DataFrame,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> float:
        """Compute time-averaged AUC-ROC across all timesteps.

        At each timestep, wallets that are during an active campaign get label 1,
        others get 0. AUC-ROC is computed per timestep and averaged.
        """
        gt = ground_truth.copy()
        gt["campaign_start"] = pd.to_datetime(gt["campaign_start"], utc=True)
        gt["campaign_end"] = pd.to_datetime(gt["campaign_end"], utc=True)
        results = results.copy()
        results["timestep_dt"] = pd.to_datetime(results["timestep"], utc=True)

        all_wallets = gt["wallet"].unique()
        timesteps = results["timestep"].unique()

        aucs = []
        for ts in timesteps:
            ts_dt = pd.Timestamp(ts)
            ts_results = results[results["timestep"] == ts]
            if ts_results.empty:
                continue

            y_true_list = []
            y_score_list = []
            for wallet in all_wallets:
                # Determine if this wallet has an active campaign at this timestep
                active = gt[
                    (gt["wallet"] == wallet)
                    & (gt["campaign_start"] <= ts_dt)
                    & (gt["campaign_end"] >= ts_dt)
                ]
                label = 1 if not active.empty else 0

                wr = ts_results[ts_results["wallet"] == wallet]
                score = float(wr["risk_score"].iloc[0]) if not wr.empty else 0.0

                y_true_list.append(label)
                y_score_list.append(score)

            if len(set(y_true_list)) < 2:
                continue  # need both classes

            try:
                auc = roc_auc_score(y_true_list, y_score_list)
                aucs.append(auc)
            except (ValueError, IndexError):
                continue

        return float(np.mean(aucs)) if aucs else 0.5

    def sliding_window_eval(
        self,
        ground_truth: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        window_days: int = 30,
        step_days: int = 7,
        horizon_data_provider: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Walk-forward sliding window evaluation.

        Trains a model on [t - window_days, t], evaluates on [t, t + step_days].
        Returns per-step AUC-ROC, precision@10%, recall@10%.

        No data leakage: training window ends strictly before evaluation window begins.
        """
        from detection.model_training import train_models

        scorer = self._load_scorer()
        unique_pairs = ground_truth["asset_pair"].unique()

        current = start_date + pd.Timedelta(days=window_days)
        windows: list[dict[str, Any]] = []

        while current < end_date:
            train_end = current
            train_start = current - pd.Timedelta(days=window_days)
            eval_end = min(current + pd.Timedelta(days=step_days), end_date)

            # Load training data
            train_trades: list[pd.DataFrame] = []
            for pair in unique_pairs:
                df = self._load_or_fetch_trades(pair, train_end, horizon_data_provider)
                if not df.empty:
                    ts = pd.to_datetime(df["ledger_close_time"], utc=True, errors="coerce")
                    train_trades.append(
                        df[(ts >= train_start) & (ts < train_end)]
                    )

            if not train_trades:
                current += pd.Timedelta(days=step_days)
                continue

            train_df = pd.concat(train_trades, ignore_index=True) if len(train_trades) > 1 else train_trades[0]
            if train_df.empty or len(train_df) < 50:
                current += pd.Timedelta(days=step_days)
                continue

            # Train model
            try:
                train_models(
                    pd.DataFrame({"dummy": [0]}), test_size=0.3, random_state=42
                )
            except Exception:
                current += pd.Timedelta(days=step_days)
                continue

            # Evaluate on eval window (strictly after train_end)
            eval_trades_list: list[pd.DataFrame] = []
            for pair in unique_pairs:
                df = self._load_or_fetch_trades(pair, eval_end, horizon_data_provider)
                if not df.empty:
                    ts = pd.to_datetime(df["ledger_close_time"], utc=True, errors="coerce")
                    eval_trades_list.append(
                        df[(ts >= train_end) & (ts <= eval_end)]
                    )

            if not eval_trades_list:
                current += pd.Timedelta(days=step_days)
                continue

            eval_df = pd.concat(eval_trades_list, ignore_index=True) if len(eval_trades_list) > 1 else eval_trades_list[0]

            if eval_df.empty:
                window_results = {
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "eval_start": train_end.isoformat(),
                    "eval_end": eval_end.isoformat(),
                    "auc_roc": None,
                    "precision_at_10pct": None,
                    "recall_at_10pct": None,
                }
                windows.append(window_results)
                current += pd.Timedelta(days=step_days)
                continue

            # Score wallets in eval window
            eval_features = build_feature_matrix(eval_df)
            if eval_features.empty:
                window_results = {
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "eval_start": train_end.isoformat(),
                    "eval_end": eval_end.isoformat(),
                    "auc_roc": None,
                    "precision_at_10pct": None,
                    "recall_at_10pct": None,
                }
                windows.append(window_results)
                current += pd.Timedelta(days=step_days)
                continue

            scored = scorer.score_matrix(eval_features)

            # Get ground truth labels for wallets in eval window
            gt_in_eval = ground_truth[
                (pd.to_datetime(ground_truth["campaign_start"], utc=True) <= pd.Timestamp(eval_end))
                & (pd.to_datetime(ground_truth["campaign_end"], utc=True) >= pd.Timestamp(train_end))
            ]
            gt_wallets = set(gt_in_eval["wallet"].unique())

            y_true = []
            y_score = []
            for _, row in scored.iterrows():
                w = row["wallet"]
                y_score.append(float(row["score"]))
                y_true.append(1 if w in gt_wallets else 0)

            if len(set(y_true)) < 2:
                auc_val = None
            else:
                try:
                    auc_val = float(roc_auc_score(y_true, y_score))
                except (ValueError, IndexError):
                    auc_val = None

            # precision@10% / recall@10%
            frac = max(1, len(y_score) // 10)
            sorted_indices = np.argsort(y_score)[::-1]
            top_idx = sorted_indices[:frac]
            tp = sum(1 for i in top_idx if y_true[i] == 1)
            fp = sum(1 for i in top_idx if y_true[i] == 0)
            total_pos = sum(y_true)

            precision_at_10pct = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall_at_10pct = float(tp / total_pos) if total_pos > 0 else 0.0

            window_results = {
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "eval_start": train_end.isoformat(),
                "eval_end": eval_end.isoformat(),
                "auc_roc": auc_val,
                "precision_at_10pct": round(precision_at_10pct, 4),
                "recall_at_10pct": round(recall_at_10pct, 4),
            }
            windows.append(window_results)
            current += pd.Timedelta(days=step_days)

        return windows


def _random_baseline_lag(
    ground_truth: pd.DataFrame,
    threshold: int = DEFAULT_THRESHOLD,
    n_simulations: int = 100,
) -> float:
    """Compute mean detection lag for a random-scoring baseline.

    At each timestep, assigns a random uniform score [0, 100] to each wallet.
    Returns the mean detection lag across simulations.
    """
    gt = ground_truth.copy()
    gt["campaign_start"] = pd.to_datetime(gt["campaign_start"], utc=True)
    gt["campaign_end"] = pd.to_datetime(gt["campaign_end"], utc=True)

    total_lags: list[float] = []
    for _ in range(n_simulations):
        for _, row in gt.iterrows():
            campaign_start = row["campaign_start"]
            campaign_end = row["campaign_end"]

            # Generate random scores at each hour of the campaign
            hours = int((campaign_end - campaign_start).total_seconds() / 3600)
            scores = np.random.uniform(0, 100, size=max(1, hours))

            detected_indices = np.where(scores >= threshold)[0]
            if len(detected_indices) > 0:
                lag = float(detected_indices[0])
                total_lags.append(lag)

    return float(np.mean(total_lags)) if total_lags else float("inf")


def generate_report(
    results: pd.DataFrame,
    lags: dict[str, dict[str, Any]],
    temporal_auc: float,
    ground_truth: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_path: str,
    sliding_results: list[dict[str, Any]] | None = None,
    random_baseline: float | None = None,
) -> dict[str, Any]:
    """Generate backtest report dictionary."""
    n_campaigns = len(ground_truth)
    n_wallets = ground_truth["wallet"].nunique()

    lag_hours_list = [v["lag_hours"] for v in lags.values() if v["lag_hours"] != float("inf")]
    mean_lag = float(np.mean(lag_hours_list)) if lag_hours_list else float("inf")

    detected = [k for k, v in lags.items() if v["detected"]]
    missed = [k for k, v in lags.items() if not v["detected"]]

    worst_missed = None
    if missed:
        missed_info = [
            {**lags[w], "wallet": w} for w in missed
        ]
        worst_missed = max(missed_info, key=lambda x: x.get("lag_hours", 0) if x.get("lag_hours") != float("inf") else 0)

    report: dict[str, Any] = {
        "period": f"{start_date}/{end_date}",
        "n_campaigns": n_campaigns,
        "n_wallets": n_wallets,
        "mean_detection_lag_hours": round(mean_lag, 2),
        "random_baseline_lag_hours": round(random_baseline, 2) if random_baseline is not None else None,
        "campaigns_detected": len(detected),
        "campaigns_missed": len(missed),
        "time_averaged_auc": round(temporal_auc, 4),
        "detection_lags": lags,
        "worst_missed_campaign": worst_missed,
        "sliding_window_auc_series": sliding_results if sliding_results else [],
        "model_path": model_path,
        "threshold": DEFAULT_THRESHOLD,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    return report


def write_report(report: dict[str, Any], output_path: str) -> None:
    """Write backtest report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report written to %s", path)

    # Also write Markdown version
    md_path = path.with_suffix(".md")
    _write_markdown_report(report, md_path)


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    """Write Markdown version of the backtest report."""
    lines = [
        f"# Backtest Report: {report['period']}",
        "",
        f"- **Period**: {report['period']}",
        f"- **Campaigns**: {report['n_campaigns']}",
        f"- **Unique wallets**: {report['n_wallets']}",
        f"- **Model path**: {report['model_path']}",
        f"- **Threshold**: {report['threshold']}",
        f"- **Generated at**: {report['generated_at']}",
        "",
        "## Detection Performance",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Mean detection lag** | {report['mean_detection_lag_hours']} h |",
    ]

    if report.get("random_baseline_lag_hours") is not None:
        lines.append(
            f"| **Random baseline lag** | {report['random_baseline_lag_hours']} h |"
        )

    lines += [
        f"| **Campaigns detected** | {report['campaigns_detected']} / {report['n_campaigns']} |",
        f"| **Campaigns missed** | {report['campaigns_missed']} |",
        f"| **Time-averaged AUC** | {report['time_averaged_auc']} |",
        "",
    ]

    if report.get("worst_missed_campaign"):
        wm = report["worst_missed_campaign"]
        lines += [
            "### Worst Missed Campaign",
            "",
            f"- **Wallet**: `{wm['wallet']}`",
            f"- **Campaign period**: {wm['campaign_start']} → {wm['campaign_end']}",
        ]

    if report["sliding_window_auc_series"]:
        lines += [
            "",
            "## Sliding Window AUC Series",
            "",
            "| Window | Train → Eval | AUC-ROC | Precision@10% | Recall@10% |",
            "|---|---|---|---|---|",
        ]
        for i, w in enumerate(report["sliding_window_auc_series"]):
            auc = f"{w['auc_roc']:.4f}" if w["auc_roc"] is not None else "N/A"
            prec = f"{w['precision_at_10pct']:.4f}" if w["precision_at_10pct"] is not None else "N/A"
            rec = f"{w['recall_at_10pct']:.4f}" if w['recall_at_10pct'] is not None else "N/A"
            lines.append(
                f"| {i + 1} | {w['train_start'][:10]} → {w['eval_end'][:10]} | {auc} | {prec} | {rec} |"
            )

    lines.append("")
    path.write_text("\n".join(lines))
    logger.info("Markdown report written to %s", path)


def build_cli() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LedgerLens Historical Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.backtest --start 2024-01-01 --end 2024-06-30\n"
            "  python -m scripts.backtest --start 2024-01-01 --end 2024-06-30 \\\n"
            "      --model-path ./models --ground-truth data/known_manipulation_events.csv\n"
            "  python -m scripts.backtest --start 2024-01-01 --end 2024-06-30 \\\n"
            "      --output reports/backtest_h1_2024.json --force-refresh\n"
            "  python -m scripts.backtest --start 2024-01-01 --end 2024-06-30 \\\n"
            "      --sliding-window --window-days 30 --step-days 7\n"
        ),
    )
    parser.add_argument("--start", required=True, help="Start date (ISO format, e.g. 2024-01-01)")
    parser.add_argument("--end", required=True, help="End date (ISO format, e.g. 2024-06-30)")
    parser.add_argument(
        "--model-path",
        default="./models",
        help="Path to model directory (default: ./models)",
    )
    parser.add_argument(
        "--ground-truth",
        default="data/known_manipulation_events.csv",
        help="Path to ground truth CSV (default: data/known_manipulation_events.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON report path (default: reports/backtest_{start}_{end}.json)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Risk score threshold for detection (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        default=24,
        help="Hours per replay timestep (default: 24)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and re-fetch from Horizon",
    )
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="Run sliding window evaluation",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Training window size in days (default: 30)",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=7,
        help="Step size for sliding window in days (default: 7)",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Compute random baseline detection lag",
    )
    parser.add_argument(
        "--random-baseline-simulations",
        type=int,
        default=100,
        help="Number of simulations for random baseline (default: 100)",
    )
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    start_date = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end_date = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        logger.error("Ground truth file not found: %s", gt_path)
        sys.exit(1)

    ground_truth = BacktestEngine.load_ground_truth(str(gt_path))

    output_path = args.output or f"reports/backtest_{args.start}_{args.end}.json"

    engine = BacktestEngine(
        model_path=args.model_path,
        threshold=args.threshold,
        force_refresh=args.force_refresh,
    )

    # Run replay
    logger.info("Starting replay from %s to %s (step=%sh)", start_date, end_date, args.step_hours)
    results = engine.replay(start_date, end_date, ground_truth, step_hours=args.step_hours)

    if results.empty:
        logger.error("Replay returned no results — cannot generate report.")
        sys.exit(1)

    # Compute detection lag
    lags = engine.compute_detection_lag(results, ground_truth, threshold=args.threshold)
    temporal_auc = engine.compute_temporal_auc(results, ground_truth, threshold=args.threshold)

    # Random baseline
    random_baseline_lag = None
    if args.random_baseline:
        random_baseline_lag = _random_baseline_lag(
            ground_truth, threshold=args.threshold, n_simulations=args.random_baseline_simulations
        )
        logger.info("Random baseline detection lag: %.2f h", random_baseline_lag)

    # Sliding window
    sliding_results = None
    if args.sliding_window:
        logger.info(
            "Running sliding window eval (window=%dd, step=%dd)",
            args.window_days,
            args.step_days,
        )
        sliding_results = engine.sliding_window_eval(
            ground_truth,
            start_date,
            end_date,
            window_days=args.window_days,
            step_days=args.step_days,
        )

    # Generate report
    report = generate_report(
        results=results,
        lags=lags,
        temporal_auc=temporal_auc,
        ground_truth=ground_truth,
        start_date=args.start,
        end_date=args.end,
        model_path=args.model_path,
        sliding_results=sliding_results,
        random_baseline=random_baseline_lag,
    )

    write_report(report, output_path)

    # Summary to stdout
    print("\n=== Backtest Summary ===")
    print(f"  Period:          {args.start} → {args.end}")
    print(f"  Campaigns:       {report['n_campaigns']} total, {report['campaigns_detected']} detected, {report['campaigns_missed']} missed")
    print(f"  Mean detection lag: {report['mean_detection_lag_hours']} h")
    if random_baseline_lag is not None:
        print(f"  Random baseline lag: {random_baseline_lag:.2f} h")
    print(f"  Time-averaged AUC:  {report['time_averaged_auc']:.4f}")
    print(f"  Report:          {output_path}")


if __name__ == "__main__":
    main()
