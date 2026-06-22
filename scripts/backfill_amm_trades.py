"""Bulk-load AMM pool trade history and compute cross-venue features.

Usage:
    python -m scripts.backfill_amm_trades \\
        --pool-ids <pool_id1> <pool_id2> \\
        --since 2024-01-01 \\
        --until 2024-06-30 \\
        --sdex-trades data/raw_trades.parquet \\
        --output data/labelled_with_cross_venue.parquet

The script joins AMM pool trade history with existing SDEX historical trades
by timestamp and computes cross-venue features for every wallet in the combined
trade set.  The result is written as Parquet to ``--output``.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config import config
from detection.cross_venue_features import (
    build_coordination_graph,
    compute_cross_venue_features,
    detect_coordinated_clusters,
)
from ingestion.amm_pool_loader import PoolNotFoundError, load_amm_pool_trades
from utils.logging import get_logger

logger = get_logger(__name__)


def _load_amm_trades_for_pools(
    pool_ids: list[str],
    since: datetime,
    until: datetime,
) -> pd.DataFrame:
    frames = []
    for pool_id in pool_ids:
        logger.info("Loading AMM trades for pool %s …", pool_id)
        try:
            df = load_amm_pool_trades(pool_id, since, until)
            if not df.empty:
                df["pool_id"] = pool_id
                frames.append(df)
                logger.info("  → %d trades loaded", len(df))
            else:
                logger.info("  → no trades in range")
        except PoolNotFoundError:
            logger.warning("Pool %s not found — skipping", pool_id)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _compute_features_for_wallets(
    sdex_trades: pd.DataFrame,
    amm_trades: pd.DataFrame,
) -> pd.DataFrame:
    if sdex_trades.empty and amm_trades.empty:
        return pd.DataFrame()

    all_trades = pd.concat([sdex_trades, amm_trades], ignore_index=True)
    wallets = set()
    for col in ("base_account", "counter_account"):
        if col in all_trades.columns:
            wallets.update(all_trades[col].dropna().tolist())
    wallets.discard("")

    logger.info("Building coordination graph for %d wallets …", len(wallets))
    graph = build_coordination_graph(sdex_trades, amm_trades, window_seconds=10)
    clusters = detect_coordinated_clusters(graph)
    logger.info("Louvain: %d clusters detected", len(clusters))

    rows = []
    for wallet in wallets:
        features = compute_cross_venue_features(wallet, sdex_trades, amm_trades, clusters, graph)
        features["wallet"] = wallet
        rows.append(features)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill AMM trade history and cross-venue features"
    )
    parser.add_argument(
        "--pool-ids",
        nargs="+",
        default=config.WATCHED_AMM_POOLS,
        help="AMM pool IDs (64-char hex). Defaults to WATCHED_AMM_POOLS from config.",
    )
    parser.add_argument(
        "--since",
        default="2024-01-01",
        help="Start date (inclusive), ISO format YYYY-MM-DD",
    )
    parser.add_argument(
        "--until",
        default="2024-06-30",
        help="End date (inclusive), ISO format YYYY-MM-DD",
    )
    parser.add_argument(
        "--sdex-trades",
        default=None,
        help="Path to existing SDEX trades Parquet (optional). Enables cross-venue features.",
    )
    parser.add_argument(
        "--output",
        default="data/labelled_with_cross_venue.parquet",
        help="Output Parquet path",
    )
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
    until = datetime.fromisoformat(args.until).replace(tzinfo=UTC)

    pool_ids: list[str] = args.pool_ids or []
    if not pool_ids:
        logger.error("No pool IDs specified. Set --pool-ids or WATCHED_AMM_POOLS in .env")
        raise SystemExit(1)

    amm_trades = _load_amm_trades_for_pools(pool_ids, since, until)
    logger.info("Total AMM trades loaded: %d", len(amm_trades))

    sdex_trades: pd.DataFrame
    if args.sdex_trades:
        sdex_path = Path(args.sdex_trades)
        if sdex_path.exists():
            sdex_trades = pd.read_parquet(sdex_path)
            logger.info("SDEX trades loaded: %d rows from %s", len(sdex_trades), sdex_path)
        else:
            logger.warning("SDEX trades file not found: %s — proceeding with AMM only", sdex_path)
            sdex_trades = pd.DataFrame()
    else:
        sdex_trades = pd.DataFrame()

    features_df = _compute_features_for_wallets(sdex_trades, amm_trades)

    if features_df.empty:
        logger.warning("No features computed — output file will not be written")
        raise SystemExit(0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    logger.info("Cross-venue features written to %s (%d wallets)", output_path, len(features_df))


if __name__ == "__main__":
    main()
