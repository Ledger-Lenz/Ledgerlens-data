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

Checkpoint / resume
--------------------
Each pool is an independent, expensive Horizon fetch. Pass ``--checkpoint-file``
to make a backfill resumable: a pool's trades are cached to Parquet next to the
checkpoint file the first time they're fetched, and reused directly on a
subsequent run instead of re-fetching; a pool whose fetch raises is recorded as
failed and retried on the next run instead of aborting the whole backfill.
Without ``--checkpoint-file`` the script behaves exactly as before. See
docs/checkpointing.md.
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
from utils.checkpoint import PipelineCheckpoint
from utils.logging import get_logger

logger = get_logger(__name__)


def _load_amm_trades_for_pools(
    pool_ids: list[str],
    since: datetime,
    until: datetime,
    checkpoint: PipelineCheckpoint | None = None,
    checkpoint_dir: Path | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Load AMM trade history for every pool, optionally resuming via *checkpoint*.

    When *checkpoint* is set: a pool already recorded as done is restored from
    its cached Parquet artifact (or skipped outright if it had no trades in
    range) instead of re-fetching; a pool whose fetch raises is recorded as
    failed — logged and retried on the next ``--checkpoint-file`` run — instead
    of propagating and aborting the whole backfill. Without a checkpoint, any
    exception other than ``PoolNotFoundError`` propagates unchanged.

    Progress callback is invoked after each pool is processed with:
        (processed_count, total_count, rows_loaded, since, until)
    """
    frames = []
    pending_ids = checkpoint.pending(pool_ids) if checkpoint is not None else pool_ids
    processed_count = 0

    if checkpoint is not None:
        for pool_id in pool_ids:
            if pool_id in pending_ids:
                continue
            processed_count += 1
            cached_path = checkpoint.artifact_path(pool_id)
            if cached_path is None:
                logger.info("Pool %s already processed (no trades) — skipping", pool_id)
            else:
                logger.info("Pool %s already fetched — reusing cached %s", pool_id, cached_path)
                frames.append(pd.read_parquet(cached_path))

            rows_loaded = sum(len(f) for f in frames)
            if progress_callback:
                progress_callback(processed_count, len(pool_ids), rows_loaded, since, until)

    for pool_id in pending_ids:
        logger.info("Loading AMM trades for pool %s …", pool_id)
        try:
            df = load_amm_pool_trades(pool_id, since, until)
        except PoolNotFoundError:
            logger.warning("Pool %s not found — skipping", pool_id)
            if checkpoint is not None:
                checkpoint.record_success(pool_id, metadata={"found": False})
            processed_count += 1
            rows_loaded = sum(len(f) for f in frames)
            if progress_callback:
                progress_callback(processed_count, len(pool_ids), rows_loaded, since, until)
            continue
        except Exception as exc:
            if checkpoint is None:
                raise
            logger.exception(
                "Pool %s failed to load — recording for retry on next --resume", pool_id
            )
            checkpoint.record_failure(pool_id, exc)
            processed_count += 1
            rows_loaded = sum(len(f) for f in frames)
            if progress_callback:
                progress_callback(processed_count, len(pool_ids), rows_loaded, since, until)
            continue

        if df.empty:
            logger.info("  → no trades in range")
            if checkpoint is not None:
                checkpoint.record_success(pool_id, metadata={"rows": 0})
            processed_count += 1
            rows_loaded = sum(len(f) for f in frames)
            if progress_callback:
                progress_callback(processed_count, len(pool_ids), rows_loaded, since, until)
            continue

        df["pool_id"] = pool_id
        logger.info("  → %d trades loaded", len(df))

        if checkpoint is not None:
            assert checkpoint_dir is not None
            artifact_path = checkpoint_dir / f"backfill_amm_trades_pool_{pool_id}.parquet"
            df.to_parquet(artifact_path, index=False)
            checkpoint.record_success(
                pool_id, artifact_path=str(artifact_path), metadata={"rows": len(df)}
            )

        frames.append(df)
        processed_count += 1
        rows_loaded = sum(len(f) for f in frames)
        if progress_callback:
            progress_callback(processed_count, len(pool_ids), rows_loaded, since, until)

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
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Path to a JSON checkpoint enabling resumable per-pool backfill. Fetched "
        "pool trades are cached to Parquet alongside this file and reused on resume. "
        "See docs/checkpointing.md.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard an existing --checkpoint-file and start over. No effect without "
        "--checkpoint-file.",
    )

    # Issue #960 — dry-run support
    from cli.dry_run import DryRunPlan, add_dry_run_argument, check_dry_run

    add_dry_run_argument(parser)
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
    until = datetime.fromisoformat(args.until).replace(tzinfo=UTC)

    pool_ids: list[str] = args.pool_ids or []
    if not pool_ids:
        logger.error("No pool IDs specified. Set --pool-ids or WATCHED_AMM_POOLS in .env")
        raise SystemExit(1)

    # --- Dry-run: report what would happen and exit early ---
    if args.dry_run:
        plan = DryRunPlan(
            command="backfill-amm-trades",
            extra={
                "pool_count": str(len(pool_ids)),
                "date_range": f"{args.since} → {args.until}",
                "output": args.output,
            },
        )
        plan.add_action(
            "fetch",
            f"Horizon AMM API for {len(pool_ids)} pool(s)",
            detail=f"pools: {', '.join(pool_ids[:3])}{'...' if len(pool_ids) > 3 else ''}",
        )
        if args.sdex_trades:
            plan.add_action("read", args.sdex_trades, detail="SDEX historical trades")
        plan.add_action(
            "write",
            args.output,
            detail="cross-venue feature matrix (Parquet)",
        )
        if args.checkpoint_file:
            plan.add_action(
                "write",
                args.checkpoint_file,
                detail="checkpoint file (JSON) + per-pool Parquet cache artifacts",
            )
        check_dry_run(args, plan)
        raise SystemExit(0)

    checkpoint: PipelineCheckpoint | None = None
    checkpoint_dir: Path | None = None
    if args.checkpoint_file:
        checkpoint_dir = Path(args.checkpoint_file).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = PipelineCheckpoint.load_or_create(
            path=args.checkpoint_file,
            pipeline="backfill_amm_trades",
            fingerprint_inputs={
                "pool_ids": sorted(pool_ids),
                "since": args.since,
                "until": args.until,
            },
            fresh=args.fresh,
        )

    amm_trades = _load_amm_trades_for_pools(
        pool_ids, since, until, checkpoint=checkpoint, checkpoint_dir=checkpoint_dir
    )
    logger.info("Total AMM trades loaded: %d", len(amm_trades))

    if checkpoint is not None:
        summary = checkpoint.summary()
        logger.info(
            "[checkpoint] Run summary: %d completed, %d failed (%s)",
            summary["completed"],
            len(summary["failed"]),
            summary["failed"] or "none",
        )

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
