"""Full LedgerLens detection pipeline entry point.

Usage:
    python run_pipeline.py --since 2024-01-01

Pipeline stages:
    1. Load historical trades per asset pair into a dict keyed by pair_id
    2. For each pair:
       a. Load order-book events (optional, skipped with ``--no-orderbook``)
       b. Load account activity and build the wallet funding graph (optional,
          skipped with ``--no-graph``)
       c. Build the per-wallet feature matrix
       d. Score each wallet with the trained ensemble (model_inference)
       e. Persist one RiskScore record per (wallet, pair_id) and optionally
          submit flagged wallets on-chain

Stage 2d requires trained models in `config.MODEL_DIR` — run
`detection/model_training.py` against a labelled dataset first. Until
models are trained, this script falls back to reporting Benford-only flags
(and persistence is skipped, since the `RiskScore` shape isn't available).
"""

import argparse
from datetime import UTC, datetime

import pandas as pd
from stellar_sdk import Asset as SdkAsset

from config import config
from detection.feature_engineering import build_feature_matrix
from detection.risk_score_store import RiskScoreStore
from detection.wallet_graph import build_funding_graph
from ingestion.account_activity_loader import load_accounts_activity
from ingestion.historical_loader import (
    load_pair_to_dataframe,
    load_watched_pairs_to_dataframe,
)
from ingestion.orderbook_loader import load_accounts_orderbook_events
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LedgerLens detection pipeline")
    parser.add_argument(
        "--since",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO date to start loading historical trades from (default: all available)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing scored wallets to RISK_SCORE_DB_URL",
    )
    parser.add_argument(
        "--no-orderbook",
        action="store_true",
        help="Skip loading order-book events (faster, but order_cancellation_rate stays 0)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip loading account activity and building the funding graph (faster, but "
        "funding_source_similarity and network_centrality stay 0)",
    )
    parser.add_argument(
        "--submit-onchain",
        action="store_true",
        help="Submit flagged wallets' RiskScore to the ledgerlens-score contract",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all pipeline stages but skip all writes (DB persist and on-chain submission).",
    )
    return parser.parse_args()


def main() -> None:
    config.validate()
    args = parse_args()

    if args.dry_run:
        logger.info("[DRY RUN] No data will be written.")

    xlm = SdkAsset.native()

    # --- Stage 1: load trades per pair ---
    logger.info("[1] Loading trades per pair: %s", config.WATCHED_ASSET_PAIRS)
    pairs_df: dict[str, pd.DataFrame] = {}
    for code, issuer in config.WATCHED_ASSET_PAIRS:
        asset = xlm if issuer == "native" else SdkAsset(code, issuer)
        if asset == xlm:
            continue
        pair_id = f"{code}:{issuer}/XLM:native"
        logger.info("    Loading pair %s", pair_id)
        pairs_df[pair_id] = load_pair_to_dataframe(asset, xlm, start_time=args.since)
        logger.info("    Loaded %d trades for %s", len(pairs_df[pair_id]), pair_id)

    if not pairs_df:
        logger.warning("No pairs configured or all pairs skipped.")
        return

    all_flagged: list[pd.DataFrame] = []
    store = RiskScoreStore() if not args.no_persist and not args.dry_run else None

    for pair_id, trades_df in pairs_df.items():
        logger.info("[pair=%s] Processing", pair_id)

        if trades_df.empty:
            logger.info("[pair=%s] No trades — skipping", pair_id)
            continue

        wallets = list(pd.unique(trades_df[["base_account", "counter_account"]].values.ravel()))

        # --- Stage 2a: order-book events ---
        orderbook_events = None
        if not args.no_orderbook:
            logger.info("[pair=%s] Loading order-book events", pair_id)
            orderbook_events = load_accounts_orderbook_events(wallets)
            logger.info("[pair=%s] Loaded %d order-book events", pair_id, len(orderbook_events))

        # --- Stage 2b: funding graph ---
        funding_graph = None
        if not args.no_graph:
            logger.info("[pair=%s] Loading account activity and building funding graph", pair_id)
            activities = load_accounts_activity(wallets)
            funding_graph = build_funding_graph(activities)
            logger.info(
                "[pair=%s] Funding graph: %d nodes, %d edges",
                pair_id,
                funding_graph.number_of_nodes(),
                funding_graph.number_of_edges(),
            )

        # --- Stage 2c: feature matrix ---
        logger.info("[pair=%s] Building feature matrix", pair_id)
        feature_matrix = build_feature_matrix(
            trades_df, orderbook_events=orderbook_events, funding_graph=funding_graph
        )
        logger.info("[pair=%s] Built features for %d wallets", pair_id, len(feature_matrix))

        # --- Stage 2d: scoring ---
        logger.info("[pair=%s] Scoring wallets", pair_id)
        try:
            from detection.model_inference import RiskScorer

            scorer = RiskScorer()
            scored = scorer.score_matrix(feature_matrix)
        except (RuntimeError, ImportError) as exc:
            logger.warning("[pair=%s] Skipping ML scoring: %s", pair_id, exc)
            logger.warning("[pair=%s] Falling back to Benford-only flags", pair_id)
            mad_cols = [c for c in feature_matrix.columns if c.startswith("benford_mad_")]
            scored = feature_matrix[["wallet"] + mad_cols].copy()
            scored["benford_flag"] = (scored[mad_cols] > 0.015).any(axis=1)

        # --- Stage 2e: persist + flag ---
        if "score" in scored:
            flagged = scored[scored["score"] >= config.RISK_SCORE_FLAG_THRESHOLD]

            if store is not None:
                for _, row in scored.iterrows():
                    store.upsert(
                        wallet=row["wallet"],
                        asset_pair=pair_id,
                        risk_score={
                            "score": row["score"],
                            "benford_flag": row["benford_flag"],
                            "ml_flag": row["ml_flag"],
                            "confidence": row["confidence"],
                        },
                    )
                logger.info("[pair=%s] Persisted %d scored wallets", pair_id, len(scored))
        else:
            flagged = scored[scored["benford_flag"]]

        logger.info("[pair=%s] Flagged wallets (%d):\n%s", pair_id, len(flagged), flagged)
        all_flagged.append(flagged)

        if args.submit_onchain:
            if args.dry_run:
                logger.warning("[pair=%s] [DRY RUN] Skipping on-chain submission", pair_id)
            elif "score" not in scored:
                logger.warning(
                    "[pair=%s] Skipping on-chain submission: no ML scores available", pair_id
                )
            else:
                submit_flagged_onchain(flagged, pair_id)

    combined_flagged = pd.concat(all_flagged, ignore_index=True) if all_flagged else pd.DataFrame()
    logger.info("Total flagged wallets across all pairs: %d", len(combined_flagged))


def submit_flagged_onchain(flagged: pd.DataFrame, pair_id: str) -> None:
    """Submit each flagged wallet's `RiskScore` to the `ledgerlens-score` contract."""
    from integrations.contract_client import LedgerLensContractClient

    client = LedgerLensContractClient()
    timestamp = int(datetime.now(UTC).timestamp())

    for _, row in flagged.iterrows():
        risk_score = {
            "score": int(row["score"]),
            "benford_flag": bool(row["benford_flag"]),
            "ml_flag": bool(row["ml_flag"]),
            "timestamp": timestamp,
            "confidence": int(row["confidence"]),
        }
        client.submit_score(wallet=row["wallet"], asset_pair=pair_id, risk_score=risk_score)

    logger.info("      Submitted %d RiskScores on-chain for %s", len(flagged), pair_id)


if __name__ == "__main__":
    main()
