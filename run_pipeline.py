"""Full LedgerLens detection pipeline entry point.

Usage:
    python run_pipeline.py --since 2024-01-01

Pipeline stages:
    1. Load historical trades for all watched asset pairs (ingestion)
    2. Build the per-wallet feature matrix (Benford + ML features)
    3. Score each wallet with the trained ensemble (model_inference)
    4. Persist scored wallets via `RiskScoreStore` and output those flagged
       above `config.RISK_SCORE_FLAG_THRESHOLD`

Stage 3 requires trained models in `config.MODEL_DIR` — run
`detection/model_training.py` against a labelled dataset first. Until
models are trained, this script falls back to reporting Benford-only flags
(and persistence is skipped, since the `RiskScore` shape isn't available).
"""

import argparse
from datetime import datetime

from config import config
from detection.feature_engineering import build_feature_matrix
from detection.risk_score_store import RiskScoreStore
from ingestion.historical_loader import load_watched_pairs_to_dataframe
from utils.logging import get_logger

logger = get_logger(__name__)


def watched_pairs_label() -> str:
    """A single label identifying the configured set of watched pairs.

    Used as the `asset_pair` key for persisted `RiskScore` records until
    per-pair feature attribution is implemented (the feature matrix is
    currently built across all watched pairs combined).
    """
    if not config.WATCHED_ASSET_PAIRS:
        return "ALL"
    return "+".join(f"{code}:{issuer}" for code, issuer in config.WATCHED_ASSET_PAIRS)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("[1/3] Loading trades for watched pairs: %s", config.WATCHED_ASSET_PAIRS)
    trades_df = load_watched_pairs_to_dataframe(start_time=args.since)
    logger.info("      Loaded %d trades", len(trades_df))

    logger.info("[2/3] Building feature matrix")
    feature_matrix = build_feature_matrix(trades_df)
    logger.info("      Built features for %d wallets", len(feature_matrix))

    logger.info("[3/3] Scoring wallets")
    try:
        from detection.model_inference import RiskScorer

        scorer = RiskScorer()
        scored = scorer.score_matrix(feature_matrix)
    except (RuntimeError, ImportError) as exc:
        logger.warning("      Skipping ML scoring: %s", exc)
        logger.warning("      Falling back to Benford-only flags")
        mad_cols = [c for c in feature_matrix.columns if c.startswith("benford_mad_")]
        scored = feature_matrix[["wallet"] + mad_cols].copy()
        scored["benford_flag"] = (scored[mad_cols] > 0.015).any(axis=1)

    if "score" in scored:
        flagged = scored[scored["score"] >= config.RISK_SCORE_FLAG_THRESHOLD]

        if not args.no_persist:
            asset_pair = watched_pairs_label()
            store = RiskScoreStore()
            for _, row in scored.iterrows():
                store.upsert(
                    wallet=row["wallet"],
                    asset_pair=asset_pair,
                    risk_score={
                        "score": row["score"],
                        "benford_flag": row["benford_flag"],
                        "ml_flag": row["ml_flag"],
                        "confidence": row["confidence"],
                    },
                )
            logger.info(
                "      Persisted %d scored wallets to %s", len(scored), config.RISK_SCORE_DB_URL
            )
    else:
        flagged = scored[scored["benford_flag"]]

    logger.info("Flagged wallets (%d):\n%s", len(flagged), flagged)


if __name__ == "__main__":
    main()
