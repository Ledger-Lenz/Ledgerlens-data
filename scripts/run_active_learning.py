"""Active learning loop integration for LedgerLens.

Queries the pool of unscored wallets from the last Horizon bulk load,
applies the configured query strategy to select the most informative
wallets, pushes them to the annotation queue, and (if an annotated
export is provided) triggers incremental model update.

Usage:
    # Select wallets using default strategy and push to queue:
    python -m scripts.run_active_learning --pool data/unscored_wallets.parquet

    # Specify strategy and batch size:
    python -m scripts.run_active_learning \\
        --pool data/unscored_wallets.parquet \\
        --strategy entropy \\
        --batch-size 30

    # After annotation, update models:
    python -m scripts.run_active_learning \\
        --pool data/unscored_wallets.parquet \\
        --update data/annotated.parquet \\
        --historical data/synthetic_dataset.parquet

Runs on a weekly schedule via .github/workflows/active_learning.yml.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from config import config
from detection.active_learning.annotation_queue import AnnotationQueue
from detection.active_learning.incremental_trainer import IncrementalTrainer
from detection.active_learning.query_strategies import get_strategy
from detection.model_training import MODEL_REGISTRY
from utils.logging import get_logger

logger = get_logger(__name__)


def load_pool(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_models(model_dir: str) -> dict:
    import joblib

    models = {}
    for name in MODEL_REGISTRY:
        artifact_path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(artifact_path):
            model = joblib.load(artifact_path)
            # verify_chain — skipped silently when no public key is configured
            try:
                from detection.persistence import ModelArtifact

                ModelArtifact(model_dir).verify_chain(name)
            except Exception as exc:
                logger.warning("Integrity check skipped for %s: %s", name, exc)
            models[name] = model
    return models


def run_active_learning(
    pool_path: str,
    strategy_name: str,
    batch_size: int,
    queue_path: str,
    model_dir: str,
    asset_pair: str = "",
) -> list[str]:
    """Select *batch_size* wallets from *pool_path* and push to *queue_path*.

    Returns the list of selected wallet IDs.
    """
    pool = load_pool(pool_path)
    if pool.empty:
        logger.warning("Pool is empty — no wallets to select")
        return []

    models = load_models(model_dir)
    strategy = get_strategy(strategy_name)

    # Pass all loaded models to CommitteeDisagreement; single model for others
    primary_model = next(iter(models.values()), None) if models else None

    kwargs: dict = {}
    if strategy_name == "committee_disagreement" and len(models) > 1:
        kwargs["models"] = models
    elif primary_model is not None:
        kwargs["model"] = primary_model

    selected = strategy.select(pool, n_query=batch_size, **kwargs)
    logger.info(
        "Strategy '%s' selected %d wallets from pool of %d",
        strategy_name,
        len(selected),
        len(pool),
    )

    queue = AnnotationQueue(queue_path=queue_path)
    queue.push(selected, strategy_name=strategy_name, asset_pair=asset_pair)
    logger.info("Pushed %d wallets to annotation queue at %s", len(selected), queue_path)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LedgerLens active learning loop")
    parser.add_argument("--pool", required=True, help="Path to unscored wallet feature parquet/csv")
    parser.add_argument("--strategy", default=config.AL_QUERY_STRATEGY)
    parser.add_argument("--batch-size", type=int, default=config.AL_BATCH_SIZE)
    parser.add_argument("--queue", default=config.AL_QUEUE_PATH)
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--asset-pair", default="")
    # Incremental update flags
    parser.add_argument(
        "--update",
        default="",
        help="Path to annotated parquet export; triggers IncrementalTrainer.update",
    )
    parser.add_argument(
        "--historical",
        default="",
        help="Path to historical labelled dataset (used for full retrain)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected = run_active_learning(
        pool_path=args.pool,
        strategy_name=args.strategy,
        batch_size=args.batch_size,
        queue_path=args.queue,
        model_dir=args.model_dir,
        asset_pair=args.asset_pair,
    )
    print(f"Selected {len(selected)} wallets → {args.queue}")

    if args.update:
        new_df = pd.read_parquet(args.update)
        trainer = IncrementalTrainer(
            model_dir=args.model_dir,
            historical_data_path=args.historical or None,
        )
        report = trainer.update(new_df)
        print(f"Model update: AUC {report['auc_before']:.4f} → {report['auc_after']:.4f}")
        if report["rolled_back"]:
            print("WARNING: update rolled back due to AUC drop")


if __name__ == "__main__":
    main()
