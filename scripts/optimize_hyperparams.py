"""CLI entry point for automated hyperparameter optimisation (Issue #213).

Runs BOHB-style TPE / NSGA-II optimisation via Optuna for each of the three
LedgerLens ensemble models (Random Forest, XGBoost, LightGBM), then writes
``models/best_hyperparams.json`` and optionally integrates the results into the
training pipeline.

Usage
-----
# Optimise all three ensemble models (default 100 trials, 2 h timeout):
    python -m scripts.optimize_hyperparams \\
        --data-path data/synthetic_dataset.parquet

# Single model, fewer trials, with Pareto-front (AUC + latency):
    python -m scripts.optimize_hyperparams \\
        --data-path data/synthetic_dataset.parquet \\
        --model xgboost --n-trials 50 --multiobjective

# Override via environment variables:
    HPARAM_SEARCH_TRIALS=200 HPARAM_SEARCH_TIMEOUT_HOURS=4 \\
    python -m scripts.optimize_hyperparams --data-path ...

Environment variables
---------------------
HPARAM_SEARCH_TRIALS        int   default 100 — total trials per model
HPARAM_SEARCH_TIMEOUT_HOURS float default 2   — wall-clock timeout (hours)
HPARAM_RANDOM_STATE         int   default 42  — RNG seed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure the repo root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config
from detection.hyperparameter_search import (
    HyperparameterSearchError,
    load_best_params,
    run_multiobjective_study,
    run_study,
    save_unified_best_hyperparams,
    select_pareto_point,
    validate_hyperparams,
)
from detection.model_training import FEATURE_COLUMNS_EXCLUDE, load_training_data
from utils.logging import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ENSEMBLE_MODELS = ["random_forest", "xgboost", "lightgbm"]

_DEFAULT_TRIALS = int(os.getenv("HPARAM_SEARCH_TRIALS", "100"))
_DEFAULT_TIMEOUT_H = float(os.getenv("HPARAM_SEARCH_TIMEOUT_HOURS", "2"))
_DEFAULT_SEED = int(os.getenv("HPARAM_RANDOM_STATE", "42"))

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise LedgerLens ensemble hyperparameters with Optuna (BOHB/TPE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the labelled feature Parquet file (output of generate_synthetic_dataset.py "
        "or build_labelled_dataset.py).",
    )
    parser.add_argument(
        "--model",
        choices=ENSEMBLE_MODELS + ["all"],
        default="all",
        help="Which model(s) to optimise. 'all' runs all three ensemble models sequentially.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=_DEFAULT_TRIALS,
        help="Number of Optuna trials per model. Overrides HPARAM_SEARCH_TRIALS env var.",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=_DEFAULT_TIMEOUT_H,
        help="Wall-clock timeout in hours per model. Overrides HPARAM_SEARCH_TIMEOUT_HOURS.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out as validation set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=_DEFAULT_SEED,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel Optuna workers per study (1–4).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory to write best_hyperparams.json and per-model JSON files. "
        "Defaults to config.MODEL_DIR.",
    )
    parser.add_argument(
        "--multiobjective",
        action="store_true",
        help="Run multi-objective optimisation (AUC-ROC + inference latency) and "
        "save the Pareto front in addition to the best single-objective params.",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.75,
        help="Minimum AUC-ROC when selecting a point from the Pareto front.",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=5.0,
        help="Maximum per-sample inference latency (ms) for Pareto point selection.",
    )
    parser.add_argument(
        "--no-improvement-patience",
        type=int,
        default=30,
        help="Stop a study after this many consecutive non-improving trials (single-obj only).",
    )
    parser.add_argument(
        "--storage-url",
        default=None,
        help="Optuna storage URL (e.g. 'sqlite:///models/optuna.db'). "
        "Defaults to 'sqlite:///<model-dir>/optuna_studies.db'.",
    )
    return parser.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────


def _prepare_validation_data(
    data_path: str,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the labelled dataset and split into train / validation."""
    df = load_training_data(data_path)
    feature_cols = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    X = df[feature_cols].select_dtypes(include=["number"])
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Dataset loaded: %d rows, %d features, train=%d val=%d",
        len(df),
        X.shape[1],
        len(X_train),
        len(X_val),
    )
    return X_train, y_train, X_val, y_val


# ──────────────────────────────────────────────────────────────────────────────
# Per-model optimisation
# ──────────────────────────────────────────────────────────────────────────────


def _optimise_model(
    model_name: str,
    validation_data: tuple,
    args: argparse.Namespace,
    model_dir: Path,
) -> dict:
    """Run the optimisation for a single model; return best params dict."""
    timeout_s = args.timeout_hours * 3600.0
    storage_url = args.storage_url or f"sqlite:///{model_dir / 'optuna_studies.db'}"

    logger.info("── Optimising %s (%d trials, %.1f h timeout) ──", model_name, args.n_trials, args.timeout_hours)

    try:
        best_params = run_study(
            model_name=model_name,
            n_trials=args.n_trials,
            validation_data=validation_data,
            n_jobs=args.n_jobs,
            storage_url=storage_url,
            timeout_seconds=timeout_s,
            random_state=args.random_state,
            no_improvement_patience=args.no_improvement_patience,
        )
    except HyperparameterSearchError as exc:
        logger.error("Optimisation failed for %s: %s", model_name, exc)
        # Fall back to previously persisted params if available.
        persisted = load_best_params(model_name, model_dir=model_dir)
        if persisted:
            logger.warning("Falling back to previously persisted params for %s.", model_name)
            return persisted
        return {}

    # Security: validate bounds before accepting
    try:
        validate_hyperparams(model_name, best_params)
    except ValueError as exc:
        logger.error(
            "Bounds validation failed for %s (this should never happen): %s",
            model_name,
            exc,
        )
        return {}

    if args.multiobjective:
        logger.info("Running multi-objective study for %s …", model_name)
        try:
            pareto = run_multiobjective_study(
                model_name=model_name,
                n_trials=max(args.n_trials // 2, 10),
                validation_data=validation_data,
                n_jobs=args.n_jobs,
                storage_url=storage_url,
                timeout_seconds=timeout_s / 2,
                random_state=args.random_state,
            )
            chosen = select_pareto_point(
                pareto,
                min_auc=args.min_auc,
                max_latency_ms=args.max_latency_ms,
            )
            if chosen:
                logger.info(
                    "Pareto point selected for %s: AUC=%.4f  latency=%.3f ms",
                    model_name,
                    chosen["auc"],
                    chosen["latency_ms"],
                )
                # Use the Pareto-selected params only if their AUC is not worse
                # than the single-objective best (within 0.5 %).
                pareto_params = chosen["params"]
                try:
                    validate_hyperparams(model_name, pareto_params)
                    best_params = pareto_params
                except ValueError:
                    logger.warning("Pareto params failed validation; keeping single-obj best.")
            else:
                logger.warning(
                    "No Pareto point met constraints (min_auc=%.2f, max_latency=%.1f ms) "
                    "for %s. Keeping single-objective best.",
                    args.min_auc,
                    args.max_latency_ms,
                    model_name,
                )
        except HyperparameterSearchError as exc:
            logger.warning("Multi-objective study failed for %s: %s. Continuing.", model_name, exc)

    return best_params


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 = success)."""
    args = _parse_args(argv)

    model_dir = Path(args.model_dir) if args.model_dir else Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Validate numeric arguments eagerly so we fail before touching the data.
    if args.n_trials <= 0:
        logger.error("--n-trials must be positive, got %d", args.n_trials)
        return 1
    if args.timeout_hours <= 0:
        logger.error("--timeout-hours must be positive, got %.2f", args.timeout_hours)
        return 1
    if not (1 <= args.n_jobs <= 4):
        logger.error("--n-jobs must be in [1, 4], got %d", args.n_jobs)
        return 1

    # Prepare shared validation data once.
    try:
        validation_data = _prepare_validation_data(
            args.data_path, args.val_size, args.random_state
        )
    except Exception as exc:
        logger.error("Failed to load training data: %s", exc)
        return 1

    models_to_run = ENSEMBLE_MODELS if args.model == "all" else [args.model]
    all_best: dict[str, dict] = {}

    for model_name in models_to_run:
        best = _optimise_model(model_name, validation_data, args, model_dir)
        if best:
            all_best[model_name] = best
            logger.info("Best params for %s: %s", model_name, json.dumps(best, indent=2))
        else:
            logger.warning("No best params obtained for %s — skipping in unified JSON.", model_name)

    if all_best:
        out_path = save_unified_best_hyperparams(all_best, model_dir=model_dir)
        logger.info("Unified best_hyperparams.json written → %s", out_path)
        print(f"\nResults written to {out_path}")
        print(json.dumps(all_best, indent=2))
    else:
        logger.error("No hyperparameters obtained for any model.")
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    sys.exit(main())
