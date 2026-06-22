"""Train DANN encoder and meta-learner with differential privacy (DP-SGD).

Runs DP-SGD via Opacus on the neural components that ingest confirmed
wash-trade wallet labels, compares against a non-private baseline, evaluates
membership inference resistance, and records achieved epsilon in metrics.json.

Usage:
    python -m scripts.train_dp_neural --model-dir ./models
    python -m scripts.train_dp_neural --epochs 20 --skip-meta
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import torch

from config import config
from detection.dann_encoder import train_dann_encoder
from detection.meta_learner import LeafEmbeddingExtractor
from detection.privacy.metrics import record_dp_metrics
from detection.privacy.meta_learner_dp import train_meta_learner_dp
from scripts.generate_synthetic_dataset import generate_synthetic_dataset
from utils.logging import get_logger

logger = get_logger(__name__)


def _split_features_labels(df):
    """Split wallet feature matrix without importing ensemble training deps."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    feature_df = df.drop(columns=[c for c in ("wallet", "label") if c in df.columns])
    return feature_df, df["label"]


def _load_leaf_embeddings(model_dir: str, df) -> np.ndarray:
    models = {}
    for name in ["random_forest", "xgboost", "lightgbm"]:
        path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    if not models:
        raise RuntimeError(
            f"No ensemble models in {model_dir}. Run model training first or use --synthetic-only."
        )
    X, _ = _split_features_labels(df)
    extractor = LeafEmbeddingExtractor(models)
    extractor.fit(X)
    return extractor.transform(X)


def _log_acceptance_checks(component: str, report) -> None:
    """Log issue #127 acceptance thresholds after a training run."""
    if report.auc_roc_degradation > 0.03:
        logger.warning(
            "%s AUC-ROC degradation %.3f exceeds 3%% acceptance threshold",
            component,
            report.auc_roc_degradation,
        )
    if report.membership_inference_success_rate > 0.55:
        logger.warning(
            "%s membership inference success %.1f%% exceeds 55%% threshold",
            component,
            report.membership_inference_success_rate * 100,
        )


def train_private_neural_components(
    *,
    model_dir: str | None = None,
    data_path: str | None = None,
    n_wallets: int = 200,
    epochs: int | None = None,
    seed: int = 42,
    skip_meta: bool = False,
    device: str | None = None,
) -> dict:
    model_dir = model_dir or config.MODEL_DIR
    epochs = epochs if epochs is not None else config.DP_EPOCHS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(model_dir, exist_ok=True)

    if data_path and os.path.exists(data_path):
        import pandas as pd

        df = pd.read_parquet(data_path)
    else:
        df = generate_synthetic_dataset(n_wallets=n_wallets, seed=seed)

    X, y = _split_features_labels(df)
    domains = (np.arange(len(y)) % 2).astype(np.float32)

    dann_report = train_dann_encoder(
        X.values,
        y.values,
        domains,
        epochs=epochs,
        seed=seed,
        use_dp=True,
        device=device,
    )
    torch.save(dann_report.model.state_dict(), os.path.join(model_dir, "dann_encoder.pt"))
    record_dp_metrics(
        model_dir,
        "dann_encoder",
        {
            "target_epsilon": config.DP_TARGET_EPSILON,
            "target_delta": config.DP_TARGET_DELTA,
            "achieved_epsilon": dann_report.achieved_epsilon,
            "max_grad_norm": config.DP_MAX_GRAD_NORM,
            "epochs": epochs,
            "auc_roc": dann_report.auc_roc,
            "baseline_auc_roc": dann_report.baseline_auc_roc,
            "auc_roc_degradation": dann_report.auc_roc_degradation,
            "membership_inference_success_rate": dann_report.membership_inference_success_rate,
        },
    )
    _log_acceptance_checks("dann_encoder", dann_report)
    logger.info(
        "DANN encoder saved; ε=%.4f auc=%.4f (baseline %.4f, Δ=%.4f)",
        dann_report.achieved_epsilon or 0.0,
        dann_report.auc_roc,
        dann_report.baseline_auc_roc,
        dann_report.auc_roc_degradation,
    )

    meta_report = None
    if not skip_meta:
        embeddings = _load_leaf_embeddings(model_dir, df)
        meta_report = train_meta_learner_dp(
            embeddings,
            y.values.astype(np.float32),
            epochs=epochs,
            seed=seed,
            use_dp=True,
            device=device,
        )
        torch.save(meta_report.model.state_dict(), os.path.join(model_dir, "maml_adapter_dp.pt"))
        record_dp_metrics(
            model_dir,
            "meta_learner",
            {
                "target_epsilon": config.DP_TARGET_EPSILON,
                "target_delta": config.DP_TARGET_DELTA,
                "achieved_epsilon": meta_report.achieved_epsilon,
                "max_grad_norm": config.DP_MAX_GRAD_NORM,
                "epochs": epochs,
                "auc_roc": meta_report.auc_roc,
                "baseline_auc_roc": meta_report.baseline_auc_roc,
                "auc_roc_degradation": meta_report.auc_roc_degradation,
                "membership_inference_success_rate": meta_report.membership_inference_success_rate,
            },
        )
        _log_acceptance_checks("meta_learner", meta_report)
        logger.info(
            "Meta-learner saved; ε=%.4f auc=%.4f (baseline %.4f, Δ=%.4f)",
            meta_report.achieved_epsilon or 0.0,
            meta_report.auc_roc,
            meta_report.baseline_auc_roc,
            meta_report.auc_roc_degradation,
        )

    return {"dann_encoder": dann_report, "meta_learner": meta_report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural components with DP-SGD")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--n-wallets", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-meta", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_private_neural_components(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_wallets=args.n_wallets,
        epochs=args.epochs,
        seed=args.seed,
        skip_meta=args.skip_meta,
        device=args.device,
    )


if __name__ == "__main__":
    main()
