"""Train the LedgerLens ensemble classifiers (RF, XGBoost, LightGBM).

Run as a script against a labelled feature matrix (see
`scripts/generate_synthetic_dataset.py` for a synthetic one, or the
"Open dataset release" roadmap item for the real thing):

    python -m detection.model_training --data-path data/synthetic_dataset.parquet

This trains each model in `MODEL_REGISTRY` with SMOTE-balanced training
data, evaluates AUC-ROC / PR-AUC / F1 on a held-out split, writes the
artifacts to `config.MODEL_DIR`, and writes `metrics.json` alongside them.

After every training run, `metrics.json` is signed with the Ed25519 private
key at `MODEL_SIGNING_PRIVATE_KEY_PATH` (if configured).

Pass `--calibrate-ensemble` to additionally run NSGA-II Pareto front search
over ensemble combination weights (see `detection/ensemble_calibrator.py`)
and write `models/pareto_front.json`.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}

FEATURE_COLUMNS_EXCLUDE = {"wallet", "label", "profile"}
PSI_N_BINS = 10
PSI_EPSILON = 1e-4


def compute_feature_distributions(
    X: pd.DataFrame,
    n_bins: int = PSI_N_BINS,
) -> dict[str, dict]:
    """Compute per-feature bin edges and expected proportions from training data.

    Each feature is discretised into `n_bins` quantile-based bins. If there are
    insufficient unique values for quantile binning, uniform-width bins are used
    as a fallback. Expected proportions are clipped to >= PSI_EPSILON to prevent
    log(0) errors in downstream PSI computation.

    Returns:
        {feature_name: {"bin_edges": list[float], "expected_proportions": list[float]}}
    """
    distributions = {}
    for col in X.columns:
        col_data = X[col].dropna().values
        if len(col_data) == 0:
            distributions[col] = {
                "bin_edges": [0.0, 1.0],
                "expected_proportions": [1.0],
            }
            continue

        if len(np.unique(col_data)) >= n_bins:
            try:
                _, bin_edges = pd.qcut(col_data, q=n_bins, retbins=True, duplicates="drop")
            except ValueError:
                bin_edges = np.histogram_bin_edges(col_data, bins=n_bins)
        else:
            bin_edges = np.histogram_bin_edges(col_data, bins=min(n_bins, len(np.unique(col_data))))

        bin_edges = np.unique(bin_edges)
        counts, _ = np.histogram(col_data, bins=bin_edges)
        total = counts.sum()
        expected = np.maximum(counts / total, PSI_EPSILON) if total > 0 else np.ones_like(counts)
        expected = expected / expected.sum()

        distributions[col] = {
            "bin_edges": bin_edges.tolist(),
            "expected_proportions": expected.tolist(),
        }

    return distributions


def compute_feature_schema_hash(feature_columns: list[str]) -> str:
    """Compute a SHA-256 hash of the sorted feature column names."""
    sorted_cols = sorted(feature_columns)
    schema_str = "\n".join(sorted_cols)
    return f"sha256:{hashlib.sha256(schema_str.encode()).hexdigest()}"


LABEL_DISTRIBUTION_BASELINE_PATH = os.path.join(
    config.MODEL_DIR, "label_distribution_baseline.json"
)


def load_training_data(path: str) -> pd.DataFrame:
    """Load a labelled feature matrix (output of `build_feature_matrix` plus
    a `label` column: 1 = wash trading, 0 = legitimate)."""
    return pd.read_parquet(path)


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]
    return df[feature_cols], df["label"]


def sha256_dataframe(df: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 of *df* (row-sorted for reproducibility)."""
    sorted_df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    h = hashlib.sha256(sorted_df.to_csv(index=False).encode()).hexdigest()
    return h


def detect_label_poisoning(
    label_distribution: dict,
    baseline_path: str | None = None,
    threshold: float | None = None,
) -> bool:
    """Return True if the wash-trade label ratio has shifted beyond *threshold*
    compared with the stored baseline.

    If no baseline file exists yet, one is written and False is returned.
    """
    baseline_path = baseline_path or LABEL_DISTRIBUTION_BASELINE_PATH
    threshold = threshold if threshold is not None else config.POISON_LABEL_RATIO_THRESHOLD

    total = sum(label_distribution.values())
    if total == 0:
        return False
    current_ratio = label_distribution.get(1, 0) / total

    if not os.path.exists(baseline_path):
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump({"wash_trade_ratio": current_ratio}, f)
        return False

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_ratio = baseline.get("wash_trade_ratio", current_ratio)
    return abs(current_ratio - baseline_ratio) > threshold


def _adversarial_augment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    aug_ratio: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Augment training data with feature-space perturbations mimicking AmountJitter."""
    if aug_ratio <= 0:
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    wash_mask = y_train == 1
    X_wash = X_train[wash_mask]
    n_aug = max(1, int(len(X_wash) * aug_ratio))

    idx = rng.choice(len(X_wash), size=n_aug, replace=True)
    X_aug = X_wash.iloc[idx].copy().reset_index(drop=True)
    noise = rng.normal(1.0, 0.005, size=X_aug.shape)
    X_aug = X_aug * noise
    y_aug = pd.Series([1] * n_aug, name=y_train.name)

    X_out = pd.concat([X_train, X_aug], ignore_index=True)
    y_out = pd.concat([y_train, y_aug], ignore_index=True)
    return X_out, y_out


def train_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    adversarial_augmentation: bool = False,
    aug_ratio: float | None = None,
) -> dict:
    """Train all models in `MODEL_REGISTRY` and return fitted estimators
    plus evaluation metrics and split info.

    Returns:
        {
          "results": {
            "random_forest": {"model": ..., "metrics": {...}},
            ...
          },
          "feature_columns": [...],
          "feature_distributions": {...},
          "n_train": int,
          "n_test": int,
          "X_test": pd.DataFrame,
          "y_test": pd.Series,
        }

    If ``adversarial_augmentation`` is True, ``auc_roc_adversarial`` is also
    included in each model's metrics dict.
    """
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if adversarial_augmentation:
        ratio = aug_ratio if aug_ratio is not None else config.ADVERSARIAL_AUG_RATIO
        if ratio <= 0:
            logger.warning(
                "ADVERSARIAL_AUG_RATIO is 0 — augmentation requested but ratio is 0. "
                "Set ADVERSARIAL_AUG_RATIO > 0 in config/.env to enable."
            )
        X_train, y_train = _adversarial_augment(X_train, y_train, ratio, random_state)
        logger.info("Adversarial augmentation: training set expanded to %d rows", len(X_train))

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    rng = np.random.default_rng(random_state)
    noise = rng.normal(1.0, 0.005, size=X_test.shape)
    X_test_adv = X_test * noise

    results = {}
    for name, model_cls in MODEL_REGISTRY.items():
        model = model_cls(random_state=random_state)
        model.fit(X_train_res, y_train_res)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        probs_adv = model.predict_proba(X_test_adv)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, probs)

        metrics = {
            "auc_roc": float(roc_auc_score(y_test, probs)),
            "pr_auc": float(auc(recall, precision)),
            "f1": float(f1_score(y_test, preds)),
        }
        if adversarial_augmentation:
            metrics["auc_roc_adversarial"] = float(roc_auc_score(y_test, probs_adv))

        results[name] = {
            "model": model,
            "metrics": metrics,
        }

    return {
        "results": results,
        "feature_columns": list(X.columns),
        "feature_distributions": compute_feature_distributions(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "X_test": X_test,
        "y_test": y_test,
    }


def save_models(results: dict, model_dir: str | None = None) -> None:
    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    to_save = results.get("results", results) if isinstance(results, dict) else results
    for name, result in to_save.items():
        joblib.dump(result["model"], os.path.join(model_dir, f"{name}.joblib"))


def save_training_artifacts(
    training_output: dict,
    data_path: str,
    model_dir: str | None = None,
) -> None:
    """Write metrics.json and model_metadata.json to the model directory."""
    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    results = training_output["results"]
    feature_columns = training_output["feature_columns"]
    feature_distributions = training_output.get("feature_distributions")

    # Save metrics.json
    metrics_path = os.path.join(model_dir, "metrics.json")
    metrics_payload = {name: result["metrics"] for name, result in results.items()}
    for name in results:
        artifact_path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(artifact_path):
            sha = hashlib.sha256()
            with open(artifact_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            metrics_payload[name]["artifact_sha256"] = sha.hexdigest()

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save model_metadata.json
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    metadata = {
        "trained_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "data_path": data_path,
        "n_training_rows": training_output["n_train"],
        "n_test_rows": training_output["n_test"],
        "feature_columns": feature_columns,
        "feature_schema_hash": compute_feature_schema_hash(feature_columns),
        "model_names": list(results.keys()),
        "python_version": sys.version.split()[0],
        "ledgerlens_version": "0.2.0",
        "feature_distributions": feature_distributions,
    }

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved model metadata to %s", metadata_path)


def save_metrics_report(
    results: dict,
    model_dir: str | None = None,
    extra: dict | None = None,
) -> str:
    """Write metrics (plus optional *extra* provenance fields) to metrics.json."""
    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "metrics.json")

    payload: dict = {name: result["metrics"] for name, result in results.items()}

    for name in results:
        artifact_path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(artifact_path):
            sha = hashlib.sha256()
            with open(artifact_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            payload[name]["artifact_sha256"] = sha.hexdigest()

    if extra:
        payload.update(extra)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LedgerLens ensemble classifiers")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--adversarial-augmentation",
        action="store_true",
        default=False,
        help=(
            "Augment training data with AmountJitter / TemporalSpreading-style "
            "perturbed copies of wash-trade rows. Augmentation ratio is controlled "
            "by ADVERSARIAL_AUG_RATIO in config / .env (default 0.0 = disabled)."
        ),
    )
    parser.add_argument(
        "--with-gnn",
        action="store_true",
        default=False,
        help=(
            "Pre-train a GraphSAGE encoder on the full training graph using "
            "contrastive link-prediction loss, then append GNN embedding features "
            "(gnn_0 … gnn_{GNN_EMBEDDING_DIM-1}) to each wallet's feature row. "
            "Requires torch and torch_geometric to be installed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir or config.MODEL_DIR

    logger.info("Loading training data from %s", args.data_path)
    df = load_training_data(args.data_path)
    logger.info("Loaded %d rows", len(df))

    data_sha = sha256_dataframe(df)
    label_dist = df["label"].value_counts().to_dict()
    logger.info("training_data_sha256=%s  label_distribution=%s", data_sha, label_dist)

    if detect_label_poisoning(label_dist):
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        os.makedirs("reports", exist_ok=True)
        alert_path = f"reports/poisoning_alert_{ts}.json"
        with open(alert_path, "w") as f:
            json.dump(
                {
                    "detected_at": ts,
                    "label_distribution": label_dist,
                    "training_data_sha256": data_sha,
                },
                f,
                indent=2,
            )
        logger.critical(
            "LABEL POISONING DETECTED — wash-trade ratio shifted beyond threshold. "
            "Training aborted. Alert written to %s",
            alert_path,
        )
        return

    # --with-gnn: pre-train GNN encoder and append embedding features
    if args.with_gnn:
        try:
            import networkx as nx

            from detection.gnn_encoder import GNNEncoder, pretrain_gnn_contrastive

            logger.info("Building wallet graph for GNN pre-training…")
            # Build a simple co-occurrence graph from wallet column for pre-training
            encoder = GNNEncoder(model_dir=model_dir, random_state=args.random_state)

            # Build a minimal funding graph from the training data
            # (wallets with label=1 form synthetic wash rings for contrastive training)
            graph = nx.DiGraph()
            wallets = df["wallet"].tolist() if "wallet" in df.columns else []
            for w in wallets:
                graph.add_node(w)

            wash_wallets = (
                df.loc[df["label"] == 1, "wallet"].tolist()
                if "wallet" in df.columns and "label" in df.columns
                else []
            )
            # Group labelled wash-trade wallets into a single synthetic ring
            wash_rings = [wash_wallets] if wash_wallets else []

            logger.info(
                "GNN pre-training: %d nodes, %d wash-trade wallets in %d ring(s)",
                graph.number_of_nodes(),
                len(wash_wallets),
                len(wash_rings),
            )

            loss_curve = pretrain_gnn_contrastive(
                encoder=encoder,
                graph=graph,
                wash_ring_wallets=wash_rings,
                random_state=args.random_state,
            )

            # Persist pre-trained encoder
            os.makedirs(model_dir, exist_ok=True)
            encoder.save()
            logger.info("GNN encoder saved to %s", model_dir)

            # Log loss curve
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            os.makedirs("reports", exist_ok=True)
            loss_report_path = f"reports/gnn_pretrain_{ts}.json"
            with open(loss_report_path, "w") as f:
                json.dump({"loss_curve": loss_curve}, f, indent=2)
            logger.info("GNN pre-training loss curve written to %s", loss_report_path)

            # Append GNN embedding features to the training DataFrame
            logger.info("Appending GNN embedding features to training data…")
            gnn_features: list[dict] = []
            for wallet in wallets:
                try:
                    emb = encoder.encode(graph, wallet)
                    gnn_features.append({f"gnn_{i}": float(emb[i]) for i in range(len(emb))})
                except Exception:
                    gnn_features.append({f"gnn_{i}": 0.0 for i in range(config.GNN_EMBEDDING_DIM)})
            gnn_df = pd.DataFrame(gnn_features, index=df.index)
            df = pd.concat([df, gnn_df], axis=1)
            logger.info("GNN embedding columns added: gnn_0 … gnn_%d", config.GNN_EMBEDDING_DIM - 1)

        except ImportError as exc:
            logger.error("--with-gnn requested but torch/torch_geometric not available: %s", exc)
            logger.error("Install torch and torch_geometric to enable GNN training.")

    training_output = train_models(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        adversarial_augmentation=args.adversarial_augmentation,
    )
    results = training_output["results"]
    for name, result in results.items():
        logger.info("%s metrics: %s", name, result["metrics"])

    save_models(results, model_dir)
    save_training_artifacts(training_output, args.data_path, model_dir)

    if args.calibrate_ensemble:
        from detection.ensemble_calibrator import EnsembleCalibrator, summarize_pareto_front

        trained_models = {name: result["model"] for name, result in results.items()}
        calibrator = EnsembleCalibrator(model_dir)
        pareto_front = calibrator.run_search(
            trained_models, training_output["X_test"], training_output["y_test"]
        )
        logger.info(summarize_pareto_front(pareto_front))

    if config.MODEL_SIGNING_PRIVATE_KEY_PATH:
        from detection.persistence import sign_metrics

        metrics_path = os.path.join(model_dir, "metrics.json")
        sig_path = sign_metrics(metrics_path, config.MODEL_SIGNING_PRIVATE_KEY_PATH)
        logger.info("Signed metrics.json → %s", sig_path)
    else:
        logger.warning("MODEL_SIGNING_PRIVATE_KEY_PATH not set — metrics.json not signed")

    logger.info("Saved models and artifacts to %s", model_dir)


if __name__ == "__main__":
    main()
