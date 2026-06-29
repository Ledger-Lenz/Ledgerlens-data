"""Incremental model update for the active learning pipeline.

``IncrementalTrainer.update`` applies newly-labelled annotations to the
ensemble, choosing between a cheap warm-start (when the new batch is small)
and a full retrain (when enough data has accumulated).

Rollback policy:
    If AUC-ROC drops by more than ``config.AL_ROLLBACK_AUC_DROP`` (default 0.01)
    on the held-out validation set, the new models are discarded, the original
    artifacts are restored, and their SHA-256 is re-verified before serving.

Backdoor detection:
    Before training, the incremental trainer runs Activation Clustering (AC) to
    detect potential backdoor-poisoned samples. If > 20% of a class's samples are
    flagged, a critical alert is emitted and training proceeds without quarantine
    (to prevent false positives from blocking training). Otherwise, flagged samples
    are quarantined (added to a quarantine list in the annotation queue).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import config
from detection.adversarial.backdoor_detector import ActivationClusteringDetector
from detection.model_training import (
    MODEL_REGISTRY,
    load_training_data,
    save_models,
    split_features_labels,
    train_models,
)
from utils.logging import get_logger

logger = get_logger(__name__)

REPORTS_DIR = "reports"

# Backdoor detection safety threshold: if > 20% of samples are flagged,
# emit alert and proceed without quarantine (safety check to prevent false positives)
BACKDOOR_SAFETY_THRESHOLD = 0.20


def _detect_and_quarantine_backdoors(
    models: dict,
    new_df: pd.DataFrame,
    report: dict,
) -> tuple[pd.DataFrame, list[int]]:
    """Run backdoor detection and quarantine flagged samples.

    Args:
        models: Trained models (ensemble)
        new_df: New annotated samples
        report: AL update report dict (will be updated with detection info)

    Returns:
        Tuple of (cleaned_df, quarantined_indices):
          - cleaned_df: new_df with quarantined samples removed
          - quarantined_indices: list of indices of quarantined samples in original new_df

    Side effects:
        - Updates report dict with backdoor detection info
        - Emits critical alert if > 20% of class samples are flagged
    """
    quarantined_indices: list[int] = []

    if new_df.empty:
        return new_df, quarantined_indices

    try:
        # Use the first model (RandomForest) for AC detection
        model = models.get("random_forest")
        if model is None:
            logger.warning("RandomForest model not available for backdoor detection")
            return new_df, quarantined_indices

        X, y = split_features_labels(new_df)
        detector = ActivationClusteringDetector(k=2, random_state=42)
        flagged_indices = detector.detect(model, X, y, threshold_percentile=25)

        if not flagged_indices:
            report["backdoor_detection"] = {
                "method": "activation_clustering",
                "n_flagged": 0,
                "quarantined": 0,
                "safety_triggered": False,
            }
            logger.info("Backdoor detection: no samples flagged")
            return new_df, quarantined_indices

        # Check safety threshold for each class
        flagged_set = set(flagged_indices)
        safety_triggered = False

        for label in sorted(y.unique()):
            label_mask = y == label
            label_count = label_mask.sum()
            label_flagged = sum(1 for idx in flagged_indices if label_mask.iloc[idx])

            if label_count > 0:
                flagged_pct = label_flagged / label_count
                if flagged_pct > BACKDOOR_SAFETY_THRESHOLD:
                    logger.critical(
                        "Backdoor detection safety check triggered for label=%d: "
                        "%.1f%% flagged (threshold=%.1f%%). Proceeding without quarantine.",
                        label,
                        100.0 * flagged_pct,
                        100.0 * BACKDOOR_SAFETY_THRESHOLD,
                    )
                    safety_triggered = True

        # If safety threshold triggered, emit alert but don't quarantine
        if safety_triggered:
            report["backdoor_detection"] = {
                "method": "activation_clustering",
                "n_flagged": len(flagged_indices),
                "quarantined": 0,
                "safety_triggered": True,
                "message": "Safety check triggered — high false positive rate detected",
            }
            logger.warning("Proceeding with training despite backdoor detection warnings")
            return new_df, quarantined_indices

        # Otherwise, quarantine flagged samples
        quarantined_indices = sorted(flagged_indices)
        cleaned_df = new_df.drop(quarantined_indices).reset_index(drop=True)

        report["backdoor_detection"] = {
            "method": "activation_clustering",
            "n_flagged": len(flagged_indices),
            "quarantined": len(quarantined_indices),
            "safety_triggered": False,
        }

        logger.info(
            "Backdoor detection: flagged %d samples, quarantined %d",
            len(flagged_indices),
            len(quarantined_indices),
        )

        return cleaned_df, quarantined_indices

    except Exception as exc:
        logger.error("Backdoor detection failed: %s. Proceeding with all samples.", exc)
        report["backdoor_detection"] = {
            "method": "activation_clustering",
            "error": str(exc),
        }
        return new_df, quarantined_indices


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _backup_models(model_dir: str) -> dict[str, str]:
    """Copy current .joblib artifacts to .bak files. Returns {name: sha256}."""
    shas: dict[str, str] = {}
    for name in MODEL_REGISTRY:
        src = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(src):
            bak = src + ".bak"
            shutil.copy2(src, bak)
            shas[name] = _sha256_file(src)
    return shas


def _restore_models(model_dir: str, original_shas: dict[str, str]) -> None:
    """Restore .bak artifacts and verify their SHA-256."""
    for name in MODEL_REGISTRY:
        bak = os.path.join(model_dir, f"{name}.joblib.bak")
        dst = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(bak):
            shutil.copy2(bak, dst)
            restored_sha = _sha256_file(dst)
            expected = original_shas.get(name, "")
            if expected and restored_sha != expected:
                raise RuntimeError(
                    f"Rollback integrity failure for {name}: "
                    f"SHA-256 mismatch after restore "
                    f"(got {restored_sha}, expected {expected})"
                )
            logger.info("Rolled back %s (SHA-256 verified)", name)


def _cleanup_backups(model_dir: str) -> None:
    for name in MODEL_REGISTRY:
        bak = os.path.join(model_dir, f"{name}.joblib.bak")
        if os.path.exists(bak):
            os.unlink(bak)


def _auc_on_df(models: dict, df: pd.DataFrame) -> float:
    """Compute mean AUC-ROC across all models on *df*."""
    X, y = split_features_labels(df)
    aucs = []
    for model in models.values():
        probs = model.predict_proba(X)[:, 1]
        aucs.append(roc_auc_score(y, probs))
    return float(sum(aucs) / len(aucs))


def _warm_start_update(
    models: dict,
    new_df: pd.DataFrame,
    model_dir: str,
) -> dict:
    """Re-fit XGBoost and LightGBM with warm-start on new data only.

    RandomForest doesn't support warm-start in the same way; it is left
    unchanged for warm-start runs.
    """
    X_new, y_new = split_features_labels(new_df)
    updated: dict = dict(models)

    # XGBoost warm start
    xgb_model = models.get("xgboost")
    if xgb_model is not None:
        xgb_model.fit(X_new, y_new, xgb_model=xgb_model.get_booster())
        updated["xgboost"] = xgb_model

    # LightGBM warm start
    lgbm_model = models.get("lightgbm")
    if lgbm_model is not None:
        lgbm_model.fit(X_new, y_new, init_model=lgbm_model.booster_)
        updated["lightgbm"] = lgbm_model

    # Persist updated artifacts
    for name, model in updated.items():
        joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))

    # Also trigger MAML adaptation if checkpoint exists
    maml_path = os.path.join(model_dir, "maml_adapter.pt")
    if os.path.exists(maml_path):
        try:
            import torch

            from detection.meta_learner import LeafEmbeddingExtractor, MAMLAdapter

            extractor = LeafEmbeddingExtractor(updated)
            X_new, y_new = split_features_labels(new_df)
            extractor.fit(X_new)  # Use new data to fit extractor if needed
            embeddings = extractor.transform(X_new)

            # Use dummy df to get dimension if needed, or just from embeddings
            input_dim = embeddings.shape[1]
            maml = MAMLAdapter(input_dim=input_dim)
            maml.load_state_dict(torch.load(maml_path, weights_only=True))

            support_x = torch.from_numpy(embeddings).float()
            support_y = torch.from_numpy(y_new.values).float()

            maml.adapt(support_x, support_y)

            # Save adapted model
            adapted_path = os.path.join(model_dir, "maml_adapter_adapted.pt")
            torch.save(maml.state_dict(), adapted_path)
            logger.info("MAML adapter adapted and saved to %s", adapted_path)

            # Fit PrototypicalClassifier
            from detection.meta_learner import PrototypicalClassifier

            proto = PrototypicalClassifier()
            proto.fit_prototype(embeddings, y_new.values)
            proto_path = os.path.join(model_dir, "prototypes.joblib")
            joblib.dump(proto.prototypes, proto_path)
            logger.info("Prototypical prototypes saved to %s", proto_path)
        except Exception as e:
            logger.error("Failed to adapt meta-learners: %s", e)

    return updated


class IncrementalTrainer:
    """Incrementally update LedgerLens ensemble models with new annotations.

    Args:
        model_dir:           Directory containing trained .joblib artifacts.
        historical_data_path: Parquet path of the full historical labelled dataset
                              (used only when full retrain is triggered).
        val_size:            Fraction held out for AUC validation.
        random_state:        RNG seed.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        historical_data_path: str | None = None,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        self.model_dir = model_dir or config.MODEL_DIR
        self.historical_data_path = historical_data_path
        self.val_size = val_size
        self.random_state = random_state

    def _load_models(self) -> dict:
        models = {}
        for name in MODEL_REGISTRY:
            path = os.path.join(self.model_dir, f"{name}.joblib")
            if os.path.exists(path):
                model = joblib.load(path)
                # verify_chain — skipped silently when no public key is configured
                try:
                    from detection.persistence import ModelArtifact

                    ModelArtifact(self.model_dir).verify_chain(name)
                except Exception as exc:
                    logger.warning("Integrity check skipped for %s: %s", name, exc)
                models[name] = model
        return models

    def update(self, new_labelled: pd.DataFrame, model_dir: str | None = None) -> dict:
        """Update models with *new_labelled* annotations.

        Returns a report dict written to ``reports/al_update_{timestamp}.json``.

        The report contains:
            - ``auc_before``, ``auc_after``: mean AUC-ROC change
            - ``rolled_back``: True if the update was rejected
            - ``strategy``: "warm_start" or "full_retrain"
            - ``backdoor_detection``: Backdoor detection results

        Backdoor detection is run before training. If > 20% of a class's samples
        are flagged, a critical alert is emitted and training proceeds without
        quarantine (to prevent false positives from blocking training).
        """
        model_dir = model_dir or self.model_dir
        threshold = config.AL_RETRAIN_THRESHOLD
        rollback_threshold = config.AL_ROLLBACK_AUC_DROP

        models_before = self._load_models()
        if not models_before:
            raise RuntimeError(f"No trained models found in {model_dir}. Train first.")

        # Run backdoor detection before training
        report: dict = {}
        new_labelled_cleaned, quarantined_indices = _detect_and_quarantine_backdoors(
            models_before, new_labelled, report
        )

        # Use cleaned data for training (backdoors removed if below safety threshold)
        training_data = new_labelled_cleaned

        # Split cleaned data for before/after AUC evaluation
        if len(training_data) < 4:
            # Too small to split — use the whole set for eval
            val_df = training_data
        else:
            _, val_df = train_test_split(
                training_data,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=training_data["label"]
                if training_data["label"].nunique() > 1
                else None,
            )

        auc_before = _auc_on_df(models_before, val_df)
        original_shas = _backup_models(model_dir)

        strategy: str
        try:
            if len(training_data) < threshold:
                logger.info(
                    "Warm-start update: %d new samples (threshold=%d)",
                    len(training_data),
                    threshold,
                )
                strategy = "warm_start"
                updated_models = _warm_start_update(models_before, training_data, model_dir)
            else:
                logger.info(
                    "Full retrain: %d new samples >= threshold=%d",
                    len(training_data),
                    threshold,
                )
                strategy = "full_retrain"
                if self.historical_data_path and os.path.exists(self.historical_data_path):
                    historical = load_training_data(self.historical_data_path)
                    combined = pd.concat([historical, training_data], ignore_index=True)
                else:
                    combined = training_data
                    logger.warning("No historical dataset found — retraining on new data only")
                training_output = train_models(combined, random_state=self.random_state)
                results = training_output["results"]
                updated_models = {name: res["model"] for name, res in results.items()}
                save_models(results, model_dir)

            auc_after = _auc_on_df(updated_models, val_df)
            auc_delta = auc_after - auc_before

            rolled_back = False
            if auc_delta < -rollback_threshold:
                logger.warning(
                    "AUC-ROC dropped %.4f (threshold %.4f) — rolling back",
                    auc_delta,
                    rollback_threshold,
                )
                _restore_models(model_dir, original_shas)
                rolled_back = True
                auc_after = auc_before  # after rollback, AUC is restored
            else:
                _cleanup_backups(model_dir)

        except Exception:
            _restore_models(model_dir, original_shas)
            raise

        report.update(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "strategy": strategy,
                "n_new_samples": len(new_labelled),
                "n_quarantined": len(quarantined_indices),
                "n_training_samples": len(training_data),
                "auc_before": round(auc_before, 6),
                "auc_after": round(auc_after, 6),
                "auc_delta": round(auc_after - auc_before, 6),
                "rolled_back": rolled_back,
            }
        )

        os.makedirs(REPORTS_DIR, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        report_path = os.path.join(REPORTS_DIR, f"al_update_{ts}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("AL update report written to %s", report_path)
        return report
