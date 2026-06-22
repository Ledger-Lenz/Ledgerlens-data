
import argparse
import os
import joblib
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import List, Tuple
from config import config
from scripts.generate_synthetic_dataset import generate_synthetic_dataset
from detection.meta_learner import LeafEmbeddingExtractor, MAMLAdapter, PrototypicalClassifier
from detection.model_training import split_features_labels
from utils.logging import get_logger

logger = get_logger(__name__)

def generate_tasks(n_tasks: int, n_support: int = 10, n_query: int = 90) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    tasks = []
    for i in range(n_tasks):
        # Vary the wash trading pattern for each task
        offset = np.random.uniform(-5, 5)
        noise = np.random.uniform(0.8, 1.2)
        df = generate_synthetic_dataset(n_wallets=n_support + n_query, seed=42+i, wash_offset=offset, wash_noise=noise)

        # Split into support and query sets
        # Assuming generate_synthetic_dataset returns balanced classes
        support_set = df.iloc[:n_support]
        query_set = df.iloc[n_support:n_support+n_query]
        tasks.append((support_set, query_set))
    return tasks

def meta_train(
    n_epochs: int = 20,
    n_tasks_per_epoch: int = 5,
    n_inner_steps: int = 5,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    model_dir: str = None,
    use_dp: bool = False,
):
    model_dir = model_dir or config.MODEL_DIR

    # Load base ensemble models to initialize extractor
    models = {}
    for name in ["random_forest", "xgboost", "lightgbm"]:
        path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            logger.warning(f"Base model {name} not found in {model_dir}. Run model_training first.")

    if not models:
        raise RuntimeError("No base models found. Cannot perform meta-training.")

    extractor = LeafEmbeddingExtractor(models)

    # Use a dummy transform to get embedding dimension
    dummy_df = generate_synthetic_dataset(n_wallets=10)
    X_dummy, _ = split_features_labels(dummy_df)
    extractor.fit(X_dummy)
    dummy_embeddings = extractor.transform(X_dummy)
    input_dim = dummy_embeddings.shape[1]

    maml = MAMLAdapter(input_dim=input_dim)
    proto = PrototypicalClassifier()

    meta_optimizer = torch.optim.Adam(maml.parameters(), lr=outer_lr)

    for epoch in range(n_epochs):
        epoch_loss = 0
        tasks = generate_tasks(n_tasks_per_epoch)

        meta_optimizer.zero_grad()
        for support_df, query_df in tasks:
            # 1. Prepare data
            X_s, y_s = split_features_labels(support_df)
            X_q, y_q = split_features_labels(query_df)

            emb_s = torch.from_numpy(extractor.transform(X_s)).float()
            y_s_t = torch.from_numpy(y_s.values).float()
            emb_q = torch.from_numpy(extractor.transform(X_q)).float()
            y_q_t = torch.from_numpy(y_q.values).float()

            # 2. Inner loop (adaptation) - using a clone for MAML
            # For simplicity in this implementation, we use a single update step
            # properly MAML would use functional-style updates or high-level libraries.
            # Here we'll do a simplified version: adapt a copy of the model.

            adapted_maml = MAMLAdapter(input_dim=input_dim)
            adapted_maml.load_state_dict(maml.state_dict())

            # Inner update
            inner_optimizer = torch.optim.SGD(adapted_maml.parameters(), lr=inner_lr)
            for _ in range(n_inner_steps):
                inner_optimizer.zero_grad()
                logits_s = adapted_maml(emb_s).squeeze(-1)
                loss_s = F.binary_cross_entropy_with_logits(logits_s, y_s_t)
                loss_s.backward()
                inner_optimizer.step()

            # 3. Outer loop (meta-update)
            logits_q = adapted_maml(emb_q).squeeze(-1)
            loss_q = F.binary_cross_entropy_with_logits(logits_q, y_q_t) / n_tasks_per_epoch

            # This is a bit tricky without functional MAML but we can
            # use the gradient from adapted_maml to update maml
            # A simpler approach is First-Order MAML (FOMAML)
            loss_q.backward()

            # Copy gradients from adapted_maml to maml
            for p, ap in zip(maml.parameters(), adapted_maml.parameters()):
                if ap.grad is not None:
                    if p.grad is None:
                        p.grad = ap.grad.clone()
                    else:
                        p.grad += ap.grad

            epoch_loss += loss_q.item() * n_tasks_per_epoch

        meta_optimizer.step()
        logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/n_tasks_per_epoch:.4f}")

    # Save checkpoints
    torch.save(maml.state_dict(), os.path.join(model_dir, "maml_adapter.pt"))
    # Prototypical classifier doesn't need "training" in this simple version
    # but we could meta-train an embedding network if we had one.
    # Here it uses frozen ensemble leaf embeddings.

    logger.info(f"Meta-training complete. Checkpoints saved to {model_dir}")

    if use_dp:
        from detection.privacy.meta_learner_dp import train_meta_learner_dp
        from detection.privacy.metrics import record_dp_metrics

        all_emb = []
        all_y = []
        for support_df, query_df in generate_tasks(3):
            for part in (support_df, query_df):
                X_part, y_part = split_features_labels(part)
                all_emb.append(extractor.transform(X_part))
                all_y.append(y_part.values)
        embeddings = np.concatenate(all_emb, axis=0)
        labels = np.concatenate(all_y, axis=0).astype(np.float32)

        dp_report = train_meta_learner_dp(
            embeddings,
            labels,
            epochs=config.DP_EPOCHS,
            use_dp=True,
        )
        torch.save(dp_report.model.state_dict(), os.path.join(model_dir, "maml_adapter_dp.pt"))
        record_dp_metrics(
            model_dir,
            "meta_learner",
            {
                "target_epsilon": config.DP_TARGET_EPSILON,
                "target_delta": config.DP_TARGET_DELTA,
                "achieved_epsilon": dp_report.achieved_epsilon,
                "max_grad_norm": config.DP_MAX_GRAD_NORM,
                "epochs": config.DP_EPOCHS,
                "auc_roc": dp_report.auc_roc,
                "baseline_auc_roc": dp_report.baseline_auc_roc,
                "auc_roc_degradation": dp_report.auc_roc_degradation,
                "membership_inference_success_rate": dp_report.membership_inference_success_rate,
            },
        )
        logger.info(
            "DP meta-learner saved (ε=%.4f, MIA=%.2f%%)",
            dp_report.achieved_epsilon or 0.0,
            dp_report.membership_inference_success_rate * 100,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--dp", action="store_true", help="Also train DP-SGD meta-learner head")
    args = parser.parse_args()

    meta_train(n_epochs=args.epochs, model_dir=args.model_dir, use_dp=args.dp)
