import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from detection.meta_learner import LeafEmbeddingExtractor, MAMLAdapter, PrototypicalClassifier


def test_maml_convergence_trivial():
    """Trivial meta-task (2-cluster Gaussian) converges in 5 inner steps."""
    torch.manual_seed(42)
    np.random.seed(42)

    input_dim = 10
    maml = MAMLAdapter(input_dim=input_dim, hidden_dim=32)

    # Task: classify two Gaussians.
    # To make it meta-learning, different tasks have different offsets.
    def get_task_data(offset):
        X0 = np.random.randn(5, input_dim) - offset
        X1 = np.random.randn(5, input_dim) + offset
        X = np.concatenate([X0, X1], axis=0)
        y = np.array([0] * 5 + [1] * 5)
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()

    # Pre-adaptation performance
    support_x, support_y = get_task_data(2.0)
    with torch.no_grad():
        logits_before = maml(support_x).squeeze(-1)
        loss_before = nn.functional.binary_cross_entropy_with_logits(logits_before, support_y)

    # Adaptation
    start_time = time.time()
    maml.adapt(support_x, support_y, n_inner_steps=5, lr=0.1)
    duration = time.time() - start_time

    # Post-adaptation performance
    with torch.no_grad():
        logits_after = maml(support_x).squeeze(-1)
        loss_after = nn.functional.binary_cross_entropy_with_logits(logits_after, support_y)

    print(f"Loss before: {loss_before:.4f}, Loss after: {loss_after:.4f}")
    assert loss_after < loss_before
    assert duration < 30.0  # Adaptation completes in < 30 seconds


def test_leaf_extraction_all_models():
    """Leaf-index embedding extraction works for RF, XGBoost, and LightGBM."""
    X = pd.DataFrame(np.random.rand(20, 10), columns=[f"f{i}" for i in range(10)])
    y = np.array([0, 1] * 10)

    rf = RandomForestClassifier(n_estimators=5).fit(X, y)
    xgb = XGBClassifier(n_estimators=5).fit(X, y)
    lgbm = LGBMClassifier(n_estimators=5).fit(X, y)

    models = {"random_forest": rf, "xgboost": xgb, "lightgbm": lgbm}
    extractor = LeafEmbeddingExtractor(models)
    extractor.fit(X)
    embeddings = extractor.transform(X)

    assert embeddings.shape[0] == 20
    assert embeddings.shape[1] > 0


def test_prototypical_separation():
    """PrototypicalClassifier achieves non-trivial separation (AUC > 0.65) with 5 support examples."""
    from sklearn.metrics import roc_auc_score

    proto = PrototypicalClassifier()

    # Generate 5 support examples per class
    support_emb = np.concatenate(
        [np.random.randn(5, 10) - 2.0, np.random.randn(5, 10) + 2.0], axis=0  # Class 0  # Class 1
    )
    support_y = np.array([0] * 5 + [1] * 5)

    proto.fit_prototype(support_emb, support_y)

    # Generate query set
    query_emb = np.concatenate(
        [np.random.randn(50, 10) - 2.0, np.random.randn(50, 10) + 2.0], axis=0
    )
    query_y = np.array([0] * 50 + [1] * 50)

    probs = proto.predict_proba(query_emb)
    auc = roc_auc_score(query_y, probs)

    print(f"Prototypical AUC: {auc:.4f}")
    assert auc > 0.65
