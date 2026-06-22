
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import joblib
import os
from config import config

class LeafEmbeddingExtractor:
    """Extracts leaf indices from a trained ensemble of models."""
    def __init__(self, models: Dict):
        self.models = models
        self.feature_columns = None

    def fit(self, X: pd.DataFrame):
        """Identify feature columns used by the models."""
        self.feature_columns = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract leaf indices and concatenate them."""
        all_leaves = []

        # Random Forest
        if "random_forest" in self.models:
            rf_leaves = self.models["random_forest"].apply(X)
            all_leaves.append(rf_leaves)

        # XGBoost
        if "xgboost" in self.models:
            xgb_model = self.models["xgboost"]
            # XGBoost sklearn wrapper's apply()
            xgb_leaves = xgb_model.apply(X)
            all_leaves.append(xgb_leaves)

        # LightGBM
        if "lightgbm" in self.models:
            lgbm_model = self.models["lightgbm"]
            lgbm_leaves = lgbm_model.predict(X, pred_leaf=True)
            all_leaves.append(lgbm_leaves)

        if not all_leaves:
            return np.array([])

        return np.concatenate(all_leaves, axis=1)

class MAMLAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.head(x)

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor, n_inner_steps: int = 10, lr: float = 0.01):
        """Fine-tune the head on a new task's support set using SGD."""
        # Create a copy of the head to adapt
        # For a simple implementation, we can just use the current parameters
        # but in MAML we often want to start from the meta-learned weights.
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.train()
        for _ in range(n_inner_steps):
            optimizer.zero_grad()
            logits = self.forward(support_x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, support_y)
            loss.backward()
            optimizer.step()

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x).squeeze(-1)
            probs = torch.sigmoid(logits)
        return probs.numpy()

class PrototypicalClassifier:
    def __init__(self):
        self.prototypes = {} # {label: prototype_vector}

    def fit_prototype(self, support_embeddings: np.ndarray, labels: np.ndarray):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            self.prototypes[label] = support_embeddings[mask].mean(axis=0)

    def predict_proba(self, X_embeddings: np.ndarray) -> np.ndarray:
        """Distance to prototype -> risk score.
        Using negative Euclidean distance as logit.
        """
        if 1 not in self.prototypes or 0 not in self.prototypes:
            # Fallback if we don't have both prototypes
            return np.zeros(len(X_embeddings))

        proto0 = self.prototypes[0]
        proto1 = self.prototypes[1]

        # Compute distances to each prototype
        dist0 = np.linalg.norm(X_embeddings - proto0, axis=1)
        dist1 = np.linalg.norm(X_embeddings - proto1, axis=1)

        # Convert distances to probabilities using softmax of negative distances
        # prob1 = exp(-dist1) / (exp(-dist0) + exp(-dist1))
        # To avoid overflow, use a temperature or just simple ratio
        # Actually, Prototypical Networks use squared Euclidean distance.
        dist0_sq = dist0**2
        dist1_sq = dist1**2

        # Softmax over negative squared distances
        max_neg_dist = np.maximum(-dist0_sq, -dist1_sq)
        exp0 = np.exp(-dist0_sq - max_neg_dist)
        exp1 = np.exp(-dist1_sq - max_neg_dist)
        probs = exp1 / (exp0 + exp1)

        return probs
