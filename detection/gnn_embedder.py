"""GraphSAGE wallet embeddings and supervised node-classification training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class WalletGraphSAGE(nn.Module):
    """Inductive GraphSAGE encoder with a binary wallet classifier head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("GraphSAGE requires at least two message-passing layers")
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [embedding_dim]
        self.convs = nn.ModuleList(
            SAGEConv(dimensions[index], dimensions[index + 1], aggr="mean")
            for index in range(num_layers)
        )
        self.dropout = dropout
        self.classifier = nn.Linear(embedding_dim, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = nn.functional.relu(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(x, edge_index)
        return self.classifier(embeddings).squeeze(-1), embeddings


def focal_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-weighted focal BCE for severely imbalanced wallet labels."""
    targets = targets.float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    probability_true = probability * targets + (1 - probability) * (1 - targets)
    alpha_true = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_true * (1 - probability_true).pow(gamma) * bce).mean()


@dataclass
class GNNTrainingResult:
    model: WalletGraphSAGE
    metrics: dict[str, float]
    embeddings: np.ndarray


def add_supervision(
    data: Data,
    labels: dict[str, int],
    *,
    test_fraction: float = 0.2,
    negative_ratio: float = 3.0,
    seed: int = 42,
) -> Data:
    """Attach labels and stratified masks, downsampling excess negatives.

    Labels are matched through ``data.wallet_ids``. Unlabelled nodes are not
    included in either mask. Negative sampling is deterministic and capped at
    ``negative_ratio`` clean wallets per positive wallet.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    positive = [i for i, wallet in enumerate(data.wallet_ids) if labels.get(wallet) == 1]
    negative = [i for i, wallet in enumerate(data.wallet_ids) if labels.get(wallet) == 0]
    if not positive or not negative:
        raise ValueError("supervision requires positive and negative labelled wallets")
    max_negative = max(1, int(len(positive) * negative_ratio))
    if len(negative) > max_negative:
        negative = rng.choice(negative, max_negative, replace=False).tolist()

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    y = torch.zeros(data.num_nodes, dtype=torch.float32, device=data.x.device)
    for indices, label in ((positive, 1.0), (negative, 0.0)):
        shuffled = np.asarray(indices)[rng.permutation(len(indices))]
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        if len(shuffled) > 1:
            n_test = min(n_test, len(shuffled) - 1)
        test_indices = shuffled[:n_test]
        train_indices = shuffled[n_test:]
        y[indices] = label
        test_mask[test_indices] = True
        train_mask[train_indices] = True
    data.y = y
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


def _mask(data: Data, name: str) -> torch.Tensor:
    mask = getattr(data, name, None)
    if mask is None:
        return torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
    return cast(torch.Tensor, mask.bool())


def evaluate_graphsage(
    model: WalletGraphSAGE, data: Data, mask_name: str = "test_mask"
) -> dict[str, float]:
    """Compute node-level ROC-AUC and PR-AUC on a graph mask."""
    model.eval()
    mask = _mask(data, mask_name)
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
    labels = data.y[mask].detach().cpu().numpy()
    probabilities = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    if len(np.unique(labels)) < 2:
        return {"auc_roc": float("nan"), "pr_auc": float("nan")}
    return {
        "auc_roc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


def train_graphsage(
    data: Data,
    *,
    hidden_dim: int = 64,
    embedding_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 200,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    alpha: float = 0.75,
    gamma: float = 2.0,
    seed: int = 42,
) -> GNNTrainingResult:
    """Train a GraphSAGE node classifier using ``data.y`` and train/test masks."""
    if not hasattr(data, "y"):
        raise ValueError("data.y is required for supervised GraphSAGE training")
    torch.manual_seed(seed)
    model = WalletGraphSAGE(
        data.num_node_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    ).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_mask = _mask(data, "train_mask")

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = focal_binary_cross_entropy(
            logits[train_mask], data.y[train_mask], alpha=alpha, gamma=gamma
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, embeddings = model(data.x, data.edge_index)
    return GNNTrainingResult(
        model=model,
        metrics=evaluate_graphsage(model, data),
        embeddings=embeddings.detach().cpu().numpy(),
    )


def embedding_feature_map(data: Data, embeddings: np.ndarray) -> dict[str, dict[str, float]]:
    """Map PyG embeddings to ensemble-ready ``gnn_embedding_0..N`` columns."""
    if embeddings.shape[0] != data.num_nodes:
        raise ValueError("embedding row count must match graph node count")
    return {
        wallet: {
            f"gnn_embedding_{index}": float(value)
            for index, value in enumerate(embeddings[node_index])
        }
        for node_index, wallet in enumerate(data.wallet_ids)
    }
