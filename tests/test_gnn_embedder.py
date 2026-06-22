from datetime import datetime
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
import torch

from detection.gnn_embedder import (
    WalletGraphSAGE,
    embedding_feature_map,
    train_graphsage,
)
from detection.wallet_graph import build_funding_graph, to_pyg_data
from ingestion.data_models import AccountActivity


def _activities() -> list[AccountActivity]:
    return [
        AccountActivity(account_id="F", account_created_at=datetime(2020, 1, 1)),
        AccountActivity(
            account_id="A", account_created_at=datetime(2021, 1, 1), funding_account="F"
        ),
        AccountActivity(account_id="B", account_created_at=datetime(2021, 1, 2)),
    ]


def test_pyg_graph_contains_funding_and_time_windowed_co_trade_edges():
    trades = pd.DataFrame(
        [
            {
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "B",
                "pair_id": "XLM/USDC",
            },
            {
                "ledger_close_time": "2024-01-01T00:03:00Z",
                "base_account": "C",
                "counter_account": "D",
                "pair_id": "XLM/USDC",
            },
        ]
    )
    data = build_funding_graph(_activities(), trades, output_format="pyg")

    edges = {
        (data.wallet_ids[source], data.wallet_ids[target])
        for source, target in data.edge_index.t().tolist()
    }
    assert ("F", "A") in edges
    assert ("A", "C") in edges
    assert ("C", "A") in edges
    assert data.x.shape == (5, 1)


def test_graphsage_embedding_shape_and_ensemble_feature_names():
    graph = nx.DiGraph([("A", "B"), ("B", "C")])
    data = to_pyg_data(graph, {wallet: [1.0, 2.0] for wallet in graph.nodes})
    model = WalletGraphSAGE(input_dim=2, embedding_dim=64)
    _, embeddings = model(data.x, data.edge_index)
    feature_map = embedding_feature_map(data, embeddings.detach().numpy())

    assert embeddings.shape == (3, 64)
    assert set(feature_map["A"]) == {f"gnn_embedding_{i}" for i in range(64)}


def test_trained_cycle_scores_above_isolated_wallets():
    graph = nx.DiGraph()
    graph.add_edges_from([("R0", "R1"), ("R1", "R2"), ("R2", "R0")])
    graph.add_nodes_from(["I0", "I1", "I2"])
    data = to_pyg_data(graph)
    data.y = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)
    data.train_mask = torch.ones(6, dtype=torch.bool)
    data.test_mask = torch.ones(6, dtype=torch.bool)

    result = train_graphsage(data, hidden_dim=16, epochs=150, seed=7)
    result.model.eval()
    with torch.no_grad():
        logits, _ = result.model(data.x, data.edge_index)
    scores = torch.sigmoid(logits).numpy()

    assert np.mean(scores[:3]) > np.mean(scores[3:])
    assert result.metrics["auc_roc"] >= 0.85


def test_batched_cpu_inference_under_50ms_per_wallet():
    graph = nx.DiGraph((f"W{i}", f"W{(i + 1) % 100}") for i in range(100))
    data = to_pyg_data(graph, {wallet: np.ones(36) for wallet in graph.nodes})
    model = WalletGraphSAGE(input_dim=36)
    model.eval()
    with torch.no_grad():
        model(data.x, data.edge_index)  # warm up
        started = perf_counter()
        model(data.x, data.edge_index)
        elapsed_per_wallet = (perf_counter() - started) / data.num_nodes
    assert elapsed_per_wallet < 0.05
