"""Wallet funding-graph features: `funding_source_similarity` and
`network_centrality`.

Builds a directed graph of "funded by" relationships from
`AccountActivity.funding_account` and derives two signals used by
`feature_engineering.compute_wallet_graph_features`:

- `funding_source_similarity`: the highest Jaccard similarity between a
  wallet's set of funding ancestors and any other wallet's funding-ancestor
  set. A high value means two wallets trace back to the same funding
  source(s) — a common pattern for sock-puppet / wash-trading rings.
- `network_centrality`: degree centrality of the wallet within the funding
  graph, a proxy for how connected/influential an account is within the
  observed funding network.

Also provides:

- `build_co_trade_graph`: builds edges between wallets that co-traded the
  same asset pair within a configurable time window.
"""

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
import re
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import re

from ingestion.data_models import AccountActivity

# Stellar account ID format: G followed by 55 uppercase base-32 chars
_STELLAR_ACCOUNT_RE = re.compile(r"^G[A-Z2-7]{55}$")


def _validate_account_id(account_id: str) -> bool:
    """Return True if *account_id* matches the Stellar account ID format."""
    return bool(_STELLAR_ACCOUNT_RE.match(account_id))


def build_funding_graph(
    activities: Iterable[AccountActivity],
    trades: pd.DataFrame | None = None,
    *,
    validate_account_ids: bool = False,
    co_trade_window: str | pd.Timedelta = "5min",
    output_format: Literal["networkx", "pyg"] = "networkx",
    node_features: pd.DataFrame | Mapping[str, Sequence[float]] | None = None,
) -> "nx.DiGraph":
    """Build the wallet graph, preserving the historical NetworkX default.

    Funding edges point from funder to funded account. When ``trades`` is
    supplied, wallets active in the same asset pair within ``co_trade_window``
    are connected in both directions. Set ``output_format="pyg"`` to obtain a
    :class:`torch_geometric.data.Data` object suitable for GraphSAGE.
    """
    if output_format not in {"networkx", "pyg"}:
        raise ValueError("output_format must be 'networkx' or 'pyg'")

    graph: nx.DiGraph = nx.DiGraph()
    for activity in activities:
        if validate_account_ids and not _validate_account_id(activity.account_id):
            continue
        graph.add_node(activity.account_id)
        if activity.funding_account:
            graph.add_edge(activity.funding_account, activity.account_id, edge_type="funding")

    if trades is not None and not trades.empty:
        _add_co_trade_edges(graph, trades, pd.Timedelta(co_trade_window))

    if output_format == "pyg":
        return to_pyg_data(graph, node_features=node_features)
    return graph


def _add_co_trade_edges(graph: nx.DiGraph, trades: pd.DataFrame, window: pd.Timedelta) -> None:
    """Add bidirectional edges between wallets co-active on an asset pair."""
    required = {"base_account", "counter_account", "ledger_close_time"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing required columns: {sorted(missing)}")

    frame = trades.copy()
    if "pair_id" not in frame:
        if not {"base_asset", "counter_asset"}.issubset(frame.columns):
            raise ValueError("trades require pair_id or base_asset/counter_asset columns")
        frame["pair_id"] = (
            frame["base_asset"].astype(str) + "/" + frame["counter_asset"].astype(str)
        )
    frame["ledger_close_time"] = pd.to_datetime(frame["ledger_close_time"], utc=True)

    for _, pair_trades in frame.groupby("pair_id", sort=False):
        records = pair_trades.sort_values("ledger_close_time").to_dict("records")
        for left_index, left in enumerate(records):
            left_time = left["ledger_close_time"]
            active_wallets = {left["base_account"], left["counter_account"]}
            for right in records[left_index + 1 :]:
                if right["ledger_close_time"] - left_time > window:
                    break
                active_wallets.update((right["base_account"], right["counter_account"]))
            for source, target in combinations(sorted(active_wallets), 2):
                graph.add_edge(source, target, edge_type="co_trade")
                graph.add_edge(target, source, edge_type="co_trade")


def to_pyg_data(
    graph: nx.DiGraph,
    node_features: pd.DataFrame | Mapping[str, Sequence[float]] | None = None,
):
    """Convert a wallet graph to PyG ``Data`` with a stable wallet mapping.

    ``node_features`` may be a feature matrix containing a ``wallet`` column,
    or a wallet-to-vector mapping. Missing wallets receive zero vectors. With
    no feature input, a constant feature is used so topology remains usable.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - exercised without GNN extra
        raise ImportError("PyG output requires torch and torch-geometric") from exc

    wallet_ids = list(graph.nodes)
    node_index = {wallet: index for index, wallet in enumerate(wallet_ids)}
    feature_map: dict[str, np.ndarray] = {}
    width = 1
    if isinstance(node_features, pd.DataFrame):
        if "wallet" not in node_features:
            raise ValueError("node feature DataFrame must contain a wallet column")
        feature_columns = [
            column
            for column in node_features.columns
            if column not in {"wallet", "label"}
            and pd.api.types.is_numeric_dtype(node_features[column])
        ]
        width = len(feature_columns) or 1
        feature_map = {
            str(row["wallet"]): row[feature_columns].to_numpy(dtype=np.float32)
            for _, row in node_features.iterrows()
        }
    elif node_features:
        feature_map = {
            str(wallet): np.asarray(values, dtype=np.float32)
            for wallet, values in node_features.items()
        }
        widths = {len(values) for values in feature_map.values()}
        if len(widths) != 1:
            raise ValueError("all node feature vectors must have equal length")
        width = widths.pop()

    default = (
        np.ones(width, dtype=np.float32) if not feature_map else np.zeros(width, dtype=np.float32)
    )
    x = (
        np.stack([feature_map.get(str(wallet), default) for wallet in wallet_ids])
        if wallet_ids
        else np.empty((0, width), dtype=np.float32)
    )
    edges = [(node_index[source], node_index[target]) for source, target in graph.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index)
    data.wallet_ids = wallet_ids
    data.node_index = node_index
    return data


def funding_source_similarity(wallet: str, graph: nx.DiGraph) -> float:
    """Highest Jaccard similarity between ``wallet``'s funding ancestors and
    any other node's funding ancestors in ``graph``.

    Returns ``0.0`` if ``wallet`` isn't in the graph or has no funding
    ancestors.

    .. deprecated::
        Use :class:`detection.gnn_encoder.GNNEncoder` embeddings instead.
        This scalar feature is preserved for backwards compatibility with the
        existing model artifact.
    """
    warnings.warn(
        "funding_source_similarity is deprecated; use GNNEncoder embeddings instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _funding_source_similarity(wallet, graph)


def _funding_source_similarity(wallet: str, graph: nx.DiGraph) -> float:
    """Internal (non-deprecated) implementation used by compute_wallet_graph_metrics."""
    if wallet not in graph:
        return 0.0

    wallet_ancestors = nx.ancestors(graph, wallet)
    if not wallet_ancestors:
        return 0.0

    best = 0.0
    for other in graph.nodes:
        if other == wallet:
            continue
        other_ancestors = nx.ancestors(graph, other)
        if not other_ancestors:
            continue
        union = wallet_ancestors | other_ancestors
        if not union:
            continue
        jaccard = len(wallet_ancestors & other_ancestors) / len(union)
        best = max(best, jaccard)

    return float(best)


def network_centrality(wallet: str, graph: nx.DiGraph) -> float:
    """Degree centrality of ``wallet`` within the funding graph.

    .. deprecated::
        Use :class:`detection.gnn_encoder.GNNEncoder` embeddings instead.
        This scalar feature is preserved for backwards compatibility with the
        existing model artifact.
    """
    warnings.warn(
        "network_centrality is deprecated; use GNNEncoder embeddings instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _network_centrality(wallet, graph)


def _network_centrality(wallet: str, graph: nx.DiGraph) -> float:
    """Internal (non-deprecated) implementation used by compute_wallet_graph_metrics."""
    if wallet not in graph or graph.number_of_nodes() < 2:
        return 0.0
    return float(nx.degree_centrality(graph)[wallet])


def compute_wallet_graph_metrics(wallet: str, graph: nx.DiGraph) -> dict:
    """Return ``{funding_source_similarity, network_centrality}`` for *wallet*.

    Calls the internal implementations directly to avoid emitting deprecation
    warnings from internal code paths.
    """
    return {
        "funding_source_similarity": _funding_source_similarity(wallet, graph),
        "network_centrality": _network_centrality(wallet, graph),
    }
