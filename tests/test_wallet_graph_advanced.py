"""Advanced wallet-graph tests (Issue #11).

Covers multi-hop ancestor traversal, Louvain community detection for
wash-trading ring identification, the new ring feature columns, and a BFS
performance benchmark.
"""

import time

import networkx as nx
import pandas as pd

from detection.feature_engineering import compute_wallet_graph_features
from detection.wallet_graph import (
    NO_RING,
    build_ring_statistics,
    detect_wash_trading_rings,
    multi_hop_ancestors,
    ring_statistics,
)


def _chain_graph() -> nx.DiGraph:
    """Funding chain A -> B -> C -> D (A funds B funds C funds D)."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return graph


def _star_ring(source: str, members: int) -> nx.DiGraph:
    """A funding source that directly funds `members` trading wallets."""
    graph = nx.DiGraph()
    for i in range(members):
        graph.add_edge(source, f"W{i}")
    return graph


def _complete_funding_graph(nodes: list[str]) -> nx.DiGraph:
    """Directed graph whose undirected projection is a complete graph."""
    graph = nx.DiGraph()
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            graph.add_edge(src, dst)
    return graph


# ---------------------------------------------------------------------------
# Multi-hop ancestor traversal
# ---------------------------------------------------------------------------


def test_multi_hop_ancestor_traversal_depth_1():
    graph = _chain_graph()
    # Only the immediate funder of D is reachable within 1 hop.
    assert multi_hop_ancestors(graph, "D", max_depth=1) == {"C"}


def test_multi_hop_ancestor_traversal_depth_3():
    graph = _chain_graph()
    # Within 3 hops the whole upstream chain is reachable.
    assert multi_hop_ancestors(graph, "D", max_depth=3) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


def test_community_detection_identifies_ring():
    # Tight cluster: source S funds 5 wallets; plus 2 isolated wallets.
    graph = _star_ring("S", 5)
    graph.add_node("ISO1")
    graph.add_node("ISO2")

    community_map = detect_wash_trading_rings(graph, min_ring_size=3)

    ring_ids = {community_map[f"W{i}"] for i in range(5)}
    assert len(ring_ids) == 1
    assert ring_ids.pop() != NO_RING
    assert community_map["ISO1"] == NO_RING
    assert community_map["ISO2"] == NO_RING


def test_community_detection_is_deterministic():
    graph = _star_ring("S", 5)
    graph.add_edge("S2", "X0")
    graph.add_edge("S2", "X1")
    graph.add_edge("S2", "X2")

    first = detect_wash_trading_rings(graph)
    second = detect_wash_trading_rings(graph)
    assert first == second


# ---------------------------------------------------------------------------
# Ring features
# ---------------------------------------------------------------------------


def test_ring_size_feature_nonzero_in_cluster():
    members = ["R0", "R1", "R2", "R3", "R4"]
    graph = _complete_funding_graph(members)  # 5-wallet ring

    community_map = detect_wash_trading_rings(graph, min_ring_size=3)
    ring_stats = build_ring_statistics(community_map, graph)

    now = pd.Timestamp.now(tz="UTC")
    for wallet in members:
        features = compute_wallet_graph_features(
            wallet, None, now, graph, community_map, ring_stats
        )
        assert features["in_wash_trading_ring"] is True
        assert features["ring_size"] == 5


def test_small_cluster_excluded_from_rings():
    graph = nx.DiGraph()
    graph.add_edge("P0", "P1")  # community of size 2

    community_map = detect_wash_trading_rings(graph, min_ring_size=3)
    ring_stats = build_ring_statistics(community_map, graph)

    now = pd.Timestamp.now(tz="UTC")
    for wallet in ("P0", "P1"):
        features = compute_wallet_graph_features(
            wallet, None, now, graph, community_map, ring_stats
        )
        assert features["in_wash_trading_ring"] is False
        assert features["ring_size"] == 0


def test_ring_internal_density():
    members = ["C0", "C1", "C2", "C3"]
    graph = _complete_funding_graph(members)  # K4 -> 6 of 6 possible edges

    community_map = {node: 0 for node in members}
    stats = ring_statistics(0, community_map, graph)

    assert stats["ring_size"] == 4
    assert stats["internal_edge_density"] == 1.0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_bfs_performance_10k_nodes():
    # Sparse random funding graph with 10,000 nodes.
    graph = nx.gnm_random_graph(10_000, 20_000, seed=7, directed=True)

    start = time.monotonic()
    multi_hop_ancestors(graph, 0, max_depth=4)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"BFS with max_depth=4 took {elapsed:.3f}s (limit 2.0s)"
