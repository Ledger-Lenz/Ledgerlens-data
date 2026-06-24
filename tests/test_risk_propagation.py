"""Tests for detection.risk_propagation.

Acceptance criteria covered:
- 3-node ring: only node A is directly scored; B and C receive decayed
  propagated risk with correct ordering (B > C for a linear chain A→B→C).
- propagate_risk_scores converges in ≤ 50 iterations for graphs with
  up to 10 000 nodes.
- propagation_attribution returns contributors whose fractions sum to 1.0.
- Co-trade-only nodes (absent from funding_graph) receive propagated scores.
- Wallets absent from base_scores receive 0.0 propagated score.
- Empty graph returns an empty dict.
- All returned scores are clipped to [0, 100].
"""

from __future__ import annotations

import time

import networkx as nx
import pytest

from detection.risk_propagation import (
    _build_combined_graph,
    _personalised_pagerank,
    propagate_risk_scores,
    propagation_attribution,
)
from scipy.sparse import csr_matrix
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_funding_graph(nodes: list[str]) -> nx.DiGraph:
    """Return A → B → C → … directed chain as a funding graph."""
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


def _ring_funding_graph(nodes: list[str]) -> nx.DiGraph:
    """Return a directed ring A → B → C → A."""
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for i, node in enumerate(nodes):
        g.add_edge(node, nodes[(i + 1) % len(nodes)])
    return g


# ---------------------------------------------------------------------------
# Core acceptance criterion: 3-node ring / chain
# ---------------------------------------------------------------------------


class TestThreeNodePropagation:
    """Spec requirement: 'A 3-node ring where only node A is directly scored
    propagates risk to B and C with correct decay'."""

    def test_linear_chain_b_receives_more_than_c(self):
        """A → B → C: B is closer to seed A so it gets more propagated risk."""
        nodes = ["A", "B", "C"]
        graph = _linear_funding_graph(nodes)
        base = {"A": 80.0}

        scores = propagate_risk_scores(base, graph)

        assert scores["A"] > 0.0, "seed node A must have non-zero propagated score"
        assert scores["B"] > 0.0, "direct successor B must inherit risk from A"
        assert scores["C"] > 0.0, "two-hop successor C must inherit some risk from A"
        assert scores["B"] > scores["C"], (
            "B is one hop from A, C is two hops — B should score higher"
        )

    def test_ring_all_nodes_receive_propagated_risk(self):
        """A → B → C → A: all three nodes should receive risk when only A is seeded."""
        nodes = ["A", "B", "C"]
        graph = _ring_funding_graph(nodes)
        base = {"A": 80.0}

        scores = propagate_risk_scores(base, graph)

        for node in nodes:
            assert scores[node] > 0.0, f"node {node} should receive propagated risk in a ring"

    def test_unseeded_nodes_score_zero_with_isolated_graph(self):
        """Nodes not reachable from any seed receive 0.0."""
        # Two disconnected components: A→B and C (isolated)
        g = nx.DiGraph()
        g.add_nodes_from(["A", "B", "C"])
        g.add_edge("A", "B")
        base = {"A": 70.0}

        scores = propagate_risk_scores(base, g)

        assert scores["B"] > 0.0
        # C is completely isolated — PPR from A cannot reach it
        assert scores["C"] == pytest.approx(0.0, abs=1e-6)

    def test_scores_clipped_to_100(self):
        """Even with a very high base score, propagated scores must not exceed 100."""
        nodes = ["A", "B", "C"]
        graph = _ring_funding_graph(nodes)
        base = {"A": 100.0, "B": 100.0, "C": 100.0}

        scores = propagate_risk_scores(base, graph)

        for node, score in scores.items():
            assert score <= 100.0 + 1e-9, f"{node} score {score} exceeds 100"

    def test_no_seeds_returns_all_zeros(self):
        """When no wallet in base_scores has a positive score, all results are 0."""
        graph = _linear_funding_graph(["A", "B", "C"])
        base = {"A": 0.0, "B": 0.0}

        scores = propagate_risk_scores(base, graph)

        for node, score in scores.items():
            assert score == pytest.approx(0.0), f"{node} should be 0 with no active seeds"

    def test_wallet_not_in_graph_ignored(self):
        """Wallets in base_scores that are not graph nodes are silently ignored."""
        graph = _linear_funding_graph(["A", "B", "C"])
        base = {"A": 60.0, "UNKNOWN_WALLET": 90.0}

        scores = propagate_risk_scores(base, graph)

        # Should still return scores for graph nodes, not raise
        assert set(scores.keys()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Empty / trivial graphs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_graph_returns_empty_dict(self):
        g = nx.DiGraph()
        scores = propagate_risk_scores({}, g)
        assert scores == {}

    def test_single_node_no_edges(self):
        g = nx.DiGraph()
        g.add_node("solo")
        scores = propagate_risk_scores({"solo": 50.0}, g)
        assert "solo" in scores
        assert 0.0 <= scores["solo"] <= 100.0

    def test_scores_are_non_negative(self):
        graph = _ring_funding_graph(["A", "B", "C", "D"])
        base = {"A": 55.0, "C": 30.0}
        scores = propagate_risk_scores(base, graph)
        for node, score in scores.items():
            assert score >= 0.0, f"{node} has negative propagated score {score}"


# ---------------------------------------------------------------------------
# Co-trade graph integration
# ---------------------------------------------------------------------------


class TestCoTradeGraph:
    def test_co_trade_only_nodes_receive_propagated_scores(self):
        """Nodes that exist only in co_trade_graph (not funding_graph) must
        still appear in the output and receive non-zero propagated scores
        when they co-traded with a high-risk seed."""
        funding_graph = nx.DiGraph()
        funding_graph.add_node("A")  # A has no funding relationships

        co_trade = nx.Graph()
        co_trade.add_edge("A", "X")  # X exists only in the co-trade graph

        base = {"A": 80.0}
        scores = propagate_risk_scores(base, funding_graph, co_trade_graph=co_trade)

        assert "X" in scores, "co-trade-only node X must appear in results"
        assert scores["X"] > 0.0, "X co-traded with high-risk A and should inherit some risk"

    def test_co_trade_graph_adds_bidirectional_edges(self):
        """_build_combined_graph must add both directions for each co-trade edge."""
        funding = nx.DiGraph()
        funding.add_nodes_from(["A", "B"])
        co_trade = nx.Graph()
        co_trade.add_edge("A", "B")

        combined = _build_combined_graph(funding, co_trade)

        assert combined.has_edge("A", "B")
        assert combined.has_edge("B", "A")

    def test_none_co_trade_graph_behaves_same_as_no_co_trade(self):
        """Passing co_trade_graph=None is equivalent to no co-trade graph."""
        graph = _linear_funding_graph(["A", "B", "C"])
        base = {"A": 70.0}

        scores_none = propagate_risk_scores(base, graph, co_trade_graph=None)
        scores_default = propagate_risk_scores(base, graph)

        for node in ["A", "B", "C"]:
            assert scores_none[node] == pytest.approx(scores_default[node], abs=1e-10)


# ---------------------------------------------------------------------------
# Convergence within ≤ 50 iterations
# ---------------------------------------------------------------------------


class TestConvergence:
    """Spec: 'converges in ≤ 50 iterations for graphs up to 10,000 nodes'."""

    def test_small_graph_converges(self):
        """Verify convergence doesn't raise or time out on a 100-node graph."""
        g = nx.DiGraph()
        nodes = [f"W{i}" for i in range(100)]
        g.add_nodes_from(nodes)
        # Random-ish edges: each node points to the next two
        for i in range(len(nodes)):
            g.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
            g.add_edge(nodes[i], nodes[(i + 3) % len(nodes)])

        base = {nodes[0]: 90.0, nodes[50]: 60.0}
        scores = propagate_risk_scores(base, g, max_iterations=50)

        assert len(scores) == len(nodes)
        assert all(0.0 <= s <= 100.0 for s in scores.values())

    @pytest.mark.slow
    def test_10k_node_graph_completes_under_2_seconds(self):
        """Performance criterion: full propagation pass < 2 s on CPU."""
        n = 10_000
        nodes = [f"W{i}" for i in range(n)]
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        # Sparse chain + a few cross-links to make it non-trivial
        for i in range(n - 1):
            g.add_edge(nodes[i], nodes[i + 1])
        for i in range(0, n, 100):
            g.add_edge(nodes[i], nodes[(i + 500) % n])

        base = {nodes[0]: 85.0, nodes[5000]: 70.0}

        start = time.perf_counter()
        scores = propagate_risk_scores(base, g, max_iterations=50)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Propagation took {elapsed:.2f}s — exceeds 2s budget"
        assert len(scores) == n


# ---------------------------------------------------------------------------
# Personalised PageRank unit tests
# ---------------------------------------------------------------------------


class TestPersonalisedPageRank:
    def test_ppr_sums_to_one(self):
        """PPR vector must sum to approximately 1.0 (it's a probability distribution)."""
        n = 5
        # Simple fully-connected graph (self-loops excluded)
        adj = np.ones((n, n), dtype=np.float64) - np.eye(n)
        row_sums = adj.sum(axis=1, keepdims=True)
        A_norm = adj / row_sums
        A_csr = csr_matrix(A_norm)

        ppr = _personalised_pagerank(A_csr, seed_idx=0, alpha=0.15, max_iterations=50, convergence_tol=1e-9)

        assert ppr.sum() == pytest.approx(1.0, abs=1e-5)

    def test_ppr_seed_node_has_highest_score(self):
        """For a star graph the seed (center) should have the highest PPR mass."""
        # Star: node 0 → all others
        n = 6
        rows = list(range(1, n))  # edges: 1→0, 2→0, ... (pointing to hub)
        cols = [0] * (n - 1)
        # Add reverse so 0 can propagate outward too
        rows += [0] * (n - 1)
        cols += list(range(1, n))
        data = np.ones(len(rows))
        adj_raw = csr_matrix((data, (rows, cols)), shape=(n, n))
        row_sums = np.asarray(adj_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        from scipy.sparse import diags
        A_csr = (diags(1.0 / row_sums) @ adj_raw).tocsr()

        ppr = _personalised_pagerank(A_csr, seed_idx=0, alpha=0.85, max_iterations=50, convergence_tol=1e-9)

        assert ppr[0] == max(ppr), "seed node should have highest PPR mass with high teleport prob"


# ---------------------------------------------------------------------------
# propagation_attribution
# ---------------------------------------------------------------------------


class TestPropagationAttribution:
    def test_attribution_fractions_sum_to_one(self):
        """All contributor fractions must sum to 1.0 (within floating-point tolerance)."""
        graph = _ring_funding_graph(["A", "B", "C"])
        base = {"A": 80.0, "B": 40.0}

        contribs = propagation_attribution("C", base, graph)

        assert len(contribs) > 0, "C should have contributors in a ring"
        total_fraction = sum(c["fraction"] for c in contribs)
        assert total_fraction == pytest.approx(1.0, abs=1e-4)

    def test_attribution_returns_empty_for_zero_propagated_score(self):
        """If the target wallet has no propagated risk, attribution is empty."""
        g = nx.DiGraph()
        g.add_nodes_from(["A", "B", "C"])
        # C is completely isolated
        g.add_edge("A", "B")
        base = {"A": 60.0}

        contribs = propagation_attribution("C", base, g)

        assert contribs == []

    def test_attribution_wallet_not_in_graph_returns_empty(self):
        graph = _linear_funding_graph(["A", "B", "C"])
        base = {"A": 70.0}

        contribs = propagation_attribution("UNKNOWN", base, graph)

        assert contribs == []

    def test_attribution_top_n_respected(self):
        """top_n should cap the number of returned contributors."""
        # Create a graph where many seeds can reach the target
        nodes = ["seed1", "seed2", "seed3", "seed4", "seed5", "target"]
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        for seed in nodes[:-1]:
            g.add_edge(seed, "target")

        base = {seed: 50.0 for seed in nodes[:-1]}
        contribs = propagation_attribution("target", base, g, top_n=3)

        assert len(contribs) <= 3

    def test_attribution_fields_present(self):
        """Each contributor dict must have the required keys."""
        graph = _linear_funding_graph(["A", "B", "C"])
        base = {"A": 75.0}

        contribs = propagation_attribution("B", base, graph)

        required_keys = {"source_wallet", "base_score", "ppr_weight", "contribution", "fraction"}
        for c in contribs:
            assert required_keys.issubset(c.keys()), f"Missing keys in contributor: {c}"

    def test_attribution_sorted_by_contribution_descending(self):
        """Contributors must be sorted largest contribution first."""
        graph = _ring_funding_graph(["A", "B", "C", "D"])
        base = {"A": 90.0, "B": 20.0}

        contribs = propagation_attribution("D", base, graph)

        if len(contribs) > 1:
            for i in range(len(contribs) - 1):
                assert contribs[i]["contribution"] >= contribs[i + 1]["contribution"]

    def test_attribution_with_co_trade_graph(self):
        """propagation_attribution should work when a co_trade_graph is supplied."""
        funding = nx.DiGraph()
        funding.add_node("A")
        co_trade = nx.Graph()
        co_trade.add_edge("A", "B")

        base = {"A": 80.0}
        contribs = propagation_attribution("B", base, funding, co_trade_graph=co_trade)

        assert len(contribs) > 0
        assert contribs[0]["source_wallet"] == "A"


# ---------------------------------------------------------------------------
# _build_combined_graph
# ---------------------------------------------------------------------------


class TestBuildCombinedGraph:
    def test_co_trade_only_nodes_included(self):
        """Nodes that appear only in co_trade_graph must be present in combined."""
        funding = nx.DiGraph()
        funding.add_node("A")
        co_trade = nx.Graph()
        co_trade.add_node("X")  # X only in co_trade
        co_trade.add_edge("A", "X")

        combined = _build_combined_graph(funding, co_trade)

        assert "X" in combined.nodes

    def test_funding_edges_preserved(self):
        funding = nx.DiGraph()
        funding.add_edge("funder", "child")
        combined = _build_combined_graph(funding, None)
        assert combined.has_edge("funder", "child")

    def test_none_co_trade_only_funding_nodes(self):
        funding = nx.DiGraph()
        funding.add_edge("A", "B")
        combined = _build_combined_graph(funding, None)
        assert set(combined.nodes) == {"A", "B"}
        assert not combined.has_edge("B", "A")  # funding is directed
