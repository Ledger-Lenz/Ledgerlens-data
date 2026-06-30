"""Tests for detection/motif_census.py."""

import time

import networkx as nx
import pytest

from detection.motif_census import (
    MotifCensusResult,
    _count_4_cycles,
    _count_star_motifs,
    _count_triangles_matrix,
    _compute_reciprocity,
    _to_undirected_simple,
    _validate_subgraph,
    compute_motif_census,
)
from detection.community_detector import enrich_communities_with_motifs


# ---------------------------------------------------------------------------
# _validate_subgraph
# ---------------------------------------------------------------------------


class TestValidateSubgraph:
    def test_all_nodes_known_passes(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        _validate_subgraph(G, {"A", "B", "C"})  # must not raise

    def test_external_node_raises(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "X")])
        with pytest.raises(ValueError, match="not in the known graph"):
            _validate_subgraph(G, {"A", "B"})

    def test_empty_subgraph_passes(self):
        G = nx.DiGraph()
        _validate_subgraph(G, {"A", "B"})

    def test_superset_known_nodes_passes(self):
        G = nx.DiGraph()
        G.add_node("A")
        _validate_subgraph(G, {"A", "B", "C", "D"})


# ---------------------------------------------------------------------------
# _count_triangles_matrix
# ---------------------------------------------------------------------------


class TestCountTrianglesMatrix:
    def test_k3_triangle_count_one(self):
        G = nx.complete_graph(3)
        count, max_t = _count_triangles_matrix(G)
        assert count == 1
        assert max_t == 1

    def test_k4_triangle_count_four(self):
        # K4 has C(4,3) = 4 triangles
        G = nx.complete_graph(4)
        count, _ = _count_triangles_matrix(G)
        assert count == 4

    def test_star_graph_no_triangles(self):
        G = nx.star_graph(5)
        count, _ = _count_triangles_matrix(G)
        assert count == 0

    def test_path_graph_no_triangles(self):
        G = nx.path_graph(5)
        count, _ = _count_triangles_matrix(G)
        assert count == 0

    def test_two_node_graph_zero(self):
        G = nx.complete_graph(2)
        count, max_t = _count_triangles_matrix(G)
        assert count == 0
        assert max_t == 0

    def test_single_node_zero(self):
        G = nx.Graph()
        G.add_node("A")
        count, max_t = _count_triangles_matrix(G)
        assert count == 0
        assert max_t == 0


# ---------------------------------------------------------------------------
# _count_4_cycles
# ---------------------------------------------------------------------------


class TestCount4Cycles:
    def test_c4_has_one_cycle(self):
        # 4-cycle graph has exactly one 4-cycle
        G = nx.cycle_graph(4)
        assert _count_4_cycles(G) == 1

    def test_k4_has_three_cycles(self):
        # K4 contains exactly 3 distinct 4-cycles
        G = nx.complete_graph(4)
        assert _count_4_cycles(G) == 3

    def test_triangle_no_4_cycle(self):
        G = nx.complete_graph(3)
        assert _count_4_cycles(G) == 0

    def test_path_graph_no_4_cycle(self):
        G = nx.path_graph(6)
        assert _count_4_cycles(G) == 0

    def test_two_nodes_returns_zero(self):
        G = nx.complete_graph(2)
        assert _count_4_cycles(G) == 0


# ---------------------------------------------------------------------------
# _count_star_motifs
# ---------------------------------------------------------------------------


class TestCountStarMotifs:
    def test_star_graph_all_open_wedges(self):
        # Star K1,4: hub has degree 4 → C(4,2)=6 wedges, 0 triangles → 6 stars
        G = nx.star_graph(4)
        count = _count_star_motifs(G, triangle_count=0)
        assert count == 6

    def test_k3_no_open_wedges(self):
        # Triangle has 3 wedges but all are closed → 3 - 3*1 = 0 open wedges
        G = nx.complete_graph(3)
        count = _count_star_motifs(G, triangle_count=1)
        assert count == 0


# ---------------------------------------------------------------------------
# _compute_reciprocity
# ---------------------------------------------------------------------------


class TestComputeReciprocity:
    def test_fully_reciprocal_digraph(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")])
        assert _compute_reciprocity(G) == 1.0

    def test_no_reciprocal_edges(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        assert _compute_reciprocity(G) == 0.0

    def test_partial_reciprocity(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "A"), ("B", "C")])
        # 2 reciprocal out of 3 total edges
        assert abs(_compute_reciprocity(G) - 2 / 3) < 1e-9

    def test_undirected_graph_returns_one(self):
        G = nx.path_graph(4)
        assert _compute_reciprocity(G) == 1.0

    def test_empty_digraph_returns_zero(self):
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B"])
        assert _compute_reciprocity(G) == 0.0


# ---------------------------------------------------------------------------
# compute_motif_census — unit tests (required by spec)
# ---------------------------------------------------------------------------


class TestComputeMotifCensus:
    """Core behavioural tests for the motif census function."""

    def test_perfect_triangle_density_one(self):
        """A K3 graph (perfect triangle) must yield triangle_density == 1.0."""
        G = nx.DiGraph()
        # All directed pairs so the undirected projection is K3.
        G.add_edges_from([("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"), ("A", "C"), ("C", "A")])
        known = set(G.nodes())
        result = compute_motif_census(G, known)
        assert result.triangle_density == pytest.approx(1.0)

    def test_directed_triangle_density_one(self):
        """A directed 3-cycle projects to K3 → triangle_density == 1.0."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        known = set(G.nodes())
        result = compute_motif_census(G, known)
        assert result.triangle_density == pytest.approx(1.0)

    def test_star_graph_star_ratio_above_0_9(self):
        """A star graph must yield star_ratio > 0.9."""
        G = nx.DiGraph()
        for i in range(10):
            G.add_edge("hub", f"leaf_{i}")
        known = set(G.nodes())
        result = compute_motif_census(G, known)
        assert result.star_ratio > 0.9

    def test_star_ratio_exactly_one_pure_star(self):
        """Pure star with no triangles → star_ratio == 1.0."""
        G = nx.DiGraph()
        for i in range(5):
            G.add_edge("hub", f"l{i}")
        known = set(G.nodes())
        result = compute_motif_census(G, known)
        assert result.star_ratio == pytest.approx(1.0)

    def test_timeout_sets_census_truncated(self, monkeypatch):
        """Exceeding the timeout budget returns partial results with census_truncated=True."""
        call_count = [0]
        real_monotonic = time.monotonic

        def fake_monotonic():
            call_count[0] += 1
            # First call establishes the deadline normally; all subsequent calls
            # report a time far in the future so every deadline check fires.
            return real_monotonic() + (10_000.0 if call_count[0] > 1 else 0.0)

        monkeypatch.setattr("detection.motif_census.time.monotonic", fake_monotonic)

        G = nx.DiGraph(nx.complete_graph(20))
        mapping = {n: f"W{n}" for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        known = set(G.nodes())
        result = compute_motif_census(G, known, timeout_seconds=5.0)
        assert result.census_truncated is True

    def test_external_node_rejected(self):
        """Subgraph with nodes outside known_nodes raises ValueError."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        with pytest.raises(ValueError, match="not in the known graph"):
            compute_motif_census(G, {"A", "B"})  # C is external

    def test_empty_graph_returns_defaults(self):
        result = compute_motif_census(nx.DiGraph(), set())
        assert result.triangle_density == 0.0
        assert result.star_ratio == 0.0
        assert result.cycle_4_count == 0
        assert result.reciprocity == 0.0
        assert result.census_truncated is False

    def test_single_node_graph(self):
        G = nx.DiGraph()
        G.add_node("W1")
        result = compute_motif_census(G, {"W1"})
        assert result.triangle_density == 0.0
        assert result.node_count == 1
        assert result.was_sampled is False

    def test_large_community_is_sampled(self):
        """Communities with > 500 nodes are subsampled; was_sampled is set."""
        G = nx.DiGraph()
        nodes = [f"W{i}" for i in range(600)]
        G.add_nodes_from(nodes)
        known = set(nodes)
        result = compute_motif_census(G, known)
        assert result.was_sampled is True
        assert result.node_count == 500

    def test_4_cycle_detected(self):
        """A pure 4-cycle graph should register cycle_4_count == 1."""
        G = nx.DiGraph(nx.cycle_graph(4))
        mapping = {n: f"W{n}" for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        known = set(G.nodes())
        result = compute_motif_census(G, known)
        assert result.cycle_4_count == 1

    def test_reciprocity_complete_bidirectional(self):
        """All edges reciprocated → reciprocity == 1.0."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")])
        result = compute_motif_census(G, set(G.nodes()))
        assert result.reciprocity == pytest.approx(1.0)

    def test_reciprocity_directed_cycle_zero(self):
        """Directed 3-cycle with no reverse edges → reciprocity == 0.0."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        result = compute_motif_census(G, set(G.nodes()))
        assert result.reciprocity == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# enrich_communities_with_motifs integration
# ---------------------------------------------------------------------------


class TestEnrichCommunitiesWithMotifs:
    def test_returns_feature_dict_per_community(self):
        G = nx.DiGraph()
        # Two triangles
        for src, dst in [("A", "B"), ("B", "C"), ("C", "A"), ("B", "A"), ("C", "B"), ("A", "C")]:
            G.add_edge(src, dst)
        community_map = {"A": 0, "B": 0, "C": 0}
        features = enrich_communities_with_motifs(G, community_map)
        assert 0 in features
        assert features[0]["triangle_density"] == pytest.approx(1.0)

    def test_noise_community_excluded(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        community_map = {"A": -1, "B": -1, "C": -1}
        features = enrich_communities_with_motifs(G, community_map)
        assert -1 not in features
        assert len(features) == 0

    def test_feature_keys_present(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        community_map = {"A": 0, "B": 0, "C": 0}
        features = enrich_communities_with_motifs(G, community_map)
        expected_keys = {
            "triangle_density", "star_ratio", "cycle_4_per_node",
            "reciprocity", "node_count", "was_sampled", "census_truncated",
        }
        assert expected_keys == set(features[0].keys())

    def test_cycle_4_per_node_normalised(self):
        """cycle_4_per_node = cycle_4_count / n; for C4 this is 1/4."""
        G = nx.DiGraph(nx.cycle_graph(4))
        mapping = {n: f"W{n}" for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        community_map = {n: 0 for n in G.nodes()}
        features = enrich_communities_with_motifs(G, community_map)
        assert features[0]["cycle_4_per_node"] == pytest.approx(1 / 4)
