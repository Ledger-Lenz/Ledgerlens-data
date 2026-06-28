"""Tests for detection/community_detector.py (Issue #280)."""

import networkx as nx
import pandas as pd
import pytest

from detection.community_detector import (
    compute_ring_concentration_score,
    detect_communities,
    validate_resolution_parameter,
)


class TestDetectCommunities:
    """Test the Louvain-based community detection."""

    def test_empty_graph(self):
        """Single-node graph returns community of size 1."""
        graph = nx.DiGraph()
        result = detect_communities(graph)
        assert result == {}

    def test_single_node_graph(self):
        """Single-node graph returns community of size 1."""
        graph = nx.DiGraph()
        graph.add_node("W1")
        result = detect_communities(graph, min_community_size=1)
        assert result == {"W1": 0}  # Single node forms its own community

    def test_two_ring_graph(self):
        """Synthetic two-ring graph: each ring is assigned a distinct community id."""
        graph = nx.DiGraph()
        # Ring 1: W1 -> W2 -> W3 -> W1
        for src, dst in [("W1", "W2"), ("W2", "W3"), ("W3", "W1")]:
            graph.add_edge(src, dst)
        # Ring 2: W4 -> W5 -> W6 -> W4
        for src, dst in [("W4", "W5"), ("W5", "W6"), ("W6", "W4")]:
            graph.add_edge(src, dst)

        result = detect_communities(graph, min_community_size=3)
        # Both rings should have >= 3 members and be assigned distinct community ids
        ring1_ids = {result["W1"], result["W2"], result["W3"]}
        ring2_ids = {result["W4"], result["W5"], result["W6"]}
        assert len(ring1_ids) == 1, "Ring 1 should have a single community id"
        assert len(ring2_ids) == 1, "Ring 2 should have a single community id"
        assert ring1_ids != ring2_ids, "Two rings should have distinct community ids"

    def test_resolution_boundary_low(self):
        """Resolution parameter boundary: 0.1 (inclusive)."""
        graph = nx.complete_graph(5)
        graph = nx.DiGraph(graph)
        result = detect_communities(graph, resolution=0.1)
        assert isinstance(result, dict)
        assert all(isinstance(cid, int) for cid in result.values())

    def test_resolution_boundary_high(self):
        """Resolution parameter boundary: 10.0 (inclusive)."""
        graph = nx.complete_graph(5)
        graph = nx.DiGraph(graph)
        result = detect_communities(graph, resolution=10.0)
        assert isinstance(result, dict)
        assert all(isinstance(cid, int) for cid in result.values())

    def test_resolution_below_min_rejected(self):
        """Resolution parameter below 0.1 is rejected."""
        graph = nx.complete_graph(5)
        graph = nx.DiGraph(graph)
        with pytest.raises(ValueError, match="resolution must be in"):
            detect_communities(graph, resolution=0.09)

    def test_resolution_above_max_rejected(self):
        """Resolution parameter above 10.0 is rejected."""
        graph = nx.complete_graph(5)
        graph = nx.DiGraph(graph)
        with pytest.raises(ValueError, match="resolution must be in"):
            detect_communities(graph, resolution=10.01)

    def test_resolution_non_numeric_rejected(self):
        """Non-numeric resolution raises ValueError."""
        graph = nx.complete_graph(5)
        graph = nx.DiGraph(graph)
        with pytest.raises(ValueError, match="resolution must be a number"):
            detect_communities(graph, resolution="invalid")

    def test_min_community_size_validation(self):
        """min_community_size < 1 raises ValueError."""
        graph = nx.complete_graph(5)
        graph = nx.DiGraph(graph)
        with pytest.raises(ValueError, match="min_community_size must be >= 1"):
            detect_communities(graph, min_community_size=0)

    def test_deterministic_results_same_seed(self):
        """Same seed produces identical partition (determinism for CI)."""
        graph = nx.DiGraph()
        for i in range(10):
            graph.add_node(f"W{i}")
            if i > 0:
                graph.add_edge(f"W{i-1}", f"W{i}")

        result1 = detect_communities(graph, seed=42, resolution=1.0)
        result2 = detect_communities(graph, seed=42, resolution=1.0)
        assert result1 == result2, "Same seed should produce identical results"

    def test_small_community_filtered(self):
        """Communities smaller than min_community_size are assigned -1."""
        graph = nx.DiGraph()
        # Ring of 3 nodes
        for src, dst in [("W1", "W2"), ("W2", "W3"), ("W3", "W1")]:
            graph.add_edge(src, dst)
        # Isolated pair
        graph.add_edge("W4", "W5")

        result = detect_communities(graph, min_community_size=3)
        # Ring of 3 should have a valid community id
        assert result["W1"] != -1
        assert result["W2"] != -1
        assert result["W3"] != -1
        # Isolated pair should have -1
        assert result["W4"] == -1
        assert result["W5"] == -1


class TestRingConcentrationScore:
    """Test intra-cluster trade volume computation."""

    def test_empty_trades(self):
        """Empty trades DataFrame returns empty dict."""
        community_map = {"W1": 0, "W2": 0}
        graph = nx.DiGraph()
        result = compute_ring_concentration_score(community_map, graph, None)
        assert result == {}

    def test_no_matching_trades(self):
        """Trades with non-matching wallets return 0.0 score."""
        community_map = {"W1": 0, "W2": 0}
        trades_df = pd.DataFrame({
            "base_account": ["W3"],
            "counter_account": ["W4"],
            "amount": [100.0],
        })
        graph = nx.DiGraph()
        result = compute_ring_concentration_score(community_map, graph, trades_df)
        # Community 0 exists but has no trades involving its members, so score is 0.0
        assert result == {0: 0.0}

    def test_intra_community_trades_high_score(self):
        """All trades within a community yield high concentration score."""
        community_map = {"W1": 0, "W2": 0, "W3": 0}
        trades_df = pd.DataFrame({
            "base_account": ["W1", "W2", "W3"],
            "counter_account": ["W2", "W3", "W1"],
            "amount": [100.0, 100.0, 100.0],
        })
        graph = nx.DiGraph()
        result = compute_ring_concentration_score(community_map, graph, trades_df)
        assert 0 in result
        assert result[0] == 1.0  # All trades are internal

    def test_mixed_trades_partial_score(self):
        """Community with both internal and external trades yields 0 < score < 1."""
        community_map = {"W1": 0, "W2": 0, "W3": 1}
        trades_df = pd.DataFrame({
            "base_account": ["W1", "W1"],
            "counter_account": ["W2", "W3"],
            "amount": [100.0, 100.0],
        })
        graph = nx.DiGraph()
        result = compute_ring_concentration_score(community_map, graph, trades_df)
        assert 0 in result
        assert 0.0 < result[0] < 1.0

    def test_non_communities_omitted(self):
        """Communities with id -1 are omitted from result."""
        community_map = {"W1": -1, "W2": -1}
        trades_df = pd.DataFrame({
            "base_account": ["W1"],
            "counter_account": ["W2"],
            "amount": [100.0],
        })
        graph = nx.DiGraph()
        result = compute_ring_concentration_score(community_map, graph, trades_df)
        assert -1 not in result


class TestValidateResolutionParameter:
    """Test resolution parameter validation."""

    def test_valid_resolution_boundaries(self):
        """Valid resolutions: 0.1, 1.0, 10.0."""
        assert validate_resolution_parameter(0.1)
        assert validate_resolution_parameter(1.0)
        assert validate_resolution_parameter(10.0)
        assert validate_resolution_parameter(5.5)

    def test_invalid_resolution_too_low(self):
        """Resolution < 0.1 is invalid."""
        assert not validate_resolution_parameter(0.09)
        assert not validate_resolution_parameter(0.0)

    def test_invalid_resolution_too_high(self):
        """Resolution > 10.0 is invalid."""
        assert not validate_resolution_parameter(10.01)
        assert not validate_resolution_parameter(100.0)

    def test_invalid_resolution_non_numeric(self):
        """Non-numeric resolution is invalid."""
        assert not validate_resolution_parameter("1.0")
        assert not validate_resolution_parameter(None)
