"""Unit tests for detection/gnn_encoder.py and detection/wallet_graph.py GNN additions.

Covers:
- build_co_trade_graph produces correct edge set on a known fixture
- GNNEncoder.encode returns shape (GNN_EMBEDDING_DIM,) and dtype float32
- GNNEncoder.encode is deterministic across two calls on the same graph
- update_node produces a result within cosine distance 0.05 of full re-encoding
  for a 1-hop change
- SHA-256 mismatch raises ModelIntegrityError
- Graceful zero-fallback when encoder artifact is absent
- Stellar account ID sanitisation in build_funding_graph and build_co_trade_graph

All GNN tests are skipped automatically when torch / torch_geometric are not
installed (CI environments without GPU/torch can still run all other tests).
"""

from __future__ import annotations

import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from config import config
from detection.wallet_graph import build_co_trade_graph, build_funding_graph
from ingestion.data_models import AccountActivity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GNN_DIM = config.GNN_EMBEDDING_DIM

# Detect whether torch / torch_geometric are available so tests can be skipped.
try:
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch and torch_geometric not installed"
)


def _stellar_id(prefix: str) -> str:
    """Generate a syntactically valid Stellar public key for testing."""
    body = prefix.upper().ljust(55, "A")[:55]
    return f"G{body}"


# 56-character valid Stellar account IDs
W_A = _stellar_id("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
W_B = _stellar_id("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
W_C = _stellar_id("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
W_D = _stellar_id("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")


def _sample_trades_df() -> pd.DataFrame:
    """Three trades: A↔B and A↔C on USDC/XLM, B↔D on AQUA/XLM."""
    now = pd.Timestamp("2024-01-01T00:00:00Z")
    return pd.DataFrame(
        [
            {
                "trade_id": "1",
                "ledger_close_time": now,
                "base_account": W_A,
                "counter_account": W_B,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "price": 0.1,
                "pair_id": "USDC:issuer/XLM:native",
            },
            {
                "trade_id": "2",
                "ledger_close_time": now + pd.Timedelta(minutes=5),
                "base_account": W_A,
                "counter_account": W_C,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 200.0,
                "price": 0.1,
                "pair_id": "USDC:issuer/XLM:native",
            },
            {
                "trade_id": "3",
                "ledger_close_time": now + pd.Timedelta(hours=2),
                "base_account": W_B,
                "counter_account": W_D,
                "base_asset": "AQUA:issuer2",
                "counter_asset": "XLM:native",
                "amount": 50.0,
                "price": 0.05,
                "pair_id": "AQUA:issuer2/XLM:native",
            },
        ]
    )


# ---------------------------------------------------------------------------
# Tests: build_co_trade_graph
# ---------------------------------------------------------------------------


class TestBuildCoTradeGraph:
    def test_edge_set_within_window(self):
        """A and C both traded USDC/XLM within the window → they get a co_trade edge."""
        df = _sample_trades_df()
        # Use a large window so A, B, C all link on USDC/XLM
        g = build_co_trade_graph(df, window_hours=24)
        # A, B, C all traded USDC/XLM within 24 h → edges among them
        assert g.has_edge(W_A, W_B) or g.has_edge(W_B, W_A)
        assert g.has_edge(W_A, W_C) or g.has_edge(W_C, W_A)

    def test_no_edge_outside_window(self):
        """Trades farther apart than the window must NOT get an edge."""
        _sample_trades_df()
        # Trade 1 (T=0) and Trade 2 (T=5min) are on USDC/XLM.
        # Trade 3 (T=2h) is on AQUA/XLM with different wallets.
        # With a 1-second window, only trades at exactly the same second can link.
        # All our test trades are minutes/hours apart, so nothing should link.
        import pandas as pd

        now = pd.Timestamp("2024-01-01T00:00:00Z")
        df2 = pd.DataFrame(
            [
                {
                    "trade_id": "a",
                    "ledger_close_time": now,
                    "base_account": W_A,
                    "counter_account": W_B,
                    "base_asset": "USDC:issuer",
                    "counter_asset": "XLM:native",
                    "amount": 100.0,
                    "price": 0.1,
                },
                {
                    "trade_id": "b",
                    "ledger_close_time": now + pd.Timedelta(hours=3),
                    "base_account": W_C,
                    "counter_account": W_D,
                    "base_asset": "USDC:issuer",
                    "counter_asset": "XLM:native",
                    "amount": 50.0,
                    "price": 0.1,
                },
            ]
        )
        # 1-hour window: trades are 3h apart → no co-trade edge
        g = build_co_trade_graph(df2, window_hours=1)
        # A and C both traded USDC/XLM but 3 hours apart → no edge
        assert not g.has_edge(W_A, W_C)
        assert not g.has_edge(W_C, W_A)

    def test_edge_attributes(self):
        """Co-trade edges must carry edge_type, weight, and timestamp."""
        df = _sample_trades_df()
        g = build_co_trade_graph(df, window_hours=24)
        for _u, _v, data in g.edges(data=True):
            assert data["edge_type"] == "co_trade"
            assert data["weight"] >= 1
            assert "timestamp" in data

    def test_empty_dataframe(self):
        g = build_co_trade_graph(pd.DataFrame(), window_hours=24)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_bidirectional_edges(self):
        """Co-trade graph must have both A→B and B→A edges."""
        df = _sample_trades_df()
        g = build_co_trade_graph(df, window_hours=24)
        if g.has_edge(W_A, W_B):
            assert g.has_edge(W_B, W_A), "Co-trade edges must be bidirectional"

    def test_rejects_invalid_stellar_ids(self):
        """Edges with non-Stellar-format account IDs must be silently dropped."""
        now = pd.Timestamp("2024-01-01T00:00:00Z")
        df = pd.DataFrame(
            [
                {
                    "trade_id": "x",
                    "ledger_close_time": now,
                    "base_account": "INVALID_ID",  # not a valid Stellar key
                    "counter_account": W_B,
                    "base_asset": "USDC:issuer",
                    "counter_asset": "XLM:native",
                    "amount": 100.0,
                    "price": 0.1,
                }
            ]
        )
        g = build_co_trade_graph(df, window_hours=24)
        assert "INVALID_ID" not in g.nodes


class TestBuildFundingGraphSanitisation:
    def test_rejects_invalid_funder(self):
        """A funding edge with a malformed funder ID must be silently dropped."""
        activities = [
            AccountActivity(
                account_id=W_A,
                account_created_at=pd.Timestamp("2021-01-01"),
                funding_account="NOT_A_STELLAR_ID",
            )
        ]
        g = build_funding_graph(activities, validate_account_ids=True)
        # Node W_A is added, but no edge from the invalid funder
        assert W_A in g.nodes
        assert g.number_of_edges() == 0

    def test_rejects_invalid_account_id(self):
        """An activity with a malformed account_id must be silently skipped."""
        activities = [
            AccountActivity(
                account_id="bad_wallet",
                account_created_at=pd.Timestamp("2021-01-01"),
            )
        ]
        g = build_funding_graph(activities, validate_account_ids=True)
        assert "bad_wallet" not in g.nodes

    def test_accepts_valid_stellar_ids(self):
        """Valid Stellar IDs must be accepted."""
        activities = [
            AccountActivity(
                account_id=W_A,
                account_created_at=pd.Timestamp("2021-01-01"),
                funding_account=W_B,
            )
        ]
        g = build_funding_graph(activities, validate_account_ids=True)
        assert W_A in g.nodes
        assert g.has_edge(W_B, W_A)


# ---------------------------------------------------------------------------
# Tests: GNNEncoder (require torch)
# ---------------------------------------------------------------------------


def _small_graph() -> nx.DiGraph:
    """Minimal funding-like graph with 4 valid Stellar wallets."""
    g = nx.DiGraph()
    g.add_edge(W_A, W_B, edge_type="funding", weight=1)
    g.add_edge(W_A, W_C, edge_type="funding", weight=1)
    g.add_edge(W_B, W_D, edge_type="co_trade", weight=2)
    return g


@requires_torch
class TestGNNEncoderEncode:
    def test_encode_shape(self):
        from detection.gnn_encoder import GNNEncoder

        enc = GNNEncoder(embedding_dim=_GNN_DIM)
        g = _small_graph()
        emb = enc.encode(g, W_B)
        assert emb.shape == (_GNN_DIM,)

    def test_encode_dtype(self):
        from detection.gnn_encoder import GNNEncoder

        enc = GNNEncoder(embedding_dim=_GNN_DIM)
        emb = enc.encode(_small_graph(), W_A)
        assert emb.dtype == np.float32

    def test_encode_deterministic(self):
        """Two calls on the same graph snapshot must return identical embeddings."""
        from detection.gnn_encoder import GNNEncoder

        enc = GNNEncoder(embedding_dim=_GNN_DIM, random_state=0)
        g = _small_graph()
        emb1 = enc.encode(g, W_A)
        emb2 = enc.encode(g, W_A)
        np.testing.assert_array_equal(emb1, emb2)

    def test_encode_wallet_not_in_graph_raises(self):
        from detection.gnn_encoder import GNNEncoder

        enc = GNNEncoder(embedding_dim=_GNN_DIM)
        g = _small_graph()
        with pytest.raises(KeyError):
            enc.encode(g, "GNOT_IN_GRAPH" + "A" * 50)


@requires_torch
class TestGNNEncoderUpdateNode:
    def test_update_node_close_to_full_encode(self):
        """update_node result must be within cosine distance 0.05 of full re-encode."""

        from detection.gnn_encoder import GNNEncoder

        enc = GNNEncoder(embedding_dim=_GNN_DIM, random_state=7)
        g = _small_graph()

        # Full encode of W_B before new edge
        enc.encode(g, W_B)

        # Simulate adding one new co-trade edge B→C
        new_edge = (W_B, W_C)
        incremental = enc.update_node(W_B, [new_edge], g)

        # Add edge to graph and do full re-encode for comparison
        g.add_edge(*new_edge, edge_type="co_trade", weight=1)
        enc._embedding_cache.clear()
        enc._last_node_order = []
        full_after = enc.encode(g, W_B)

        # Cosine distance between incremental and full_after should be small
        norm_i = np.linalg.norm(incremental)
        norm_f = np.linalg.norm(full_after)
        if norm_i > 0 and norm_f > 0:
            cos_sim = float(np.dot(incremental, full_after) / (norm_i * norm_f))
            cos_dist = 1.0 - cos_sim
        else:
            cos_dist = 0.0

        assert cos_dist <= 0.05, (
            f"Cosine distance {cos_dist:.4f} exceeds threshold 0.05 — "
            "incremental update diverged from full re-encoding"
        )

    def test_update_node_shape(self):
        from detection.gnn_encoder import GNNEncoder

        enc = GNNEncoder(embedding_dim=_GNN_DIM, random_state=3)
        g = _small_graph()
        result = enc.update_node(W_A, [(W_A, W_D)], g)
        assert result.shape == (_GNN_DIM,)


@requires_torch
class TestGNNEncoderPersistence:
    def test_save_and_load(self, tmp_path):
        from detection.gnn_encoder import GNNEncoder

        enc = GNNEncoder(embedding_dim=_GNN_DIM, model_dir=str(tmp_path))
        enc.save()
        assert os.path.exists(os.path.join(str(tmp_path), "gnn_encoder.pt"))
        assert os.path.exists(os.path.join(str(tmp_path), "metrics.json"))

        # Load into a fresh encoder — weights should load cleanly
        enc2 = GNNEncoder(embedding_dim=_GNN_DIM, model_dir=str(tmp_path))
        enc2.load()

    def test_sha256_mismatch_raises_model_integrity_error(self, tmp_path):
        """Tampered artifact must raise ModelIntegrityError on load."""
        from detection.gnn_encoder import GNNEncoder
        from detection.persistence import ModelIntegrityError

        enc = GNNEncoder(embedding_dim=_GNN_DIM, model_dir=str(tmp_path))
        enc.save()

        # Corrupt the artifact
        artifact_path = os.path.join(str(tmp_path), "gnn_encoder.pt")
        with open(artifact_path, "ab") as f:
            f.write(b"\x00\x00")

        enc2 = GNNEncoder(embedding_dim=_GNN_DIM, model_dir=str(tmp_path))
        with pytest.raises(ModelIntegrityError):
            enc2.load()

    def test_missing_artifact_raises_file_not_found(self, tmp_path):
        from detection.gnn_encoder import GNNEncoder

        enc = GNNEncoder(embedding_dim=_GNN_DIM, model_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            enc.load()


# ---------------------------------------------------------------------------
# Tests: Graceful zero-fallback in feature_engineering
# ---------------------------------------------------------------------------


class TestGNNGracefulFallback:
    """compute_graph_embedding_features must return zeros when encoder is absent."""

    def test_build_feature_vector_gnn_absent(self):
        """When gnn_encoder=None, gnn_0…gnn_31 must all be 0.0."""
        from detection.feature_engineering import build_feature_vector

        wallet = W_A
        trades = pd.DataFrame(
            [
                {
                    "trade_id": "1",
                    "ledger_close_time": "2024-01-01T00:00:00Z",
                    "base_account": W_A,
                    "counter_account": W_B,
                    "base_asset": "USDC:issuer",
                    "counter_asset": "XLM:native",
                    "amount": 100.0,
                    "price": 0.1,
                }
            ]
        )
        row = build_feature_vector(wallet, trades, gnn_encoder=None)
        for i in range(config.GNN_EMBEDDING_DIM):
            assert row[f"gnn_{i}"] == 0.0, f"gnn_{i} must be 0.0 when encoder is absent"

    @requires_torch
    def test_build_feature_vector_gnn_present(self):
        """When encoder is provided and wallet is in graph, gnn_ columns must be non-trivially set."""
        from detection.feature_engineering import build_feature_vector
        from detection.gnn_encoder import GNNEncoder

        wallet = W_A
        trades = pd.DataFrame(
            [
                {
                    "trade_id": "1",
                    "ledger_close_time": "2024-01-01T00:00:00Z",
                    "base_account": W_A,
                    "counter_account": W_B,
                    "base_asset": "USDC:issuer",
                    "counter_asset": "XLM:native",
                    "amount": 100.0,
                    "price": 0.1,
                }
            ]
        )
        graph = _small_graph()
        enc = GNNEncoder(embedding_dim=_GNN_DIM, random_state=0)
        row = build_feature_vector(wallet, trades, funding_graph=graph, gnn_encoder=enc)

        # Keys must be present
        for i in range(_GNN_DIM):
            assert f"gnn_{i}" in row

        # At least some values should be non-zero (untrained weights still produce non-zero outputs)
        values = [row[f"gnn_{i}"] for i in range(_GNN_DIM)]
        assert any(v != 0.0 for v in values), "Expected non-zero GNN features for wallet in graph"
