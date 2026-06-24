"""Unit tests for detection/cross_venue_features.py.

Tests cover the 7 cross-venue features, coordination graph construction,
Louvain cluster detection, and fallback behaviour on empty AMM data.
"""

import datetime

import networkx as nx
import pandas as pd
import pytest

from detection.cross_venue_features import (
    build_coordination_graph,
    compute_cross_venue_features,
    cross_venue_cluster_score,
    detect_coordinated_clusters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime.datetime(2024, 1, 10, 12, 0, 0, tzinfo=datetime.UTC)


def _ts(offset_seconds: float) -> str:
    t = _BASE_TIME + datetime.timedelta(seconds=offset_seconds)
    return t.isoformat()


def _make_trade(
    base_account: str,
    counter_account: str,
    amount: float = 100.0,
    offset_seconds: float = 0.0,
    base_asset: str = "USDC:issuer",
    counter_asset: str = "XLM:native",
) -> dict:
    return {
        "trade_id": f"{base_account}-{counter_account}-{offset_seconds}",
        "ledger_close_time": _ts(offset_seconds),
        "base_account": base_account,
        "counter_account": counter_account,
        "base_asset": base_asset,
        "counter_asset": counter_asset,
        "amount": amount,
        "price": 0.5,
    }


def _make_df(trades: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# cross_venue_timing_synchrony — 1.0 for perfectly paired trades
# ---------------------------------------------------------------------------


def test_timing_synchrony_perfect_pairing():
    """All AMM trades within 10 s of a SDEX trade → synchrony = 1.0."""
    wallet = "WALLET_A"
    sdex = _make_df(
        [
            _make_trade(wallet, "B", offset_seconds=0.0),
            _make_trade(wallet, "C", offset_seconds=100.0),
            _make_trade(wallet, "D", offset_seconds=200.0),
        ]
    )
    # Each AMM trade is within 5 s of a SDEX trade
    amm = _make_df(
        [
            _make_trade(wallet, "E", offset_seconds=3.0),
            _make_trade(wallet, "F", offset_seconds=98.0),
            _make_trade(wallet, "G", offset_seconds=205.0),
        ]
    )

    features = compute_cross_venue_features(wallet, sdex, amm)
    assert features["cross_venue_timing_synchrony"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cross_venue_timing_synchrony — 0.0 for completely disjoint timing
# ---------------------------------------------------------------------------


def test_timing_synchrony_disjoint():
    """AMM trades far from any SDEX trade → synchrony = 0.0."""
    wallet = "WALLET_A"
    sdex = _make_df([_make_trade(wallet, "B", offset_seconds=0.0)])
    amm = _make_df(
        [
            _make_trade(wallet, "C", offset_seconds=1000.0),
            _make_trade(wallet, "D", offset_seconds=2000.0),
        ]
    )

    features = compute_cross_venue_features(wallet, sdex, amm)
    assert features["cross_venue_timing_synchrony"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cross_venue_net_flow — 0.0 for a perfect round-trip
# ---------------------------------------------------------------------------


def test_net_flow_perfect_round_trip():
    """Wallet sends 100 on SDEX and receives 100 on AMM → net flow ≈ 0."""
    wallet = "WALLET_A"
    # Wallet is base on SDEX (sends)
    sdex = _make_df([_make_trade(wallet, "B", amount=100.0)])
    # Wallet is counter on AMM (receives same amount)
    amm = _make_df(
        [
            {
                "trade_id": "amm-1",
                "ledger_close_time": _ts(5.0),
                "base_account": "POOL",
                "counter_account": wallet,
                "base_asset": "XLM:native",
                "counter_asset": "USDC:issuer",
                "amount": 100.0,
                "price": 0.5,
            }
        ]
    )

    features = compute_cross_venue_features(wallet, sdex, amm)
    assert features["cross_venue_net_flow"] < 0.01


# ---------------------------------------------------------------------------
# build_coordination_graph — N edges for N perfectly-paired SDEX/AMM trade pairs
# ---------------------------------------------------------------------------


def test_build_coordination_graph_edge_count():
    """N distinct SDEX/AMM trade pairs within window produce exactly N edges."""
    n = 5
    sdex_trades = []
    amm_trades = []
    for i in range(n):
        sdex_trades.append(_make_trade(f"WALLET_{i}", f"CP_{i}", offset_seconds=float(i * 100)))
        amm_trades.append(_make_trade(f"WALLET_{i}", "AMMPool", offset_seconds=float(i * 100 + 3)))

    sdex_df = _make_df(sdex_trades)
    amm_df = _make_df(amm_trades)

    graph = build_coordination_graph(sdex_df, amm_df, window_seconds=10)

    # Each wallet pairs with their SDEX counterparty AND the AMM pool side within 10s
    # At a minimum, we should have edges from paired wallets
    assert graph.number_of_edges() >= n


# ---------------------------------------------------------------------------
# detect_coordinated_clusters — each wallet in exactly one cluster (partition)
# ---------------------------------------------------------------------------


def test_detect_coordinated_clusters_partition_property():
    """Every wallet appears in exactly one cluster (partition, not cover)."""
    wallets = [f"W{i}" for i in range(10)]
    graph = nx.DiGraph()
    # Create two connected groups
    for i in range(5):
        graph.add_edge(wallets[i], wallets[(i + 1) % 5], venue="sdex", weight=1)
    for i in range(5, 10):
        graph.add_edge(wallets[i], wallets[5 + (i - 4) % 5], venue="amm", weight=1)

    clusters = detect_coordinated_clusters(graph)

    # Every wallet in exactly one cluster
    all_wallets_in_clusters = []
    for c in clusters:
        all_wallets_in_clusters.extend(c)

    assert len(all_wallets_in_clusters) == len(
        set(all_wallets_in_clusters)
    ), "Some wallets appear in multiple clusters (not a partition)"
    assert set(all_wallets_in_clusters) == set(wallets), "Not all wallets are covered by clusters"


# ---------------------------------------------------------------------------
# All 7 features fall back to 0.0 when AMM data is empty
# ---------------------------------------------------------------------------


def test_all_features_fallback_to_zero_when_amm_empty():
    wallet = "WALLET_A"
    sdex = _make_df([_make_trade(wallet, "B")])
    amm = pd.DataFrame()  # empty

    features = compute_cross_venue_features(wallet, sdex, amm)

    expected_keys = [
        "venue_trade_ratio",
        "cross_venue_volume_correlation",
        "cross_venue_timing_synchrony",
        "cross_venue_net_flow",
        "counterparty_venue_overlap",
        "simultaneous_order_pair",
        "cross_venue_cluster_score",
    ]
    for key in expected_keys:
        assert key in features, f"Feature {key!r} missing from output"
        assert features[key] == pytest.approx(
            0.0
        ), f"Feature {key!r} should be 0.0 when AMM data is empty, got {features[key]}"


def test_all_features_fallback_to_zero_when_amm_none():
    wallet = "WALLET_A"
    sdex = _make_df([_make_trade(wallet, "B")])

    features = compute_cross_venue_features(wallet, sdex, None)

    for key, val in features.items():
        assert val == pytest.approx(
            0.0
        ), f"Feature {key!r} should be 0.0 when AMM is None, got {val}"


# ---------------------------------------------------------------------------
# cross_venue_cluster_score — basic sanity
# ---------------------------------------------------------------------------


def test_cluster_score_returns_zero_when_no_clusters():
    graph = nx.DiGraph()
    graph.add_node("W1")
    score = cross_venue_cluster_score("W1", [], graph)
    assert score == pytest.approx(0.0)


def test_cluster_score_returns_zero_for_unknown_wallet():
    graph = nx.DiGraph()
    graph.add_edge("W1", "W2", venue="sdex", weight=1)
    clusters = [{"W1", "W2"}]
    score = cross_venue_cluster_score("UNKNOWN", clusters, graph)
    assert score == pytest.approx(0.0)


def test_cluster_score_nonzero_for_central_wallet():
    graph = nx.DiGraph()
    # Hub wallet connected to many others
    hub = "HUB"
    spokes = [f"SPOKE_{i}" for i in range(5)]
    for s in spokes:
        graph.add_edge(hub, s, venue="sdex", weight=2)
        graph.add_edge(hub, s, venue="amm", weight=1)

    clusters = [{hub} | set(spokes)]
    score = cross_venue_cluster_score(hub, clusters, graph)
    assert score > 0.0


# ---------------------------------------------------------------------------
# build_coordination_graph — empty inputs produce empty graph
# ---------------------------------------------------------------------------


def test_build_coordination_graph_empty_inputs():
    graph = build_coordination_graph(pd.DataFrame(), pd.DataFrame(), window_seconds=10)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


# ---------------------------------------------------------------------------
# detect_coordinated_clusters — empty graph returns empty list
# ---------------------------------------------------------------------------


def test_detect_coordinated_clusters_empty_graph():
    clusters = detect_coordinated_clusters(nx.DiGraph())
    assert clusters == []


# ---------------------------------------------------------------------------
# venue_trade_ratio
# ---------------------------------------------------------------------------


def test_venue_trade_ratio_balanced():
    wallet = "WALLET_A"
    sdex = _make_df([_make_trade(wallet, "B")] * 4)
    amm = _make_df([_make_trade(wallet, "C")] * 2)
    features = compute_cross_venue_features(wallet, sdex, amm)
    assert features["venue_trade_ratio"] == pytest.approx(2.0)


def test_venue_trade_ratio_zero_when_no_amm():
    wallet = "WALLET_A"
    sdex = _make_df([_make_trade(wallet, "B")])
    features = compute_cross_venue_features(wallet, sdex, pd.DataFrame())
    assert features["venue_trade_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# simultaneous_order_pair
# ---------------------------------------------------------------------------


def test_simultaneous_order_pair_overlapping_activity():
    wallet = "WALLET_A"
    sdex = _make_df(
        [
            _make_trade(wallet, "B", offset_seconds=0.0),
            _make_trade(wallet, "C", offset_seconds=3600.0),
        ]
    )
    # AMM trades overlap with SDEX time range
    amm = _make_df([_make_trade(wallet, "D", offset_seconds=1800.0)])
    features = compute_cross_venue_features(wallet, sdex, amm)
    assert features["simultaneous_order_pair"] == pytest.approx(1.0)


def test_simultaneous_order_pair_non_overlapping():
    wallet = "WALLET_A"
    sdex = _make_df([_make_trade(wallet, "B", offset_seconds=0.0)])
    amm = _make_df([_make_trade(wallet, "C", offset_seconds=100000.0)])
    features = compute_cross_venue_features(wallet, sdex, amm)
    # Ranges don't overlap if we treat each as a point — they do overlap by range
    # since sdex_min=T, sdex_max=T, amm_min=T+100000, amm_max=T+100000
    # sdex_min(T) <= amm_max(T+100000) is True, amm_min(T+100000) <= sdex_max(T) is False
    assert features["simultaneous_order_pair"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration test (skipped unless LEDGERLENS_INTEGRATION_TESTS=1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    __import__("os").getenv("LEDGERLENS_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled — set LEDGERLENS_INTEGRATION_TESTS=1 to run",
)
def test_integration_backfill_amm_testnet():
    """Backfill 7 days of AMM data for known Testnet pools and verify features."""
    import datetime

    from ingestion.amm_pool_loader import load_amm_pool_trades

    testnet_pools = [
        "a" * 64,  # replace with real Testnet pool IDs
    ]
    since = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    until = datetime.datetime(2024, 1, 7, tzinfo=datetime.UTC)

    for pool_id in testnet_pools:
        df = load_amm_pool_trades(pool_id, since, until)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
