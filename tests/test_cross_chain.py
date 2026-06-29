"""Unit and integration tests for cross-chain identity resolution."""

from __future__ import annotations

import base64
from datetime import datetime, UTC
import networkx as nx
import pytest

from detection.cross_chain.behavioral_matcher import BehavioralMatcher, to_timestamp
from detection.cross_chain.bridge_detector import BridgeDetector, bytes_to_base58
from detection.cross_chain.identity_graph import IdentityGraph
from detection.cross_chain.resolver import resolve, resolve_risk_scores
from detection.persistence import Base, get_engine, get_session_factory
from detection.risk_propagation import propagate_risk_scores


@pytest.fixture
def db_url(tmp_path):
    db_file = tmp_path / "test_identity.db"
    return f"sqlite:///{db_file}"


@pytest.fixture
def session_factory(db_url):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return get_session_factory(engine)


# ===========================================================================
# 1. Identity Graph & Resolver Tests
# ===========================================================================

def test_identity_graph_nodes_and_edges(session_factory, db_url):
    graph = IdentityGraph(session_factory)

    # Add nodes
    graph.add_node("GSTELLAR1", "stellar", risk_score=15.0)
    graph.add_node("0xETH1", "ethereum", risk_score=85.0)
    graph.add_node("SOL1", "solana", risk_score=45.0)

    # Add edges
    graph.add_edge("GSTELLAR1", "0xETH1", "bridge", confidence=1.0, metadata={"tx": "tx1"})
    graph.add_edge("0xETH1", "SOL1", "behavioral", confidence=0.85)

    # Check component of GSTELLAR1
    comp = graph.get_connected_component("GSTELLAR1")
    assert len(comp["eth"]) == 1
    assert comp["eth"][0]["address"] == "0xeth1"
    assert comp["eth"][0]["risk_score"] == 85.0

    assert len(comp["sol"]) == 1
    assert comp["sol"][0]["address"] == "SOL1"
    assert comp["sol"][0]["risk_score"] == 45.0

    # Test Resolver APIs
    resolved_addresses = resolve("GSTELLAR1", db_url=db_url)
    assert resolved_addresses["eth"] == ["0xeth1"]
    assert resolved_addresses["sol"] == ["SOL1"]

    scores = resolve_risk_scores("GSTELLAR1", db_url=db_url)
    assert scores["0xeth1"] == 85.0
    assert scores["SOL1"] == 45.0


# ===========================================================================
# 2. Bridge Detector Tests
# ===========================================================================

def test_bridge_detector_memo_parsing():
    detector = BridgeDetector()

    # EVM Text
    assert detector.parse_memo_address("text", "0x71C7656EC7ab88b098defB751B7401B5f6d1476B") == (
        "0x71c7656ec7ab88b098defb751b7401b5f6d1476b",
        "ethereum",
    )
    # EVM Text without 0x
    assert detector.parse_memo_address("text", "71C7656EC7ab88b098defB751B7401B5f6d1476B") == (
        "0x71c7656ec7ab88b098defb751b7401b5f6d1476b",
        "ethereum",
    )
    # Solana Text
    sol_addr = "HN7cABFi4JZZuN861HT27G5hU3yvJy7as59s3n54NDry"
    assert detector.parse_memo_address("text", sol_addr) == (sol_addr, "solana")

    # EVM Padded Hash (Left padded)
    padded_hex = "00000000000000000000000071C7656EC7ab88b098defB751B7401B5f6d1476B"
    assert detector.parse_memo_address("hash", padded_hex) == (
        "0x71c7656ec7ab88b098defb751b7401b5f6d1476b",
        "ethereum",
    )

    # EVM Padded Hash (Right padded)
    right_padded_hex = "71C7656EC7ab88b098defB751B7401B5f6d1476B000000000000000000000000"
    assert detector.parse_memo_address("hash", right_padded_hex) == (
        "0x71c7656ec7ab88b098defb751b7401b5f6d1476b",
        "ethereum",
    )

    # Base64 Padded EVM Hash
    b64_val = base64.b64encode(bytes.fromhex(padded_hex)).decode()
    assert detector.parse_memo_address("hash", b64_val) == (
        "0x71c7656ec7ab88b098defb751b7401b5f6d1476b",
        "ethereum",
    )

    # Solana raw 32-byte key inside hash
    raw_key = bytes([1] * 32)
    expected_sol = bytes_to_base58(raw_key)
    assert detector.parse_memo_address("hash", raw_key.hex()) == (expected_sol, "solana")


def test_bridge_detector_link_detection():
    detector = BridgeDetector(anchor_addresses=["GANCHOR"])

    txs = [
        # Deposit: user is 'from'
        {
            "id": "tx1",
            "source_account": "GUSER1",
            "memo_type": "text",
            "memo": "0x71C7656EC7ab88b098defB751B7401B5f6d1476B",
            "from": "GUSER1",
            "to": "GANCHOR",
        },
        # Withdrawal: user is 'to'
        {
            "id": "tx2",
            "source_account": "GANCHOR",
            "memo_type": "text",
            "memo": "HN7cABFi4JZZuN861HT27G5hU3yvJy7as59s3n54NDry",
            "from": "GANCHOR",
            "to": "GUSER2",
        },
        # Invalid / skipped
        {
            "id": "tx3",
            "source_account": "GUSER3",
            "memo_type": "text",
            "memo": "hello world",
        }
    ]

    links = detector.detect_bridge_links(txs)
    assert len(links) == 2

    assert links[0]["stellar_address"] == "GUSER1"
    assert links[0]["linked_address"] == "0x71c7656ec7ab88b098defb751b7401b5f6d1476b"
    assert links[0]["chain"] == "ethereum"

    assert links[1]["stellar_address"] == "GUSER2"
    assert links[1]["linked_address"] == "HN7cABFi4JZZuN861HT27G5hU3yvJy7as59s3n54NDry"
    assert links[1]["chain"] == "solana"


# ===========================================================================
# 3. Behavioral Matcher Tests
# ===========================================================================

def test_behavioral_matcher_amount_fingerprint():
    stellar_txs = [
        {"wallet": "GUSER1", "timestamp": 1700000000, "amount": 100.0, "id": "s1"},
        {"wallet": "GUSER2", "timestamp": 1700000100, "amount": 250.0, "id": "s2"},
    ]
    external_txs = [
        {"wallet": "0xETH1", "timestamp": 1700000010, "amount": 100.05, "chain": "ethereum", "id": "e1"},
        {"wallet": "0xETH2", "timestamp": 1700000200, "amount": 250.0, "chain": "ethereum", "id": "e2"},
    ]

    # Matching with 0.1% tolerance and 60s window
    # s1 (100.0) and e1 (100.05): diff = 0.05 (0.05/100 = 0.0005 <= 0.001), dt = 10s <= 60s -> MATCH!
    # s2 (250.0) and e2 (250.0): diff = 0, dt = 100s > 60s -> NO MATCH!
    links = BehavioralMatcher.match_amount_fingerprints(stellar_txs, external_txs, tolerance=0.001, window_seconds=60.0)
    assert len(links) == 1
    assert links[0]["stellar_address"] == "GUSER1"
    assert links[0]["linked_address"] == "0xETH1"
    assert round(links[0]["confidence"], 4) == 0.9995


def test_behavioral_matcher_timing_correlation():
    # Construct matching time patterns (counts per hour)
    # 10 hours, highly correlated counts
    times_s = []
    times_ext = []
    times_uncorrelated = []

    pattern_s = [1, 2, 0, 4, 1, 0, 3, 2, 0, 1]
    pattern_ext = [1, 2, 0, 5, 1, 0, 3, 2, 0, 1]
    pattern_unc = [5, 0, 4, 0, 1, 3, 0, 1, 5, 2]

    start_time = 1700000000.0
    for hour in range(10):
        hour_start = start_time + hour * 3600.0
        # Add transactions for Stellar
        for _ in range(pattern_s[hour]):
            times_s.append(hour_start + 1800.0)
        # Add transactions for EVM
        for _ in range(pattern_ext[hour]):
            times_ext.append(hour_start + 1850.0)
        # Add uncorrelated transactions for EVM2
        for _ in range(pattern_unc[hour]):
            times_uncorrelated.append(hour_start + 1200.0)

    stellar_txs = [{"wallet": "GUSER1", "timestamp": t} for t in times_s]
    external_txs = (
        [{"wallet": "0xETH1", "timestamp": t, "chain": "ethereum"} for t in times_ext] +
        [{"wallet": "0xETH2", "timestamp": t, "chain": "ethereum"} for t in times_uncorrelated]
    )

    links = BehavioralMatcher.match_timing_correlation(
        stellar_txs, external_txs, bin_size_seconds=3600.0, min_common_bins=5, threshold=0.8
    )

    # GUSER1 should match 0xETH1 but NOT 0xETH2
    assert len(links) == 1
    assert links[0]["stellar_address"] == "GUSER1"
    assert links[0]["linked_address"] == "0xETH1"
    assert links[0]["confidence"] >= 0.8


# ===========================================================================
# 4. Risk Propagation Integration Tests
# ===========================================================================

def test_risk_propagation_with_cross_chain(session_factory, db_url):
    # Initialize DB with cross-chain link
    graph = IdentityGraph(session_factory)
    graph.add_node("GSTELLAR3", "stellar", risk_score=0.0)
    graph.add_node("0xETH_FLAGGED", "ethereum", risk_score=95.0)
    graph.add_edge("GSTELLAR3", "0xETH_FLAGGED", "bridge")

    # Build Stellar graph: GSTELLAR1 -> GSTELLAR2 -> GSTELLAR3
    funding = nx.DiGraph()
    funding.add_edge("GSTELLAR1", "GSTELLAR2")
    funding.add_edge("GSTELLAR2", "GSTELLAR3")

    co_trade = nx.Graph()
    co_trade.add_edge("GSTELLAR2", "GSTELLAR3")
    co_trade.add_edge("GSTELLAR1", "GSTELLAR2")

    base_scores = {"GSTELLAR3": 10.0}

    # Run propagation with cross chain db
    propagated_scores = propagate_risk_scores(
        base_scores=base_scores,
        funding_graph=funding,
        co_trade_graph=co_trade,
        alpha=0.15,
        db_url=db_url,
    )

    # Without cross chain db, propagation only uses GSTELLAR3's base score of 10.0
    normal_scores = propagate_risk_scores(
        base_scores=base_scores,
        funding_graph=funding,
        co_trade_graph=co_trade,
        alpha=0.15,
    )

    # With cross chain db, GSTELLAR3's base seed risk is updated to 95.0, so all scores should be higher
    assert propagated_scores["GSTELLAR3"] > 14.0
    assert propagated_scores["GSTELLAR2"] > 0.0
    assert propagated_scores["GSTELLAR1"] > 0.0
    assert propagated_scores["GSTELLAR3"] > normal_scores["GSTELLAR3"] * 9
    assert normal_scores["GSTELLAR3"] <= 10.0


# ===========================================================================
# 5. False Positive Rate Test
# ===========================================================================

def test_false_positive_rate():
    # Build 100 mock pairs of activity records (50 true matches, 50 false matches)
    # Verify that the false positive rate is under 5% (i.e. <= 2 false matches)
    stellar_txs = []
    external_txs = []
    
    # 50 True matches: identical/highly similar amounts & timing
    for i in range(50):
        time_s = 1700000000 + i * 1000
        stellar_txs.append({"wallet": f"GSTEL_{i}", "timestamp": time_s, "amount": 100.0 + i})
        external_txs.append({"wallet": f"GEXT_{i}", "timestamp": time_s + 5, "amount": 100.0 + i, "chain": "ethereum"})

    # 50 False matches: random amounts & timing
    for i in range(50, 100):
        time_s = 1700000000 + i * 1000
        stellar_txs.append({"wallet": f"GSTEL_{i}", "timestamp": time_s, "amount": 100.0 + i})
        external_txs.append({"wallet": f"GEXT_{i}", "timestamp": time_s + 500, "amount": 999.0 - i, "chain": "ethereum"})

    # Evaluate matchers
    detected_links = BehavioralMatcher.match_amount_fingerprints(
        stellar_txs, external_txs, tolerance=0.001, window_seconds=60.0
    )

    # A false positive is a link where GSTEL_{i} is linked to GEXT_{j} where i != j,
    # or GSTEL_{i} is linked to GEXT_{i} for i >= 50 (which we set up as false matches).
    false_positives = 0
    true_positives = 0

    for link in detected_links:
        s_idx = int(link["stellar_address"].split("_")[1])
        ext_idx = int(link["linked_address"].split("_")[1])
        if s_idx == ext_idx and s_idx < 50:
            true_positives += 1
        else:
            false_positives += 1

    fp_rate = false_positives / 50.0  # 50 actual negative pairs
    assert fp_rate < 0.05
    assert true_positives >= 45  # Should identify most true matches
