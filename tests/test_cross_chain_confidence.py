"""Tests for confidence-scored, evidence-linked cross-chain identity edges (Issue #879)."""

from __future__ import annotations

import pytest

from benchmarks.cross_chain import build_identity_pairs, evaluate_identity_confidence
from config import config
from detection.cross_chain.confidence import Evidence, combine_confidence, precision_recall
from detection.cross_chain.identity_graph import IdentityGraph
from detection.cross_chain.resolver import get_link_evidence, resolve, resolve_risk_scores
from detection.persistence import Base, get_engine, get_session_factory


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'identity_confidence.db'}"


@pytest.fixture
def graph(db_url):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return IdentityGraph(get_session_factory(engine))


def test_single_evidence_is_strength_times_reliability():
    assert combine_confidence([Evidence("bridge", 1.0)]) == pytest.approx(1.0)
    assert combine_confidence([Evidence("timing_correlation", 0.9)]) == pytest.approx(0.63)
    assert combine_confidence([Evidence("amount_fingerprint", 1.0)]) == pytest.approx(0.55)


def test_corroborating_evidence_raises_confidence():
    amount = Evidence("amount_fingerprint", 1.0)
    timing = Evidence("timing_correlation", 0.9)
    combined = combine_confidence([amount, timing])
    assert combined > max(amount.probability, timing.probability)
    assert combined == pytest.approx(1 - 0.45 * 0.37)


def test_no_evidence_is_zero_confidence():
    assert combine_confidence([]) == 0.0


def test_strength_is_clamped():
    assert Evidence("bridge", 1.7).probability == 1.0
    assert Evidence("bridge", -0.3).probability == 0.0


def test_edge_evidence_retrievable_for_forensics(graph):
    graph.add_edge("GUSER", "0xABC", "bridge", confidence=1.0, metadata={"tx_id": "tx1"})
    graph.add_edge("0xabc", "GUSER", "timing_correlation", confidence=0.9, metadata={"r": 0.9})

    evidence = graph.get_edge_evidence("GUSER", "0xabc")
    assert {e.evidence_type for e in evidence} == {"bridge", "timing_correlation"}
    bridge = next(e for e in evidence if e.evidence_type == "bridge")
    assert bridge.metadata["tx_id"] == "tx1"
    assert graph.get_link_confidence("GUSER", "0xabc") == pytest.approx(1.0)


def test_get_link_evidence_resolver(graph, db_url):
    graph.add_edge("GUSER", "0xabc", "amount_fingerprint", confidence=1.0, metadata={"t": 1})
    report = get_link_evidence("GUSER", "0xabc", db_url=db_url)
    assert report["confidence"] == pytest.approx(0.55)
    assert report["evidence"][0]["evidence_type"] == "amount_fingerprint"
    assert report["evidence"][0]["metadata"]["t"] == 1


def test_add_links_stores_detector_payload(graph):
    links = [
        {
            "stellar_address": "GUSER",
            "linked_address": "0xabc",
            "chain": "ethereum",
            "tx_id": "tx9",
            "memo": "0xabc",
            "confidence": 1.0,
        }
    ]
    assert graph.add_links(links, "bridge") == 1
    [evidence] = graph.get_edge_evidence("GUSER", "0xabc")
    assert evidence.metadata["tx_id"] == "tx9"
    assert evidence.metadata["chain"] == "ethereum"


def test_low_confidence_links_excluded_from_risk_propagation(graph, db_url):
    graph.add_node("0xstrong", "ethereum", risk_score=90.0)
    graph.add_node("0xweak", "ethereum", risk_score=80.0)
    graph.add_edge("GUSER", "0xstrong", "bridge", confidence=1.0)
    graph.add_edge("GUSER", "0xweak", "amount_fingerprint", confidence=1.0)

    scores = resolve_risk_scores("GUSER", db_url=db_url)
    assert scores == {"0xstrong": 90.0}

    # Weak link is still visible to investigators via resolve().
    assert set(resolve("GUSER", db_url=db_url)["eth"]) == {"0xstrong", "0xweak"}


def test_threshold_blocks_transitive_traversal(graph, db_url):
    mid = "0x" + "1" * 40
    sol = "SoLanaAddr1111111111111111111111111"
    graph.add_node(sol, "solana", risk_score=70.0)
    graph.add_node(mid, "ethereum", risk_score=10.0)
    graph.add_edge("GUSER", mid, "timing_correlation", confidence=0.8)
    graph.add_edge(mid, sol, "bridge", confidence=1.0)

    assert resolve_risk_scores("GUSER", db_url=db_url) == {}
    assert resolve_risk_scores("GUSER", db_url=db_url, min_confidence=0.5) == {
        mid: 10.0,
        sol: 70.0,
    }


def test_threshold_configurable(graph, db_url, monkeypatch):
    graph.add_node("0xweak", "ethereum", risk_score=80.0)
    graph.add_edge("GUSER", "0xweak", "amount_fingerprint", confidence=1.0)
    monkeypatch.setattr(config, "CROSS_CHAIN_MIN_CONFIDENCE", 0.5)
    assert resolve_risk_scores("GUSER", db_url=db_url) == {"0xweak": 80.0}


def test_component_nodes_carry_confidence(graph):
    graph.add_edge("GUSER", "0x" + "a" * 40, "bridge", confidence=1.0)
    component = graph.get_connected_component("GUSER")
    assert component["eth"][0]["confidence"] == pytest.approx(1.0)


def test_precision_recall_unordered_pairs():
    scored = {("a", "b"): 0.9, ("c", "d"): 0.7, ("e", "f"): 0.2}
    truth = {("b", "a"), ("e", "f")}
    result = precision_recall(scored, truth, 0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_labelled_benchmark_default_threshold_is_f1_optimal():
    results = evaluate_identity_confidence(build_identity_pairs())
    best = max(results, key=lambda r: r["f1"])
    assert best["threshold"] == pytest.approx(config.CROSS_CHAIN_MIN_CONFIDENCE)
    assert best["precision"] > 0.7
    assert best["recall"] > 0.6
