"""Tests for mechanism-aware bridge detection (Issue #880)."""

from __future__ import annotations

import pytest

from benchmarks.cross_chain import build_bridge_transactions, evaluate_bridge_mechanisms
from detection.cross_chain.bridge_detector import BridgeDetector
from detection.cross_chain.bridge_mechanisms import (
    LIQUIDITY_POOL,
    LOCK_AND_MINT,
    UNKNOWN_MECHANISM,
    BridgeMechanismHandler,
    analyze_bridge_transaction,
    classify_bridge_mechanism,
    get_bridge_mechanisms,
    register_bridge_mechanism,
    unregister_bridge_mechanism,
)

LOCK_MINT_TX = {
    "id": "lm1",
    "events": [
        {"type": "lock", "amount": 100.0, "chain": "stellar"},
        {"type": "mint", "amount": 100.0, "chain": "ethereum"},
    ],
}

POOL_TX = {
    "id": "lp1",
    "events": [
        {"type": "pool_deposit", "amount": 100.0, "chain": "stellar"},
        {"type": "pool_withdraw", "amount": 99.7, "chain": "solana"},
    ],
    "pool_balances": {
        "stellar": {"before": 1000.0, "after": 1100.0},
        "solana": {"before": 1000.0, "after": 900.3},
    },
}


def test_classifies_lock_and_mint():
    assert classify_bridge_mechanism(LOCK_MINT_TX)[0] == LOCK_AND_MINT


def test_classifies_liquidity_pool():
    assert classify_bridge_mechanism(POOL_TX)[0] == LIQUIDITY_POOL


def test_unrecognised_transaction_is_unknown():
    tx = {"events": [{"type": "transfer", "amount": 5.0, "chain": "stellar"}]}
    result = analyze_bridge_transaction(tx)
    assert result["mechanism"] == UNKNOWN_MECHANISM
    assert result["is_bridge"] is False


def test_lock_and_mint_features_and_detection():
    result = analyze_bridge_transaction(LOCK_MINT_TX)
    assert result["features"]["mint_burn_ratio"] == pytest.approx(1.0)
    assert result["is_bridge"] is True


def test_lock_and_mint_same_chain_is_not_bridge():
    tx = {
        "events": [
            {"type": "lock", "amount": 100.0, "chain": "stellar"},
            {"type": "mint", "amount": 100.0, "chain": "stellar"},
        ]
    }
    assert analyze_bridge_transaction(tx)["is_bridge"] is False


def test_lock_and_mint_amount_mismatch_is_not_bridge():
    tx = {
        "events": [
            {"type": "burn", "amount": 100.0, "chain": "stellar"},
            {"type": "mint", "amount": 99.5, "chain": "ethereum"},
        ]
    }
    assert analyze_bridge_transaction(tx)["is_bridge"] is False


def test_liquidity_pool_features_and_detection():
    result = analyze_bridge_transaction(POOL_TX)
    assert result["features"]["fee_ratio"] == pytest.approx(0.003)
    assert result["features"]["source_pool_delta"] == pytest.approx(100.0)
    assert result["features"]["destination_pool_delta"] == pytest.approx(99.7)
    assert result["is_bridge"] is True


def test_liquidity_pool_same_chain_swap_is_not_bridge():
    tx = {
        "events": [
            {"type": "pool_deposit", "amount": 100.0, "chain": "stellar"},
            {"type": "pool_withdraw", "amount": 99.7, "chain": "stellar"},
        ]
    }
    assert analyze_bridge_transaction(tx)["is_bridge"] is False


def test_liquidity_pool_inconsistent_balances_is_not_bridge():
    tx = {**POOL_TX, "pool_balances": {**POOL_TX["pool_balances"]}}
    tx["pool_balances"]["stellar"] = {"before": 1000.0, "after": 1150.0}
    assert analyze_bridge_transaction(tx)["is_bridge"] is False


def test_registered_handler_extensibility():
    @register_bridge_mechanism
    class MessagePassingHandler(BridgeMechanismHandler):
        name = "message_passing"

        def classification_score(self, tx):
            return 1.0 if tx.get("message_id") else 0.0

        def extract_features(self, tx):
            return {"has_message": 1.0}

        def detect(self, features):
            return features["has_message"]

    try:
        assert "message_passing" in get_bridge_mechanisms()
        result = analyze_bridge_transaction({"id": "m1", "message_id": "abc", "events": []})
        assert result["mechanism"] == "message_passing"
        assert result["is_bridge"] is True
    finally:
        unregister_bridge_mechanism("message_passing")
    assert "message_passing" not in get_bridge_mechanisms()


def test_bridge_detector_classify_transactions():
    results = BridgeDetector().classify_transactions([LOCK_MINT_TX, POOL_TX])
    assert [r["mechanism"] for r in results] == [LOCK_AND_MINT, LIQUIDITY_POOL]


def test_detect_bridge_links_attaches_mechanism():
    tx = {
        **LOCK_MINT_TX,
        "source_account": "GUSER",
        "memo_type": "text",
        "memo": "0x" + "a" * 40,
    }
    [link] = BridgeDetector().detect_bridge_links([tx])
    assert link["mechanism"] == LOCK_AND_MINT
    assert link["mechanism_confidence"] == pytest.approx(1.0)


def test_detect_bridge_links_without_events_unchanged():
    tx = {"source_account": "GUSER", "memo_type": "text", "memo": "0x" + "a" * 40}
    [link] = BridgeDetector().detect_bridge_links([tx])
    assert "mechanism" not in link


def test_labelled_benchmark_per_mechanism():
    report = evaluate_bridge_mechanisms(build_bridge_transactions())
    assert report["classification_accuracy"] >= 0.95
    for mechanism in (LOCK_AND_MINT, LIQUIDITY_POOL):
        specific = report["per_mechanism"][mechanism]["mechanism_specific"]
        unified = report["per_mechanism"][mechanism]["unified"]
        assert specific["precision"] >= unified["precision"]
        assert specific["recall"] >= 0.95
