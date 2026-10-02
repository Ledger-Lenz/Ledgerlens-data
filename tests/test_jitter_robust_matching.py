"""Regression tests for timing-jitter-robust cross-chain hop matching (Issue #882)."""

from __future__ import annotations

import pytest

from detection.cross_chain.behavioral_matcher import BehavioralMatcher
from scripts.cross_chain_evasion_simulator import compare, evaluate, run_naive, run_robust, simulate

FEE = 0.003


def _leg(tx_id, wallet, ts, amount, chain=None):
    rec = {"id": tx_id, "wallet": wallet, "timestamp": ts, "amount": amount}
    if chain:
        rec["chain"] = chain
    return rec


def test_non_jittered_transfer_matches():
    links = BehavioralMatcher.match_jitter_robust(
        [_leg("s1", "GA", 1_000.0, 1234.5678)],
        [_leg("e1", "0xa", 1_020.0, 1234.5678 * (1 - FEE), "ethereum")],
        fee_models=[(0.0, FEE)],
    )
    assert len(links) == 1
    assert links[0]["linked_address"] == "0xa"
    assert links[0]["metadata"]["type"] == "jitter_robust_fingerprint"


def test_jittered_transfer_matches_where_fixed_window_fails():
    s = [_leg("s1", "GA", 1_000.0, 1234.5678)]
    e = [_leg("e1", "0xa", 1_000.0 + 3 * 3600, 1234.5678 * (1 - FEE), "ethereum")]
    assert BehavioralMatcher.match_amount_fingerprints(s, e, tolerance=FEE + 0.0005) == []
    links = BehavioralMatcher.match_jitter_robust(s, e, fee_models=[(0.0, FEE)])
    assert [lk["metadata"]["external_tx_id"] for lk in links] == ["e1"]


def test_beyond_max_delay_or_before_source_not_matched():
    s = [_leg("s1", "GA", 10_000.0, 1234.5678)]
    late = [_leg("e1", "0xa", 10_000.0 + 7 * 3600, 1234.5678, "ethereum")]
    early = [_leg("e2", "0xb", 10_000.0 - 3600, 1234.5678, "ethereum")]
    assert BehavioralMatcher.match_jitter_robust(s, late) == []
    assert BehavioralMatcher.match_jitter_robust(s, early) == []


def test_fee_mismatch_not_matched():
    links = BehavioralMatcher.match_jitter_robust(
        [_leg("s1", "GA", 0.0, 1000.37)],
        [_leg("e1", "0xa", 30.0, 1000.37 * 0.98, "ethereum")],
        fee_models=[(0.0, FEE)],
    )
    assert links == []


def test_assignment_is_one_to_one_and_prefers_closer_delay():
    s = [_leg("s1", "GA", 0.0, 777.123), _leg("s2", "GB", 50.0, 777.123)]
    e = [_leg("e1", "0xa", 30.0, 777.123, "ethereum"), _leg("e2", "0xb", 80.0, 777.123, "ethereum")]
    links = BehavioralMatcher.match_jitter_robust(s, e, min_confidence=0.0)
    pairs = {(lk["metadata"]["stellar_tx_id"], lk["metadata"]["external_tx_id"]) for lk in links}
    assert len({p[0] for p in pairs}) == len(pairs) == len({p[1] for p in pairs})
    assert ("s1", "e1") in pairs


def test_popular_round_amount_is_down_weighted():
    # A lone 1000.0 match with a jittered delay is ambiguous when 1000.0 is a
    # popular amount across the external set; a distinctive amount is not.
    background = [
        _leg(f"b{i}", f"0xbg{i}", 100_000.0 + i * 5_000, 1000.0, "ethereum") for i in range(40)
    ]
    popular = BehavioralMatcher.match_jitter_robust(
        [_leg("s1", "GA", 0.0, 1000.0)],
        [_leg("e1", "0xa", 2 * 3600, 1000.0, "ethereum"), *background],
    )
    distinctive = BehavioralMatcher.match_jitter_robust(
        [_leg("s1", "GA", 0.0, 1043.3171)],
        [_leg("e1", "0xa", 2 * 3600, 1043.3171, "ethereum"), *background],
    )
    assert popular == []
    assert len(distinctive) == 1


def test_invalid_match_prior():
    with pytest.raises(ValueError):
        BehavioralMatcher.match_jitter_robust([], [], match_prior=1.0)


@pytest.mark.parametrize("jitter", [0.0, 3600.0])
def test_simulated_scenario_recall_and_precision(jitter):
    sc = simulate(jitter_seconds=jitter, fee_rate=FEE, seed=7)
    robust = evaluate(run_robust(sc, FEE), sc.truth)
    assert robust["recall"] >= 0.8
    assert robust["precision"] >= 0.95


def test_recall_margin_over_naive_without_precision_collapse():
    """Locks in the documented improvement (docs/cross_chain_jitter_matching.md)."""
    rows = compare(jitters=(0.0, 1800.0, 14400.0), seeds=2, fee_rate=FEE)
    by_jitter = {r["jitter_seconds"]: r for r in rows}
    # Non-jittered: no regression versus the fixed-window matcher.
    assert by_jitter[0.0]["robust_recall"] >= by_jitter[0.0]["naive_recall"]
    assert by_jitter[0.0]["robust_precision"] >= 0.97
    for jitter in (1800.0, 14400.0):
        r = by_jitter[jitter]
        assert r["robust_recall"] - r["naive_recall"] >= 0.6
        assert r["robust_precision"] >= 0.95


def test_naive_matcher_behaviour_unchanged():
    sc = simulate(jitter_seconds=0.0, fee_rate=FEE, seed=3)
    assert evaluate(run_naive(sc, FEE), sc.truth)["recall"] >= 0.9
