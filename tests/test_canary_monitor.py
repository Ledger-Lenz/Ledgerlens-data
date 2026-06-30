"""Tests for ModelCanaryMonitor (issue #240)."""

import hashlib

import pytest

from detection.model_inference import ModelCanaryMonitor


def _hash(wallet: str) -> str:
    return hashlib.sha256(wallet.encode()).hexdigest()


def make_monitor() -> ModelCanaryMonitor:
    return ModelCanaryMonitor(champion_version="v1.0", canary_version="v1.1-rc1")


def test_log_score_pair_stores_entry():
    monitor = make_monitor()
    monitor.log_score_pair(_hash("G1"), champion_score=50.0, canary_score=55.0)
    pairs = monitor.score_pairs()
    assert len(pairs) == 1
    assert pairs[0]["delta"] == pytest.approx(5.0)


def test_histogram_distribution_correct():
    """Log 100 pairs with known deltas; confirm distribution is accurate."""
    monitor = make_monitor()
    for i in range(100):
        monitor.log_score_pair(_hash(f"G{i}"), champion_score=float(i), canary_score=float(i) + 1.0)

    pairs = monitor.score_pairs()
    assert len(pairs) == 100
    deltas = [p["delta"] for p in pairs]
    assert all(d == pytest.approx(1.0) for d in deltas)


def test_promotion_readiness_passes_below_threshold():
    monitor = make_monitor()
    for i in range(100):
        # delta = 5 for all — well below p95 limit of 15
        monitor.log_score_pair(_hash(f"G{i}"), champion_score=50.0, canary_score=55.0)

    report = monitor.promotion_readiness()
    assert report["ready"] is True
    assert report["p95_delta"] == pytest.approx(5.0)
    assert report["pair_count"] == 100


def test_promotion_readiness_fails_above_threshold():
    """p95 delta > 15 must set ready=False."""
    monitor = make_monitor()
    for i in range(100):
        # 95 % of pairs have delta 20, exceeding the limit
        delta = 20.0 if i < 96 else 1.0
        monitor.log_score_pair(_hash(f"G{i}"), champion_score=50.0, canary_score=50.0 + delta)

    report = monitor.promotion_readiness()
    assert report["ready"] is False
    assert report["p95_delta"] > 15.0
    assert "p95 score delta" in (report["reason"] or "")


def test_promotion_readiness_no_pairs():
    monitor = make_monitor()
    report = monitor.promotion_readiness()
    assert report["ready"] is False
    assert report["pair_count"] == 0


def test_top_divergent_wallets_returns_n():
    monitor = make_monitor()
    for i in range(50):
        monitor.log_score_pair(_hash(f"G{i}"), champion_score=0.0, canary_score=float(i))

    top = monitor.top_divergent_wallets(n=10)
    assert len(top) == 10
    # Highest delta should be first
    assert top[0]["delta"] >= top[-1]["delta"]


def test_disagreement_rate():
    monitor = make_monitor()
    # 30 pairs with delta > 15, 70 with delta <= 15
    for i in range(30):
        monitor.log_score_pair(_hash(f"Hi{i}"), champion_score=0.0, canary_score=20.0)
    for i in range(70):
        monitor.log_score_pair(_hash(f"Lo{i}"), champion_score=50.0, canary_score=55.0)

    rate = monitor.disagreement_rate(band_threshold=15.0)
    assert rate == pytest.approx(0.30)


def test_wallet_addresses_not_stored():
    """Score pairs must not contain raw wallet addresses."""
    monitor = make_monitor()
    raw_wallet = "GBCFXNZQN2P7YBZFPKG4TMZQNHEFGQJZRSVSXFSEXAMPLEWALLET12345"
    wallet_hash = _hash(raw_wallet)
    monitor.log_score_pair(wallet_hash, champion_score=40.0, canary_score=45.0)

    for pair in monitor.score_pairs():
        assert raw_wallet not in str(pair.values())
