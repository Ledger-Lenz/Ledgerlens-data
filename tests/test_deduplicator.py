"""Tests for alerts.deduplicator.deduplicate (Issue #282)."""

from alerts.deduplicator import deduplicate

WALLET_A = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
WALLET_B = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBWHF"
PAIR = "XLM:USDC"


def _alert(wallet, detector, risk_score, detected_at, **evidence):
    return {
        "wallet_address": wallet,
        "asset_pair": PAIR,
        "detector": detector,
        "risk_score": risk_score,
        "detected_at": detected_at,
        "evidence": evidence,
    }


def test_same_wallet_within_window_produces_one_grouped_alert():
    stream = [
        _alert(WALLET_A, "benford", 60, 100.0, chi2=18.4),
        _alert(WALLET_A, "gnn", 85, 110.0, embedding_distance=0.92),
    ]

    grouped = list(deduplicate(stream, window_seconds=60))

    assert len(grouped) == 1
    alert = grouped[0]
    assert alert["wallet_address"] == WALLET_A
    assert alert["detectors"] == ["benford", "gnn"]
    assert alert["risk_score"] == 85
    assert alert["evidence"] == {"chi2": 18.4, "embedding_distance": 0.92}
    assert alert["detected_at"] == 100.0


def test_different_wallets_produce_two_separate_alerts():
    stream = [
        _alert(WALLET_A, "benford", 60, 100.0),
        _alert(WALLET_B, "isolation_forest", 70, 101.0),
    ]

    grouped = list(deduplicate(stream, window_seconds=60))

    assert len(grouped) == 2
    wallets = {alert["wallet_address"] for alert in grouped}
    assert wallets == {WALLET_A, WALLET_B}


def test_alert_after_window_expiry_starts_new_group():
    stream = [
        _alert(WALLET_A, "benford", 60, 0.0),
        _alert(WALLET_A, "gnn", 85, 200.0),  # well past the 60s silence window
    ]

    grouped = list(deduplicate(stream, window_seconds=60))

    assert len(grouped) == 2
    assert grouped[0]["detectors"] == ["benford"]
    assert grouped[1]["detectors"] == ["gnn"]


def test_out_of_order_delivery_within_window_still_merges():
    stream = [
        _alert(WALLET_A, "gnn", 85, 110.0),
        _alert(WALLET_A, "benford", 60, 100.0),  # arrives second but happened first
    ]

    grouped = list(deduplicate(stream, window_seconds=60))

    assert len(grouped) == 1
    assert grouped[0]["detected_at"] == 100.0
    assert grouped[0]["detectors"] == ["benford", "gnn"]


def test_no_alert_is_ever_suppressed():
    stream = [
        _alert(WALLET_A, "benford", 60, 0.0),
        _alert(WALLET_A, "gnn", 85, 10.0),
        _alert(WALLET_B, "isolation_forest", 40, 500.0),
    ]

    grouped = list(deduplicate(stream, window_seconds=60))
    total_raw_detectors = sum(len(alert["detectors"]) for alert in grouped)

    assert total_raw_detectors == len(stream)
