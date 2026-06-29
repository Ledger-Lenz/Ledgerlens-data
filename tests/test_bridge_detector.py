"""Tests for bridge wash-trade detection (Issue #278)."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from detection.cross_chain.bridge_detector import (
    detect_bridge_wash_trade,
    load_bridge_anchors,
    _validate_stellar_address,
)

ANCHOR_A = "GCEZWKCA5VLDNRLN3RPRJMRZOX3Z6G5CHCGYWDEAVJJCSBVALM2XVKXB"
WALLET = "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"
NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _tx(frm, to, amount=100.0, hours_offset=0):
    return {
        "from": frm,
        "to": to,
        "amount": amount,
        "timestamp": (NOW + timedelta(hours=hours_offset)).isoformat(),
    }


def test_round_trip_within_window_produces_ratio_one():
    """Wallet sends 100 XLM to bridge, receives 100 XLM back within 24 h → ratio = 1.0."""
    txns = [
        _tx(WALLET, ANCHOR_A, 100.0, hours_offset=0),
        _tx(ANCHOR_A, WALLET, 100.0, hours_offset=20),  # 20 h later, within 72 h window
    ]
    result = detect_bridge_wash_trade(WALLET, txns, bridge_contracts=[ANCHOR_A])
    assert result["bridge_round_trip_ratio"] == 1.0
    assert result["bridge_round_trips"] == 1
    assert result["total_bridge_txns"] == 2


def test_outbound_only_produces_ratio_zero():
    """Wallet only sends to bridges (no inbound) → ratio = 0.0."""
    txns = [
        _tx(WALLET, ANCHOR_A, 100.0, hours_offset=0),
        _tx(WALLET, ANCHOR_A, 50.0, hours_offset=5),
    ]
    result = detect_bridge_wash_trade(WALLET, txns, bridge_contracts=[ANCHOR_A])
    assert result["bridge_round_trip_ratio"] == 0.0
    assert result["bridge_round_trips"] == 0


def test_no_bridge_txns_returns_zero():
    txns = [_tx("OTHER_WALLET", WALLET, 100.0)]
    result = detect_bridge_wash_trade(WALLET, txns, bridge_contracts=[ANCHOR_A])
    assert result["bridge_round_trip_ratio"] == 0.0
    assert result["total_bridge_txns"] == 0


def test_round_trip_outside_window_not_matched():
    """Return after 73 h (> default 72 h window) must not match."""
    txns = [
        _tx(WALLET, ANCHOR_A, 100.0, hours_offset=0),
        _tx(ANCHOR_A, WALLET, 100.0, hours_offset=73),
    ]
    result = detect_bridge_wash_trade(WALLET, txns, bridge_contracts=[ANCHOR_A], window_hours=72)
    assert result["bridge_round_trip_ratio"] == 0.0


def test_validate_stellar_address():
    assert _validate_stellar_address(ANCHOR_A)
    assert _validate_stellar_address(WALLET)
    assert not _validate_stellar_address("INVALID")
    assert not _validate_stellar_address("G" + "A" * 54)  # 55 chars (too short)
    assert not _validate_stellar_address("G" + "A" * 56)  # 57 chars (too long)


def test_load_bridge_anchors_validates_addresses():
    good = {"anchors": [{"address": ANCHOR_A, "name": "Test"}]}
    bad = {"anchors": [{"address": "NOTAVALIDADDRESS", "name": "Bad"}]}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(good, f)
        good_path = f.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        bad_path = f.name

    try:
        anchors = load_bridge_anchors(good_path)
        assert ANCHOR_A in anchors

        with pytest.raises(ValueError, match="Invalid Stellar G-address"):
            load_bridge_anchors(bad_path)
    finally:
        os.unlink(good_path)
        os.unlink(bad_path)


def test_invalid_bridge_contract_raises():
    with pytest.raises(ValueError, match="Invalid bridge anchor address"):
        detect_bridge_wash_trade(WALLET, [], bridge_contracts=["NOT_VALID"])
