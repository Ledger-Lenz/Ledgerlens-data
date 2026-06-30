"""Tests for IncidentResponder (monitoring/incident_responder.py).

Covers:
- Idempotency: same alert fires twice → exactly one incident record
- simulate() produces an incident with the same structure as handle_alert()
- Low-severity alerts are not processed
- Webhook payload never contains raw wallet address
"""

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest

from monitoring.incident_responder import (
    IncidentResponder,
    _alert_fingerprint,
    _hash_wallet,
)

WALLET = "GABC1234567890EXAMPLEWALLETADDRESS000000000000000000000001"
HIGH_RISK_ALERT = {"risk_score": 95, "alert_type": "high_risk_wallet", "benford_mad": 0.0}
LOW_RISK_ALERT = {"risk_score": 50, "alert_type": "low_activity", "benford_mad": 0.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_responder(dedup_window: int = 3600) -> IncidentResponder:
    store: dict = {}
    return IncidentResponder(
        incident_store=store,
        dedup_window_seconds=dedup_window,
    )


# ---------------------------------------------------------------------------
# 1. Idempotency — same alert fired twice creates exactly one incident record
# ---------------------------------------------------------------------------


def test_idempotency_single_incident_on_duplicate_alert():
    responder = _make_responder()

    incident1 = responder.handle_alert(WALLET, HIGH_RISK_ALERT)
    incident2 = responder.handle_alert(WALLET, HIGH_RISK_ALERT)

    assert incident1 is not None, "First alert should create an incident"
    assert incident2 is None, "Second alert within dedup window should be suppressed"
    assert len(responder.incidents) == 1


def test_idempotency_no_duplicate_notification():
    responder = _make_responder()
    notifications: list[dict] = []

    original_post = responder._post_webhook

    def mock_post(incident, params):
        notifications.append(incident.to_dict())

    responder._post_webhook = mock_post  # type: ignore[method-assign]

    responder.handle_alert(WALLET, HIGH_RISK_ALERT)
    responder.handle_alert(WALLET, HIGH_RISK_ALERT)

    assert len(notifications) == 1, "Notification must be sent exactly once"


def test_after_dedup_window_expires_new_incident_is_created():
    responder = _make_responder(dedup_window=0)  # window of 0 s always allows re-trigger

    incident1 = responder.handle_alert(WALLET, HIGH_RISK_ALERT)
    time.sleep(0.01)
    incident2 = responder.handle_alert(WALLET, HIGH_RISK_ALERT)

    assert incident1 is not None
    assert incident2 is not None
    assert incident1.incident_id != incident2.incident_id


# ---------------------------------------------------------------------------
# 2. --simulate mode produces same report structure as a live alert
# ---------------------------------------------------------------------------


def test_simulate_produces_incident_with_same_structure_as_live():
    responder = _make_responder()

    # Live alert
    live_incident = responder.handle_alert(WALLET, HIGH_RISK_ALERT)

    # Simulate (with a different wallet to avoid dedup collision)
    wallet2 = WALLET[:-1] + "2"
    store2: dict = {}
    responder2 = IncidentResponder(incident_store=store2)
    sim_incident = responder2.simulate(wallet2)

    assert sim_incident is not None
    assert live_incident is not None

    # Both must share the same top-level structure
    live_keys = set(live_incident.to_dict().keys())
    sim_keys = set(sim_incident.to_dict().keys())
    assert live_keys == sim_keys, "Simulated incident must have same keys as live incident"


def test_simulate_without_wallet_still_works():
    responder = _make_responder()
    incident = responder.simulate(WALLET, risk_score=98)
    assert incident is not None
    assert incident.risk_score == 98


# ---------------------------------------------------------------------------
# 3. Low-severity alerts are ignored
# ---------------------------------------------------------------------------


def test_low_severity_alert_not_processed():
    responder = _make_responder()
    result = responder.handle_alert(WALLET, LOW_RISK_ALERT)
    assert result is None
    assert len(responder.incidents) == 0


def test_benford_mad_above_threshold_is_high_severity():
    responder = _make_responder()
    alert = {"risk_score": 50, "alert_type": "benford_spike", "benford_mad": 0.06}
    incident = responder.handle_alert(WALLET, alert)
    assert incident is not None


# ---------------------------------------------------------------------------
# 4. Webhook payload must not contain raw wallet address
# ---------------------------------------------------------------------------


def test_webhook_payload_contains_no_raw_wallet():
    posted_payloads: list[dict] = []

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        responder = IncidentResponder(
            webhook_url="https://hooks.example.com/test",
            incident_store={},
        )
        incident = responder.handle_alert(WALLET, HIGH_RISK_ALERT)

        assert mock_post.called
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]

        # Raw wallet address must not appear in any string value
        payload_str = str(payload)
        assert WALLET not in payload_str, "Raw wallet address must not appear in webhook payload"
        assert "wallet_hash" in payload


def test_webhook_url_http_raises():
    with pytest.raises(ValueError, match="HTTPS"):
        IncidentResponder(webhook_url="http://hooks.example.com/test")


# ---------------------------------------------------------------------------
# 5. Incident record fields
# ---------------------------------------------------------------------------


def test_incident_record_has_expected_fields():
    responder = _make_responder()
    incident = responder.handle_alert(WALLET, HIGH_RISK_ALERT)
    assert incident is not None
    d = incident.to_dict()
    for field in ("incident_id", "wallet_hash", "alert_fingerprint", "alert_type", "risk_score", "created_at", "status"):
        assert field in d, f"Missing field: {field}"
    assert d["wallet_hash"] == _hash_wallet(WALLET)
    assert WALLET not in str(d)


def test_hash_wallet_is_deterministic():
    h1 = _hash_wallet(WALLET)
    h2 = _hash_wallet(WALLET)
    assert h1 == h2
    assert len(h1) == 16  # first 16 hex chars


def test_alert_fingerprint_is_deterministic():
    fp1 = _alert_fingerprint(WALLET, "high_risk_wallet", 95)
    fp2 = _alert_fingerprint(WALLET, "high_risk_wallet", 95)
    assert fp1 == fp2
