"""Tests for GovernanceAuditListener and AuditTrail (#950).

Test plan
---------
1. ``test_governance_event_recorded`` — events are captured via ``on_governance_event``.
2. ``test_justification_correlation`` — end-to-end correlation of a
   ``parameter_change`` event.
3. ``test_missing_justification_alert_fires_after_grace_period`` — alert fires
   only after the grace period has elapsed.
4. ``test_no_alert_within_grace_period`` — no premature alert during the window.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from detection.governance_audit_trail import AuditEntry, AuditTrail
from integrations.governance_audit_listener import GovernanceAuditListener


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_listener(
    alert_fn: MagicMock | None = None,
    grace_period_seconds: int = 300,
) -> tuple[GovernanceAuditListener, AuditTrail, MagicMock]:
    if alert_fn is None:
        alert_fn = MagicMock()
    store = AuditTrail()
    listener = GovernanceAuditListener(
        audit_store=store,
        alert_fn=alert_fn,
        grace_period_seconds=grace_period_seconds,
    )
    return listener, store, alert_fn


def _governance_event(
    event_id: str = "gov-001",
    event_type: str = "parameter_change",
    event_data: dict | None = None,
    timestamp: float | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "event_data": event_data or {"parameter": "RISK_SCORE_FLAG_THRESHOLD", "new_value": 75},
        "timestamp": timestamp or time.time(),
    }


# ---------------------------------------------------------------------------
# 1. Governance event recording
# ---------------------------------------------------------------------------


class TestGovernanceEventRecorded:
    def test_event_stored_in_audit_trail(self):
        listener, store, _ = make_listener()
        listener.on_governance_event(_governance_event("gov-001"))
        entry = store.get_entry("gov-001")
        assert entry is not None
        assert isinstance(entry, AuditEntry)

    def test_event_type_preserved(self):
        listener, store, _ = make_listener()
        listener.on_governance_event(_governance_event("gov-002", event_type="threshold_changed"))
        entry = store.get_entry("gov-002")
        assert entry.event_type == "threshold_changed"

    def test_event_data_preserved(self):
        data = {"parameter": "RISK_SCORE_FLAG_THRESHOLD", "old_value": 70, "new_value": 75}
        listener, store, _ = make_listener()
        listener.on_governance_event(_governance_event("gov-003", event_data=data))
        entry = store.get_entry("gov-003")
        assert entry.event_data == data

    def test_timestamp_preserved(self):
        ts = time.time() - 100.0
        listener, store, _ = make_listener()
        listener.on_governance_event(_governance_event("gov-004", timestamp=ts))
        entry = store.get_entry("gov-004")
        assert abs(entry.timestamp - ts) < 1.0

    def test_duplicate_event_idempotent(self):
        listener, store, _ = make_listener()
        listener.on_governance_event(_governance_event("gov-005"))
        listener.on_governance_event(_governance_event("gov-005"))
        assert len(store.list_entries()) == 1

    def test_event_without_id_ignored(self):
        listener, store, _ = make_listener()
        listener.on_governance_event({"event_type": "parameter_change"})
        assert len(store.list_entries()) == 0

    def test_justification_initially_none(self):
        listener, store, _ = make_listener()
        listener.on_governance_event(_governance_event("gov-006"))
        entry = store.get_entry("gov-006")
        assert entry.justification is None
        assert not entry.is_correlated()


# ---------------------------------------------------------------------------
# 2. Justification correlation
# ---------------------------------------------------------------------------


class TestJustificationCorrelation:
    """End-to-end test: governance event recorded, justification submitted,
    entry shows correlated state."""

    def test_submit_justification_links_to_event(self):
        listener, store, _ = make_listener()
        listener.on_governance_event(_governance_event("gov-010"))
        listener.submit_justification(
            event_id="gov-010",
            justification="Threshold raised to 75 based on Q3 precision/recall analysis.",
            submitter="alice@example.com",
        )
        entry = store.get_entry("gov-010")
        assert entry.justification == "Threshold raised to 75 based on Q3 precision/recall analysis."
        assert entry.submitter == "alice@example.com"
        assert entry.is_correlated()

    def test_correlated_at_set(self):
        listener, store, _ = make_listener()
        before = time.time()
        listener.on_governance_event(_governance_event("gov-011"))
        listener.submit_justification("gov-011", "Approved by governance vote.", "bob@example.com")
        entry = store.get_entry("gov-011")
        assert entry.correlated_at is not None
        assert entry.correlated_at >= before

    def test_parameter_change_end_to_end(self):
        """Full end-to-end: parameter_change event correlated with a justification."""
        listener, store, _ = make_listener()
        event = {
            "event_id": "gov-012",
            "event_type": "parameter_change",
            "event_data": {"parameter": "RISK_SCORE_FLAG_THRESHOLD", "new_value": 80},
            "timestamp": time.time(),
        }
        listener.on_governance_event(event)
        listener.submit_justification(
            "gov-012",
            justification="Risk appetite review recommends 80.",
            submitter="carol@example.com",
        )
        entry = store.get_entry("gov-012")
        assert entry.event_type == "parameter_change"
        assert entry.event_data["new_value"] == 80
        assert entry.is_correlated()
        assert entry.submitter == "carol@example.com"

    def test_no_alert_after_justification_submitted(self):
        """After correlation, run_once should not fire an alert for this event."""
        listener, store, alert_fn = make_listener(grace_period_seconds=0)
        listener.on_governance_event(_governance_event("gov-013"))
        listener.submit_justification("gov-013", "Justified.", "dave@example.com")
        listener.run_once()
        alert_fn.assert_not_called()

    def test_submit_justification_for_unknown_event_raises(self):
        listener, _, _ = make_listener()
        with pytest.raises(KeyError):
            listener.submit_justification("nonexistent", "Justification.", "eve@example.com")


# ---------------------------------------------------------------------------
# 3. Missing justification alert fires after grace period
# ---------------------------------------------------------------------------


class TestMissingJustificationAlertFiresAfterGracePeriod:
    def test_alert_fires_for_overdue_event(self):
        """An event older than the grace period with no justification triggers alert."""
        listener, store, alert_fn = make_listener(grace_period_seconds=60)
        old_ts = time.time() - 120  # 120s ago > 60s grace period
        listener.on_governance_event(_governance_event("gov-020", timestamp=old_ts))
        listener.run_once()
        alert_fn.assert_called_once()

    def test_alert_details_contain_event_id(self):
        listener, store, alert_fn = make_listener(grace_period_seconds=0)
        listener.on_governance_event(_governance_event("gov-021"))
        # Grace period = 0 → already overdue immediately
        time.sleep(0.01)
        listener.run_once()
        details = alert_fn.call_args[0][0]
        assert details["event_id"] == "gov-021"

    def test_alert_details_contain_severity(self):
        listener, store, alert_fn = make_listener(grace_period_seconds=0)
        listener.on_governance_event(_governance_event("gov-022"))
        time.sleep(0.01)
        listener.run_once()
        details = alert_fn.call_args[0][0]
        assert details["severity"] in ("HIGH", "CRITICAL")

    def test_alert_details_contain_grace_period(self):
        listener, store, alert_fn = make_listener(grace_period_seconds=300)
        old_ts = time.time() - 400
        listener.on_governance_event(_governance_event("gov-023", timestamp=old_ts))
        listener.run_once()
        details = alert_fn.call_args[0][0]
        assert details["grace_period_seconds"] == 300

    def test_alert_details_contain_message(self):
        listener, store, alert_fn = make_listener(grace_period_seconds=0)
        listener.on_governance_event(_governance_event("gov-024"))
        time.sleep(0.01)
        listener.run_once()
        details = alert_fn.call_args[0][0]
        assert "message" in details and len(details["message"]) > 0

    def test_multiple_overdue_events_multiple_alerts(self):
        listener, store, alert_fn = make_listener(grace_period_seconds=0)
        old_ts = time.time() - 10
        for i in range(3):
            listener.on_governance_event(_governance_event(f"gov-030-{i}", timestamp=old_ts))
        time.sleep(0.01)
        listener.run_once()
        assert alert_fn.call_count == 3

    def test_check_missing_justifications_returns_alerts(self):
        listener, store, alert_fn = make_listener(grace_period_seconds=0)
        listener.on_governance_event(_governance_event("gov-025"))
        time.sleep(0.01)
        alerts = listener.check_missing_justifications()
        assert len(alerts) == 1
        assert alerts[0]["event_id"] == "gov-025"


# ---------------------------------------------------------------------------
# 4. No alert within grace period
# ---------------------------------------------------------------------------


class TestNoAlertWithinGracePeriod:
    def test_no_alert_for_recent_event(self):
        """A fresh event within the grace period must NOT trigger an alert."""
        listener, store, alert_fn = make_listener(grace_period_seconds=300)
        # Event just happened (now)
        listener.on_governance_event(_governance_event("gov-040", timestamp=time.time()))
        listener.run_once()
        alert_fn.assert_not_called()

    def test_no_alert_just_before_grace_period_expires(self):
        """Event just under the grace period threshold — no alert yet."""
        listener, store, alert_fn = make_listener(grace_period_seconds=300)
        # 299s ago — still within grace period
        ts = time.time() - 299
        listener.on_governance_event(_governance_event("gov-041", timestamp=ts))
        listener.run_once()
        alert_fn.assert_not_called()

    def test_alert_fires_just_after_grace_period_expires(self):
        """Event just past the grace period threshold — alert fires."""
        listener, store, alert_fn = make_listener(grace_period_seconds=300)
        # 301s ago — past grace period
        ts = time.time() - 301
        listener.on_governance_event(_governance_event("gov-042", timestamp=ts))
        listener.run_once()
        alert_fn.assert_called_once()

    def test_run_once_alias_matches_check_missing(self):
        """run_once() is a thin alias for check_missing_justifications()."""
        listener, store, alert_fn = make_listener(grace_period_seconds=0)
        listener.on_governance_event(_governance_event("gov-043"))
        time.sleep(0.01)
        alerts_via_run_once = listener.run_once()
        assert len(alerts_via_run_once) == 1


# ---------------------------------------------------------------------------
# 5. AuditTrail unit tests
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_record_and_get_entry(self):
        store = AuditTrail()
        store.record_governance_event("ev-1", "param_change", {"v": 10}, time.time())
        entry = store.get_entry("ev-1")
        assert entry is not None
        assert entry.event_id == "ev-1"

    def test_get_entry_unknown_returns_none(self):
        store = AuditTrail()
        assert store.get_entry("nonexistent") is None

    def test_add_justification(self):
        store = AuditTrail()
        store.record_governance_event("ev-2", "param_change", {}, time.time())
        store.add_justification("ev-2", "Because reasons.", "alice")
        entry = store.get_entry("ev-2")
        assert entry.is_correlated()
        assert entry.justification == "Because reasons."

    def test_add_justification_unknown_raises(self):
        store = AuditTrail()
        with pytest.raises(KeyError):
            store.add_justification("no-such", "j", "alice")

    def test_get_uncorrelated_events_empty_when_all_correlated(self):
        store = AuditTrail()
        store.record_governance_event("ev-3", "param_change", {}, time.time() - 1000)
        store.add_justification("ev-3", "j", "alice")
        assert store.get_uncorrelated_events(older_than_seconds=0) == []

    def test_get_uncorrelated_events_returns_overdue(self):
        store = AuditTrail()
        store.record_governance_event("ev-4", "param_change", {}, time.time() - 100)
        result = store.get_uncorrelated_events(older_than_seconds=50)
        assert len(result) == 1
        assert result[0]["event_id"] == "ev-4"

    def test_list_entries_ordered_newest_first(self):
        store = AuditTrail()
        store.record_governance_event("ev-5", "t1", {}, 1000.0)
        store.record_governance_event("ev-6", "t2", {}, 2000.0)
        entries = store.list_entries()
        assert entries[0].event_id == "ev-6"
        assert entries[1].event_id == "ev-5"

    def test_list_entries_respects_limit(self):
        store = AuditTrail()
        for i in range(10):
            store.record_governance_event(f"ev-{i}", "t", {}, float(i))
        assert len(store.list_entries(limit=3)) == 3

    def test_as_dict(self):
        entry = AuditEntry(event_id="x", event_type="y", event_data={"k": "v"}, timestamp=1.0)
        d = entry.as_dict()
        assert d["event_id"] == "x"
        assert d["event_type"] == "y"
        assert d["justification"] is None
