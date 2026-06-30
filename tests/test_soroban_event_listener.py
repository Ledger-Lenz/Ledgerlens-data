"""Unit tests for integrations/soroban_event_listener.py."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from integrations.soroban_event_listener import (
    ContractEvent,
    ContractEventRecord,
    EventWatermark,
    SorobanEventListener,
    _hash_address,
    check_stale_score_alert,
    get_watermark,
    parse_contract_event,
    persist_event,
    set_watermark,
    _get_session_factory,
    EVENT_HMAC_SECRET,
)

# ---------------------------------------------------------------------------
# Load synthetic fixtures
# ---------------------------------------------------------------------------

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "soroban_events.json"

with open(FIXTURES_PATH) as _f:
    FIXTURES = json.load(_f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path):
    """In-memory SQLite session factory."""
    return _get_session_factory(f"sqlite:///{tmp_path}/test.db")


def _expected_hash(address: str) -> str:
    return hmac.new(
        EVENT_HMAC_SECRET.encode(), address.encode(), hashlib.sha256
    ).hexdigest()


# ---------------------------------------------------------------------------
# 1. Parsing: score_read
# ---------------------------------------------------------------------------

class TestParseScoreRead:
    def test_returns_contract_event(self):
        event = parse_contract_event(FIXTURES["score_read"])
        assert isinstance(event, ContractEvent)

    def test_event_type(self):
        event = parse_contract_event(FIXTURES["score_read"])
        assert event.event_type == "score_read"

    def test_ledger_sequence(self):
        event = parse_contract_event(FIXTURES["score_read"])
        assert event.ledger_sequence == 54321

    def test_timestamp_is_utc(self):
        event = parse_contract_event(FIXTURES["score_read"])
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.year == 2026

    def test_score_parsed(self):
        event = parse_contract_event(FIXTURES["score_read"])
        assert event.score == 75

    def test_asset_pair(self):
        event = parse_contract_event(FIXTURES["score_read"])
        assert "USDC" in (event.asset_pair or "")

    def test_wallet_id_is_hashed(self):
        event = parse_contract_event(FIXTURES["score_read"])
        raw_wallet = "GBTEST12345678901234567890123456789012345678901234567890"
        assert event.wallet_id_hash == _expected_hash(raw_wallet)
        # Must not contain raw address
        assert raw_wallet not in event.wallet_id_hash

    def test_consumer_address_is_hashed(self):
        event = parse_contract_event(FIXTURES["score_read"])
        raw_consumer = "GCONSUMER1234567890123456789012345678901234567890123456"
        assert event.consumer_address_hash == _expected_hash(raw_consumer)
        assert raw_consumer not in (event.consumer_address_hash or "")

    def test_raw_wallet_not_in_event_fields(self):
        """Raw G-addresses must not leak into any string field of the event."""
        event = parse_contract_event(FIXTURES["score_read"])
        raw = "GBTEST12345678901234567890123456789012345678901234567890"
        for field_val in [
            event.wallet_id_hash,
            event.consumer_address_hash,
            event.asset_pair,
            event.event_type,
        ]:
            assert raw not in str(field_val or "")


# ---------------------------------------------------------------------------
# 2. Parsing: score_updated
# ---------------------------------------------------------------------------

class TestParseScoreUpdated:
    def test_event_type(self):
        event = parse_contract_event(FIXTURES["score_updated"])
        assert event.event_type == "score_updated"

    def test_score(self):
        event = parse_contract_event(FIXTURES["score_updated"])
        assert event.score == 82

    def test_no_consumer_address(self):
        event = parse_contract_event(FIXTURES["score_updated"])
        assert event.consumer_address_hash is None

    def test_ledger(self):
        event = parse_contract_event(FIXTURES["score_updated"])
        assert event.ledger_sequence == 54322


# ---------------------------------------------------------------------------
# 3. Parsing: threshold_updated
# ---------------------------------------------------------------------------

class TestParseThresholdUpdated:
    def test_event_type(self):
        event = parse_contract_event(FIXTURES["threshold_updated"])
        assert event.event_type == "threshold_updated"

    def test_thresholds(self):
        event = parse_contract_event(FIXTURES["threshold_updated"])
        assert event.old_threshold == 70
        assert event.new_threshold == 75

    def test_score_is_none(self):
        event = parse_contract_event(FIXTURES["threshold_updated"])
        assert event.score is None


# ---------------------------------------------------------------------------
# 4. Unknown event returns None
# ---------------------------------------------------------------------------

class TestParseUnknownEvent:
    def test_returns_none(self):
        result = parse_contract_event(FIXTURES["unknown_event"])
        assert result is None

    def test_empty_dict_returns_none(self):
        assert parse_contract_event({}) is None

    def test_no_topics_returns_none(self):
        assert parse_contract_event({"ledger": "1", "value": {}}) is None


# ---------------------------------------------------------------------------
# 5. Horizon effects shape (fallback backend)
# ---------------------------------------------------------------------------

class TestParseHorizonShape:
    def test_horizon_score_read_parsed(self):
        event = parse_contract_event(FIXTURES["horizon_score_read"])
        assert event is not None
        assert event.event_type == "score_read"
        assert event.score == 60

    def test_horizon_consumer_hashed(self):
        event = parse_contract_event(FIXTURES["horizon_score_read"])
        raw = "GCONSUMER1234567890123456789012345678901234567890123456"
        assert event.consumer_address_hash == _expected_hash(raw)


# ---------------------------------------------------------------------------
# 6. Privacy: _hash_address
# ---------------------------------------------------------------------------

class TestHashAddress:
    def test_deterministic(self):
        assert _hash_address("GTEST") == _hash_address("GTEST")

    def test_different_inputs_different_hashes(self):
        assert _hash_address("GWALLET1") != _hash_address("GWALLET2")

    def test_output_is_hex(self):
        h = _hash_address("GTEST")
        int(h, 16)  # raises ValueError if not valid hex

    def test_length_64(self):
        assert len(_hash_address("GTEST")) == 64


# ---------------------------------------------------------------------------
# 7. Persistence
# ---------------------------------------------------------------------------

class TestPersistEvent:
    def test_persist_stores_record(self, db):
        event = parse_contract_event(FIXTURES["score_read"])
        persist_event(event, db)
        with db() as session:
            records = session.query(ContractEventRecord).all()
        assert len(records) == 1
        assert records[0].event_type == "score_read"
        assert records[0].score == 75

    def test_raw_address_not_in_db(self, db):
        event = parse_contract_event(FIXTURES["score_read"])
        persist_event(event, db)
        with db() as session:
            rec = session.query(ContractEventRecord).first()
        raw = "GBTEST12345678901234567890123456789012345678901234567890"
        assert raw not in (rec.wallet_id_hash or "")
        assert raw not in (rec.consumer_address_hash or "")

    def test_persist_multiple_events(self, db):
        for fixture_key in ("score_read", "score_updated", "threshold_updated"):
            event = parse_contract_event(FIXTURES[fixture_key])
            persist_event(event, db)
        with db() as session:
            count = session.query(ContractEventRecord).count()
        assert count == 3


# ---------------------------------------------------------------------------
# 8. Watermark
# ---------------------------------------------------------------------------

class TestWatermark:
    def test_default_watermark_is_zero(self, db):
        assert get_watermark("CTEST", db) == 0

    def test_set_and_get(self, db):
        set_watermark("CTEST", 12345, db)
        assert get_watermark("CTEST", db) == 12345

    def test_update_watermark(self, db):
        set_watermark("CTEST", 100, db)
        set_watermark("CTEST", 200, db)
        assert get_watermark("CTEST", db) == 200

    def test_independent_per_contract(self, db):
        set_watermark("CONTRACT_A", 50, db)
        set_watermark("CONTRACT_B", 99, db)
        assert get_watermark("CONTRACT_A", db) == 50
        assert get_watermark("CONTRACT_B", db) == 99


# ---------------------------------------------------------------------------
# 9. Stale score alert
# ---------------------------------------------------------------------------

class TestStaleScoreAlert:
    def _make_score_read_event(self, score: int) -> ContractEvent:
        raw = dict(FIXTURES["score_read"])
        # Override score in value map
        raw["value"]["value"][0]["val"]["value"] = score
        return parse_contract_event(raw)

    def test_alert_fires_when_delta_exceeds_threshold(self):
        event = self._make_score_read_event(50)
        dispatcher = MagicMock()
        fired = check_stale_score_alert(event, current_score=80, dispatcher=dispatcher, threshold=20)
        assert fired is True
        dispatcher.dispatch.assert_called_once()

    def test_alert_does_not_fire_below_threshold(self):
        event = self._make_score_read_event(65)
        dispatcher = MagicMock()
        fired = check_stale_score_alert(event, current_score=80, dispatcher=dispatcher, threshold=20)
        assert fired is False
        dispatcher.dispatch.assert_not_called()

    def test_alert_does_not_fire_at_exact_threshold(self):
        event = self._make_score_read_event(60)
        dispatcher = MagicMock()
        fired = check_stale_score_alert(event, current_score=80, dispatcher=dispatcher, threshold=20)
        assert fired is False

    def test_alert_fires_above_threshold(self):
        event = self._make_score_read_event(30)
        dispatcher = MagicMock()
        fired = check_stale_score_alert(event, current_score=80, dispatcher=dispatcher, threshold=20)
        assert fired is True

    def test_no_alert_for_non_score_read_event(self):
        event = parse_contract_event(FIXTURES["score_updated"])
        dispatcher = MagicMock()
        fired = check_stale_score_alert(event, current_score=80, dispatcher=dispatcher, threshold=20)
        assert fired is False
        dispatcher.dispatch.assert_not_called()

    def test_alert_payload_contains_delta(self):
        event = self._make_score_read_event(40)
        dispatcher = MagicMock()
        check_stale_score_alert(event, current_score=80, dispatcher=dispatcher, threshold=20)
        call_args = dispatcher.dispatch.call_args
        payload = call_args[0][1]
        assert payload["delta"] == 40
        assert payload["stale_consumption"] is True

    def test_no_alert_when_current_score_is_none(self):
        event = self._make_score_read_event(50)
        dispatcher = MagicMock()
        fired = check_stale_score_alert(event, current_score=None, dispatcher=dispatcher, threshold=20)
        assert fired is False

    def test_default_threshold_is_20(self):
        """Verify the configured default alert threshold is 20 points."""
        from integrations.soroban_event_listener import STALE_SCORE_ALERT_THRESHOLD
        assert STALE_SCORE_ALERT_THRESHOLD == 20

    def test_alert_uses_absolute_delta(self):
        """Score going down should also trigger an alert."""
        event = self._make_score_read_event(80)
        dispatcher = MagicMock()
        fired = check_stale_score_alert(event, current_score=50, dispatcher=dispatcher, threshold=20)
        assert fired is True


# ---------------------------------------------------------------------------
# 10. SorobanEventListener.process_batch integration
# ---------------------------------------------------------------------------

class TestProcessBatch:
    def test_process_batch_persists_and_returns_events(self, db, tmp_path):
        listener = SorobanEventListener(
            contract_id="CTEST",
            db_url=f"sqlite:///{tmp_path}/batch.db",
        )
        raw_events = [FIXTURES["score_read"], FIXTURES["score_updated"]]
        parsed = listener.process_batch(raw_events)
        assert len(parsed) == 2
        assert {e.event_type for e in parsed} == {"score_read", "score_updated"}

    def test_process_batch_advances_watermark(self, tmp_path):
        listener = SorobanEventListener(
            contract_id="CTEST",
            db_url=f"sqlite:///{tmp_path}/wm.db",
        )
        listener.process_batch([FIXTURES["score_read"], FIXTURES["score_updated"]])
        wm = get_watermark("CTEST", listener._session_factory)
        assert wm == 54322  # max ledger in the batch

    def test_process_batch_fires_stale_alert(self, tmp_path):
        dispatcher = MagicMock()

        def _current_score(wallet_hash):
            return 99  # far from consumed score of 75 → delta = 24 > 20

        listener = SorobanEventListener(
            contract_id="CTEST",
            db_url=f"sqlite:///{tmp_path}/alert.db",
            dispatcher=dispatcher,
            current_score_fn=_current_score,
            stale_threshold=20,
        )
        listener.process_batch([FIXTURES["score_read"]])
        dispatcher.dispatch.assert_called_once()

    def test_process_batch_skips_unknown_events(self, tmp_path):
        listener = SorobanEventListener(
            contract_id="CTEST",
            db_url=f"sqlite:///{tmp_path}/unk.db",
        )
        parsed = listener.process_batch([FIXTURES["unknown_event"]])
        assert parsed == []

    def test_process_batch_empty(self, tmp_path):
        listener = SorobanEventListener(
            contract_id="CTEST",
            db_url=f"sqlite:///{tmp_path}/empty.db",
        )
        parsed = listener.process_batch([])
        assert parsed == []


# ---------------------------------------------------------------------------
# 11. Listener stop / background thread
# ---------------------------------------------------------------------------

class TestListenerLifecycle:
    def test_stop_halts_loop(self, tmp_path):
        import threading, time

        listener = SorobanEventListener(
            contract_id="CTEST",
            db_url=f"sqlite:///{tmp_path}/lc.db",
            poll_interval=0.05,
        )
        with patch.object(listener, "_fetch_raw_events", return_value=[]):
            t = listener.start_background()
            time.sleep(0.15)
            listener.stop()
            t.join(timeout=1.0)
        assert not t.is_alive()
