"""Tests for reorg-aware event processing (#948).

Covers the new :class:`~integrations.soroban_event_listener.ReorgAwareEventProcessor`
and :class:`~integrations.soroban_event_listener.PendingEvent` additions.

Test plan
---------
1. ``test_normal_event_processing`` — events are held in the pending buffer
   until ``current_ledger_seq`` advances past the confirmation depth, then
   finalized.
2. ``test_reorg_retraction`` — events buffered for a ledger are removed when
   ``handle_reorg`` is called with that ledger sequence; they do not appear in
   the finalized set.
3. ``test_no_duplicate_after_reorg_reprocess`` — after a reorg, re-feeding
   canonical-chain events produces exactly one finalized entry per event (no
   duplicates).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from integrations.soroban_event_listener import (
    CONFIRMATION_DEPTH,
    ContractEvent,
    PendingEvent,
    ReorgAwareEventProcessor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_event(ledger: int, event_type: str = "score_updated") -> dict:
    """Build a minimal raw Soroban RPC event dict at *ledger*."""
    return {
        "id": f"synthetic:{ledger}:{event_type}",
        "ledger": ledger,
        "ledgerClosedAt": "2026-09-28T19:00:00Z",
        "contractId": "CTEST1234",
        "topic": [{"type": "symbol", "value": event_type}],
        "value": {"type": "i32", "value": 42},
    }


def _make_contract_event(ledger: int) -> ContractEvent:
    """Build a minimal :class:`ContractEvent` at *ledger*."""
    from datetime import UTC, datetime

    return ContractEvent(
        event_type="score_updated",
        ledger_sequence=ledger,
        timestamp=datetime.now(UTC),
        event_id=f"ev-{ledger}",
    )


# ---------------------------------------------------------------------------
# 1. Normal event processing
# ---------------------------------------------------------------------------


class TestNormalEventProcessing:
    """Events should finalize only after CONFIRMATION_DEPTH ledgers have passed."""

    def test_event_stays_pending_before_confirmation_depth(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        event = _make_raw_event(ledger=100)
        # Feed event at tip=100; not yet confirmed (100 - 100 = 0 < 2)
        proc.process_event(event, current_ledger_seq=100)
        assert len(proc._finalized) == 0
        assert 100 in proc._pending

    def test_event_stays_pending_one_ledger_before_threshold(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        event = _make_raw_event(ledger=100)
        # Tip at 101: 101 - 100 = 1 < 2 → still pending
        proc.process_event(event, current_ledger_seq=101)
        assert len(proc._finalized) == 0

    def test_event_finalizes_exactly_at_confirmation_depth(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        event = _make_raw_event(ledger=100)
        # Feed at tip=100 (pending), then advance tip to 102 (100 + depth = 102)
        proc.process_event(event, current_ledger_seq=100)
        proc.finalize_events(current_ledger_seq=102)
        assert len(proc._finalized) == 1
        assert proc._finalized[0].ledger_sequence == 100

    def test_pending_buffer_cleared_after_finalization(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        event = _make_raw_event(ledger=100)
        proc.process_event(event, current_ledger_seq=100)
        proc.finalize_events(current_ledger_seq=102)
        assert 100 not in proc._pending

    def test_multiple_events_same_ledger_all_finalized(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        for i in range(3):
            proc.process_event(_make_raw_event(ledger=100, event_type=f"ev_{i}"), 100)
        proc.finalize_events(102)
        assert len(proc._finalized) == 3

    def test_on_finalized_callback_invoked(self):
        callback = MagicMock()
        proc = ReorgAwareEventProcessor(confirmation_depth=2, on_finalized=callback)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=102)
        callback.assert_called_once()
        ev: PendingEvent = callback.call_args[0][0]
        assert isinstance(ev, PendingEvent)
        assert ev.ledger_sequence == 100

    def test_contract_event_input_supported(self):
        """ReorgAwareEventProcessor should also accept ContractEvent instances."""
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        ev = _make_contract_event(ledger=200)
        proc.process_event(ev, current_ledger_seq=202)
        assert len(proc._finalized) == 1
        assert proc._finalized[0].ledger_sequence == 200

    def test_default_confirmation_depth_is_2(self):
        assert CONFIRMATION_DEPTH == 2

    def test_process_event_returns_none(self):
        """process_event is fire-and-forget; it has no meaningful return value."""
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        result = proc.process_event(_make_raw_event(100), 100)
        assert result is None

    def test_finalize_events_returns_newly_finalized(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        proc.process_event(_make_raw_event(100), 100)
        proc.process_event(_make_raw_event(101), 101)
        newly = proc.finalize_events(103)
        # Both ledgers (100, 101) are now ≥ depth below tip=103
        assert len(newly) == 2


# ---------------------------------------------------------------------------
# 2. Reorg retraction
# ---------------------------------------------------------------------------


class TestReorgRetraction:
    """Events from orphaned ledgers must be removed and must not appear in finalized."""

    def test_handle_reorg_removes_buffered_events(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        assert 100 in proc._pending
        proc.handle_reorg([100])
        assert 100 not in proc._pending

    def test_retracted_events_not_in_finalized(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.handle_reorg([100])
        # Even after advancing the tip, the retracted events should not finalize
        proc.finalize_events(current_ledger_seq=110)
        assert len(proc._finalized) == 0

    def test_handle_reorg_returns_retracted_events(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        retracted = proc.handle_reorg([100])
        assert len(retracted) == 1
        assert retracted[0].ledger_sequence == 100

    def test_handle_reorg_multiple_ledgers(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        for ledger in (100, 101):
            proc.process_event(_make_raw_event(ledger=ledger), current_ledger_seq=101)
        retracted = proc.handle_reorg([100, 101])
        assert len(retracted) == 2
        proc.finalize_events(110)
        assert len(proc._finalized) == 0

    def test_handle_reorg_nonexistent_ledger_is_noop(self):
        """Calling handle_reorg with a ledger not in pending should not raise."""
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        retracted = proc.handle_reorg([999])
        assert retracted == []

    def test_retracted_sequence_recorded(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.handle_reorg([100])
        assert 100 in proc._retracted_sequences

    def test_events_from_non_orphaned_ledger_still_finalize(self):
        """Only the specified orphaned ledgers should be retracted."""
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.process_event(_make_raw_event(ledger=101), current_ledger_seq=101)
        # Only ledger 100 is orphaned; 101 is canonical
        proc.handle_reorg([100])
        proc.finalize_events(103)
        assert len(proc._finalized) == 1
        assert proc._finalized[0].ledger_sequence == 101

    def test_on_finalized_callback_not_called_for_retracted(self):
        callback = MagicMock()
        proc = ReorgAwareEventProcessor(confirmation_depth=2, on_finalized=callback)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.handle_reorg([100])
        proc.finalize_events(110)
        callback.assert_not_called()


# ---------------------------------------------------------------------------
# 3. No duplicate after reorg + reprocess
# ---------------------------------------------------------------------------


class TestNoDuplicateAfterReorgReprocess:
    """After a reorg, reprocessing events from the canonical chain must not
    produce duplicate entries in the finalized output.
    """

    def test_canonical_event_finalizes_exactly_once(self):
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        # Step 1: receive event at ledger 100 (canonical candidate)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        # Step 2: reorg — ledger 100 was orphaned
        proc.handle_reorg([100])
        # Step 3: re-process the canonical replacement event at ledger 100
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        # Step 4: advance tip past confirmation depth
        proc.finalize_events(current_ledger_seq=102)
        # Must have exactly ONE finalized event (the canonical one)
        assert len(proc._finalized) == 1

    def test_multiple_reorgs_single_finalized_entry(self):
        """Simulates a double reorg on ledger 100 — still only one final entry."""
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.handle_reorg([100])
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.handle_reorg([100])
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.finalize_events(102)
        assert len(proc._finalized) == 1

    def test_different_ledger_events_all_finalize_without_duplicates(self):
        """Events from different ledgers should each appear exactly once."""
        proc = ReorgAwareEventProcessor(confirmation_depth=2)
        for ledger in (100, 101, 102):
            proc.process_event(_make_raw_event(ledger=ledger), current_ledger_seq=ledger)
        # Reorg ledger 101; re-process it
        proc.handle_reorg([101])
        proc.process_event(_make_raw_event(ledger=101), current_ledger_seq=101)
        proc.finalize_events(104)
        # Should have exactly 3 finalized events (100, 101, 102)
        assert len(proc._finalized) == 3
        finalized_seqs = [ev.ledger_sequence for ev in proc._finalized]
        assert sorted(finalized_seqs) == [100, 101, 102]

    def test_callback_invoked_once_per_canonical_event(self):
        """The on_finalized callback should be called once per unique canonical event."""
        callback = MagicMock()
        proc = ReorgAwareEventProcessor(confirmation_depth=2, on_finalized=callback)
        # Reorg cycle for ledger 100
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.handle_reorg([100])
        proc.process_event(_make_raw_event(ledger=100), current_ledger_seq=100)
        proc.finalize_events(102)
        callback.assert_called_once()
