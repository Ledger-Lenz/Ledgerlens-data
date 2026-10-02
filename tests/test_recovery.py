"""Tests for pipeline/recovery.py (Issue #578).

Covers:
- StageTracker: status accumulation, summary, has_failures
- RecoveryManager: stage context manager (skip/complete/fail paths), resume_info
- rollback_partial_writes: deletes rows, handles missing delete method
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pipeline.idempotency import CheckpointStore
from pipeline.recovery import RecoveryManager, StageResult, StageTracker, rollback_partial_writes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    return CheckpointStore(db_url="sqlite:///:memory:")


@pytest.fixture()
def rm(store):
    return RecoveryManager(store)


@pytest.fixture()
def pair_id():
    return "USDC:GA5Z.../XLM:native"


@pytest.fixture()
def run_id(pair_id):
    return CheckpointStore.make_run_id(pair_id, "recovery-test")


# ---------------------------------------------------------------------------
# StageTracker
# ---------------------------------------------------------------------------


class TestStageTracker:
    def test_empty_summary(self):
        t = StageTracker()
        s = t.summary()
        assert s["completed_count"] == 0
        assert s["skipped_count"] == 0
        assert s["failed_count"] == 0
        assert s["total_wall_seconds"] == 0.0

    def test_records_completed_stage(self):
        t = StageTracker()
        t.record(StageResult("ingest", "completed", wall_seconds=1.5))
        assert t.summary()["completed_count"] == 1

    def test_records_skipped_stage(self):
        t = StageTracker()
        t.record(StageResult("ingest", "skipped", wall_seconds=0.001))
        assert t.summary()["skipped_count"] == 1

    def test_records_failed_stage(self):
        t = StageTracker()
        t.record(StageResult("scoring", "failed", error="OOM"))
        assert t.summary()["failed_count"] == 1

    def test_has_failures_false_when_all_pass(self):
        t = StageTracker()
        t.record(StageResult("ingest", "completed"))
        assert t.has_failures() is False

    def test_has_failures_true_when_any_fail(self):
        t = StageTracker()
        t.record(StageResult("ingest", "completed"))
        t.record(StageResult("scoring", "failed", error="crash"))
        assert t.has_failures() is True

    def test_total_wall_time_accumulates(self):
        t = StageTracker()
        t.record(StageResult("ingest", "completed", wall_seconds=1.0))
        t.record(StageResult("features", "completed", wall_seconds=2.5))
        assert abs(t.summary()["total_wall_seconds"] - 3.5) < 0.01


# ---------------------------------------------------------------------------
# RecoveryManager — stage context manager
# ---------------------------------------------------------------------------


class TestRecoveryManagerStage:
    def test_stage_executes_on_first_run(self, rm, run_id, pair_id):
        executed = []
        with rm.stage(run_id, pair_id, "ingest") as ctx:
            if not ctx.skip:
                executed.append("ran")
        assert executed == ["ran"]

    def test_stage_skipped_on_second_run(self, rm, run_id, pair_id):
        with rm.stage(run_id, pair_id, "ingest"):
            pass
        executed = []
        with rm.stage(run_id, pair_id, "ingest") as ctx:
            if not ctx.skip:
                executed.append("ran")
        assert executed == []

    def test_stage_records_result(self, rm, run_id, pair_id, store):
        with rm.stage(run_id, pair_id, "scoring") as ctx:
            ctx.set_result({"wallet_count": 10})
        assert store.get_result(run_id, pair_id, "scoring") == {"wallet_count": 10}

    def test_stage_records_failure_and_propagates_exception(self, rm, run_id, pair_id):
        with pytest.raises(RuntimeError, match="test crash"):
            with rm.stage(run_id, pair_id, "ingest"):
                raise RuntimeError("test crash")
        assert rm.has_failures(run_id, pair_id) is True

    def test_failed_stage_status_in_summary(self, rm, run_id, pair_id):
        with pytest.raises(ValueError):
            with rm.stage(run_id, pair_id, "features"):
                raise ValueError("bad features")
        summary = rm.run_summary(run_id, pair_id)
        failed = [s for s in summary["stages"] if s["stage"] == "features"]
        assert len(failed) == 1
        assert failed[0]["status"] == "failed"
        assert "bad features" in failed[0]["error"]

    def test_skipped_stage_in_summary(self, rm, run_id, pair_id):
        # Complete the stage in run_id so the second call skips it.
        with rm.stage(run_id, pair_id, "ingest"):
            pass
        # New RecoveryManager shares same store — stage already marked done.
        rm2 = RecoveryManager(rm._store)
        with rm2.stage(run_id, pair_id, "ingest"):
            pass
        summary = rm2.run_summary(run_id, pair_id)
        skipped = [s for s in summary["stages"] if s["stage"] == "ingest"]
        assert skipped[0]["status"] == "skipped"

    def test_wall_time_is_non_negative(self, rm, run_id, pair_id):
        with rm.stage(run_id, pair_id, "ingest"):
            pass
        summary = rm.run_summary(run_id, pair_id)
        for s in summary["stages"]:
            assert s["wall_seconds"] >= 0

    def test_force_reruns_completed_stage(self, rm, run_id, pair_id):
        calls = []
        with rm.stage(run_id, pair_id, "ingest"):
            calls.append(1)
        with rm.stage(run_id, pair_id, "ingest", force=True) as ctx:
            if not ctx.skip:
                calls.append(2)
        assert calls == [1, 2]


# ---------------------------------------------------------------------------
# RecoveryManager — resume_info
# ---------------------------------------------------------------------------


class TestResumeInfo:
    def test_no_resume_on_fresh_run(self, rm, run_id, pair_id):
        info = rm.resume_info(run_id, pair_id)
        assert info["will_resume"] is False
        assert info["completed_stages"] == []

    def test_resume_info_shows_completed_stages(self, rm, run_id, pair_id):
        with rm.stage(run_id, pair_id, "ingest"):
            pass
        with rm.stage(run_id, pair_id, "orderbook"):
            pass
        info = rm.resume_info(run_id, pair_id)
        assert "ingest" in info["completed_stages"]
        assert "orderbook" in info["completed_stages"]

    def test_resume_detects_first_incomplete_stage(self, rm, run_id, pair_id):
        with rm.stage(run_id, pair_id, "ingest"):
            pass
        info = rm.resume_info(run_id, pair_id)
        assert info["first_incomplete_stage"] == "orderbook"

    def test_will_resume_true_after_partial_run(self, rm, run_id, pair_id):
        with rm.stage(run_id, pair_id, "ingest"):
            pass
        info = rm.resume_info(run_id, pair_id)
        assert info["will_resume"] is True

    def test_will_resume_false_all_complete(self, rm, run_id, pair_id, store):
        from pipeline.idempotency import PIPELINE_STAGES

        for s in PIPELINE_STAGES:
            store.mark_done(run_id, pair_id, s)
        info = rm.resume_info(run_id, pair_id)
        assert info["will_resume"] is False
        assert info["first_incomplete_stage"] is None


# ---------------------------------------------------------------------------
# rollback_partial_writes
# ---------------------------------------------------------------------------


class TestRollbackPartialWrites:
    def test_deletes_listed_wallets(self):
        mock_store = MagicMock()
        mock_store.delete.return_value = True
        deleted = rollback_partial_writes(mock_store, ["W1", "W2", "W3"], "USDC/XLM")
        assert deleted == 3
        assert mock_store.delete.call_count == 3

    def test_handles_missing_delete_method(self):
        """Stores without .delete() log a warning and return 0 — no crash."""
        mock_store = MagicMock(spec=[])  # no .delete
        result = rollback_partial_writes(mock_store, ["W1"], "USDC/XLM")
        assert result == 0

    def test_handles_delete_returning_false(self):
        mock_store = MagicMock()
        mock_store.delete.return_value = False  # row did not exist
        deleted = rollback_partial_writes(mock_store, ["W1", "W2"], "USDC/XLM")
        assert deleted == 0

    def test_handles_delete_exception_gracefully(self):
        mock_store = MagicMock()
        mock_store.delete.side_effect = Exception("db locked")
        # Must not raise
        deleted = rollback_partial_writes(mock_store, ["W1"], "USDC/XLM")
        assert deleted == 0

    def test_empty_wallets_list(self):
        mock_store = MagicMock()
        deleted = rollback_partial_writes(mock_store, [], "USDC/XLM")
        assert deleted == 0
        mock_store.delete.assert_not_called()

    def test_partial_failure_continues(self):
        """If one delete fails the rest still execute."""
        mock_store = MagicMock()
        mock_store.delete.side_effect = [True, Exception("oops"), True]
        deleted = rollback_partial_writes(mock_store, ["W1", "W2", "W3"], "USDC/XLM")
        # W1 and W3 succeeded, W2 errored → 2 deleted
        assert deleted == 2


# ---------------------------------------------------------------------------
# Post-recovery consistency verification (Issue #920)
# ---------------------------------------------------------------------------


def _simulate_recovery(rm, run_id, pair_id, source_rows, recovered_rows):
    """Run an ingest stage that 'recovers' *recovered_rows* and verify it
    against the source-of-truth *source_rows*."""
    from pipeline.recovery import StageSnapshot

    with rm.stage(run_id, pair_id, "ingest") as ctx:
        if not ctx.skip:
            ctx.set_result({"row_count": len(recovered_rows)})
    return rm.complete_recovery(
        run_id,
        pair_id,
        expected={"ingest": StageSnapshot.from_records(source_rows)},
        actual={"ingest": StageSnapshot.from_records(recovered_rows)},
        recovered_range=("2026-09-01T00:00:00Z", "2026-09-01T06:00:00Z"),
    )


_ROWS = [{"trade_id": i, "amount": str(i * 10)} for i in range(5)]


def test_clean_recovery_passes_and_resumes_automatically(rm, run_id, pair_id):
    report = _simulate_recovery(rm, run_id, pair_id, _ROWS, list(reversed(_ROWS)))

    assert report.passed
    assert report.recovered_stages == ["ingest"]
    assert rm.can_auto_resume(run_id, pair_id)
    rendered = report.render()
    assert "Result:           PASS" in rendered
    assert "ALLOWED" in rendered

    # Normal processing continues without raising.
    with rm.stage(run_id, pair_id, "features") as ctx:
        ctx.set_result({"wallet_count": 3})


def test_injected_count_inconsistency_blocks_auto_resumption(rm, run_id, pair_id):
    from pipeline.recovery import RecoveryBlockedError

    report = _simulate_recovery(rm, run_id, pair_id, _ROWS, _ROWS[:-1])

    assert not report.passed
    assert report.failed_stages() == ["ingest"]
    assert not rm.can_auto_resume(run_id, pair_id)
    assert "record count mismatch" in report.render()
    assert "BLOCKED" in report.render()

    with pytest.raises(RecoveryBlockedError):
        with rm.stage(run_id, pair_id, "features"):
            pass


def test_injected_checksum_inconsistency_is_detected(rm, run_id, pair_id):
    tampered = [dict(r) for r in _ROWS]
    tampered[2]["amount"] = "999"
    report = _simulate_recovery(rm, run_id, pair_id, _ROWS, tampered)

    assert not report.passed
    assert any("checksum mismatch" in i for i in report.checks[0].issues)


def test_missing_stage_after_recovery_fails(rm, run_id, pair_id):
    from pipeline.recovery import StageSnapshot, verify_recovery

    report = verify_recovery(
        run_id,
        pair_id,
        expected={"ingest": StageSnapshot(5), "features": StageSnapshot(3)},
        actual={"ingest": StageSnapshot(5)},
    )
    assert report.failed_stages() == ["features"]


def test_manual_approval_unblocks_resumption(rm, run_id, pair_id):
    _simulate_recovery(rm, run_id, pair_id, _ROWS, _ROWS[:-1])
    rm.approve_resume(run_id, pair_id, reviewer="oncall@ledgerlens")

    assert rm.can_auto_resume(run_id, pair_id)
    report = rm.recovery_report(run_id, pair_id)
    assert report.to_dict()["approved_by"] == "oncall@ledgerlens"
    assert report.to_dict()["result"] == "FAIL"
    with rm.stage(run_id, pair_id, "features"):
        pass


def test_block_is_scoped_to_run_and_pair(rm, run_id, pair_id):
    _simulate_recovery(rm, run_id, pair_id, _ROWS, _ROWS[:-1])
    assert rm.can_auto_resume(run_id, "OTHER:pair")
    with rm.stage(run_id, "OTHER:pair", "ingest"):
        pass


def test_checksum_is_order_independent():
    from pipeline.recovery import compute_checksum

    assert compute_checksum(_ROWS) == compute_checksum(list(reversed(_ROWS)))
    assert compute_checksum(_ROWS) != compute_checksum(_ROWS[:-1])
