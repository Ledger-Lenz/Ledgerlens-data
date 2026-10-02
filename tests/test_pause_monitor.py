"""Tests for integrations/pause_monitor.py — off-chain pause parity check (#949).

Test plan
---------
1. ``test_no_alert_when_states_match`` — on-chain paused + pipeline paused = no alert.
2. ``test_alert_on_mismatch_paused_onchain_running_pipeline`` — on-chain says
   paused but pipeline is still running → alert fires.
3. ``test_alert_on_mismatch_running_onchain_paused_pipeline`` — on-chain says
   running but pipeline is halted → alert fires.
4. ``test_no_false_positive_during_transition`` — a brief transition window
   (state queried mid-flip) does not produce a spurious alert when the correct
   behaviour of the transition is respected.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from integrations.pause_monitor import PauseMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_monitor(
    onchain_paused: bool,
    pipeline_running: bool,
    alert_fn: MagicMock | None = None,
) -> tuple[PauseMonitor, MagicMock]:
    """Create a PauseMonitor with mocked contract_client and pipeline_state_fn."""
    if alert_fn is None:
        alert_fn = MagicMock()

    contract_client = MagicMock()
    contract_client.is_paused.return_value = onchain_paused

    monitor = PauseMonitor(
        contract_client=contract_client,
        pipeline_state_fn=lambda: pipeline_running,
        alert_fn=alert_fn,
        poll_interval_seconds=30,
    )
    return monitor, alert_fn


# ---------------------------------------------------------------------------
# 1. No alert when states match
# ---------------------------------------------------------------------------


class TestNoAlertWhenStatesMatch:
    def test_both_paused_no_alert(self):
        """On-chain paused, pipeline halted → in parity, no alert."""
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=False)
        result = monitor.run_once()
        assert result is True
        alert_fn.assert_not_called()

    def test_both_running_no_alert(self):
        """On-chain running, pipeline running → in parity, no alert."""
        monitor, alert_fn = make_monitor(onchain_paused=False, pipeline_running=True)
        result = monitor.run_once()
        assert result is True
        alert_fn.assert_not_called()

    def test_parity_check_returns_true(self):
        monitor, _ = make_monitor(onchain_paused=False, pipeline_running=True)
        assert monitor.check_parity() is True

    def test_get_onchain_pause_state_delegates_to_client(self):
        monitor, _ = make_monitor(onchain_paused=True, pipeline_running=False)
        assert monitor.get_onchain_pause_state() is True

    def test_get_pipeline_operational_state_delegates_to_fn(self):
        monitor, _ = make_monitor(onchain_paused=False, pipeline_running=True)
        assert monitor.get_pipeline_operational_state() is True


# ---------------------------------------------------------------------------
# 2. Alert: on-chain paused, pipeline still running
# ---------------------------------------------------------------------------


class TestAlertPausedOnchainRunningPipeline:
    def test_alert_fires(self):
        """On-chain pause active, pipeline still running → mismatch alert."""
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=True)
        result = monitor.run_once()
        assert result is False
        alert_fn.assert_called_once()

    def test_alert_details_severity_is_critical(self):
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=True)
        monitor.run_once()
        details = alert_fn.call_args[0][0]
        assert details["severity"] == "CRITICAL"

    def test_alert_details_onchain_paused_true(self):
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=True)
        monitor.run_once()
        details = alert_fn.call_args[0][0]
        assert details["onchain_paused"] is True

    def test_alert_details_pipeline_paused_false(self):
        """When pipeline is running despite on-chain pause, pipeline_paused=False."""
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=True)
        monitor.run_once()
        details = alert_fn.call_args[0][0]
        assert details["pipeline_paused"] is False

    def test_alert_details_has_detected_at(self):
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=True)
        monitor.run_once()
        details = alert_fn.call_args[0][0]
        assert "detected_at" in details
        assert details["detected_at"]  # non-empty string

    def test_alert_details_has_message(self):
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=True)
        monitor.run_once()
        details = alert_fn.call_args[0][0]
        assert "message" in details
        assert len(details["message"]) > 0


# ---------------------------------------------------------------------------
# 3. Alert: on-chain running, pipeline halted
# ---------------------------------------------------------------------------


class TestAlertRunningOnchainPausedPipeline:
    def test_alert_fires(self):
        """On-chain running, pipeline halted → mismatch alert."""
        monitor, alert_fn = make_monitor(onchain_paused=False, pipeline_running=False)
        result = monitor.run_once()
        assert result is False
        alert_fn.assert_called_once()

    def test_alert_details_severity_is_critical(self):
        monitor, alert_fn = make_monitor(onchain_paused=False, pipeline_running=False)
        monitor.run_once()
        details = alert_fn.call_args[0][0]
        assert details["severity"] == "CRITICAL"

    def test_alert_details_onchain_paused_false(self):
        monitor, alert_fn = make_monitor(onchain_paused=False, pipeline_running=False)
        monitor.run_once()
        details = alert_fn.call_args[0][0]
        assert details["onchain_paused"] is False

    def test_alert_details_pipeline_paused_true(self):
        """Pipeline halted when it shouldn't be → pipeline_paused=True."""
        monitor, alert_fn = make_monitor(onchain_paused=False, pipeline_running=False)
        monitor.run_once()
        details = alert_fn.call_args[0][0]
        assert details["pipeline_paused"] is True

    def test_check_parity_returns_false(self):
        monitor, _ = make_monitor(onchain_paused=False, pipeline_running=False)
        assert monitor.check_parity() is False


# ---------------------------------------------------------------------------
# 4. No false positive during a brief transition window
# ---------------------------------------------------------------------------


class TestNoFalsePositiveDuringTransition:
    """A single poll that catches a legitimate state transition mid-flip should
    not fire an alert because the transition itself is expected.

    Modelling the transition: the on-chain state flips from "paused=True" to
    "paused=False", and the pipeline has already responded by resuming.  From
    the monitor's perspective, both states are "running" — this is parity, so
    no alert should fire.
    """

    def test_pause_lifted_and_pipeline_resumed_no_alert(self):
        """After an unpause: onchain=False, pipeline=running → parity."""
        monitor, alert_fn = make_monitor(onchain_paused=False, pipeline_running=True)
        result = monitor.run_once()
        assert result is True
        alert_fn.assert_not_called()

    def test_pause_initiated_and_pipeline_halted_no_alert(self):
        """After a pause: onchain=True, pipeline=halted → parity."""
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=False)
        result = monitor.run_once()
        assert result is True
        alert_fn.assert_not_called()

    def test_no_alert_on_repeated_consistent_ticks(self):
        """Multiple consecutive ticks in a consistent state should never alert."""
        monitor, alert_fn = make_monitor(onchain_paused=False, pipeline_running=True)
        for _ in range(5):
            monitor.run_once()
        alert_fn.assert_not_called()

    def test_alert_fires_only_once_per_tick_on_mismatch(self):
        """Even if check_parity is called twice in one tick, only one alert fires
        per call (no alert dedup within PauseMonitor itself)."""
        monitor, alert_fn = make_monitor(onchain_paused=True, pipeline_running=True)
        monitor.run_once()
        monitor.run_once()
        # Two ticks, two mismatches → two alerts (dedup is the caller's job)
        assert alert_fn.call_count == 2


# ---------------------------------------------------------------------------
# 5. Async loop management
# ---------------------------------------------------------------------------


class TestAsyncLoop:
    def test_start_stop(self):
        """start() runs the loop and stop() terminates it after a tick."""
        monitor, alert_fn = make_monitor(onchain_paused=False, pipeline_running=True)

        async def _run():
            task = asyncio.create_task(monitor.start())
            await asyncio.sleep(0.05)
            monitor.stop()
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
        # No mismatch → no alert
        alert_fn.assert_not_called()
        assert monitor._running is False

    def test_pause_contract_id_passed_to_client(self):
        """When pause_contract_id is set, is_paused is called with it."""
        alert_fn = MagicMock()
        client = MagicMock()
        client.is_paused.return_value = False

        monitor = PauseMonitor(
            contract_client=client,
            pipeline_state_fn=lambda: True,
            alert_fn=alert_fn,
            pause_contract_id="CPAUSE123",
        )
        monitor.run_once()
        client.is_paused.assert_called_once_with("CPAUSE123")
