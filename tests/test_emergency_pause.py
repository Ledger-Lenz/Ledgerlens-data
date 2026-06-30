"""Tests for emergency pause watchdog and contract client methods (issue #241)."""

import time
from unittest.mock import MagicMock, call, patch

import pytest

from monitoring.emergency_watchdog import EmergencyWatchdog


_PAUSE_CONTRACT = "CPAUSE5678"
_SIGNING_KEY = "SAUQSDM4BPSOWVJJM7RAHPSGXDX5YLRYNZCZ5QP33EVB6WDAAVJJRJHG"


def make_watchdog(**kwargs) -> EmergencyWatchdog:
    defaults = {
        "pause_contract_id": _PAUSE_CONTRACT,
        "signing_key": _SIGNING_KEY,
        "anomaly_score_threshold": 95,
        "anomaly_rate_threshold": 0.90,
        "window_seconds": 60,
    }
    defaults.update(kwargs)
    return EmergencyWatchdog(**defaults)


def test_no_pause_below_anomaly_rate():
    watchdog = make_watchdog()
    # 80% of scores above threshold — below 90% rate → no pause
    for i in range(80):
        watchdog.record_score(f"hash{i}", 96)
    for i in range(20):
        watchdog.record_score(f"hash_lo{i}", 50)

    proposed = watchdog.check()
    assert proposed is False


def test_pause_proposed_above_anomaly_rate():
    """When > 90% of scores exceed 95, a pause proposal is submitted."""
    proposed_ids = []

    def on_proposed(proposal_id: int, reason: str) -> None:
        proposed_ids.append(proposal_id)

    watchdog = make_watchdog(on_pause_proposed=on_proposed)

    # Inject 95 anomalous scores out of 100
    for i in range(95):
        watchdog.record_score(f"hash{i}", 99)
    for i in range(5):
        watchdog.record_score(f"hash_lo{i}", 10)

    with patch(
        "integrations.contract_client.LedgerLensContractClient"
    ) as MockClient:
        instance = MockClient.return_value
        instance.initiate_emergency_pause.return_value = 42

        proposed = watchdog.check()

    assert proposed is True
    instance.initiate_emergency_pause.assert_called_once()
    call_kwargs = instance.initiate_emergency_pause.call_args
    assert call_kwargs.kwargs["pause_contract_id"] == _PAUSE_CONTRACT
    assert "Anomalous" in call_kwargs.kwargs["reason"]
    assert proposed_ids == [42]


def test_pause_proposed_only_once():
    """A second check should not propose a second pause."""
    watchdog = make_watchdog()
    for i in range(100):
        watchdog.record_score(f"hash{i}", 99)

    with patch("integrations.contract_client.LedgerLensContractClient") as MockClient:
        instance = MockClient.return_value
        instance.initiate_emergency_pause.return_value = 1

        watchdog.check()
        watchdog.check()

    assert instance.initiate_emergency_pause.call_count == 1


def test_pipeline_halt_event_via_listener():
    """Receiving contract_paused event must set listener.is_paused = True."""
    from integrations.soroban_event_listener import SorobanEventListener

    halted = []

    listener = SorobanEventListener(
        governance_contract_id="CGOV",
        pause_contract_id=_PAUSE_CONTRACT,
        on_contract_paused=lambda reason: halted.append(reason),
    )

    fake_event = {
        "contractId": _PAUSE_CONTRACT,
        "topic": [{"sym": "c_paused"}, {"u64": 0}],
        "value": {"str": "test pause reason"},
        "ledger": 1000,
    }

    with patch.object(listener, "_fetch_events", return_value=[fake_event]):
        listener.poll_once()

    assert listener.is_paused is True
    assert halted == ["test pause reason"]


def test_contract_paused_event_emitted(tmp_path):
    """initiate_emergency_pause must call the contract's initiate_pause entry point."""
    from unittest.mock import patch as _patch

    from integrations.contract_client import LedgerLensContractClient

    with _patch("integrations.contract_client.ContractClient"):
        client = LedgerLensContractClient(
            contract_id="CCONTRACT",
            rpc_url="https://soroban-testnet.stellar.org",
            network_passphrase="Test SDF Network ; September 2015",
            submitter_secret=_SIGNING_KEY,
        )

    mock_pause_client = MagicMock()
    mock_tx = MagicMock()
    mock_pause_client.invoke.return_value = mock_tx

    with (
        _patch("integrations.contract_client.ContractClient", return_value=mock_pause_client),
        _patch("integrations.contract_client.scval.to_native", return_value=7),
    ):
        proposal_id = client.initiate_emergency_pause(
            pause_contract_id=_PAUSE_CONTRACT,
            reason="Bug in scoring pipeline",
            signing_key=_SIGNING_KEY,
        )

    mock_pause_client.invoke.assert_called_once()
    args, kwargs = mock_pause_client.invoke.call_args
    assert args[0] == "initiate_pause"
    assert len(args[1]) == 2
    mock_tx.sign_and_submit.assert_called_once()
    assert proposal_id == 7
