"""Tests for multi-sig governance client methods (issue #238)."""

from unittest.mock import MagicMock, patch

import pytest

from integrations.contract_client import LedgerLensContractClient


def make_client(**kwargs) -> LedgerLensContractClient:
    defaults = {
        "contract_id": "CCONTRACT",
        "rpc_url": "https://soroban-testnet.stellar.org",
        "network_passphrase": "Test SDF Network ; September 2015",
        "submitter_secret": "SAUQSDM4BPSOWVJJM7RAHPSGXDX5YLRYNZCZ5QP33EVB6WDAAVJJRJHG",
    }
    defaults.update(kwargs)
    with patch("integrations.contract_client.ContractClient"):
        return LedgerLensContractClient(**defaults)


_PROPOSER_SECRET = "SAUQSDM4BPSOWVJJM7RAHPSGXDX5YLRYNZCZ5QP33EVB6WDAAVJJRJHG"
_APPROVER_SECRET = "SBFZNUYBQI7ZMH5VNXBJGMZXJXMJMZXJXMJMZXJXMJMZXJXMJMZXJXMJ"
_GOV_CONTRACT = "CGOV1234"
_PAUSE_CONTRACT = "CPAUSE5678"


def test_propose_threshold_change_calls_governance_contract():
    client = make_client()
    mock_governance_client = MagicMock()
    mock_tx = MagicMock()
    mock_governance_client.invoke.return_value = mock_tx
    mock_tx.sign_and_submit.return_value = MagicMock()

    with (
        patch("integrations.contract_client.ContractClient", return_value=mock_governance_client),
        patch("integrations.contract_client.scval.to_native", return_value=0),
    ):
        proposal_id = client.propose_threshold_change(
            governance_contract_id=_GOV_CONTRACT,
            new_threshold=75,
            proposer_secret=_PROPOSER_SECRET,
        )

    mock_governance_client.invoke.assert_called_once()
    args, kwargs = mock_governance_client.invoke.call_args
    assert args[0] == "propose_threshold_change"
    assert len(args[1]) == 2
    assert proposal_id == 0


def test_approve_threshold_change_returns_true_on_quorum():
    client = make_client()
    mock_governance_client = MagicMock()
    mock_tx = MagicMock()
    mock_governance_client.invoke.return_value = mock_tx

    with (
        patch("integrations.contract_client.ContractClient", return_value=mock_governance_client),
        patch("integrations.contract_client.scval.to_native", return_value=True),
    ):
        applied = client.approve_threshold_change(
            governance_contract_id=_GOV_CONTRACT,
            proposal_id=0,
            approver_secret=_PROPOSER_SECRET,
        )

    mock_governance_client.invoke.assert_called_once()
    args, _ = mock_governance_client.invoke.call_args
    assert args[0] == "approve_threshold_change"
    assert applied is True


def test_approve_threshold_change_returns_false_below_quorum():
    client = make_client()
    mock_governance_client = MagicMock()
    mock_tx = MagicMock()
    mock_governance_client.invoke.return_value = mock_tx

    with (
        patch("integrations.contract_client.ContractClient", return_value=mock_governance_client),
        patch("integrations.contract_client.scval.to_native", return_value=False),
    ):
        applied = client.approve_threshold_change(
            governance_contract_id=_GOV_CONTRACT,
            proposal_id=0,
            approver_secret=_PROPOSER_SECRET,
        )

    assert applied is False


def test_initiate_emergency_pause_calls_pause_contract():
    client = make_client()
    mock_pause_client = MagicMock()
    mock_tx = MagicMock()
    mock_pause_client.invoke.return_value = mock_tx

    with (
        patch("integrations.contract_client.ContractClient", return_value=mock_pause_client),
        patch("integrations.contract_client.scval.to_native", return_value=0),
    ):
        proposal_id = client.initiate_emergency_pause(
            pause_contract_id=_PAUSE_CONTRACT,
            reason="Anomalous scores detected",
            signing_key=_PROPOSER_SECRET,
        )

    mock_pause_client.invoke.assert_called_once()
    args, kwargs = mock_pause_client.invoke.call_args
    assert args[0] == "initiate_pause"
    assert len(args[1]) == 2
    assert "source" in kwargs and "signer" in kwargs
    assert proposal_id == 0


def test_approve_emergency_pause_emits_paused_event():
    client = make_client()
    mock_pause_client = MagicMock()
    mock_tx = MagicMock()
    mock_pause_client.invoke.return_value = mock_tx

    with (
        patch("integrations.contract_client.ContractClient", return_value=mock_pause_client),
        patch("integrations.contract_client.scval.to_native", return_value=True),
    ):
        paused = client.approve_emergency_pause(
            pause_contract_id=_PAUSE_CONTRACT,
            proposal_id=0,
            signing_key=_PROPOSER_SECRET,
        )

    args, _ = mock_pause_client.invoke.call_args
    assert args[0] == "approve_pause"
    assert paused is True
