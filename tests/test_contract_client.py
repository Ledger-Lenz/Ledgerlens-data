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


def test_requires_contract_id():
    with patch("integrations.contract_client.ContractClient"):
        with pytest.raises(ValueError):
            LedgerLensContractClient(contract_id="", submitter_secret="S...")


def test_submit_score_requires_submitter_secret():
    client = make_client(submitter_secret="")
    with pytest.raises(ValueError):
        client.submit_score(
            "GBC7IT5A5IFEADNWESIJFR4F4AW35BV7YIM42G6TR2W43IF3JTFBWRPD",
            "USDC:issuer/XLM:native",
            {
                "score": 80,
                "benford_flag": True,
                "ml_flag": True,
                "confidence": 80,
                "timestamp": 123,
            },
        )


def test_submit_score_invokes_contract():
    client = make_client()
    mock_tx = MagicMock()
    client._client.invoke.return_value = mock_tx

    result = client.submit_score(
        "GBC7IT5A5IFEADNWESIJFR4F4AW35BV7YIM42G6TR2W43IF3JTFBWRPD",
        "USDC:issuer/XLM:native",
        {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 80, "timestamp": 123},
    )

    client._client.invoke.assert_called_once()
    args, kwargs = client._client.invoke.call_args
    assert args[0] == "submit_score"
    assert len(args[1]) == 7
    assert "source" in kwargs and "signer" in kwargs
    mock_tx.sign_and_submit.assert_called_once()
    assert result is mock_tx.sign_and_submit.return_value


def test_submit_score_with_commitment_invokes_attested_contract():
    client = make_client()
    mock_tx = MagicMock()
    client._client.invoke.return_value = mock_tx

    result = client.submit_score(
        "GBC7IT5A5IFEADNWESIJFR4F4AW35BV7YIM42G6TR2W43IF3JTFBWRPD",
        "USDC:issuer/XLM:native",
        {
            "score": 80,
            "benford_flag": True,
            "ml_flag": True,
            "confidence": 80,
            "timestamp": 123,
        },
        commitment="deadbeef",
        trade_data_hash="cafebabe",
        model_version_hash="sha256:feedface",
    )

    client._client.invoke.assert_called_once()
    args, kwargs = client._client.invoke.call_args
    assert args[0] == "submit_score_with_commitment"
    assert len(args[1]) == 10
    assert "source" in kwargs and "signer" in kwargs
    mock_tx.sign_and_submit.assert_called_once()
    assert result is mock_tx.sign_and_submit.return_value


def test_submit_score_with_commitment_requires_hashes():
    client = make_client()

    with pytest.raises(ValueError):
        client.submit_score(
            "GBC7IT5A5IFEADNWESIJFR4F4AW35BV7YIM42G6TR2W43IF3JTFBWRPD",
            "USDC:issuer/XLM:native",
            {
                "score": 80,
                "benford_flag": True,
                "ml_flag": True,
                "confidence": 80,
                "timestamp": 123,
            },
            commitment="deadbeef",
        )


def test_get_score_invokes_contract_and_parses_result():
    client = make_client()
    mock_tx = MagicMock()
    client._client.invoke.return_value = mock_tx

    with patch(
        "integrations.contract_client.scval.to_native", return_value={"score": 42}
    ) as to_native:
        result = client.get_score(
            "GBC7IT5A5IFEADNWESIJFR4F4AW35BV7YIM42G6TR2W43IF3JTFBWRPD", "USDC:issuer/XLM:native"
        )

    client._client.invoke.assert_called_once()
    args, kwargs = client._client.invoke.call_args
    assert args[0] == "get_score"
    assert kwargs.get("simulate") is True
    to_native.assert_called_once_with(mock_tx.result.return_value)
    assert result == {"score": 42}
