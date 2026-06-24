"""Unit tests for scripts/testnet_setup.py — no network access."""

import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest

import scripts.testnet_setup as setup

# ---------------------------------------------------------------------------
# 1. Skips deployment when LEDGERLENS_CONTRACT_ID is already set
# ---------------------------------------------------------------------------


def test_setup_skips_deployment_if_contract_id_set(tmp_path):
    env_file = str(tmp_path / ".env.testnet")

    with (
        patch.dict(
            os.environ,
            {
                "LEDGERLENS_CONTRACT_ID": "CA_ALREADY_DEPLOYED",
                "LEDGERLENS_SUBMITTER_SECRET": "SAUQSDM4BPSOWVJJM7RAHPSGXDX5YLRYNZCZ5QP33EVB6WDAAVJJRJHG",
            },
        ),
        patch("scripts.testnet_setup.load_dotenv"),
        patch("scripts.testnet_setup.fund_account") as mock_fund,
        patch("scripts.testnet_setup.deploy_contract") as mock_deploy,
    ):
        contract_id = setup.run(
            wasm_path="dummy.wasm",
            skip_hash_check=True,
            env_file=env_file,
        )

    mock_fund.assert_called_once()
    mock_deploy.assert_not_called()
    assert contract_id == "CA_ALREADY_DEPLOYED"


# ---------------------------------------------------------------------------
# 2. Writes .env.testnet with the correct keys
# ---------------------------------------------------------------------------


def test_setup_writes_env_file(tmp_path):
    env_file = str(tmp_path / ".env.testnet")
    fake_secret = "SAUQSDM4BPSOWVJJM7RAHPSGXDX5YLRYNZCZ5QP33EVB6WDAAVJJRJHG"

    with (
        patch.dict(
            os.environ, {"LEDGERLENS_CONTRACT_ID": "", "LEDGERLENS_SUBMITTER_SECRET": fake_secret}
        ),
        patch("scripts.testnet_setup.load_dotenv"),
        patch("scripts.testnet_setup.fund_account"),
        patch("scripts.testnet_setup.deploy_contract", return_value="CDEPLOYEDCONTRACT123"),
    ):
        setup.run(
            wasm_path="dummy.wasm",
            skip_hash_check=True,
            env_file=env_file,
        )

    content = open(env_file).read()
    assert "LEDGERLENS_CONTRACT_ID=CDEPLOYEDCONTRACT123" in content
    assert "LEDGERLENS_SUBMITTER_SECRET=" in content


# ---------------------------------------------------------------------------
# 3. SHA-256 verification rejects bad hash
# ---------------------------------------------------------------------------


def test_wasm_sha256_verification_rejects_bad_hash(tmp_path):
    wasm_file = tmp_path / "bad.wasm"
    wasm_file.write_bytes(b"this is not the real wasm")

    wrong_hash = "0" * 64  # definitely wrong

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup.verify_wasm_hash(str(wasm_file), wrong_hash)


def test_wasm_sha256_verification_accepts_correct_hash(tmp_path):
    content = b"real wasm bytes"
    wasm_file = tmp_path / "good.wasm"
    wasm_file.write_bytes(content)
    correct_hash = hashlib.sha256(content).hexdigest()

    # Should not raise
    setup.verify_wasm_hash(str(wasm_file), correct_hash)


# ---------------------------------------------------------------------------
# 4. Friendbot retries on 429 and succeeds on second attempt
# ---------------------------------------------------------------------------


def test_friendbot_funding_retries_on_rate_limit():
    rate_limited = MagicMock()
    rate_limited.status_code = 429

    success = MagicMock()
    success.status_code = 200

    with (
        patch(
            "scripts.testnet_setup.requests.get", side_effect=[rate_limited, success]
        ) as mock_get,
        patch("scripts.testnet_setup.time.sleep") as mock_sleep,
    ):
        setup.fund_account("GPUBLICKEY")

    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(setup._RETRY_DELAY)


def test_friendbot_funding_raises_after_max_retries():
    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.raise_for_status.side_effect = Exception("429")

    with (
        patch("scripts.testnet_setup.requests.get", return_value=rate_limited),
        patch("scripts.testnet_setup.time.sleep"),
    ):
        with pytest.raises(Exception):  # noqa: B017
            setup.fund_account("GPUBLICKEY")
