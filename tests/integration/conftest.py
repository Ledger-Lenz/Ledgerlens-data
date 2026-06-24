"""Integration test fixtures.

All integration tests are skipped unless LEDGERLENS_INTEGRATION_TESTS=1 so
that `make test` never hits the Testnet.  Environment variables (or a
.env.testnet file) must supply:

    LEDGERLENS_CONTRACT_ID    — deployed testnet contract
    LEDGERLENS_SUBMITTER_SECRET — funded testnet keypair
    SOROBAN_RPC_URL           — defaults to https://soroban-testnet.stellar.org
"""

import os

import pytest
from dotenv import load_dotenv

# Load .env.testnet if present so local runs don't need manual export.
load_dotenv(".env.testnet")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if os.getenv("LEDGERLENS_INTEGRATION_TESTS") != "1":
        skip = pytest.mark.skip(reason="Set LEDGERLENS_INTEGRATION_TESTS=1 to run")
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def contract_id() -> str:
    value = os.environ.get("LEDGERLENS_CONTRACT_ID", "")
    if not value:
        pytest.skip("LEDGERLENS_CONTRACT_ID not set")
    return value


@pytest.fixture(scope="session")
def submitter_secret() -> str:
    value = os.environ.get("LEDGERLENS_SUBMITTER_SECRET", "")
    if not value:
        pytest.skip("LEDGERLENS_SUBMITTER_SECRET not set")
    return value


@pytest.fixture(scope="session")
def rpc_url() -> str:
    return os.environ.get("SOROBAN_RPC_URL", "https://soroban-testnet.stellar.org")


@pytest.fixture(scope="session")
def live_client(contract_id, submitter_secret, rpc_url):
    """A real LedgerLensContractClient pointed at Testnet."""
    from stellar_sdk import Network

    from integrations.contract_client import LedgerLensContractClient

    return LedgerLensContractClient(
        contract_id=contract_id,
        rpc_url=rpc_url,
        network_passphrase=Network.TESTNET_NETWORK_PASSPHRASE,
        submitter_secret=submitter_secret,
    )
