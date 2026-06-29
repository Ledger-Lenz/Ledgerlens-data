"""Testnet account and contract setup for LedgerLens integration testing.

Usage:
    python -m scripts.testnet_setup --wasm-path ledgerlens_score.wasm [options]

Steps:
    1. Fund submitter account via Horizon Testnet Friendbot.
    2. Verify WASM SHA-256 hash against --wasm-sha256 (required unless --skip-hash-check).
    3. Deploy ledgerlens-score contract (skip if LEDGERLENS_CONTRACT_ID already set).
    4. Write LEDGERLENS_CONTRACT_ID and LEDGERLENS_SUBMITTER_SECRET to .env.testnet.
    5. Print the contract ID to stdout.
"""

import argparse
import hashlib
import os
import sys
import time

import requests
from dotenv import load_dotenv
from stellar_sdk import Keypair, Network
from stellar_sdk.contract import ContractClient

FRIENDBOT_URL = "https://friendbot.stellar.org"
TESTNET_RPC_URL = "https://soroban-testnet.stellar.org"
TESTNET_PASSPHRASE = Network.TESTNET_NETWORK_PASSPHRASE
_MAX_FRIENDBOT_RETRIES = 3
_RETRY_DELAY = 5


def fund_account(public_key: str) -> None:
    """Fund `public_key` via Friendbot with up to 3 retries on 429."""
    for attempt in range(1, _MAX_FRIENDBOT_RETRIES + 1):
        resp = requests.get(FRIENDBOT_URL, params={"addr": public_key}, timeout=30)
        if resp.status_code == 200:
            return
        if resp.status_code == 429 and attempt < _MAX_FRIENDBOT_RETRIES:
            print(
                f"Friendbot rate-limited (429), retrying in {_RETRY_DELAY}s "
                f"(attempt {attempt}/{_MAX_FRIENDBOT_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(_RETRY_DELAY)
            continue
        resp.raise_for_status()


def verify_wasm_hash(wasm_path: str, expected_sha256: str) -> None:
    """Raise ValueError if the WASM file's SHA-256 doesn't match `expected_sha256`."""
    with open(wasm_path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {wasm_path}: expected {expected_sha256}, got {actual}"
        )


def deploy_contract(
    wasm_path: str,
    submitter_secret: str,
    salt: str | None = None,
    rpc_url: str = TESTNET_RPC_URL,
    network_passphrase: str = TESTNET_PASSPHRASE,
) -> str:
    """Upload WASM and instantiate the contract. Returns the contract ID."""
    keypair = Keypair.from_secret(submitter_secret)
    client = ContractClient.from_wasm_file(
        wasm_path,
        rpc_url=rpc_url,
        network_passphrase=network_passphrase,
        source=keypair.public_key,
        signer=keypair,
        salt=salt.encode() if salt else None,
    )
    return str(client.contract_id)


def write_env_file(contract_id: str, submitter_secret: str, path: str = ".env.testnet") -> None:
    """Write LEDGERLENS_CONTRACT_ID and LEDGERLENS_SUBMITTER_SECRET to `path`."""
    with open(path, "w") as f:
        f.write(f"LEDGERLENS_CONTRACT_ID={contract_id}\n")
        f.write(f"LEDGERLENS_SUBMITTER_SECRET={submitter_secret}\n")


def run(
    wasm_path: str,
    expected_sha256: str | None = None,
    salt: str | None = None,
    env_file: str = ".env.testnet",
    rpc_url: str = TESTNET_RPC_URL,
    network_passphrase: str = TESTNET_PASSPHRASE,
    skip_hash_check: bool = False,
) -> str:
    """Run the full setup flow. Returns the contract ID."""
    load_dotenv()

    # 1. Resolve or generate the submitter keypair
    submitter_secret = os.getenv("LEDGERLENS_SUBMITTER_SECRET", "")
    if submitter_secret:
        keypair = Keypair.from_secret(submitter_secret)
    else:
        keypair = Keypair.random()
        submitter_secret = keypair.secret
        print(f"Generated fresh keypair: {keypair.public_key}", file=sys.stderr)

    # 2. Fund via Friendbot
    print(f"Funding {keypair.public_key} via Friendbot...", file=sys.stderr)
    fund_account(keypair.public_key)

    # 3. Check if contract is already deployed
    contract_id = os.getenv("LEDGERLENS_CONTRACT_ID", "")
    if contract_id:
        print(
            f"LEDGERLENS_CONTRACT_ID already set ({contract_id}), skipping deployment.",
            file=sys.stderr,
        )
    else:
        # 4. Verify WASM hash
        if not skip_hash_check:
            if not expected_sha256:
                raise ValueError("--wasm-sha256 is required unless --skip-hash-check is passed")
            verify_wasm_hash(wasm_path, expected_sha256)

        # 5. Deploy contract
        print(f"Deploying contract from {wasm_path}...", file=sys.stderr)
        contract_id = deploy_contract(
            wasm_path=wasm_path,
            submitter_secret=submitter_secret,
            salt=salt,
            rpc_url=rpc_url,
            network_passphrase=network_passphrase,
        )

    # 6. Write .env.testnet
    write_env_file(contract_id, submitter_secret, path=env_file)
    print(f"Written {env_file}", file=sys.stderr)

    # 7. Print contract ID to stdout (for CI capture)
    print(contract_id)
    return contract_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up testnet account and deploy contract")
    parser.add_argument(
        "--wasm-path",
        default="ledgerlens_score.wasm",
        help="Path to the pre-compiled ledgerlens-score WASM file",
    )
    parser.add_argument(
        "--wasm-sha256", default=None, help="Expected SHA-256 hash of the WASM file (hex string)"
    )
    parser.add_argument(
        "--salt",
        default=None,
        help="Deployment salt for deterministic contract ID (e.g. 'ci-testnet')",
    )
    parser.add_argument(
        "--env-file",
        default=".env.testnet",
        help="Path to write environment variables to (default: .env.testnet)",
    )
    parser.add_argument("--rpc-url", default=TESTNET_RPC_URL, help="Soroban RPC endpoint")
    parser.add_argument(
        "--network-passphrase", default=TESTNET_PASSPHRASE, help="Stellar network passphrase"
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip WASM SHA-256 verification (not recommended for production)",
    )
    args = parser.parse_args()

    run(
        wasm_path=args.wasm_path,
        expected_sha256=args.wasm_sha256,
        salt=args.salt,
        env_file=args.env_file,
        rpc_url=args.rpc_url,
        network_passphrase=args.network_passphrase,
        skip_hash_check=args.skip_hash_check,
    )


if __name__ == "__main__":
    main()
