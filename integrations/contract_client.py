"""Client for the `ledgerlens-score` Soroban contract.

Wraps `stellar_sdk.contract.ContractClient` to call the two functions
documented in the README's "Shared Contracts" section:

  - `submit_score(wallet, asset_pair, score, benford_flag, ml_flag,
    timestamp, confidence)` — writes a `RiskScore` record on-chain. Requires
    `LEDGERLENS_SUBMITTER_SECRET` (an authorized service-account secret key).
    - `submit_score_with_commitment(...)` — writes the same score plus a
        deterministic commitment and attestation metadata.
  - `get_score(wallet, asset_pair)` — permissionless read of the on-chain
    `RiskScore`.

`wallet` is a Stellar account ID (`G...`), `asset_pair` is the
`CODE:ISSUER/CODE:ISSUER` string from `ingestion.data_models.Asset.pair_id`.
"""

from stellar_sdk import Keypair, Network, scval
from stellar_sdk.contract import ContractClient

from config import config

_NETWORK_PASSPHRASES = {
    "PUBLIC": Network.PUBLIC_NETWORK_PASSPHRASE,
    "TESTNET": Network.TESTNET_NETWORK_PASSPHRASE,
}


class LedgerLensContractClient:
    """Thin wrapper around the `ledgerlens-score` contract's invocations."""

    def __init__(
        self,
        contract_id: str | None = None,
        rpc_url: str | None = None,
        network_passphrase: str | None = None,
        submitter_secret: str | None = None,
    ):
        self.contract_id = contract_id or config.LEDGERLENS_CONTRACT_ID
        if not self.contract_id:
            raise ValueError("LEDGERLENS_CONTRACT_ID is not configured")

        self.rpc_url = rpc_url or config.SOROBAN_RPC_URL
        self.network_passphrase = network_passphrase or _NETWORK_PASSPHRASES.get(
            config.STELLAR_NETWORK, Network.TESTNET_NETWORK_PASSPHRASE
        )
        self.submitter_secret = submitter_secret or config.LEDGERLENS_SUBMITTER_SECRET

        self._client = ContractClient(
            contract_id=self.contract_id,
            rpc_url=self.rpc_url,
            network_passphrase=self.network_passphrase,
        )

    def submit_score(
        self,
        wallet: str,
        asset_pair: str,
        risk_score: dict,
        *,
        commitment: str | None = None,
        trade_data_hash: str | None = None,
        model_version_hash: str | None = None,
    ) -> object:
        """Submit a `RiskScore` record (the dict shape from `RiskScorer.score()`,
        plus an integer `timestamp`) for `(wallet, asset_pair)`.

        When `commitment` is provided, the client calls the attested contract
        entry point and includes the commitment metadata alongside the score.
        Returns the parsed contract result. Requires `LEDGERLENS_SUBMITTER_SECRET`.
        """
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")

        signer = Keypair.from_secret(self.submitter_secret)

        params = [
            scval.to_address(wallet),
            scval.to_string(asset_pair),
            scval.to_uint32(int(risk_score["score"])),
            scval.to_bool(bool(risk_score["benford_flag"])),
            scval.to_bool(bool(risk_score["ml_flag"])),
            scval.to_uint64(int(risk_score["timestamp"])),
            scval.to_uint32(int(risk_score["confidence"])),
        ]

        if commitment is None:
            tx = self._client.invoke(
                "submit_score",
                params,
                source=signer.public_key,
                signer=signer,
            )
        else:
            if trade_data_hash is None or model_version_hash is None:
                raise ValueError(
                    "trade_data_hash and model_version_hash are required when commitment is set"
                )
            params = params + [
                scval.to_string(commitment),
                scval.to_string(trade_data_hash),
                scval.to_string(model_version_hash),
            ]
            tx = self._client.invoke(
                "submit_score_with_commitment",
                params,
                source=signer.public_key,
                signer=signer,
            )
        return tx.sign_and_submit()

    def submit_score_with_commitment(
        self,
        wallet: str,
        asset_pair: str,
        risk_score: dict,
        commitment: str,
        trade_data_hash: str,
        model_version_hash: str,
    ) -> object:
        """Explicit attested-submit helper for callers that already built a receipt."""
        return self.submit_score(
            wallet,
            asset_pair,
            risk_score,
            commitment=commitment,
            trade_data_hash=trade_data_hash,
            model_version_hash=model_version_hash,
        )

    def get_score(self, wallet: str, asset_pair: str) -> dict:
        """Read the on-chain `RiskScore` for `(wallet, asset_pair)`."""
        params = [scval.to_address(wallet), scval.to_string(asset_pair)]
        tx = self._client.invoke("get_score", params, simulate=True)
        return scval.to_native(tx.result())
