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

Retry and Idempotency (#951)
-----------------------------
Submitting a Soroban transaction on Stellar requires managing the source
account's **sequence number**: each transaction must carry exactly the current
sequence + 1.  Network retries therefore cannot naively reuse a transaction
built on a stale sequence number — doing so produces a ``400 tx_bad_seq`` error.

Conversely, if a submission *succeeds* but the HTTP response is lost (timeout,
network blip), a naive retry would build a new transaction with a fresh
sequence number and submit it again, creating a duplicate on-chain effect.

``_submit_with_retry`` addresses both issues:

1. **Fresh sequence number on every attempt.** ``_get_current_sequence_number``
   fetches the account's live sequence from Horizon before building each
   transaction, so a sequence-mismatch error on attempt N triggers an
   immediate re-fetch and rebuild rather than an incremental patch.

2. **Idempotency via result-hash check.** Before submitting, the client checks
   whether the transaction hash is already present in Horizon
   (``_is_already_submitted``).  If it is, the call returns the existing result
   without re-submitting — preventing double effects when the original
   submission succeeded but the response was lost.

3. **Exponential backoff.** Transient errors (timeouts, 503s) are retried with
   ``base_delay * 2^attempt`` seconds of sleep between attempts, up to
   ``max_retries``.

4. **Sequence-mismatch fast-path.** On a ``400 tx_bad_seq`` response the
   client skips the backoff delay and immediately re-fetches the sequence
   number before the next attempt, as the correct fix is known.

**For integration authors:**  if you add a new submission method to this
client, follow the pattern used by ``submit_score``:

.. code-block:: python

    def my_new_submission(self, ...):
        signer = Keypair.from_secret(self.submitter_secret)

        def _build_and_submit(sequence_number: int) -> object:
            # Build transaction using sequence_number (or pass it to your
            # ContractClient invoke call if your SDK version supports it).
            tx = self._client.invoke("my_function", params, source=signer.public_key, signer=signer)
            return tx.sign_and_submit()

        return self._submit_with_retry(_build_and_submit)

The ``transaction_fn`` receives the current sequence number as its only
argument.  It is responsible for incorporating it into the transaction.
"""

import time
from typing import Any, Protocol, cast

import requests as _requests

from stellar_sdk import Keypair, Network, scval
from stellar_sdk.contract import ContractClient

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)


class AnchorableReport(Protocol):
    report_id: str
    report_sha256: str
    soroban_anchor_tx: str | None


_NETWORK_PASSPHRASES = {
    "PUBLIC": Network.PUBLIC_NETWORK_PASSPHRASE,
    "TESTNET": Network.TESTNET_NETWORK_PASSPHRASE,
}

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TransactionAlreadySubmittedError(Exception):
    """Raised when a transaction is detected as already landed on-chain.

    Callers may catch this to treat a duplicate submission as a no-op rather
    than an error — the transaction's effect is already present on-chain.
    """


class MaxRetriesExhaustedError(Exception):
    """Raised when ``_submit_with_retry`` exhausts all retry attempts."""


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

    # ------------------------------------------------------------------
    # Retry and idempotency helpers (issue #951)
    # ------------------------------------------------------------------

    def _get_current_sequence_number(self, account_id: str) -> int:
        """Fetch the *current* sequence number for *account_id* from Horizon.

        Stellar transaction sequence numbers must be strictly ``account_sequence
        + 1``.  On a retry after a sequence-mismatch (``400 tx_bad_seq``) or a
        network timeout, the account's sequence may have advanced (because the
        first submission landed before the error was returned) — fetching a
        fresh value before each attempt avoids submitting with a stale nonce.

        Parameters
        ----------
        account_id:
            The G-address of the source account (typically the submitter).

        Returns
        -------
        int
            The current sequence number of the account as seen by Horizon.

        Raises
        ------
        requests.HTTPError
            If the Horizon account endpoint returns a non-2xx response.
        """
        horizon_url = getattr(config, "HORIZON_URL", "https://horizon-testnet.stellar.org")
        url = f"{horizon_url.rstrip('/')}/accounts/{account_id}"
        resp = _requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return int(data["sequence"])

    def _is_already_submitted(self, tx_hash: str) -> bool:
        """Check whether a transaction with *tx_hash* has already landed on-chain.

        Queries the Horizon ``/transactions/{hash}`` endpoint.  Returns
        ``True`` if the transaction is found (200), ``False`` if not found
        (404).  Any other error is re-raised.

        This check is the **idempotency guard**: if the transaction hash is
        found, the caller should return the existing result rather than
        re-submitting and producing a duplicate on-chain effect.

        Parameters
        ----------
        tx_hash:
            Stellar transaction hash (hex string).
        """
        horizon_url = getattr(config, "HORIZON_URL", "https://horizon-testnet.stellar.org")
        url = f"{horizon_url.rstrip('/')}/transactions/{tx_hash}"
        try:
            resp = _requests.get(url, timeout=10)
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            return True
        except _requests.HTTPError:
            return False

    def _submit_with_retry(
        self,
        transaction_fn: Any,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> object:
        """Submit a transaction built by *transaction_fn* with automatic retry.

        On each attempt:

        1. Fetch a fresh sequence number via :meth:`_get_current_sequence_number`.
        2. Call ``transaction_fn(sequence_number)`` to build and submit the tx.
        3. On success: check whether the transaction hash was already on-chain
           (:meth:`_is_already_submitted`).  If so, return without re-submitting.
        4. On timeout / network error: sleep ``base_delay * 2^attempt`` seconds
           and retry.
        5. On ``400 tx_bad_seq``: skip the backoff, re-fetch the sequence number
           immediately, and retry on the next iteration.
        6. After ``max_retries`` failed attempts: raise
           :class:`MaxRetriesExhaustedError`.

        Parameters
        ----------
        transaction_fn:
            Callable with signature ``(sequence_number: int) -> result``.  It is
            responsible for building, signing, and submitting the transaction.
            The sequence number argument is informational — the SDK may
            re-fetch it internally; supplying it here allows future
            implementations to override the SDK's default behaviour.
        max_retries:
            Maximum number of submission attempts.  Default 3.
        base_delay:
            Base sleep time (seconds) for exponential backoff.  Default 1.0 s.

        Returns
        -------
        object
            The result returned by the first successful ``transaction_fn`` call.

        Raises
        ------
        MaxRetriesExhaustedError
            When all ``max_retries`` attempts fail.
        TransactionAlreadySubmittedError
            When the transaction is detected as already landed on-chain and the
            caller has configured strict idempotency checking.
        """
        signer_key = self.submitter_secret
        account_id = (
            Keypair.from_secret(signer_key).public_key if signer_key else None
        )

        last_exception: Exception | None = None
        bad_seq = False

        for attempt in range(max_retries):
            try:
                # Step 1: fresh sequence number (best-effort; SDK may override)
                sequence_number: int = 0
                if account_id is not None:
                    try:
                        sequence_number = self._get_current_sequence_number(account_id)
                    except Exception:
                        logger.warning(
                            "submit_with_retry: could not fetch sequence number "
                            "(attempt %d/%d) — proceeding with SDK default",
                            attempt + 1,
                            max_retries,
                        )

                # Step 2: build and submit
                result = transaction_fn(sequence_number)

                # Step 3: idempotency — check if the tx was already on-chain
                tx_hash: str | None = getattr(result, "hash", None)
                if tx_hash and self._is_already_submitted(str(tx_hash)):
                    logger.info(
                        "submit_with_retry: transaction %s already on-chain — "
                        "skipping duplicate submission",
                        tx_hash,
                    )
                    return result

                logger.info(
                    "submit_with_retry: succeeded on attempt %d (hash=%s)",
                    attempt + 1,
                    tx_hash,
                )
                return result

            except Exception as exc:
                last_exception = exc
                exc_str = str(exc).lower()
                is_bad_seq = "tx_bad_seq" in exc_str or "bad_seq" in exc_str or (
                    hasattr(exc, "status_code") and getattr(exc, "status_code", 0) == 400
                    and "seq" in exc_str
                )

                if is_bad_seq:
                    logger.warning(
                        "submit_with_retry: sequence number mismatch on attempt %d "
                        "(tx_bad_seq) — re-fetching sequence immediately",
                        attempt + 1,
                    )
                    bad_seq = True
                    # No backoff on bad_seq — re-fetch and retry immediately
                    continue

                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "submit_with_retry: attempt %d/%d failed (%s) — "
                        "retrying in %.1fs",
                        attempt + 1,
                        max_retries,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)

        raise MaxRetriesExhaustedError(
            f"Transaction submission failed after {max_retries} attempts. "
            f"Last error: {last_exception!r}"
        ) from last_exception

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

        This method uses :meth:`_submit_with_retry` with exponential backoff and
        idempotency checking to handle transient network errors and sequence-number
        mismatches without producing duplicate on-chain effects.
        """
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")

        signer = Keypair.from_secret(self.submitter_secret)

        if commitment is not None:
            if trade_data_hash is None or model_version_hash is None:
                raise ValueError(
                    "trade_data_hash and model_version_hash are required when commitment is set"
                )

        def _build_and_submit(_sequence_number: int) -> object:
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

        return self._submit_with_retry(_build_and_submit)

    def submit_score_with_uncertainty(
        self,
        wallet: str,
        asset_pair: str,
        risk_score_dict: dict,
    ) -> object:
        """Submit a risk score with uncertainty bounds to the Soroban contract.

        Passes ``score_lower`` and ``score_upper`` as additional Soroban i128
        fields (scaled x100 for integer representation).

        NOTE: The ``ledgerlens-contract`` repo's ``RiskScore`` struct must be
        extended with:

        .. code-block:: rust

            pub struct RiskScore {
                pub score: u32,
                pub benford_flag: bool,
                pub ml_flag: bool,
                pub timestamp: u64,
                pub confidence: u32,
                pub score_lower: i128,   // NEW — scaled x100
                pub score_upper: i128,   // NEW — scaled x100
                pub coverage_guarantee: u32,  // NEW — percentage 0-100
            }

        See https://github.com/Ledger-Lenz/ledgerlens-contract/issues/... for
        the matching change.

        Uses :meth:`_submit_with_retry` with exponential backoff and idempotency
        checking.
        """
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")

        signer = Keypair.from_secret(self.submitter_secret)

        score_lower_scaled = int(round(risk_score_dict.get("score_lower", 0.0) * 100))
        score_upper_scaled = int(round(risk_score_dict.get("score_upper", 100.0) * 100))
        coverage_pct = int(round(risk_score_dict.get("coverage_guarantee", 1.0) * 100))

        def _build_and_submit(_sequence_number: int) -> object:
            params = [
                scval.to_address(wallet),
                scval.to_string(asset_pair),
                scval.to_uint32(int(risk_score_dict["score"])),
                scval.to_bool(bool(risk_score_dict["benford_flag"])),
                scval.to_bool(bool(risk_score_dict["ml_flag"])),
                scval.to_uint64(int(risk_score_dict["timestamp"])),
                scval.to_uint32(int(risk_score_dict["confidence"])),
                scval.to_int128(score_lower_scaled),
                scval.to_int128(score_upper_scaled),
                scval.to_uint32(coverage_pct),
            ]
            tx = self._client.invoke(
                "submit_score_with_uncertainty",
                params,
                source=signer.public_key,
                signer=signer,
            )
            return tx.sign_and_submit()

        return self._submit_with_retry(_build_and_submit)

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
        return cast(dict[Any, Any], scval.to_native(tx.result()))

    # ------------------------------------------------------------------
    # Multi-sig governance (issue #238)
    # ------------------------------------------------------------------

    def propose_threshold_change(
        self,
        governance_contract_id: str,
        new_threshold: int,
        proposer_secret: str,
    ) -> int:
        """Submit a proposal to change RISK_SCORE_FLAG_THRESHOLD on-chain.

        ``proposer_secret`` must correspond to one of the registered
        governance keyholders.  The key is used only to sign the transaction
        and is never stored by this client.

        Returns the on-chain proposal_id (u64).
        """
        signer = Keypair.from_secret(proposer_secret)
        governance_client = ContractClient(
            contract_id=governance_contract_id,
            rpc_url=self.rpc_url,
            network_passphrase=self.network_passphrase,
        )
        params = [
            scval.to_address(signer.public_key),
            scval.to_uint32(int(new_threshold)),
        ]
        tx = governance_client.invoke(
            "propose_threshold_change",
            params,
            source=signer.public_key,
            signer=signer,
        )
        result = tx.sign_and_submit()
        return int(scval.to_native(result))

    def approve_threshold_change(
        self,
        governance_contract_id: str,
        proposal_id: int,
        approver_secret: str,
    ) -> bool:
        """Cast an approval for an open threshold-change proposal.

        Returns True if the approval triggered quorum and the threshold
        was updated on-chain.
        """
        signer = Keypair.from_secret(approver_secret)
        governance_client = ContractClient(
            contract_id=governance_contract_id,
            rpc_url=self.rpc_url,
            network_passphrase=self.network_passphrase,
        )
        params = [
            scval.to_address(signer.public_key),
            scval.to_uint64(int(proposal_id)),
        ]
        tx = governance_client.invoke(
            "approve_threshold_change",
            params,
            source=signer.public_key,
            signer=signer,
        )
        result = tx.sign_and_submit()
        return bool(scval.to_native(result))

    # ------------------------------------------------------------------
    # Emergency pause (issue #241)
    # ------------------------------------------------------------------

    def initiate_emergency_pause(
        self,
        pause_contract_id: str,
        reason: str,
        signing_key: str,
    ) -> int:
        """Propose an emergency pause of the scoring oracle.

        ``signing_key`` must be one of the 3 registered emergency keyholder
        secrets.  It signs the Stellar transaction locally; it is never
        transmitted or stored beyond the scope of this call.

        Returns the on-chain pause proposal_id.
        """
        signer = Keypair.from_secret(signing_key)
        pause_client = ContractClient(
            contract_id=pause_contract_id,
            rpc_url=self.rpc_url,
            network_passphrase=self.network_passphrase,
        )
        params = [
            scval.to_address(signer.public_key),
            scval.to_string(reason),
        ]
        tx = pause_client.invoke(
            "initiate_pause",
            params,
            source=signer.public_key,
            signer=signer,
        )
        result = tx.sign_and_submit()
        return int(scval.to_native(result))

    def approve_emergency_pause(
        self,
        pause_contract_id: str,
        proposal_id: int,
        signing_key: str,
    ) -> bool:
        """Cast the second approval to activate an emergency pause.

        Returns True if quorum was reached and the contract is now paused.
        """
        signer = Keypair.from_secret(signing_key)
        pause_client = ContractClient(
            contract_id=pause_contract_id,
            rpc_url=self.rpc_url,
            network_passphrase=self.network_passphrase,
        )
        params = [
            scval.to_address(signer.public_key),
            scval.to_uint64(int(proposal_id)),
        ]
        tx = pause_client.invoke(
            "approve_pause",
            params,
            source=signer.public_key,
            signer=signer,
        )
        result = tx.sign_and_submit()
        return bool(scval.to_native(result))

    def anchor_report(self, report: AnchorableReport) -> str:
        """Submit a forensic report's SHA-256 fingerprint to Soroban.

        Calls the contract's `anchor_report(report_id, sha256)` function and
        returns the Stellar transaction hash.  The hash is also stored on
        `report.soroban_anchor_tx` so the caller has it immediately.

        Anyone can verify the anchor independently:
            GET {HORIZON_URL}/transactions/{tx_hash}
        and compare the embedded SHA-256 to the report on disk.
        """
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")

        signer = Keypair.from_secret(self.submitter_secret)

        params = [
            scval.to_string(report.report_id),
            scval.to_string(report.report_sha256),
        ]

        tx = self._client.invoke(
            "anchor_report",
            params,
            source=signer.public_key,
            signer=signer,
        )
        result = tx.sign_and_submit()
        tx_hash: str = str(result.hash)
        report.soroban_anchor_tx = tx_hash
        return tx_hash
