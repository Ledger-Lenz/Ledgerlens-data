"""Tests for idempotent retry and nonce handling in contract_client.py (#951).

Test plan
---------
1. ``test_retry_after_timeout_single_onchain_effect`` — mock: first call times
   out, second succeeds; verify submit called twice but idempotent check
   prevents a double on-chain effect.
2. ``test_stale_sequence_number_recovery`` — mock: first call gets a
   ``tx_bad_seq`` error, re-fetches the sequence number, second call succeeds.
3. ``test_max_retries_exhausted_raises`` — after ``max_retries`` failures,
   ``MaxRetriesExhaustedError`` propagates.
4. ``test_already_submitted_idempotency`` — if transaction already landed
   (detected via ``_is_already_submitted``), do not resubmit.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from integrations.contract_client import (
    LedgerLensContractClient,
    MaxRetriesExhaustedError,
    TransactionAlreadySubmittedError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECRET = "SAUQSDM4BPSOWVJJM7RAHPSGXDX5YLRYNZCZ5QP33EVB6WDAAVJJRJHG"
_CONTRACT = "CCONTRACT12345"
_RPC_URL = "https://soroban-testnet.stellar.org"
_NETWORK = "Test SDF Network ; September 2015"


def make_client(**kwargs) -> LedgerLensContractClient:
    defaults = {
        "contract_id": _CONTRACT,
        "rpc_url": _RPC_URL,
        "network_passphrase": _NETWORK,
        "submitter_secret": _SECRET,
    }
    defaults.update(kwargs)
    with patch("integrations.contract_client.ContractClient"):
        return LedgerLensContractClient(**defaults)


def _fake_result(hash_val: str = "abc123") -> MagicMock:
    """Create a mock transaction result with a ``hash`` attribute."""
    result = MagicMock()
    result.hash = hash_val
    return result


# ---------------------------------------------------------------------------
# 1. Retry after timeout — single on-chain effect
# ---------------------------------------------------------------------------


class TestRetryAfterTimeoutSingleOnchainEffect:
    def test_second_attempt_succeeds_after_timeout(self):
        """First transaction_fn call raises TimeoutError; second succeeds."""
        client = make_client()
        ok_result = _fake_result("deadbeef")

        call_count = 0

        def _tx_fn(seq: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("connection timed out")
            return ok_result

        with (
            patch.object(client, "_get_current_sequence_number", return_value=12345),
            patch.object(client, "_is_already_submitted", return_value=False),
            patch("integrations.contract_client.time.sleep"),
        ):
            result = client._submit_with_retry(_tx_fn, max_retries=3, base_delay=0.01)

        assert result is ok_result
        assert call_count == 2

    def test_idempotency_check_called_after_success(self):
        """After a successful submission, _is_already_submitted is called."""
        client = make_client()
        ok_result = _fake_result("cafebabe")

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch.object(
                client, "_is_already_submitted", return_value=False
            ) as mock_idempotency,
            patch("integrations.contract_client.time.sleep"),
        ):
            client._submit_with_retry(lambda seq: ok_result)

        mock_idempotency.assert_called_once_with("cafebabe")

    def test_no_double_submission_when_idempotency_detects_already_landed(self):
        """If _is_already_submitted returns True after success, result is returned
        without re-submitting (the effect was already applied)."""
        client = make_client()
        ok_result = _fake_result("badf00d")
        call_count = 0

        def _tx_fn(seq: int) -> object:
            nonlocal call_count
            call_count += 1
            return ok_result

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch.object(client, "_is_already_submitted", return_value=True),
        ):
            result = client._submit_with_retry(_tx_fn)

        # Should have called _tx_fn exactly once and returned early
        assert call_count == 1
        assert result is ok_result

    def test_exponential_backoff_applied_between_retries(self):
        """Verify that time.sleep is called with increasing delays."""
        client = make_client()
        call_count = 0

        def _failing_tx(seq: int) -> object:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("network error")

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch("integrations.contract_client.time.sleep") as mock_sleep,
        ):
            with pytest.raises(MaxRetriesExhaustedError):
                client._submit_with_retry(_failing_tx, max_retries=3, base_delay=1.0)

        # Should have slept twice (after attempt 1 and 2; not after the final failure)
        assert mock_sleep.call_count == 2
        sleep_args = [c[0][0] for c in mock_sleep.call_args_list]
        # Exponential: 1.0, 2.0
        assert sleep_args[0] == pytest.approx(1.0)
        assert sleep_args[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 2. Stale sequence number recovery
# ---------------------------------------------------------------------------


class TestStaleSequenceNumberRecovery:
    def test_bad_seq_triggers_refetch_and_retry(self):
        """On tx_bad_seq, sequence number is re-fetched before next attempt."""
        client = make_client()
        ok_result = _fake_result("goodseq")
        call_count = 0
        seq_fetch_count = 0

        def _tx_fn(seq: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("tx_bad_seq: sequence number mismatch")
            return ok_result

        def _mock_get_seq(account_id: str) -> int:
            nonlocal seq_fetch_count
            seq_fetch_count += 1
            return 100 + seq_fetch_count  # returns a new value each call

        with (
            patch.object(client, "_get_current_sequence_number", side_effect=_mock_get_seq),
            patch.object(client, "_is_already_submitted", return_value=False),
            patch("integrations.contract_client.time.sleep"),
        ):
            result = client._submit_with_retry(_tx_fn, max_retries=3, base_delay=0.01)

        assert result is ok_result
        assert call_count == 2
        # Sequence should have been fetched at least twice (once before attempt 1,
        # once before attempt 2 after the bad_seq error)
        assert seq_fetch_count >= 2

    def test_bad_seq_no_backoff_sleep(self):
        """tx_bad_seq errors should NOT trigger exponential backoff sleep."""
        client = make_client()
        ok_result = _fake_result("fixed")
        call_count = 0

        def _tx_fn(seq: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("tx_bad_seq")
            return ok_result

        with (
            patch.object(client, "_get_current_sequence_number", return_value=42),
            patch.object(client, "_is_already_submitted", return_value=False),
            patch("integrations.contract_client.time.sleep") as mock_sleep,
        ):
            client._submit_with_retry(_tx_fn, max_retries=3, base_delay=1.0)

        # No sleep on bad_seq — immediate re-fetch and retry
        mock_sleep.assert_not_called()

    def test_sequence_number_passed_to_transaction_fn(self):
        """_submit_with_retry passes the fetched sequence number to transaction_fn."""
        client = make_client()
        received_seq: list[int] = []

        def _tx_fn(seq: int) -> object:
            received_seq.append(seq)
            return _fake_result()

        with (
            patch.object(client, "_get_current_sequence_number", return_value=9999),
            patch.object(client, "_is_already_submitted", return_value=False),
        ):
            client._submit_with_retry(_tx_fn)

        assert received_seq == [9999]


# ---------------------------------------------------------------------------
# 3. Max retries exhausted
# ---------------------------------------------------------------------------


class TestMaxRetriesExhausted:
    def test_raises_after_max_retries(self):
        """After max_retries consecutive failures, MaxRetriesExhaustedError is raised."""
        client = make_client()
        call_count = 0

        def _always_fail(seq: int) -> object:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("permanent failure")

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch("integrations.contract_client.time.sleep"),
        ):
            with pytest.raises(MaxRetriesExhaustedError):
                client._submit_with_retry(_always_fail, max_retries=3, base_delay=0.01)

        assert call_count == 3

    def test_error_message_mentions_last_exception(self):
        client = make_client()

        def _always_fail(seq: int) -> object:
            raise ValueError("something went wrong")

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch("integrations.contract_client.time.sleep"),
        ):
            with pytest.raises(MaxRetriesExhaustedError) as exc_info:
                client._submit_with_retry(_always_fail, max_retries=2, base_delay=0.01)

        assert "something went wrong" in str(exc_info.value)

    def test_max_retries_one_attempts_once(self):
        client = make_client()
        call_count = 0

        def _fail(seq: int) -> object:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("nope")

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch("integrations.contract_client.time.sleep"),
        ):
            with pytest.raises(MaxRetriesExhaustedError):
                client._submit_with_retry(_fail, max_retries=1, base_delay=0.01)

        assert call_count == 1


# ---------------------------------------------------------------------------
# 4. Already-submitted idempotency
# ---------------------------------------------------------------------------


class TestAlreadySubmittedIdempotency:
    def test_returns_existing_result_when_already_on_chain(self):
        """If _is_already_submitted returns True, result is returned without
        further submission attempts."""
        client = make_client()
        ok_result = _fake_result("existing_tx")
        call_count = 0

        def _tx_fn(seq: int) -> object:
            nonlocal call_count
            call_count += 1
            return ok_result

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch.object(client, "_is_already_submitted", return_value=True),
        ):
            result = client._submit_with_retry(_tx_fn, max_retries=3)

        assert result is ok_result
        # The tx_fn was called once (to get the result / hash), but not again
        assert call_count == 1

    def test_is_already_submitted_called_with_correct_hash(self):
        client = make_client()
        tx_hash = "specific_hash_abc"
        ok_result = _fake_result(tx_hash)

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch.object(
                client, "_is_already_submitted", return_value=False
            ) as mock_check,
        ):
            client._submit_with_retry(lambda seq: ok_result)

        mock_check.assert_called_once_with(tx_hash)

    def test_result_without_hash_attribute_does_not_crash(self):
        """Results without a hash attribute should not cause _submit_with_retry
        to crash — idempotency check is skipped."""
        client = make_client()
        no_hash_result = MagicMock(spec=[])  # No 'hash' attribute

        with (
            patch.object(client, "_get_current_sequence_number", return_value=1),
            patch.object(client, "_is_already_submitted") as mock_check,
        ):
            result = client._submit_with_retry(lambda seq: no_hash_result)

        # _is_already_submitted should not have been called (no hash to check)
        mock_check.assert_not_called()
        assert result is no_hash_result

    def test_submit_score_uses_retry(self):
        """submit_score() is expected to use _submit_with_retry internally."""
        client = make_client()
        ok_result = _fake_result()

        risk_score = {
            "score": 75,
            "benford_flag": True,
            "ml_flag": True,
            "timestamp": 1727544055,
            "confidence": 85,
        }

        with (
            patch.object(client, "_submit_with_retry", return_value=ok_result) as mock_retry,
        ):
            result = client.submit_score(
                "GBCTEST1234567890123456789012345678901234567890123456",
                "USDC:GABC/XLM:native",
                risk_score,
            )

        mock_retry.assert_called_once()
        assert result is ok_result


# ---------------------------------------------------------------------------
# 5. _get_current_sequence_number unit test
# ---------------------------------------------------------------------------


class TestGetCurrentSequenceNumber:
    def test_returns_parsed_sequence_from_horizon(self):
        client = make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"sequence": "12345678"}
        mock_response.raise_for_status = MagicMock()

        with patch("integrations.contract_client._requests.get", return_value=mock_response):
            seq = client._get_current_sequence_number("GTEST")

        assert seq == 12345678

    def test_raises_on_non_200_response(self):
        import requests as real_requests

        client = make_client()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = real_requests.HTTPError("404")

        with patch("integrations.contract_client._requests.get", return_value=mock_response):
            with pytest.raises(real_requests.HTTPError):
                client._get_current_sequence_number("GBADACCOUNT")
