"""Unit tests for AlertDispatcher (streaming/alert_dispatcher.py).

All seven required tests are present and run without live Horizon or model
artifacts — the dispatcher is isolated via mocks and capsys.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from streaming.alert_dispatcher import AlertDispatcher

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

WALLET = "GABC1234567890EXAMPLEWALLETADDRESS"
PAIR_ID = "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"
ABOVE_THRESHOLD = {"score": 83, "benford_flag": True, "ml_flag": True, "confidence": 76}
BELOW_THRESHOLD = {"score": 50, "benford_flag": False, "ml_flag": False, "confidence": 30}
THRESHOLD = 70


# ---------------------------------------------------------------------------
# 1. stdout — above threshold
# ---------------------------------------------------------------------------


def test_dispatch_stdout_above_threshold(capsys):
    dispatcher = AlertDispatcher(channel="stdout", threshold=THRESHOLD)
    dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

    out = capsys.readouterr().out
    assert "[ALERT]" in out
    assert WALLET in out
    assert "score=83" in out
    assert "benford=True" in out
    assert "ml=True" in out
    assert "confidence=76" in out


# ---------------------------------------------------------------------------
# 2. stdout — below threshold, nothing printed
# ---------------------------------------------------------------------------


def test_dispatch_suppressed_below_threshold(capsys):
    dispatcher = AlertDispatcher(channel="stdout", threshold=THRESHOLD)
    dispatcher.dispatch(WALLET, BELOW_THRESHOLD, PAIR_ID)

    out = capsys.readouterr().out
    assert out == ""


# ---------------------------------------------------------------------------
# 3. Dedup — second dispatch within cooldown is swallowed
# ---------------------------------------------------------------------------


def test_dedup_within_cooldown():
    dispatcher = AlertDispatcher(channel="stdout", threshold=THRESHOLD, alert_cooldown_seconds=3600)
    with patch.object(dispatcher, "_deliver") as mock_deliver:
        dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)
        dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

    mock_deliver.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Dedup — second dispatch fires after cooldown expires
# ---------------------------------------------------------------------------


def test_dedup_allows_after_cooldown_expires():
    with patch("streaming.alert_dispatcher.time") as mock_time:
        mock_time.time.return_value = 1000.0
        dispatcher = AlertDispatcher(
            channel="stdout", threshold=THRESHOLD, alert_cooldown_seconds=3600
        )
        with patch.object(dispatcher, "_deliver") as mock_deliver:
            dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)
            # Advance time past the cooldown window (1000 + 3600 = 4600)
            mock_time.time.return_value = 4601.0
            dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

        assert mock_deliver.call_count == 2


# ---------------------------------------------------------------------------
# 5. Webhook — http:// URL rejected at construction
# ---------------------------------------------------------------------------


def test_webhook_rejects_http_url():
    with pytest.raises(ValueError, match="https://"):
        AlertDispatcher(channel="webhook", webhook_url="http://example.com")


# ---------------------------------------------------------------------------
# 6. Webhook — correct payload posted to HTTPS endpoint
# ---------------------------------------------------------------------------


def test_webhook_posts_correct_payload():
    with patch("streaming.alert_dispatcher.requests") as mock_requests:
        mock_response = MagicMock()
        mock_requests.post.return_value = mock_response

        dispatcher = AlertDispatcher(
            channel="webhook",
            webhook_url="https://hooks.example.com/alert",
            threshold=THRESHOLD,
        )
        dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

    mock_requests.post.assert_called_once()
    call_args = mock_requests.post.call_args
    assert call_args[0][0] == "https://hooks.example.com/alert"

    payload = call_args[1]["json"]
    assert payload["wallet"] == WALLET
    assert payload["score"] == 83
    assert payload["benford_flag"] is True
    assert payload["ml_flag"] is True
    assert payload["pair_id"] == PAIR_ID


# ---------------------------------------------------------------------------
# 7. WebSocket — injected ws_client.send() called with valid JSON
# ---------------------------------------------------------------------------


def test_websocket_channel_calls_send():
    ws_client = MagicMock()
    dispatcher = AlertDispatcher(channel="websocket", ws_client=ws_client, threshold=THRESHOLD)
    dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

    ws_client.send.assert_called_once()
    raw_message = ws_client.send.call_args[0][0]
    sent = json.loads(raw_message)

    assert sent["wallet"] == WALLET
    assert sent["score"] == 83
    assert sent["benford_flag"] is True
    assert sent["ml_flag"] is True
    assert sent["pair_id"] == PAIR_ID
    assert "confidence" in sent


# ---------------------------------------------------------------------------
# 8. Webhook retry — mock requests.post to return 500 twice then 200
# ---------------------------------------------------------------------------


def test_webhook_retries_on_500(tmp_path):
    dlq_file = tmp_path / "test_dlq.ndjson"
    with patch("streaming.alert_dispatcher.config") as mock_config, \
         patch("streaming.alert_dispatcher.requests") as mock_requests, \
         patch("streaming.alert_dispatcher.time.sleep") as mock_sleep:

        mock_config.ALERT_DEAD_LETTER_PATH = str(dlq_file)

        # mock responses: two 500s then a 200
        mock_resp_500_1 = MagicMock()
        mock_resp_500_1.status_code = 500
        mock_resp_500_1.raise_for_status.side_effect = requests.HTTPError("500 Server Error", response=mock_resp_500_1)

        mock_resp_500_2 = MagicMock()
        mock_resp_500_2.status_code = 500
        mock_resp_500_2.raise_for_status.side_effect = requests.HTTPError("500 Server Error", response=mock_resp_500_2)

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.raise_for_status.side_effect = None

        mock_requests.post.side_effect = [mock_resp_500_1, mock_resp_500_2, mock_resp_200]

        dispatcher = AlertDispatcher(
            channel="webhook",
            webhook_url="https://hooks.example.com/alert",
            threshold=THRESHOLD,
            base_delay=0.1
        )
        dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

        # It should call post 3 times
        assert mock_requests.post.call_count == 3
        # It should sleep twice
        assert mock_sleep.call_count == 2
        # Since it succeeded on 3rd attempt, DLQ file should not exist
        assert not dlq_file.exists()


# ---------------------------------------------------------------------------
# 9. Webhook retry — HTTP 4xx does not retry, gets written to DLQ
# ---------------------------------------------------------------------------


def test_webhook_no_retry_on_400(tmp_path):
    dlq_file = tmp_path / "test_dlq.ndjson"
    with patch("streaming.alert_dispatcher.config") as mock_config, \
         patch("streaming.alert_dispatcher.requests") as mock_requests, \
         patch("streaming.alert_dispatcher.time.sleep") as mock_sleep:

        mock_config.ALERT_DEAD_LETTER_PATH = str(dlq_file)

        mock_resp_400 = MagicMock()
        mock_resp_400.status_code = 400
        mock_resp_400.raise_for_status.side_effect = requests.HTTPError("400 Bad Request", response=mock_resp_400)

        mock_requests.post.side_effect = [mock_resp_400]

        dispatcher = AlertDispatcher(
            channel="webhook",
            webhook_url="https://hooks.example.com/alert",
            threshold=THRESHOLD,
            base_delay=0.1
        )
        dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

        # It should call post once (no retry)
        assert mock_requests.post.call_count == 1
        # It should not sleep
        assert mock_sleep.call_count == 0
        # DLQ file should contain the failed alert payload
        assert dlq_file.exists()
        with open(dlq_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        alert = json.loads(lines[0])
        assert alert["wallet"] == WALLET
        assert alert["score"] == ABOVE_THRESHOLD["score"]


# ---------------------------------------------------------------------------
# 10. Webhook retry — retries on network/connection errors
# ---------------------------------------------------------------------------


def test_webhook_retries_on_connection_error(tmp_path):
    dlq_file = tmp_path / "test_dlq.ndjson"
    with patch("streaming.alert_dispatcher.config") as mock_config, \
         patch("streaming.alert_dispatcher.requests") as mock_requests, \
         patch("streaming.alert_dispatcher.time.sleep") as mock_sleep:

        mock_config.ALERT_DEAD_LETTER_PATH = str(dlq_file)

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.raise_for_status.side_effect = None

        mock_requests.post.side_effect = [
            requests.exceptions.ConnectionError("Connection timed out"),
            mock_resp_200
        ]

        dispatcher = AlertDispatcher(
            channel="webhook",
            webhook_url="https://hooks.example.com/alert",
            threshold=THRESHOLD,
            base_delay=0.1
        )
        dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

        # It should call post 2 times
        assert mock_requests.post.call_count == 2
        # It should sleep once
        assert mock_sleep.call_count == 1
        assert not dlq_file.exists()


# ---------------------------------------------------------------------------
# 11. Webhook retry — exhausts all 3 retries and writes to DLQ
# ---------------------------------------------------------------------------


def test_webhook_exhausts_retries_and_writes_to_dlq(tmp_path):
    dlq_file = tmp_path / "test_dlq.ndjson"
    with patch("streaming.alert_dispatcher.config") as mock_config, \
         patch("streaming.alert_dispatcher.requests") as mock_requests, \
         patch("streaming.alert_dispatcher.time.sleep") as mock_sleep:

        mock_config.ALERT_DEAD_LETTER_PATH = str(dlq_file)

        # 4 failures (1 initial + 3 retries)
        failures = []
        for _ in range(4):
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error", response=mock_resp)
            failures.append(mock_resp)

        mock_requests.post.side_effect = failures

        dispatcher = AlertDispatcher(
            channel="webhook",
            webhook_url="https://hooks.example.com/alert",
            threshold=THRESHOLD,
            max_retries=3,
            base_delay=0.1
        )
        dispatcher.dispatch(WALLET, ABOVE_THRESHOLD, PAIR_ID)

        # It should call post 4 times
        assert mock_requests.post.call_count == 4
        # It should sleep 3 times
        assert mock_sleep.call_count == 3
        # DLQ file should contain the failed alert payload
        assert dlq_file.exists()
        with open(dlq_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        alert = json.loads(lines[0])
        assert alert["wallet"] == WALLET
