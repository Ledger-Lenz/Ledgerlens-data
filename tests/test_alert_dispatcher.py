"""Unit tests for AlertDispatcher (streaming/alert_dispatcher.py).

All seven required tests are present and run without live Horizon or model
artifacts — the dispatcher is isolated via mocks and capsys.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

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
