"""Live integration tests for LedgerLensContractClient against Testnet.

Run with:
    LEDGERLENS_INTEGRATION_TESTS=1 pytest tests/integration/test_contract_client_live.py -v
"""

import time
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.integration

# Stable testnet wallet + pair used across all tests in this module.
_WALLET = "GBTESTWALLETLEDGERLENS0000000000000000000000000000000001"
_PAIR = "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVV/XLM:native"

_RETRIES = 3
_RETRY_DELAY = 5


def _with_retry(fn):
    """Call fn(), retrying up to _RETRIES times on exception."""
    last_exc = None
    for _ in range(_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(_RETRY_DELAY)
    raise last_exc  # type: ignore[misc]


def _ts() -> int:
    return int(datetime.now(UTC).timestamp())


@pytest.mark.timeout(120)
def test_submit_score_live(live_client):
    """submit_score succeeds without raising."""
    risk_score = {
        "score": 42,
        "benford_flag": False,
        "ml_flag": False,
        "timestamp": _ts(),
        "confidence": 50,
    }
    _with_retry(lambda: live_client.submit_score(_WALLET, _PAIR, risk_score))


@pytest.mark.timeout(120)
def test_get_score_live_after_submit(live_client):
    """get_score returns the values that were submitted."""
    risk_score = {
        "score": 55,
        "benford_flag": False,
        "ml_flag": True,
        "timestamp": _ts(),
        "confidence": 60,
    }
    _with_retry(lambda: live_client.submit_score(_WALLET, _PAIR, risk_score))

    result = _with_retry(lambda: live_client.get_score(_WALLET, _PAIR))

    assert result["score"] == 55
    assert result["ml_flag"] is True
    assert result["confidence"] == 60


@pytest.mark.timeout(120)
def test_submit_score_updates_existing(live_client):
    """Second submit overwrites the first; get_score returns latest values."""
    first = {
        "score": 30,
        "benford_flag": False,
        "ml_flag": False,
        "timestamp": _ts(),
        "confidence": 30,
    }
    second = {
        "score": 99,
        "benford_flag": True,
        "ml_flag": True,
        "timestamp": _ts(),
        "confidence": 95,
    }

    _with_retry(lambda: live_client.submit_score(_WALLET, _PAIR, first))
    _with_retry(lambda: live_client.submit_score(_WALLET, _PAIR, second))

    result = _with_retry(lambda: live_client.get_score(_WALLET, _PAIR))
    assert result["score"] == 99


@pytest.mark.timeout(120)
def test_score_benford_flag_persisted(live_client):
    """benford_flag=True round-trips correctly."""
    risk_score = {
        "score": 70,
        "benford_flag": True,
        "ml_flag": False,
        "timestamp": _ts(),
        "confidence": 70,
    }
    _with_retry(lambda: live_client.submit_score(_WALLET, _PAIR, risk_score))
    result = _with_retry(lambda: live_client.get_score(_WALLET, _PAIR))
    assert result["benford_flag"] is True


@pytest.mark.timeout(120)
def test_confidence_field_persisted(live_client):
    """confidence=87 round-trips without truncation."""
    risk_score = {
        "score": 65,
        "benford_flag": False,
        "ml_flag": False,
        "timestamp": _ts(),
        "confidence": 87,
    }
    _with_retry(lambda: live_client.submit_score(_WALLET, _PAIR, risk_score))
    result = _with_retry(lambda: live_client.get_score(_WALLET, _PAIR))
    assert result["confidence"] == 87


@pytest.mark.timeout(120)
def test_invalid_wallet_rejected(live_client):
    """Submitting a malformed wallet address raises an exception."""
    risk_score = {
        "score": 50,
        "benford_flag": False,
        "ml_flag": False,
        "timestamp": _ts(),
        "confidence": 50,
    }
    with pytest.raises(Exception):  # noqa: B017
        _with_retry(lambda: live_client.submit_score("NOT_A_VALID_WALLET", _PAIR, risk_score))
