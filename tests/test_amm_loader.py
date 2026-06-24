"""Unit tests for ingestion/amm_pool_loader.py.

All Horizon API responses are mocked — no live network calls.
"""

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ingestion.amm_pool_loader import (
    PoolNotFoundError,
    _validate_pool_id,
    list_active_pools,
    load_amm_pool_trades,
)

VALID_POOL_ID = "a" * 64
ANOTHER_POOL_ID = "b" * 64

SINCE = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
UNTIL = datetime.datetime(2024, 1, 31, tzinfo=datetime.UTC)

_SAMPLE_RECORD = {
    "id": "trade-001",
    "paging_token": "12345-0",
    "ledger_close_time": "2024-01-10T12:00:00Z",
    "base_account": "GBASE123",
    "counter_account": "GCOUNTER456",
    "base_asset_type": "credit_alphanum4",
    "base_asset_code": "USDC",
    "base_asset_issuer": "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
    "counter_asset_type": "native",
    "counter_asset_code": None,
    "counter_asset_issuer": None,
    "base_amount": "100.0000000",
    "counter_amount": "50.0000000",
    "price": {"n": "2", "d": "1"},
}


def _make_page(records, has_next=False):
    return {
        "_embedded": {"records": records},
        "_links": {
            "next": {"href": "http://next" if has_next else ""},
        },
    }


# ---------------------------------------------------------------------------
# Pool ID validation
# ---------------------------------------------------------------------------


def test_validate_pool_id_accepts_valid():
    _validate_pool_id(VALID_POOL_ID)


def test_validate_pool_id_rejects_short():
    with pytest.raises(ValueError):
        _validate_pool_id("abc123")


def test_validate_pool_id_rejects_uppercase():
    with pytest.raises(ValueError):
        _validate_pool_id("A" * 64)


def test_validate_pool_id_rejects_non_hex():
    with pytest.raises(ValueError):
        _validate_pool_id("z" * 64)


# ---------------------------------------------------------------------------
# load_amm_pool_trades — happy path: correct schema
# ---------------------------------------------------------------------------


def test_load_amm_pool_trades_returns_correct_schema(monkeypatch):
    page = _make_page([_SAMPLE_RECORD])
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = page
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_resp):
        df = load_amm_pool_trades(VALID_POOL_ID, SINCE, UNTIL)

    expected_cols = {
        "trade_id",
        "ledger_close_time",
        "base_account",
        "counter_account",
        "base_asset",
        "counter_asset",
        "amount",
        "price",
    }
    assert expected_cols.issubset(
        set(df.columns)
    ), f"Missing columns: {expected_cols - set(df.columns)}"
    assert len(df) == 1
    assert df.iloc[0]["base_account"] == "GBASE123"
    assert df.iloc[0]["counter_account"] == "GCOUNTER456"
    assert (
        df.iloc[0]["base_asset"] == "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
    )
    assert df.iloc[0]["counter_asset"] == "XLM:native"
    assert float(df.iloc[0]["amount"]) == pytest.approx(100.0)
    assert float(df.iloc[0]["price"]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# load_amm_pool_trades — empty response returns empty DataFrame (not None)
# ---------------------------------------------------------------------------


def test_load_amm_pool_trades_empty_response_returns_dataframe(monkeypatch):
    page = _make_page([])
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = page
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_resp):
        df = load_amm_pool_trades(VALID_POOL_ID, SINCE, UNTIL)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    expected_cols = {
        "trade_id",
        "ledger_close_time",
        "base_account",
        "counter_account",
        "base_asset",
        "counter_asset",
        "amount",
        "price",
    }
    assert expected_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# load_amm_pool_trades — HTTP 404 raises PoolNotFoundError (not generic)
# ---------------------------------------------------------------------------


def test_load_amm_pool_trades_404_raises_pool_not_found(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"status": 404}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_resp):
        with pytest.raises(PoolNotFoundError):
            load_amm_pool_trades(VALID_POOL_ID, SINCE, UNTIL)


def test_load_amm_pool_trades_404_not_generic_exception(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_resp):
        with pytest.raises(PoolNotFoundError):
            load_amm_pool_trades(VALID_POOL_ID, SINCE, UNTIL)

        # Verify it's NOT just a generic exception
        try:
            load_amm_pool_trades(VALID_POOL_ID, SINCE, UNTIL)
        except PoolNotFoundError:
            pass
        except Exception as e:
            pytest.fail(f"Expected PoolNotFoundError, got {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# load_amm_pool_trades — deduplication by paging_token
# ---------------------------------------------------------------------------


def test_load_amm_pool_trades_deduplicates_by_paging_token(monkeypatch):
    duplicate = dict(_SAMPLE_RECORD)
    page = _make_page([_SAMPLE_RECORD, duplicate])  # same paging_token twice
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = page
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_resp):
        df = load_amm_pool_trades(VALID_POOL_ID, SINCE, UNTIL)

    assert len(df) == 1


# ---------------------------------------------------------------------------
# load_amm_pool_trades — invalid pool ID raises ValueError (not API call)
# ---------------------------------------------------------------------------


def test_load_amm_pool_trades_invalid_pool_id_raises_value_error():
    with pytest.raises(ValueError):
        load_amm_pool_trades("not-a-valid-pool-id", SINCE, UNTIL)


# ---------------------------------------------------------------------------
# list_active_pools
# ---------------------------------------------------------------------------


def test_list_active_pools_returns_pool_ids(monkeypatch):
    records = [
        {"id": VALID_POOL_ID, "paging_token": "1"},
        {"id": ANOTHER_POOL_ID, "paging_token": "2"},
    ]
    page = {
        "_embedded": {"records": records},
        "_links": {"next": {"href": ""}},
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = page
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.Session.get", return_value=mock_resp):
        pool_ids = list_active_pools(
            "USDC", "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
        )

    assert VALID_POOL_ID in pool_ids
    assert ANOTHER_POOL_ID in pool_ids
