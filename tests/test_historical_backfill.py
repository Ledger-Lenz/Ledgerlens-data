"""Resumable, chunked historical backfill (issue #915).

Horizon is replaced by an in-memory fake that honours the call builder's
``cursor`` exactly like the real paginated trades endpoint, so resumption is
exercised through the same paging-token path production uses.
"""

import json
from unittest.mock import patch

import pandas as pd
import pytest
from stellar_sdk import Asset as SdkAsset

from cli.main import main as cli_main
from ingestion.historical_loader import (
    backfill_status,
    backfill_trades,
    parquet_chunk_sink,
)

VALID_BASE = "GCGPQMCLRXCUPCL3AVMYUUQML2WVC7A5M6HO5RKYSU4CIA7O7SI4VKWE"
VALID_COUNTER = "GB2HHLFDCBSBDAMU2QRDU4AJV63WQE2DWT7MBZWZRQDFYUXJXIPPUG7M"
VALID_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
BASE_ASSET = SdkAsset("USDC", VALID_ISSUER)
COUNTER_ASSET = SdkAsset.native()


def _record(i: int) -> dict:
    return {
        "id": f"t{i:03d}",
        "paging_token": f"t{i:03d}",
        "ledger_close_time": f"2024-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
        "base_account": VALID_BASE,
        "counter_account": VALID_COUNTER,
        "base_asset_code": "USDC",
        "base_asset_issuer": VALID_ISSUER,
        "counter_asset_code": "",
        "counter_asset_issuer": None,
        "base_amount": "100.0",
        "counter_amount": "50.0",
        "price": {"n": 1, "d": 2},
    }


class FakeHorizon:
    """Serves ``records`` page by page, starting after the builder's cursor."""

    def __init__(self, records: list[dict], crash_on_call: int | None = None):
        self.records = records
        self.crash_on_call = crash_on_call
        self.calls = 0

    def __call__(self, call_builder):
        self.calls += 1
        if self.calls == self.crash_on_call:
            raise RuntimeError("simulated crash mid-backfill")
        cursor = call_builder.params.get("cursor")
        tokens = [r["paging_token"] for r in self.records]
        start = tokens.index(cursor) + 1 if cursor else 0
        page = self.records[start : start + int(call_builder.params["limit"])]
        more = start + len(page) < len(self.records)
        return {
            "_embedded": {"records": page},
            "_links": {"next": {"href": "next" if more else ""}},
        }


def _run(horizon: FakeHorizon, checkpoint, sink, **kwargs):
    with patch("ingestion.historical_loader._fetch_page", side_effect=horizon):
        return backfill_trades(
            BASE_ASSET, COUNTER_ASSET, checkpoint, sink, chunk_size=7, limit_per_page=5, **kwargs
        )


def _written_trade_ids(output_dir) -> list[str]:
    files = sorted(output_dir.glob("chunk-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files])["trade_id"].tolist()


def test_interrupted_backfill_resumes_without_gap_or_duplication(tmp_path):
    records = [_record(i) for i in range(25)]
    checkpoint = tmp_path / "backfill.json"
    output = tmp_path / "chunks"
    sink = parquet_chunk_sink(output)

    # Pages hold 5 records and chunks 7, so the crash on the 4th page fetch
    # lands mid-chunk: chunks 0 and 1 (records 0-13) are committed, chunk 2
    # was partially read when the process "died".
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(FakeHorizon(records, crash_on_call=4), checkpoint, sink)

    status = backfill_status(checkpoint)
    assert status["chunks_completed"] == 2
    assert status["cursor"] == "t013"
    assert _written_trade_ids(output) == [f"t{i:03d}" for i in range(14)]

    resumed = FakeHorizon(records)
    summary = _run(resumed, checkpoint, sink)

    assert summary["completed"] == 4  # 7 + 7 + 7 + 4 records
    assert _written_trade_ids(output) == [r["id"] for r in records]
    # The resumed run started from the checkpointed cursor, not from scratch.
    assert resumed.calls == 3  # records 14-18, 19-23, 24


def test_failed_chunk_is_recorded_and_retried_on_next_run(tmp_path):
    records = [_record(i) for i in range(10)]
    checkpoint = tmp_path / "backfill.json"
    written: dict[int, list[str]] = {}
    fail_once = {1}

    def flaky_sink(index, trades):
        written[index] = [t.trade_id for t in trades]
        if index in fail_once:
            fail_once.discard(index)
            raise OSError("disk full")
        return None

    with pytest.raises(OSError):
        _run(FakeHorizon(records), checkpoint, flaky_sink)
    assert backfill_status(checkpoint)["failed_chunks"]["chunk-000001"]["attempts"] == 1

    _run(FakeHorizon(records), checkpoint, flaky_sink)

    status = backfill_status(checkpoint)
    assert status["failed_chunks"] == {}
    assert status["trades"] == 10
    assert [tid for i in sorted(written) for tid in written[i]] == [r["id"] for r in records]


def test_completed_backfill_picks_up_only_new_trades(tmp_path):
    checkpoint = tmp_path / "backfill.json"
    written: dict[int, list[str]] = {}

    def sink(index, trades):
        written[index] = [t.trade_id for t in trades]
        return None

    _run(FakeHorizon([_record(i) for i in range(9)]), checkpoint, sink)
    _run(FakeHorizon([_record(i) for i in range(12)]), checkpoint, sink)

    assert [tid for i in sorted(written) for tid in written[i]] == [f"t{i:03d}" for i in range(12)]


def test_backfill_status_command_reports_progress(tmp_path, capsys):
    checkpoint = tmp_path / "backfill.json"
    with pytest.raises(RuntimeError):
        _run(
            FakeHorizon([_record(i) for i in range(25)], crash_on_call=4),
            checkpoint,
            parquet_chunk_sink(tmp_path / "chunks"),
        )

    assert cli_main(["backfill-status", "--checkpoint-file", str(checkpoint), "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["pair"] == f"USDC:{VALID_ISSUER}/XLM:native"
    assert status["chunks_completed"] == 2
    assert status["trades"] == 14
    assert status["last_ledger_close_time"] == "2024-01-01T00:00:13+00:00"

    assert cli_main(["backfill-status", "--checkpoint-file", str(checkpoint)]) == 0
    out = capsys.readouterr().out
    assert "Chunks completed: 2 (last: 1)" in out
    assert "Resume cursor: t013" in out


def test_backfill_status_command_fails_for_missing_checkpoint(tmp_path, capsys):
    missing = tmp_path / "nope.json"
    assert cli_main(["backfill-status", "--checkpoint-file", str(missing)]) == 1
    assert "Cannot read backfill checkpoint" in capsys.readouterr().err


def test_chunk_size_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        backfill_trades(
            BASE_ASSET, COUNTER_ASSET, tmp_path / "c.json", lambda i, t: None, chunk_size=0
        )
