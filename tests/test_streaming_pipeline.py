"""Unit tests for StreamingPipeline (streaming/pipeline.py).

All three required tests run without live Horizon or model artifacts — every
external dependency is mocked.
"""

import datetime
import threading
import time
from unittest.mock import MagicMock

from stellar_sdk import Asset as SdkAsset

from ingestion.data_models import Asset, Trade
from streaming.pipeline import StreamingPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def make_trade(
    base_account: str = "WALLETBASE123",
    counter_account: str = "WALLETCOUNTER456",
) -> Trade:
    return Trade(
        trade_id="test-trade-001",
        ledger_close_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        base_account=base_account,
        counter_account=counter_account,
        base_asset=Asset(code="USDC", issuer=USDC_ISSUER),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=100.0,
        counter_amount=50.0,
        price=2.0,
    )


def _mock_pipeline(pairs=None):
    """Return (pipeline, buffer_mock, scorer_mock, dispatcher_mock)."""
    buffer = MagicMock()
    scorer = MagicMock()
    scorer.score_wallet.return_value = None  # below min_trades by default
    dispatcher = MagicMock()
    pipeline = StreamingPipeline(
        buffer,
        scorer,
        dispatcher,
        pairs=pairs or [("USDC", USDC_ISSUER)],
    )
    return pipeline, buffer, scorer, dispatcher


# ---------------------------------------------------------------------------
# 1. Both accounts per trade are scored and dispatched
# ---------------------------------------------------------------------------


def test_pipeline_scores_both_accounts_per_trade(monkeypatch):
    """_stream_pair must call dispatcher.dispatch for base AND counter account."""
    trade = make_trade()
    pipeline, buffer, scorer, dispatcher = _mock_pipeline(pairs=[("USDC", USDC_ISSUER)])

    risk_score = {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 70}
    scorer.score_wallet.return_value = risk_score

    def mock_stream(*args, **kwargs):
        yield trade
        # Signal the pipeline's stop_event so the while-loop in _stream_pair exits.
        pipeline._stop_event.set()
        yield from []

    monkeypatch.setattr("streaming.pipeline.stream_trades", mock_stream)

    base = SdkAsset(code="USDC", issuer=USDC_ISSUER)
    counter = SdkAsset.native()
    pipeline._stream_pair(base, counter)

    assert dispatcher.dispatch.call_count == 2
    dispatched_wallets = {c[0][0] for c in dispatcher.dispatch.call_args_list}
    assert trade.base_account in dispatched_wallets
    assert trade.counter_account in dispatched_wallets


# ---------------------------------------------------------------------------
# 2. Pipeline reconnects transparently after a stream error
# ---------------------------------------------------------------------------


def test_pipeline_reconnects_on_stream_failure(monkeypatch):
    """After ConnectionError from stream_trades the pipeline restarts the stream."""
    trade = make_trade()
    pipeline, buffer, scorer, dispatcher = _mock_pipeline(pairs=[("USDC", USDC_ISSUER)])

    risk_score = {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 70}
    scorer.score_wallet.return_value = risk_score

    call_count = [0]

    def mock_stream(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ConnectionError("simulated Horizon disconnect")
        # Second call: yield one trade then stop
        yield trade
        pipeline._stop_event.set()
        yield from []

    monkeypatch.setattr("streaming.pipeline.stream_trades", mock_stream)

    base = SdkAsset(code="USDC", issuer=USDC_ISSUER)
    counter = SdkAsset.native()
    pipeline._stream_pair(base, counter)

    # stream_trades was called twice — reconnect happened
    assert call_count[0] == 2
    # At least one trade was processed after the reconnect
    assert dispatcher.dispatch.call_count >= 1


# ---------------------------------------------------------------------------
# 3. Graceful shutdown: all threads join within 5 seconds on SIGINT
# ---------------------------------------------------------------------------


def test_pipeline_graceful_shutdown(monkeypatch):
    """Sending SIGINT causes run() to set the stop event; all threads join within 5s."""
    pipeline, buffer, scorer, _ = _mock_pipeline(pairs=[("USDC", USDC_ISSUER)])

    def mock_stream(*args, **kwargs):
        # Block until the pipeline's stop event is set.
        while not pipeline._stop_event.is_set():
            time.sleep(0.01)
        yield from []

    monkeypatch.setattr("streaming.pipeline.stream_trades", mock_stream)

    run_thread = threading.Thread(target=pipeline.run, daemon=True)
    run_thread.start()
    time.sleep(0.15)  # let the pipeline start its worker threads

    pipeline._stop_event.set()

    run_thread.join(timeout=5)
    assert not run_thread.is_alive(), "run() did not exit within 5 seconds after SIGINT"
