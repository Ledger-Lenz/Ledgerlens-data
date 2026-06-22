"""Backend-selection tests for streaming/pipeline.py.

These assert that ``STREAMING_BACKEND=sse`` (the default) drives the threaded
Horizon SSE pipeline and never imports the Kafka stack — operators without Kafka
must keep running unchanged.
"""

import datetime
import sys
from unittest.mock import MagicMock

import streaming.pipeline as pipeline_module
from ingestion.data_models import Asset, Trade
from streaming.pipeline import StreamingPipeline

USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def make_trade() -> Trade:
    return Trade(
        trade_id="test-trade-001",
        ledger_close_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        base_account="WALLETBASE123",
        counter_account="WALLETCOUNTER456",
        base_asset=Asset(code="USDC", issuer=USDC_ISSUER),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=100.0,
        counter_amount=50.0,
        price=2.0,
    )


def _mock_pipeline():
    buffer = MagicMock()
    scorer = MagicMock()
    scorer.score_wallet.return_value = {
        "score": 80,
        "benford_flag": True,
        "ml_flag": True,
        "confidence": 70,
    }
    dispatcher = MagicMock()
    pipeline = StreamingPipeline(buffer, scorer, dispatcher, pairs=[("USDC", USDC_ISSUER)])
    return pipeline, buffer, scorer, dispatcher


def test_sse_backend_runs_threaded_pipeline_without_kafka(monkeypatch):
    """Default sse backend dispatches via the threaded path with no Kafka import."""
    monkeypatch.setattr(pipeline_module.config, "STREAMING_BACKEND", "sse")
    # Any attempt to import confluent_kafka must blow up — proving the sse path
    # never reaches the Kafka modules.
    monkeypatch.setitem(sys.modules, "confluent_kafka", None)

    pipeline, buffer, scorer, dispatcher = _mock_pipeline()

    def mock_stream(*args, **kwargs):
        yield make_trade()
        pipeline._stop_event.set()
        yield from []

    monkeypatch.setattr("streaming.pipeline.stream_trades", mock_stream)

    # Drive a single pair directly — this is the threaded backend's worker body.
    pipeline._run_sse()

    assert dispatcher.dispatch.call_count == 2
    # The Kafka worker module was never imported by the sse path.
    assert "streaming.kafka_worker" not in sys.modules or sys.modules.get("confluent_kafka") is None


def test_run_dispatches_to_sse_backend_by_default(monkeypatch):
    """run() routes to the threaded sse backend unless STREAMING_BACKEND=kafka."""
    monkeypatch.setattr(pipeline_module.config, "STREAMING_BACKEND", "sse")
    pipeline, _, _, _ = _mock_pipeline()

    called = {"sse": False, "kafka": False}
    monkeypatch.setattr(pipeline, "_run_sse", lambda: called.__setitem__("sse", True))
    monkeypatch.setattr(pipeline, "_run_kafka", lambda: called.__setitem__("kafka", True))

    pipeline.run()

    assert called["sse"] is True
    assert called["kafka"] is False


def test_run_dispatches_to_kafka_backend_when_configured(monkeypatch):
    monkeypatch.setattr(pipeline_module.config, "STREAMING_BACKEND", "kafka")
    pipeline, _, _, _ = _mock_pipeline()

    called = {"sse": False, "kafka": False}
    monkeypatch.setattr(pipeline, "_run_sse", lambda: called.__setitem__("sse", True))
    monkeypatch.setattr(pipeline, "_run_kafka", lambda: called.__setitem__("kafka", True))

    pipeline.run()

    assert called["kafka"] is True
    assert called["sse"] is False
