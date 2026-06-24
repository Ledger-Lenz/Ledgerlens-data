"""Unit tests for streaming/kafka_worker.py (KafkaWorker).

confluent_kafka.Consumer is mocked — no live broker is required. The scorer,
dispatcher, and feature buffer are mocked so the test focuses on offset-commit
and lag-alerting semantics.
"""

import datetime
import logging
from unittest.mock import MagicMock

import pytest

from ingestion.avro_codec import load_schema, serialize


def _avro_value(trade_id: str = "trade-001") -> bytes:
    record = {
        "trade_id": trade_id,
        "base_account": "WALLETBASE123",
        "counter_account": "WALLETCOUNTER456",
        "base_amount": 100.5,
        "counter_amount": 50.25,
        "price": 2.0,
        "asset_pair": "USDC:GISSUER/XLM:native",
        "ledger_close_time": datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        "ingestion_timestamp_ms": 1704110400000,
    }
    return serialize(record, load_schema())


def _make_msg(*, offset: int = 5, topic: str = "ledgerlens.trades.USDC_X") -> MagicMock:
    msg = MagicMock()
    msg.topic.return_value = topic
    msg.partition.return_value = 0
    msg.offset.return_value = offset
    msg.value.return_value = _avro_value()
    msg.error.return_value = None
    return msg


def _make_worker(consumer, *, score=None, dispatch_side_effect=None, lag_high=100):
    from streaming.kafka_worker import KafkaWorker

    scorer = MagicMock()
    scorer.score_wallet.return_value = score
    dispatcher = MagicMock()
    if dispatch_side_effect is not None:
        dispatcher.dispatch.side_effect = dispatch_side_effect
    buffer = MagicMock()

    consumer.get_watermark_offsets.return_value = (0, lag_high)

    worker = KafkaWorker(scorer, dispatcher, buffer, consumer=consumer)
    return worker, scorer, dispatcher


# ---------------------------------------------------------------------------
# 1. Offset committed exactly once per message after scorer + dispatcher
# ---------------------------------------------------------------------------


def test_offset_committed_once_after_dispatch():
    consumer = MagicMock()
    score = {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 70}
    worker, scorer, dispatcher = _make_worker(consumer, score=score)

    msg = _make_msg()
    worker.process_message(msg)

    # Both accounts scored + dispatched, but the offset is committed once.
    assert dispatcher.dispatch.call_count == 2
    assert consumer.commit.call_count == 1
    consumer.commit.assert_called_once_with(message=msg, asynchronous=False)


def test_offset_committed_once_even_when_below_threshold():
    """Successful processing with no alert still commits exactly once."""
    consumer = MagicMock()
    worker, scorer, dispatcher = _make_worker(consumer, score=None)

    worker.process_message(_make_msg())

    dispatcher.dispatch.assert_not_called()
    assert consumer.commit.call_count == 1


# ---------------------------------------------------------------------------
# 2. Offset NOT committed if AlertDispatcher.dispatch raises
# ---------------------------------------------------------------------------


def test_offset_not_committed_when_dispatch_raises():
    consumer = MagicMock()
    score = {"score": 90, "benford_flag": True, "ml_flag": True, "confidence": 80}
    worker, scorer, dispatcher = _make_worker(
        consumer, score=score, dispatch_side_effect=RuntimeError("dispatch boom")
    )

    with pytest.raises(RuntimeError, match="dispatch boom"):
        worker.process_message(_make_msg())

    consumer.commit.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Lag above threshold emits a CRITICAL log (worker does not crash)
# ---------------------------------------------------------------------------


def test_lag_above_threshold_emits_critical(caplog):
    consumer = MagicMock()
    # High watermark 10_000 with offset 0 → lag ~9_999, well above default 500.
    worker, scorer, dispatcher = _make_worker(consumer, score=None, lag_high=10_000)

    with caplog.at_level(logging.CRITICAL, logger="streaming.kafka_worker"):
        worker.process_message(_make_msg(offset=0))

    assert any("exceeds threshold" in r.message for r in caplog.records)
    # The message was still processed and committed — no crash.
    assert consumer.commit.call_count == 1


def test_dlq_topic_is_skipped_not_scored():
    consumer = MagicMock()
    worker, scorer, dispatcher = _make_worker(consumer, score=None)

    msg = _make_msg(topic="ledgerlens.trades.dlq")
    worker.process_message(msg)

    scorer.score_wallet.assert_not_called()
    # Skipped messages are committed so they don't block the partition.
    consumer.commit.assert_called_once_with(message=msg, asynchronous=False)
