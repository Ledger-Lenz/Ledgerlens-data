"""Unit tests for ingestion/kafka_producer.py (HorizonKafkaProducer).

confluent_kafka.Producer is mocked — no live broker is required.
"""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest
from confluent_kafka import KafkaException

from ingestion.avro_codec import deserialize, load_schema
from ingestion.data_models import Asset, Trade
from ingestion.kafka_producer import HorizonKafkaProducer

USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def make_trade(trade_id: str = "trade-001") -> Trade:
    return Trade(
        trade_id=trade_id,
        ledger_close_time=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        base_account="WALLETBASE123",
        counter_account="WALLETCOUNTER456",
        base_asset=Asset(code="USDC", issuer=USDC_ISSUER),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=100.5,
        counter_amount=50.25,
        price=2.0,
    )


def _producer_with_mock():
    mock = MagicMock()
    producer = HorizonKafkaProducer(producer=mock)
    return producer, mock


# ---------------------------------------------------------------------------
# 1. Produced message deserialises back to the original trade dict
# ---------------------------------------------------------------------------


def test_produced_message_round_trips():
    producer, mock = _producer_with_mock()
    trade = make_trade()

    producer.produce_trade(trade)

    mock.produce.assert_called_once()
    kwargs = mock.produce.call_args.kwargs
    # Topic is per-pair and sanitised; key is the base account (wallet_id).
    assert kwargs["topic"] == f"ledgerlens.trades.USDC_{USDC_ISSUER}_XLM_native"
    assert kwargs["key"] == b"WALLETBASE123"

    decoded = deserialize(kwargs["value"], load_schema())
    assert decoded["trade_id"] == trade.trade_id
    assert decoded["base_account"] == trade.base_account
    assert decoded["counter_account"] == trade.counter_account
    assert decoded["base_amount"] == pytest.approx(trade.base_amount)
    assert decoded["counter_amount"] == pytest.approx(trade.counter_amount)
    assert decoded["price"] == pytest.approx(trade.price)
    assert decoded["asset_pair"] == f"USDC:{USDC_ISSUER}/XLM:native"
    assert decoded["ledger_close_time"] == trade.ledger_close_time


# ---------------------------------------------------------------------------
# 2. KafkaException triggers retry with backoff
# ---------------------------------------------------------------------------


def test_kafka_exception_triggers_retry_with_backoff():
    producer, mock = _producer_with_mock()
    # First produce raises, second succeeds.
    mock.produce.side_effect = [KafkaException(), None]

    with patch("time.sleep") as mock_sleep:
        producer.produce_trade(make_trade())

    assert mock.produce.call_count == 2
    mock_sleep.assert_called_once()  # one backoff between the two attempts


def test_kafka_exception_exhausts_retries_and_raises():
    producer, mock = _producer_with_mock()
    mock.produce.side_effect = KafkaException()

    with patch("time.sleep"):
        with pytest.raises(KafkaException):
            producer.produce_trade(make_trade())

    assert mock.produce.call_count == 5  # max_attempts on _produce


# ---------------------------------------------------------------------------
# 3. Serialisation failure routes to the DLQ topic, not the main topic
# ---------------------------------------------------------------------------


def test_serialisation_failure_routes_to_dlq():
    producer, mock = _producer_with_mock()

    with patch(
        "ingestion.kafka_producer.serialize",
        side_effect=ValueError("bad field"),
    ):
        producer.produce_trade(make_trade())

    mock.produce.assert_called_once()
    kwargs = mock.produce.call_args.kwargs
    assert kwargs["topic"] == "ledgerlens.trades.dlq"

    # No message was produced to any main per-pair topic.
    for call in mock.produce.call_args_list:
        assert call.kwargs["topic"] == "ledgerlens.trades.dlq"

    # The DLQ envelope carries the raw payload and the failure reason.
    envelope = json.loads(kwargs["value"].decode("utf-8"))
    assert envelope["reason"] == "bad field"
    assert envelope["raw"]["trade_id"] == "trade-001"
    assert ("reason", b"bad field") in kwargs["headers"]
