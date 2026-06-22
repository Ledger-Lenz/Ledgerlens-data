"""End-to-end Kafka streaming integration test.

Skipped unless ``LEDGERLENS_INTEGRATION_TESTS=1`` (see tests/integration/conftest.py).
Requires a running stack — start it with:

    docker-compose up --scale ledgerlens-scorer=3

The test produces 1,000 synthetic trades through ``HorizonKafkaProducer`` and
asserts that all 1,000 are reflected in the scorer fleet's
``kafka_messages_consumed_total`` metric (0 data loss), scraped from the
Prometheus endpoint exposed by the scorer replicas.
"""

import datetime
import os
import time

import pytest
import requests

pytestmark = pytest.mark.integration

TOTAL_TRADES = 1_000
USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def _synthetic_trade(i: int):
    from ingestion.data_models import Asset, Trade

    return Trade(
        trade_id=f"synthetic-{i:06d}",
        ledger_close_time=datetime.datetime.now(tz=datetime.UTC),
        base_account=f"WALLET{i % 50:03d}",
        counter_account=f"WALLETC{i % 37:03d}",
        base_asset=Asset(code="USDC", issuer=USDC_ISSUER),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=100.0 + i,
        counter_amount=50.0 + i,
        price=2.0,
    )


def _consumed_total(prometheus_url: str) -> float:
    resp = requests.get(
        f"{prometheus_url}/api/v1/query",
        params={"query": "sum(kafka_messages_consumed_total)"},
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json()["data"]["result"]
    return float(result[0]["value"][1]) if result else 0.0


def test_thousand_trades_zero_loss():
    from ingestion.kafka_producer import HorizonKafkaProducer

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    prometheus_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    if not bootstrap:
        pytest.skip("KAFKA_BOOTSTRAP_SERVERS not set — start docker-compose first")

    baseline = _consumed_total(prometheus_url)

    producer = HorizonKafkaProducer(bootstrap_servers=bootstrap)
    for i in range(TOTAL_TRADES):
        producer.produce_trade(_synthetic_trade(i))
    remaining = producer.flush(timeout=30.0)
    assert remaining == 0, f"{remaining} messages failed to flush to Kafka"

    # Allow the 3 scorer replicas to drain the topics.
    deadline = time.time() + 90
    consumed = 0.0
    while time.time() < deadline:
        consumed = _consumed_total(prometheus_url) - baseline
        if consumed >= TOTAL_TRADES:
            break
        time.sleep(2)

    assert (
        consumed >= TOTAL_TRADES
    ), f"expected >= {TOTAL_TRADES} consumed, got {consumed} (data loss)"
