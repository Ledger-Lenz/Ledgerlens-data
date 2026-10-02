"""Tests for stream-level ingestion anomaly detection (Issue #913)."""

import datetime
import os

import numpy as np
import pandas as pd
import pytest

from alerts.router import AlertRouter, RouteDestination, RoutingRule
from ingestion.data_quality import STREAM_QUALITY_DETECTOR, StreamQualityMonitor

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
SOURCE = "horizon_trades"


def _batch(rows: int, null_ratio: float = 0.0, amount_mean: float = 100.0) -> pd.DataFrame:
    amounts = [amount_mean] * rows
    for i in range(int(rows * null_ratio)):
        amounts[i] = None
    return pd.DataFrame({"trade_id": [f"t{i}" for i in range(rows)], "amount": amounts})


@pytest.fixture
def delivered() -> list[tuple[RouteDestination, dict]]:
    return []


@pytest.fixture
def monitor(delivered) -> StreamQualityMonitor:
    router = AlertRouter(
        rules=[
            RoutingRule(
                name="stream-quality",
                detectors={STREAM_QUALITY_DETECTOR},
                destinations=[RouteDestination("slack", "#data-quality")],
            )
        ]
    )
    return StreamQualityMonitor(
        router=router,
        dispatch=lambda dest, alert: delivered.append((dest, alert)),
        key_fields=["amount"],
        distribution_fields=["amount"],
    )


def _warm_up(monitor: StreamQualityMonitor, batches: int = 24) -> datetime.datetime:
    """Feed normally fluctuating volume (~1000 ± 3%) and return the next timestamp."""
    rng = np.random.default_rng(42)
    for i in range(batches):
        rows = int(rng.normal(1000, 30))
        assert (
            monitor.observe_batch(SOURCE, _batch(rows), at=T0 + datetime.timedelta(minutes=i)) == []
        )
    return T0 + datetime.timedelta(minutes=batches)


def test_normal_fluctuation_does_not_alert(monitor, delivered):
    at = _warm_up(monitor)
    rng = np.random.default_rng(7)
    for i in range(50):
        rows = int(rng.normal(1000, 30))
        anomalies = monitor.observe_batch(
            SOURCE, _batch(rows), at=at + datetime.timedelta(minutes=i)
        )
        assert anomalies == []
    assert delivered == []


@pytest.mark.parametrize(("rows", "direction"), [(5000, "spike"), (80, "drop")])
def test_volume_anomaly_routes_alert_with_context(monitor, delivered, rows, direction):
    at = _warm_up(monitor)

    anomalies = monitor.observe_batch(SOURCE, _batch(rows), at=at)

    assert [(a.metric, a.direction) for a in anomalies] == [("volume", direction)]
    assert len(delivered) == 1
    destination, alert = delivered[0]
    assert destination == RouteDestination("slack", "#data-quality")
    assert alert["detectors"] == [STREAM_QUALITY_DETECTOR]
    assert alert["source"] == SOURCE
    assert alert["metric"] == "volume"
    assert alert["direction"] == direction
    assert alert["observed"] == rows
    assert 900 < alert["baseline"] < 1100
    assert abs(alert["magnitude"]) > monitor.z_threshold


def test_null_rate_shift_alerts_even_when_volume_is_normal(monitor, delivered):
    at = _warm_up(monitor)

    anomalies = monitor.observe_batch(SOURCE, _batch(1000, null_ratio=0.4), at=at)

    assert {a.metric for a in anomalies} == {"null_rate:amount"}
    assert delivered[0][1]["direction"] == "spike"


def test_distribution_shift_alerts(monitor, delivered):
    at = _warm_up(monitor)

    anomalies = monitor.observe_batch(SOURCE, _batch(1000, amount_mean=1e6), at=at)

    assert [a.metric for a in anomalies] == ["mean:amount"]


def test_no_alerts_before_min_history(monitor, delivered):
    monitor.observe_batch(SOURCE, _batch(1000), at=T0)
    assert monitor.observe_batch(SOURCE, _batch(10), at=T0) == []
    assert delivered == []


def test_acknowledged_window_suppresses_alert_spam(monitor, delivered):
    at = _warm_up(monitor)
    monitor.acknowledge(
        SOURCE, at, at + datetime.timedelta(hours=1), reason="planned Horizon maintenance"
    )

    for i in range(30):
        anomalies = monitor.observe_batch(SOURCE, _batch(5), at=at + datetime.timedelta(minutes=i))
        assert anomalies and all(a.suppressed_by is not None for a in anomalies)
    assert delivered == []

    # Once the window has expired the same drop alerts again.
    monitor.observe_batch(SOURCE, _batch(5), at=at + datetime.timedelta(hours=2))
    assert len(delivered) == 1


def test_suppression_is_scoped_to_source_and_metric(monitor, delivered):
    at = _warm_up(monitor)
    monitor.acknowledge("other_source", at, at + datetime.timedelta(hours=1), reason="n/a")
    monitor.acknowledge(
        SOURCE, at, at + datetime.timedelta(hours=1), reason="n/a", metrics=["null_rate:amount"]
    )

    monitor.observe_batch(SOURCE, _batch(5), at=at)

    assert [alert["metric"] for _, alert in delivered] == ["volume"]


def test_anomalies_do_not_pollute_baseline_and_reset_clears_it(monitor, delivered):
    at = _warm_up(monitor)
    for i in range(3):
        monitor.observe_batch(SOURCE, _batch(5000), at=at + datetime.timedelta(minutes=i))
    assert len(delivered) == 3

    monitor.reset_baseline(SOURCE)
    for _ in range(monitor.min_history + 1):
        assert monitor.observe_batch(SOURCE, _batch(5000), at=at) == []


def test_acknowledge_rejects_inverted_window(monitor):
    with pytest.raises(ValueError):
        monitor.acknowledge(SOURCE, T0, T0, reason="bad")


def test_shipped_routing_config_routes_stream_anomalies():
    path = os.path.join(os.path.dirname(__file__), "..", "alerts", "routing_config.yaml")
    delivered: list = []
    monitor = StreamQualityMonitor(
        router=AlertRouter.from_yaml(path), dispatch=lambda d, a: delivered.append(d)
    )
    at = _warm_up(monitor)

    monitor.observe_batch(SOURCE, _batch(5000), at=at)

    assert [(d.channel, d.target) for d in delivered] == [("slack", "#data-quality")]
