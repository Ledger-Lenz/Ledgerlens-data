"""Tests for per-stage latency budgets and SLO-burn alerting (Issue #921)."""

from __future__ import annotations

import math
import os

import pytest

from alerts.router import AlertRouter, RouteDestination, RoutingRule
from monitoring.latency_budget import (
    E2E_DETECTION_LATENCY_SLO_SECONDS,
    PIPELINE_LATENCY_STAGES,
    SLO_BURN_DETECTOR,
    STAGE_LATENCY_SLOS,
    LatencyBudgetTracker,
    StageSLO,
    validate_stage_slos,
)

_FAST = {"ingestion": 0.5, "streaming": 0.3, "feature": 0.8, "scoring": 1.0, "alert": 0.2}


def _router() -> AlertRouter:
    return AlertRouter(
        rules=[
            RoutingRule(
                name="slo-burn",
                detectors={SLO_BURN_DETECTOR},
                stop_on_match=True,
                destinations=[RouteDestination("pagerduty", "pipeline-oncall")],
            )
        ]
    )


def test_stage_slos_sum_to_e2e_target():
    assert set(STAGE_LATENCY_SLOS) == set(PIPELINE_LATENCY_STAGES)
    total = sum(s.budget_seconds for s in STAGE_LATENCY_SLOS.values())
    assert math.isclose(total, E2E_DETECTION_LATENCY_SLO_SECONDS)


def test_validate_rejects_budgets_not_summing_to_target():
    bad = dict(STAGE_LATENCY_SLOS)
    bad["scoring"] = StageSLO("scoring", 9.0)
    with pytest.raises(ValueError, match="sum to"):
        validate_stage_slos(bad)


def test_validate_rejects_missing_stage():
    bad = {k: v for k, v in STAGE_LATENCY_SLOS.items() if k != "alert"}
    with pytest.raises(ValueError, match="missing"):
        validate_stage_slos(bad)


def test_breakdown_sums_to_observed_e2e_latency():
    tracker = LatencyBudgetTracker()
    e2e = [tracker.record_event({**_FAST, "scoring": 1.0 + i * 0.01}) for i in range(50)]
    breakdown = tracker.breakdown()

    assert math.isclose(breakdown["stage_sum_seconds"], sum(e2e) / len(e2e))
    assert math.isclose(breakdown["stage_sum_seconds"], breakdown["e2e_mean_seconds"])
    assert math.isclose(sum(s["share_of_total"] for s in breakdown["stages"].values()), 1.0)


def test_healthy_pipeline_fires_no_alerts():
    tracker = LatencyBudgetTracker(router=_router())
    for _ in range(100):
        tracker.record_event(_FAST)
    assert tracker.check_slo_burn() == []


def test_slow_stage_triggers_only_its_own_alert_routed_via_router():
    delivered = []
    tracker = LatencyBudgetTracker(
        router=_router(), dispatch=lambda alert, dests: delivered.append((alert, dests))
    )
    # The feature stage blows its 2.5s budget on 10% of events, pushing the
    # end-to-end latency over target -- no other stage should be blamed.
    for i in range(200):
        tracker.record_event({**_FAST, "feature": 6.0 if i % 10 == 0 else 0.8})

    fired = tracker.check_slo_burn()

    assert [a.stage for a in fired] == ["feature"]
    assert fired[0].burn_rate == pytest.approx(10.0)
    assert fired[0].destinations[0].target == "pipeline-oncall"
    assert delivered and delivered[0][0]["stage"] == "feature"
    assert delivered[0][0]["observed_p99_seconds"] == 6.0
    for stage in PIPELINE_LATENCY_STAGES:
        if stage != "feature":
            assert tracker.burn_rate(stage) == 0.0


def test_min_samples_suppresses_cold_start_alerts():
    tracker = LatencyBudgetTracker(router=_router(), min_samples=20)
    for _ in range(5):
        tracker.record("scoring", 30.0)
    assert tracker.check_slo_burn() == []


def test_time_stage_records_observation():
    tracker = LatencyBudgetTracker()
    with tracker.time_stage("alert"):
        pass
    assert tracker.breakdown()["stages"]["alert"]["mean_seconds"] >= 0.0
    with pytest.raises(KeyError):
        tracker.record("unknown", 1.0)


def test_shipped_routing_config_routes_slo_burn_to_oncall_only():
    path = os.path.join(os.path.dirname(__file__), "..", "alerts", "routing_config.yaml")
    router = AlertRouter.from_yaml(path)
    tracker = LatencyBudgetTracker(router=router)
    for _ in range(50):
        tracker.record_event({**_FAST, "ingestion": 5.0})

    fired = tracker.check_slo_burn()
    assert [a.stage for a in fired] == ["ingestion"]
    assert router.explain(fired[0].alert) == ["latency-slo-burn-oncall"]
    assert {d.target for d in fired[0].destinations} == {
        "ledgerlens-pipeline-oncall",
        "#ledgerlens-ops",
    }
