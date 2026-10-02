# End-to-End Latency Budgets and Per-Stage SLOs

Issue #921. Implementation: `monitoring/latency_budget.py`.

## End-to-end target

The detection-latency SLO is **p99 < 10 seconds** from ledger event
ingestion to alert delivery (see `docs/load_testing.md`). That budget is
split across the five pipeline stages below. The per-stage budgets must sum
to exactly the end-to-end target; `validate_stage_slos()` runs at import time
and rejects any change that breaks this.

## Per-stage SLOs

| Stage       | p99 budget | Objective | Rationale |
|-------------|-----------:|----------:|-----------|
| `ingestion` | 2.0 s | 99% | Horizon streaming plus retry/backoff on transient errors. One backoff retry (≈1 s) has to fit inside the budget. |
| `streaming` | 1.5 s | 99% | Kafka produce/consume hop and dedup/exactly-once check. Should be ≪1 s in steady state; the headroom covers consumer-group rebalances. |
| `feature`   | 2.5 s | 99% | Rolling-window and graph features: the most variable stage because cost grows with wallet neighbourhood size. |
| `scoring`   | 3.0 s | 99% | Model ensemble inference plus Benford checks. This is the largest share because it is the core of detection and the stage we least want to degrade. |
| `alert`     | 1.0 s | 99% | Dedup, consensus escalation, routing (`alerts/router.py`) and dispatch to external channels. |
| **Total**   | **10.0 s** | | Equals the end-to-end detection-latency target. |

## Metrics

All stages emit into a single histogram, labelled consistently by `stage`, so
the per-stage values aggregate cleanly:

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `ledgerlens_stage_latency_seconds` | Histogram | `stage` | Time spent in one stage for one event. |
| `ledgerlens_stage_latency_budget_breaches_total` | Counter | `stage` | Number of events where the stage took longer than its budget. |
| `ledgerlens_stage_latency_slo_burn_rate` | Gauge | `stage` | Rolling in-process burn rate, set by `LatencyBudgetTracker.check_slo_burn()`. |
| `ledgerlens_stage_latency_budget_seconds` | Gauge | `stage` | Configured budget, used to draw the budget line on the dashboard. |

To instrument a stage, wrap it with `tracker.time_stage("<stage>")`, or report
all stages of an event at once with `tracker.record_event({...})`.

## Dashboard

`monitoring/grafana/dashboards/latency_budget.json` ("LedgerLens — End-to-End
Latency Budget") has four panels:

- A stacked per-stage mean latency panel. The stacked series sum to the
  dashed end-to-end total.
- Each stage's share of total latency.
- Per-stage p99 plotted against its budget.
- Per-stage burn rate.

## SLO-burn alerting

**Burn rate** = (fraction of events over the stage budget) / (1 − objective).
A burn rate of 1.0 means the stage is using its error budget exactly on pace.

Burn rate is calculated **per stage, against that stage's own budget**. When
one stage slows down, the end-to-end latency rises, but no other stage's
burn rate changes, so only the slow stage alerts.

- **In-process:** `LatencyBudgetTracker.check_slo_burn()` fires once burn
  rate ≥ 2.0 with at least 20 samples in the window. The alert
  (`detectors: ["latency_slo_burn"]`) goes through `AlertRouter`, and the
  shipped `alerts/routing_config.yaml` rule `latency-slo-burn-oncall` sends it
  only to the pipeline on-call rotation.
- **Prometheus:** `monitoring/alert_rules.yml`, group
  `ledgerlens_stage_latency_slo`, uses multi-window (1h and 5m) burn-rate
  alerts: `StageLatencySLOBurnWarning` (> 2×, for 15m) and
  `StageLatencySLOBurnCritical` (> 14.4×, for 2m).

## Responding to an SLO-burn alert

1. On the latency-budget dashboard, find the stage named in the alert's
   `stage` label.
2. Compare that stage's p99 with its budget, and look at its share of total
   latency, to judge how much of the end-to-end SLO is at risk.
3. Look for stage-specific causes: Horizon rate limiting (ingestion), consumer
   lag (streaming), unusually large wallet graphs (feature), a model rollout
   (scoring), or slow external channels (alert).
4. Changing a budget means moving budget between stages, because the budgets
   must still sum to the end-to-end target. Update this document in the same
   change.
