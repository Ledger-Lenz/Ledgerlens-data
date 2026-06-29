# LedgerLens Monitoring

## SLO Dashboard (issue #197)

### Dashboard: `grafana/dashboards/slo_dashboard.json`

The SLO (Service Level Objective) dashboard surfaces aggregate system-wide detection quality commitments: detection latency, recall on confirmed fraud, false positive rate, and Benford computation throughput.

#### Panels

| Panel | Type | SLO Target | Description |
|---|---|---|---|
| Scoring Latency (p50, p95, p99) | Time series | p99 < 5s | End-to-end scoring latency percentiles. Target addresses fraud investigation use case (not real-time blocking). |
| Recall on Confirmed Wash Trades | Gauge | >= 85% | Fraction of manually-confirmed wash trading events detected (scored >= 70). Protects against missing manipulation cases. |
| False Positive Rate on Clean Wallets | Gauge | < 5% | Fraction of manually-confirmed clean wallets flagged as suspicious. Keeps analyst alert burden manageable. |
| Benford Computation Throughput | Time series | > 0.1 ops/s | Rate of Benford computations per second. Falls to near-zero if data ingestion or feature pipeline is blocked. |
| Alert Threshold Drift (RL Controller) | Time series | — | Deviation of adaptive threshold from 30-day rolling mean. Helps track feedback loop stability. |
| Confirmed Wash Trades (24h) | Stat | — | Cumulative confirmed fraud cases in last 24h. |
| Confirmed Clean Wallets (24h) | Stat | — | Cumulative confirmed legitimate traders in last 24h. |
| Scoring Errors (24h) | Stat | 0 | Count of scoring pipeline errors. Green if 0; yellow if 1–4; red if ≥ 5. |
| P99 Latency Alert Status | Stat | — | Shows active `ScoringLatencyP99High` alert. Green if no alerts; red if firing. |

#### Optional Filtering

Use the **Asset Pair** dropdown at the top to drill down to a specific pair. Leave empty to view aggregate system metrics.

#### Metrics Used

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `ledgerlens_score_duration_seconds` | Histogram | `asset_pair` | Existing; per-pair scoring latency. |
| `ledgerlens_confirmed_wash_trades_total` | Counter | `asset_pair` | NEW (issue #197); incremented when analysts confirm a wallet is conducting wash trading. |
| `ledgerlens_confirmed_clean_wallets_total` | Counter | `asset_pair` | NEW (issue #197); incremented when analysts confirm a wallet is legitimate. |
| `ledgerlens_false_negative_wash_trades_total` | Counter | `asset_pair` | NEW (issue #197); incremented when analysts find a confirmed wash trader that our system missed. |
| `ledgerlens_false_positive_wallets_total` | Counter | `asset_pair` | NEW (issue #197); incremented when analysts find a confirmed clean wallet that our system flagged. |
| `ledgerlens_scoring_errors_total` | Counter | `asset_pair` | NEW (issue #197); incremented on scoring pipeline exceptions. |
| `ledgerlens_benford_computation_total` | Counter | `asset_pair, status` | Existing; incremented by Benford engine. |
| `ledgerlens_rl_threshold_current` | Gauge | `asset_pair` | NEW (future); for RL controller threshold drift visualization. |

#### Sample Prometheus Queries

**Recall calculation (24h):**
```promql
sum(increase(ledgerlens_confirmed_wash_trades_total[24h]))
/
(
  sum(increase(ledgerlens_confirmed_wash_trades_total[24h]))
  + sum(increase(ledgerlens_false_negative_wash_trades_total[24h]))
)
```

**False positive rate calculation (24h):**
```promql
sum(increase(ledgerlens_false_positive_wallets_total[24h]))
/
sum(increase(ledgerlens_confirmed_clean_wallets_total[24h]))
```

### Alert Rules: `alert_rules.yml`

**New SLO alerting group: `ledgerlens_slo_monitoring`**

| Alert | Condition | Duration | Severity | Runbook |
|---|---|---|---|---|
| `ScoringLatencyP99High` | p99 latency > 5s | 10 min | critical | docs/monitoring.md#runbook-scoring-latency |
| `WashTradeRecallBelowSLO` | Recall < 85% | 1 hour | warning | docs/monitoring.md#runbook-recall |
| `WashTradeFalsePositiveRateHigh` | False positive rate > 5% | 1 hour | warning | docs/monitoring.md#runbook-false-positive |
| `BenfordThroughputCritical` | Throughput < 0.1 ops/s | 15 min | critical | docs/monitoring.md#runbook-throughput |

#### Alert Threshold Rationale

- **p99 latency 5 seconds**: SLO target for fraud investigation workflows (asynchronous). Exceeding this indicates a systemic issue (ingestion backlog, model latency, or infrastructure contention).
- **Recall 85%**: Conservative baseline to ensure significant fraud cases are caught. Anything lower than this deserves immediate investigation (model drift, training data quality issue).
- **False positive rate 5%**: Keeps manual review workload for analysts manageable (~5 in 100 alerts are false positives). Higher rates lead to alert fatigue and missed true positives.
- **Benford throughput 0.1 ops/s**: Near-critical threshold indicating either (a) data source dried up or (b) pipeline is hung. Investigation required either way.

## Per-Asset-Pair Health Dashboard (issue #276)

### Dashboard: `grafana/dashboards/per_pair_health.json`

The per-pair health dashboard (`LedgerLens — Per-Asset-Pair Health`) surfaces detection quality issues at the asset-pair level so operators can identify which specific trading pairs are experiencing problems without digging through aggregate metrics.

#### Panels

| Panel | Type | Description |
|---|---|---|
| Scoring Latency Heatmap | Heatmap | p95 scoring latency per pair over time |
| Benford MAD Time Series | Time series | MAD vs. asset-class baseline per pair; >0.015 is non-conforming |
| Alert Volume: Confirmed vs FP | Time series | Rate of confirmed alerts and false positives per pair |
| Pair Health Score | Gauge | Composite 0–1 health score per pair |
| Risk Score Distribution | Histogram | Distribution of 0–100 risk scores per pair over the last hour |

#### Filtering

Use the **Asset Pair** Grafana variable dropdown at the top of the dashboard to filter all panels to a specific pair. The variable queries `label_values(ledgerlens_risk_score_distribution_bucket, asset_pair)`.

### Composite Health Score Formula

```
health = (latency_health × 0.4) + (benford_health × 0.4) + (fp_rate_health × 0.2)
```

Where:

- **latency_health** = `1 - clamp(p95_latency_seconds / 0.5, 0, 1)`
  — 1.0 when p95 < 0 ms; 0.0 when p95 ≥ 500ms.
- **benford_health** = `1 - clamp(benford_MAD / 0.03, 0, 1)`
  — 1.0 when MAD = 0; 0.0 when MAD ≥ 0.03 (2× non-conformity threshold).
- **fp_rate_health** = `1 - clamp(false_positive_rate / confirmed_rate, 0, 1)`
  — penalises pairs with high false-positive-to-confirmed-alert ratios.

A composite score below **0.7** for more than **30 minutes** fires the `PairHealthScoreLow` Prometheus alert.

### Metrics

All three metrics carry the `asset_pair` label in canonical `CODE:ISSUER/CODE:ISSUER` sorted-alphabetical format:

| Metric | Type | Labels |
|---|---|---|
| `ledgerlens_score_duration_seconds` | Histogram | `asset_pair` |
| `ledgerlens_benford_computation_total` | Counter | `asset_pair`, `status` |
| `ledgerlens_risk_score_distribution` | Histogram | `asset_pair` |

Labels **never** include wallet addresses — only aggregate pair identifiers.

### Alert Rules: `alert_rules.yml`

| Alert | Condition | Duration | Severity |
|---|---|---|---|
| `PairHealthScoreLow` | composite health < 0.7 | 30 min | warning |
| `PairScoringLatencyHigh` | p95 latency > 500ms | 10 min | warning |
| `PairBenfordNonConforming` | MAD > 0.015 | 15 min | info |

### Alert Threshold Rationale

- **0.7 health score**: below this level at least one major component (latency or Benford freshness) is significantly degraded; investigation is warranted.
- **30-minute duration**: filters out transient spikes from brief data ingestion gaps without delaying response to sustained degradation.
- **p95 500ms latency**: 10× the typical p95 under normal load; indicates a systemic issue rather than isolated slow requests.
