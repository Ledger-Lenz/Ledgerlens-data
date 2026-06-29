# LedgerLens Monitoring

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
