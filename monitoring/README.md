# LedgerLens Monitoring

## Capacity Planning Dashboard (Issue #242)

### Dashboard: `grafana/dashboards/capacity_planning.json`

The capacity planning dashboard (`LedgerLens — Capacity Planning`) projects compute requirements from trade volume growth so operators can make proactive infrastructure scaling decisions.

#### Panels

| Panel | Type | Description |
|---|---|---|
| CPU Usage by Component | Time series | CPU utilisation ratio (0–1) per pipeline component |
| Memory Usage | Time series | Process RSS memory in bytes |
| Trade Ingestion Rate | Time series | Trade events/s per asset pair |
| CPU Trend Projection | Time series | 7-day linear regression projection with 95% CI band and 80% threshold line |
| Days to 80% CPU | Stat | Estimated days until average CPU reaches 80% (from recording rule) |
| Memory Growth Rate | Stat | Per-second rate of RSS growth projected over 7 days |

#### Interpreting the Dashboard

**CPU Trend Projection panel**: The solid line shows the current CPU utilisation; the shaded band is the ±1.96σ confidence interval derived from the 7-day rolling standard deviation. The dashed red line marks the 80% capacity threshold. The trend is meaningful only after at least 7 days of history; before that, the projection will be noisy.

**Days to 80% CPU counter**: A green value (≥ 30 days) means capacity is comfortable. Yellow (14–30 days) means planning should begin. Red (< 14 days) means immediate action is needed (add compute resources or reduce batch sizes).

**What the 80% threshold means**: Operating above 80% CPU leaves insufficient headroom for burst traffic (e.g. a new exchange listing that causes a 2–3× spike in trade volume). Below 80% the pipeline can absorb moderate spikes without dropping events.

**Adding compute resources**: Scale the `ledgerlens-scorer` replicas in `docker-compose.yml` (or your orchestrator) by incrementing the replica count. The Kafka partition-key assignment distributes load across replicas automatically.

**Step-change events**: A sudden volume increase (e.g. a new exchange listing) will invalidate the linear trend for several days until the regression window rolls past the step. The confidence interval band will widen significantly during this period — treat the "Days to 80%" counter as unreliable until the band narrows again (typically 2–3 days after the event).

#### Prometheus Recording Rules

The linear-regression recording rules are defined in `alert_rules.yml` under the `ledgerlens_capacity_planning` group:

| Recording Rule | Description |
|---|---|
| `ledgerlens:cpu_usage_trend_slope` | Per-second derivative of CPU usage (least-squares over 7 days) |
| `ledgerlens:capacity_days_to_80pct_cpu` | Projected days until average CPU hits 80% |
| `ledgerlens:cpu_projection_ci_halfwidth` | 95% CI half-width on the projection |

Recording rules are evaluated every 5 minutes. The 7-day `[7d]` range vector requires Prometheus to retain at least 7 days of raw samples.

#### Security

The capacity metrics endpoint (`/metrics` on port 9100) is only scraped by the internal Prometheus instance. It is not exposed externally — the `WS_ALLOW_EXTERNAL` setting and external load balancer configuration must not route this port to the public internet.

#### Metrics Registered

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ledgerlens_cpu_usage_ratio` | Gauge | `component` | CPU usage ratio (0.0–1.0) |
| `ledgerlens_memory_usage_bytes` | Gauge | — | Process RSS in bytes |
| `ledgerlens_trades_per_second` | Gauge | `asset_pair` | Trade event ingestion rate |

Update these metrics from the pipeline process using `monitoring.capacity_metrics`:

```python
from monitoring.capacity_metrics import set_cpu_usage, set_memory_usage, set_trades_per_second

set_cpu_usage("benford", 0.42)
set_memory_usage(512 * 1024 * 1024)
set_trades_per_second("XLM/USDC", 24.5)
```

---

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
