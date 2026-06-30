# LedgerLens Canary Deployment Runbook

## Overview

When a new model version is deployed in shadow mode, `ModelCanaryMonitor`
(in `detection/model_inference.py`) records per-wallet score pairs
`(champion_score, canary_score)`.  This runbook explains how to interpret
the Grafana dashboard and make a data-driven promotion decision.

## Dashboard

Open **LedgerLens Canary Model Monitor** in Grafana
(`monitoring/grafana/dashboards/canary_monitor.json`).

| Panel | What to look for |
|---|---|
| Score Difference Distribution | Should be a tight peak near 0. A long right tail signals the canary disagrees with the champion on high-risk wallets. |
| Score Pairs Logged | Should grow steadily. A plateau means shadow scoring stopped. |
| p95 Score Delta (stat) | **Key metric** — must be ≤ 15 to pass promotion check. |
| Score Delta Quantiles | p50, p95, p99 over time. A sudden spike indicates a pipeline issue, not a model shift. |

## Promotion Readiness Check

Run this from a Python shell or notebook after the canary has scored at
least several hundred wallets:

```python
from detection.model_inference import ModelCanaryMonitor

monitor = ModelCanaryMonitor(
    champion_version="v1.4.0",
    canary_version="v1.5.0-rc1",
)
# ... (monitor.log_score_pair called during shadow period) ...

report = monitor.promotion_readiness()
print(report)
# {"ready": True, "pair_count": 500, "p95_delta": 8.3, "mean_delta": 3.1, ...}
```

**Promote** if `ready=True` (p95 delta ≤ 15 points).

**Hold** if `ready=False`.  Review `top_divergent_wallets()` to understand
where the canary diverges most:

```python
for w in monitor.top_divergent_wallets(n=10):
    print(w)
```

## Statistical Test Recommendation

Use a **Kolmogorov-Smirnov (KS) test** to assess whether champion and canary
score distributions differ significantly:

```python
from scipy import stats

champion_scores = [p["champion_score"] for p in monitor.score_pairs()]
canary_scores   = [p["canary_score"]   for p in monitor.score_pairs()]
ks_stat, p_value = stats.ks_2samp(champion_scores, canary_scores)
print(f"KS statistic: {ks_stat:.3f}  p-value: {p_value:.4f}")
```

A p-value < 0.05 means the distributions are statistically different.
Combine with the p95 delta check: if distributions differ but the p95 delta
is ≤ 15 and the canary AUC is higher, promotion is still appropriate.

## Prometheus Metric

The Prometheus Summary metric `ledgerlens_canary_score_delta_seconds` exposes
the quantile distribution of absolute score deltas.  Alert if:

```
histogram_quantile(0.95, rate(ledgerlens_canary_score_delta_seconds_bucket[10m])) > 15
```

## Rollback

If the canary is promoted and a regression is detected:

1. Redeploy the champion model artifacts to `config.MODEL_DIR`.
2. Restart the scoring service.
3. File an incident report documenting what the p95 delta was post-promotion.
