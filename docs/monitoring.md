# Monitoring

## Drift Detection — Sliding Window Covariance Shift (MMD)

`monitoring/drift_detector.py` implements a `CovarianceShiftDetector` that compares the current feature distribution against a reference window using the **Maximum Mean Discrepancy (MMD)** statistic.

### MMD Statistic

MMD measures the distance between two distributions in a reproducing kernel Hilbert space (RKHS):

```
MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
```

where `k` is an RBF kernel with bandwidth selected via the **median heuristic** (bandwidth = median of pairwise distances). This gives an unbiased, parameter-free estimate of distribution shift.

**Why MMD over KL divergence?**
- KL divergence requires density estimation (histogram binning), which introduces quantisation error and fails for low-sample or continuous distributions.
- MMD operates directly on samples with no binning, is well-defined even when the two distributions have non-overlapping support, and has stronger theoretical guarantees for finite-sample hypothesis testing.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `DRIFT_REFERENCE_WINDOW_HOURS` | `168` (7 days) | Look-back window for the reference distribution |
| `DRIFT_TEST_WINDOW_HOURS` | `1` | Current window to compare against reference |
| `DRIFT_CHECK_INTERVAL_MINUTES` | `30` | Background task check frequency |

### Prometheus Gauge

`ledgerlens_feature_drift_detected` — set to `1` when drift is detected across any feature, `0` when stable.

### Interpreting the Gauge

- **0** — all per-feature MMD values are ≤ threshold; model inputs are stable.
- **1** — at least one feature exceeds the MMD threshold; a warning is logged with the top-5 drifted features ranked by MMD contribution. Consider triggering a retrain.

Drift reports contain only feature names and MMD statistics — no raw feature values are included.
