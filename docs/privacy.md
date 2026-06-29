# Differential Privacy in LedgerLens

LedgerLens applies differential privacy (DP) noise to aggregate statistics computed during training data preparation. This prevents individual wallet activity from being reconstructed from logged or shared statistics.

## Overview

`privacy/dp_aggregator.py` exposes a `DPAggregator` class that wraps standard aggregations (mean, count, histogram) with calibrated DP noise.

Two mechanisms are used:

| Aggregation | Mechanism | Privacy guarantee |
|---|---|---|
| Mean | Laplace | Pure DP (ε, 0) |
| Count | Laplace | Pure DP (ε, 0) |
| Histogram | Gaussian | Approximate DP (ε, δ) |

## Sensitivity Analysis

Sensitivity quantifies the maximum change in an aggregate from adding or removing one wallet's data.

- **Mean** (L1 sensitivity): `Δ_mean = (feature_max − feature_min) / n`  
  Computed analytically per feature using known range bounds.
- **Count** (L1 sensitivity): `Δ_count = 1`  
  A single wallet can change the count by exactly 1.
- **Histogram bin count** (L2 sensitivity): `Δ_bin = 1`  
  Adding/removing one record shifts exactly one bin by ±1.

## Noise Calibration

### Laplace mechanism (mean, count)

Noise `Lap(0, Δ/ε)` is added, where:

- `Δ` = sensitivity (computed above)
- `ε` = `DP_AGGREGATOR_EPSILON` (default `1.0`)

### Gaussian mechanism (histogram)

Noise `N(0, σ²)` per bin, with:

```
σ = Δ · √(2 ln(1.25/δ)) / ε
```

where `δ` = `DP_AGGREGATOR_DELTA` (default `1e-5`).

## Privacy Budget Accounting

Each call to `private_mean`, `private_count`, or `private_histogram` consumes one unit of the configured `ε` (and `δ` for histograms). The cumulative budget is tracked in `DPAggregator.budget_consumed()` and logged to the training run metadata at `models/dp_training_stats.json`:

```json
{
  "epsilon_used": 12.3,
  "delta_used": 3e-5,
  "queries": 37,
  "feature_dp_stats": { ... }
}
```

The training script logs `epsilon_used`, `delta_used`, and `queries` at INFO level after every run.

## Epsilon Selection Guidance

| Use case | Recommended ε | Rationale |
|---|---|---|
| Internal analytics (not shared) | 2.0–4.0 | Loose bound; utility prioritised |
| Shared aggregate Benford statistics | 0.5–1.0 | Standard DP publication threshold |
| High-sensitivity regulatory release | 0.1–0.3 | Conservative; significant noise |

For sharing aggregate Benford statistics externally, **ε = 1.0** is a reasonable starting point — it is a widely accepted default in the DP literature and provides meaningful noise against reconstruction attacks while preserving the directional accuracy of the Benford deviation signal.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DP_AGGREGATOR_EPSILON` | `1.0` | Privacy budget ε (must be > 0) |
| `DP_AGGREGATOR_DELTA` | `1e-5` | Privacy budget δ (must be in (0, 0.5)) |

Both are validated at startup by `Config.validate()`.

## Reproducibility

Pass `random_seed` to `DPAggregator` for deterministic noise in tests:

```python
agg = DPAggregator(epsilon=1.0, delta=1e-5, random_seed=42)
```

Leave `random_seed=None` (default) in production for unpredictable noise.
