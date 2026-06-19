# Feature Drift Detection & Automated Retraining

## Overview

LedgerLens uses a **Population Stability Index (PSI)** based drift detection
system to monitor when the production feature distribution diverges from the
training-time distribution. When significant drift is detected, the system
automatically retrains the ensemble models and, if the new models meet quality
thresholds, promotes them to production.

## PSI Methodology

### Formula

```
PSI = Σ_i (observed_i - expected_i) × ln(observed_i / expected_i)
```

Where `i` iterates over bins of each feature's value distribution.

### Reference Implementation

The PSI computation in `detection/drift_monitor.py` follows the methodology
described in:

- Yurdakul, B. (2017) "Statistical Properties of Population Stability Index",
  *WUSS Conference*, San Francisco, CA.
  [PDF](https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf)

This is the standard reference for PSI in credit scoring and tabular ML
systems.

### Bin Strategy

Each feature is discretised into **10 quantile-based bins** using the training
(reference) distribution. The same bin edges are applied to the current
distribution to compute observed proportions. If quantile binning fails due to
insufficient unique values, uniform-width bins are used as a fallback.

### Zero-Frequency Handling

Both expected and observed proportions are clipped to ≥ 1e-4 before
computation to prevent log(0) and division-by-zero errors (the standard
"small constant fix" used in production PSI implementations).

## Thresholds

| PSI Range | Classification | Action |
|-----------|---------------|--------|
| PSI < 0.1 | No significant drift | No action |
| 0.1 ≤ PSI < 0.25 | Moderate drift | Monitor (logged) |
| PSI ≥ 0.25 | Significant drift | **Trigger retraining** |

These thresholds follow industry-standard practice from consumer credit
modelling and have been validated across multiple domains.

## Promotion Gate Logic

When drift triggers retraining, the new ensemble models are evaluated against
a held-out test set and compared to the current production model's metrics
(from `metrics.json`).

### Promotion Criteria

A new model is promoted **only if**:

```
AUC-ROC_new ≥ AUC-ROC_old - 0.01
AND
F1_new ≥ F1_old - 0.01
```

For **every model** in the ensemble (Random Forest, XGBoost, LightGBM).

The 0.01 (1 percentage point) tolerance accounts for stochastic variation in
training. If any model degrades beyond this tolerance, the new models are
archived but **not** promoted, and the old production models remain live.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No drift detected |
| 2 | Retrained and promoted |
| 3 | Retrained but not promoted (regression) |
| 1 | Fatal error |

## Architecture

```
                          ┌──────────────────────┐
                          │  model_metadata.json  │
                          │  (reference dist.)    │
                          └──────┬───────────────┘
                                  │
                                  ▼
┌──────────────┐    ┌─────────────────────────┐
│  Horizon API  │───▶│  get_feature_data()     │
│  (last N days)│    │  → feature_matrix       │
└──────────────┘    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  DriftMonitor.compute()  │
                    │  → DriftReport           │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  any_drift_detected?     │
                    └───┬───────────┬─────────┘
                   No   │           │  Yes
                        ▼           ▼
                  Exit 0     ┌─────────────────┐
                             │  Archive old     │
                             │  models          │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │  Retrain on      │
                             │  latest labelled │
                             │  dataset         │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │  Evaluate vs.    │
                             │  old metrics     │
                             └────────┬────────┘
                                      │
                          ┌───────────▼───────────┐
                          │  Metrics within        │
                          │  tolerance?            │
                          └───┬───────────┬───────┘
                         Yes  │           │  No
                              ▼           ▼
                      Promote +     Archive only
                      Exit 2        Exit 3
```

## File Layout

| Path | Purpose |
|------|---------|
| `detection/drift_monitor.py` | `DriftMonitor` class + `DriftReport` dataclass |
| `scripts/retrain_if_drifted.py` | CLI entry point for drift detection + retraining |
| `scripts/list_model_versions.py` | List archived model versions with metrics |
| `models/archive/{timestamp}/` | Archived model artifacts |
| `reports/drift_report_{timestamp}.json` | Per-run drift detection report |
| `reports/retrain_report_{timestamp}.json` | Full retrain + promotion decision report |
| `.github/workflows/retrain.yml` | Scheduled weekly retraining workflow |

## How to Interpret Drift Reports

A `drift_report_{timestamp}.json` contains:

```json
{
  "generated_at": "2026-06-18T02:00:00Z",
  "any_drift_detected": true,
  "n_features_checked": 25,
  "n_features_drifted": 3,
  "features": [
    {"feature": "benford_mad_24h", "psi": 0.32, "drift_flag": true},
    {"feature": "counterparty_concentration_ratio", "psi": 0.05, "drift_flag": false}
  ]
}
```

- **`n_features_drifted`**: Count of features with PSI ≥ 0.25. If > 0,
  retraining trigger fires.
- **`psi`**: The PSI value for a single feature. Values above 0.25 indicate
  significant distribution shift.
- **`drift_flag`**: `true` when PSI ≥ 0.25.

## Testing

```bash
pytest tests/test_drift_monitor.py tests/test_retrain_trigger.py -v
```

Eight tests cover:
1. PSI below threshold (identical distributions)
2. PSI above threshold (shifted distribution)
3. Zero-frequency bin handling
4. Report written to JSON with correct schema
5. Promotion gate blocks regression
6. Promotion gate allows improvement
7. Archive created before promotion
8. All exit codes (0, 2, 3, 1)
