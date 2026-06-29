# Uncertainty Quantification: Conformal Prediction

## What It Is

Every LedgerLens risk score is now accompanied by a **prediction interval** —
a range `[score_lower, score_upper]` that is guaranteed to contain the
"true" risk score with at least **90% probability** (configurable via the
`alpha` parameter, default 0.10).

A score of **72 ± 3** tells you the model is confident; a score of
**72 ± 40** tells you the model is uncertain and should be investigated
further. This is critical for compliance, regulatory, and operational use
cases.

## Why Conformal Prediction?

**Conformal prediction** (Angelopoulos & Bates, 2023) is the only framework
that provides:

- **Distribution-free** guarantees — no assumptions about the data
  distribution
- **Finite-sample validity** — the guarantee holds for any sample size
- **Model-agnostic** — works with any classifier or regressor

This is especially important for LedgerLens because:

1. **Class imbalance** — wash trades are rare (~5% of the dataset)
2. **Distribution shift** — the synthetic training data may not perfectly
   reflect live Testnet/Mainnet distributions
3. **Regulatory requirements** — regulators demand confidence estimates
   alongside fraud flags

## How It Works (Plain Language)

### Calibration Phase (Training)

1. After the train/test split, we reserve an additional **calibration split**
   (10% of training data, stratified by label).
2. For each row in the calibration split, we ask the model for its prediction
   and compute a **nonconformity score**:
   - **Classification**: `1 - softmax_probability(true_label)`
   - **Regression**: `|true_value - predicted_value|`
3. We take the `(1 - alpha)`-quantile of all nonconformity scores — this is
   `q_hat`, the calibration threshold.
4. The threshold `q_hat` is saved to a JSON artifact alongside the model.

### Inference Phase

For every new score:

1. The model makes its prediction.
2. We compute the interval: `[score - q_hat, score + q_hat]`.
3. **Guarantee**: The true risk score falls inside this interval for at
   least `(1 - alpha)` of all test examples, regardless of the data
   distribution.

## Reading the Output

### Standard Score Response

```json
{
  "score": 72,
  "score_lower": 69,
  "score_upper": 75,
  "coverage_guarantee": 0.90,
  "prediction_set": [1],
  "benford_flag": true,
  "ml_flag": true,
  "confidence": 85
}
```

| Field | Meaning |
|---|---|
| `score` | Point estimate (0–100) |
| `score_lower` | Lower bound of the 90% prediction interval |
| `score_upper` | Upper bound of the 90% prediction interval |
| `coverage_guarantee` | The probability that the true score lies within the interval |
| `prediction_set` | Set of class labels the model considers plausible (classification only) |

### Interpreting Interval Width

| Width | Meaning | Action |
|---|---|---|
| **≤ 5 points** | High confidence | Score can be relied upon |
| **5–15 points** | Moderate confidence | Investigate before acting |
| **> 15 points** | Low confidence | Treat as uncertain — gather more evidence |

## Artifact Integrity

Each calibration artifact (`{model_name}_conformal.json`) contains a
SHA-256 hash of its content. On load, the hash is verified. If the artifact
has been tampered with, a `CalibrationIntegrityError` is raised and the
system falls back to maximally conservative bounds `[0, 100]`.

```json
{
  "alpha": 0.10,
  "q_hat": 0.0421,
  "n_cal": 45,
  "random_state": 42,
  "mode": "classification",
  "feature_columns": ["benford_chi_square_1h", ...],
  "classes": [0, 1],
  "sha256": "a1b2c3d4..."
}
```

## Fallback Behavior

When a calibration artifact is missing (e.g., first run after model training
without calibration), `score_with_uncertainty` does **not crash**. Instead it
returns maximally conservative bounds:

```json
{
  "score_lower": 0.0,
  "score_upper": 100.0,
  "coverage_guarantee": 1.0,
  "prediction_set": []
}
```

A warning is logged so operators know the artifact is missing.

## Reproducibility

The calibration split index range is logged in `metrics.json`:

```json
{
  "random_forest": {
    "conformal_empirical_coverage": 0.9111,
    "conformal_q_hat": 0.0421,
    "calibration_split_size": 45,
    "calibration_split_index_range": "0..44",
    ...
  }
}
```

This allows auditors to reproduce the calibration without access to the
raw training data.

## References

- Angelopoulos, A.N. & Bates, S. (2023) "Conformal prediction: A gentle
  introduction." *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Angelopoulos, A.N. et al. (2021) "Uncertainty sets for image classifiers
  using conformal prediction." *ICLR 2021*.
- Romano, Y., Patterson, E. & Candès, E. (2019) "Conformalized quantile
  regression." *NeurIPS 2019*.
