# Score normalization guarantees

`PerPairScoreNormaliser` calibrates each raw anomaly score against the rolling
window for one allowlisted asset pair. When the window has at least
`SCORE_NORM_MIN_SAMPLES` observations, the result is
`(number of window scores strictly below x + 0.5) / n`.

## Guarantees

* **Monotonic within model, pair, and calibration window:** if `x <= y`, the
  normalized score for `x` is no greater than the score for `y`.
* **Bounded after calibration:** the result is positive and at most
  `(n + 0.5) / n`; values inside the observed window are strictly below `1.0`.
* **Raw-score pass-through before calibration:** with fewer than 50 samples,
  the raw value is returned and `normalisation_skipped=True`. Callers must not
  treat this value as a percentile.
* **No cross-model absolute guarantee:** percentile calibration preserves rank,
  but does not make scores from different model families or different asset-pair
  windows interchangeable. A model can produce a different score distribution
  for the same transaction population.

The monotonic and bounded properties are covered with Hypothesis over every
model type registered by the scoring stack (`random_forest`, `xgboost`, and
`lightgbm`). The model type is metadata for the calibration contract; the
percentile algorithm itself is intentionally shared.

## Compatibility matrix

`Y` means direct numeric comparison is allowed only when the pair and rolling
window are the same. `N` means consumers should compare ranks within each model,
calibrate to a shared reference population, or use an ensemble calibrator.

| normalized output | random_forest | xgboost | lightgbm |
|---|---:|---:|---:|
| **random_forest** | Y | N | N |
| **xgboost** | N | Y | N |
| **lightgbm** | N | N | Y |

Use `is_directly_comparable()` from `detection.score_normaliser` rather than
assuming all values in `[0, 1]` are cross-model comparable.
