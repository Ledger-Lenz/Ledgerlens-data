# Adversarial Robustness & Backdoor Detection

## Overview

The active learning pipeline in LedgerLens is vulnerable to data poisoning attacks: a malicious annotator could inject backdoor-poisoned examples (wash trades with a specific trigger pattern) labelled as clean, causing the model to misclassify any input containing that trigger. This document describes the Activation Clustering (AC) defence used to detect and quarantine poisoned samples before training.

## Activation Clustering (AC) Defence

### Theory

AC is a post-training defence that identifies backdoor samples by clustering penultimate-layer activations. The key insight is that:

- **Clean samples** within a class have consistent activation patterns
- **Backdoor samples** (with a trigger pattern) form a cohesive minority cluster with distinct activation patterns
- By running k-means clustering (k=2) on activations, we can isolate the minority cluster and flag its members as potential backdoors

### Implementation

The `ActivationClusteringDetector` class in `detection/adversarial/backdoor_detector.py` implements AC:

```python
from detection.adversarial.backdoor_detector import ActivationClusteringDetector

detector = ActivationClusteringDetector(k=2, random_state=42)
flagged_indices = detector.detect(model, X, y, threshold_percentile=25)
```

#### Workflow

1. **Extract penultimate-layer activations** from the trained model:
   - RandomForest: leaf indices
   - XGBoost: raw model predictions (pre-sigmoid)
   - LightGBM: raw scores

2. **Cluster per-class activations** using k-means with k=2:
   - Each class is clustered independently to find class-specific backdoors
   - Activations are standardized before clustering

3. **Flag minority cluster members**:
   - For each class, identify the smaller cluster
   - Apply a safety threshold: if the minority cluster > 25th percentile of cluster sizes, skip flagging (to avoid false positives)
   - Flag samples in the remaining minority cluster

4. **Generate detection report** with statistics

### Parameters

- `k=2`: Number of clusters (one majority, one potential backdoor)
- `threshold_percentile=25`: Minimum cluster size threshold (if minority cluster exceeds this percentile, it's likely not a backdoor)
- `random_state=42`: Reproducibility seed

## Integration with Active Learning

The backdoor detection is integrated into the incremental training pipeline:

```python
# In detection/active_learning/incremental_trainer.py
trainer = IncrementalTrainer()
new_labelled = queue.export_labelled("data/new_annotations.parquet")
report = trainer.update(new_labelled)  # Runs AC before training
```

### Workflow

1. **Before training**, run AC on newly-annotated samples
2. **If > 20% of a class is flagged**:
   - This indicates a high false positive rate (safety check)
   - Emit a critical alert
   - Proceed without quarantine to prevent training blockage
3. **Otherwise**:
   - Quarantine flagged samples (add `quarantine=True` to annotation queue)
   - Train on cleaned data (backdoors removed)

### 20% Safety Threshold

The 20% threshold is a safeguard against false positives:

- AC is designed for scenarios with small numbers of backdoors (<10% of data)
- If > 20% of samples are flagged, the detector is likely making false positives
- In this case, we emit an alert but proceed with training, trusting that false positives are diluted in the training signal

Example:
```
Label=1 (wash trades): 100 samples, 25 flagged (25%)
→ CRITICAL ALERT: "Safety check triggered: 25.0% flagged (threshold 20%)"
→ Train with all 100 samples (no quarantine)
```

## Assumptions & Limitations

### Assumptions

1. **Backdoor samples form a cohesive minority cluster** in activation space
2. **Clean samples have consistent activation patterns** within each class
3. **Trigger patterns are detectable at the penultimate layer**, not hidden in post-hoc processing

### Known Limitations

#### Clean-Label Attacks

AC **does NOT detect clean-label attacks**, where:

- Backdoor samples have correct labels (wash trades labelled as wash trades)
- The trigger pattern is designed to affect only specific inputs
- Example: a wash trade with feature pattern X triggers misclassification of feature pattern Y

In clean-label attacks, the backdoor samples are indistinguishable from legitimate data at the activation level, so no minority cluster emerges.

**Mitigation**: Combine with other defences (e.g., certified robustness training, trigger reverse-engineering)

#### Sparse Triggers

If the trigger pattern is very rare in the data, AC may fail to isolate it:

- Few backdoor samples → minority cluster is very small → easy to mistake for noise
- Solution: Use ensemble of defences or increase backdoor sample injection during training to trigger AC

#### Multi-Modal Backdoors

If the backdoor samples span multiple distinct patterns (multi-modal), AC may split them across multiple clusters:

- Solution: Use k > 2 or combine with other clustering methods

## False Positive Mitigation

AC can produce false positives in several scenarios:

1. **Natural minority patterns**: Some legitimate trades may have unusual activation patterns
2. **Mislabelled data**: Incorrectly-labelled trades may appear as a minority cluster
3. **Imbalanced classes**: In heavily-imbalanced datasets, minority clusters are more likely

### Best Practices

1. **Monitor quarantine rates**: If > 5% of samples are consistently quarantined, review the detector threshold or collect more annotated data
2. **Use percentile threshold**: The `threshold_percentile=25` parameter prevents overflagging; adjust based on empirical false positive rates
3. **Operator override**: Use `scripts/inspect_quarantine.py dismiss --wallet GA...` to override false positives
4. **Log and audit**: All quarantine decisions are logged with reason (`quarantine_reason` field)

## Usage Examples

### Example 1: Run AC Detection Manually

```python
from detection.adversarial.backdoor_detector import ActivationClusteringDetector
from detection.active_learning.incremental_trainer import IncrementalTrainer
import pandas as pd

# Load trained models
trainer = IncrementalTrainer(model_dir="models/")
models = trainer._load_models()

# Load new annotations
new_data = pd.read_parquet("data/new_annotations.parquet")

# Run AC detection
detector = ActivationClusteringDetector(k=2, random_state=42)
flagged = detector.detect(models["random_forest"], new_data.drop(columns=["label"]), new_data["label"])

# Generate report
report = detector.report(new_data.drop(columns=["label"]), new_data["label"], flagged)
print(f"Flagged {report['n_flagged']} samples ({report['flagged_percentage']:.1f}%)")
```

### Example 2: Review Quarantined Samples

```bash
# List all quarantined samples
python scripts/inspect_quarantine.py list

# Print summary
python scripts/inspect_quarantine.py summary

# Dismiss false positive
python scripts/inspect_quarantine.py dismiss --wallet GA...

# Export for analysis
python scripts/inspect_quarantine.py export --output reports/quarantine_analysis.json
```

### Example 3: Integrate with Active Learning Pipeline

```python
from detection.active_learning.annotation_queue import AnnotationQueue
from detection.active_learning.incremental_trainer import IncrementalTrainer

# Add new annotation
queue = AnnotationQueue()
queue.annotate("GABCD...", label=1, annotator_id="alice", notes="wash trade pattern")

# Export and train (AC runs automatically)
new_labelled = queue.export_labelled("data/new_annotations.parquet")
trainer = IncrementalTrainer()
report = trainer.update(new_labelled)

# Check report
if report.get("backdoor_detection", {}).get("n_flagged"):
    print(f"Flagged {report['backdoor_detection']['n_flagged']} potential backdoors")
    if report["backdoor_detection"].get("safety_triggered"):
        print("⚠ Safety threshold triggered — check quarantine for false positives")
```

## Performance Considerations

### Computational Cost

- AC runs k-means clustering once per class
- For a model with 20 features and 100 samples: ~10ms per class
- Total overhead per training round: ~100-200ms (negligible vs. model training)

### Accuracy Tradeoff

- Quarantine rate should be < 5% for normal datasets
- Higher quarantine rates (> 10%) indicate either:
  - High backdoor injection (unusual)
  - False positive detector (check threshold settings)
  - Mislabelled data (review annotations)

## Testing

Unit tests for AC detection are in `tests/test_backdoor_detector.py`:

```bash
# Run tests
pytest tests/test_backdoor_detector.py -v

# Key test: inject 10 backdoors into 100 clean samples
# AC must flag >= 8 of 10 (80% recall)

# Safety test: verify 20% threshold prevents training blockage
```

## Recommendations

1. **Enable AC by default** in production active learning pipelines
2. **Monitor quarantine rates** weekly; investigate > 5% rates
3. **Use operator override sparingly** — log and review all dismissals
4. **Combine with other defences**:
   - Certified robustness training
   - Trigger reverse-engineering (advanced)
   - Data validation and quality checks
5. **Plan for clean-label attacks**: Use ensemble of defences or manual review of high-uncertainty samples

## References

- Wang et al. (2019) "Activation Clustering: An Approach to Detecting Backdoor Attacks"
  https://arxiv.org/abs/1811.03728
- Chen et al. (2019) "Spectral Signatures in Backdoor Attacks"
  https://arxiv.org/abs/1811.00636
- Turner et al. (2018) "Clean-Label Backdoor Attacks on Video Recognition Models"
  https://arxiv.org/abs/1912.02765

---

## FGSM Adversarial Training (Issue #191)

### Motivation

A sophisticated wash-trade operator who reverse-engineers LedgerLens's scoring
system could craft transactions specifically designed to move their Benford and
ML features away from the suspicious region. The existing `FGSMAttack` /
`PGDAttack` classes measure *evasion success rates* — they tell us the model is
vulnerable but do not fix it. **Adversarial training** makes the model robust
to such targeted perturbations by augmenting the training set with the
worst-case examples it would encounter under attack.

### Feature-Space FGSM (`feature_space_fgsm`)

Unlike classical image-space FGSM, LedgerLens operates on a tabular feature
vector of engineered metrics (Benford MAD, counterparty concentration, etc.).
`detection.adversarial.attack.feature_space_fgsm` applies the FGSM step in
this **normalised feature space**:

```python
from detection.adversarial.attack import feature_space_fgsm
from detection.adversarial.robustness import feature_scale_from_matrix

scale = feature_scale_from_matrix(df)  # per-feature std (for L-inf budget)
perturbed = feature_space_fgsm(
    feature_row,   # pd.Series — the wallet's feature vector
    epsilon=0.1,   # L-inf budget in standardised units
    scorer=scorer, # RiskScorer instance
    feature_scale=scale,
)
```

The perturbation direction is **score-increasing** (towards higher risk), so
the augmented example sits near the decision boundary from the clean side.
Training on it forces the model to classify correctly under worst-case
feature-space perturbations.

#### Epsilon Bounds and Feature Validity

Epsilon must be positive. The perturbation budget per feature is
`epsilon × scale[feature]`. Post-perturbation, the following validity
constraints are enforced:

| Constraint | Affected features |
|---|---|
| `≥ 0` | `benford_chi_square_*`, `benford_mad_*`, all rate/probability features, ring density |
| Immutable (scale=0) | `account_age_days` — time cannot be reversed |
| L-inf budget | All features: `|Δ| ≤ epsilon × scale` |

Generating physically impossible feature values (e.g. trade count < 0) is
prevented by the non-negativity clip.

### Adversarial Training Step (`adversarial_training_step`)

`detection.adversarial.robustness.adversarial_training_step` generates
adversarial examples for a training batch and mixes them in at a configurable
ratio (`adv_ratio`, default 0.5 — 50% of wash-trade rows get an adversarial
copy appended):

```python
from detection.adversarial.robustness import adversarial_training_step

X_aug, y_aug = adversarial_training_step(
    X_batch, y_batch, scorer,
    epsilon=0.1,
    adv_ratio=0.5,   # ADV_TRAINING_RATIO
)
```

### Multi-Epoch Adversarial Training Loop (`run_adversarial_training`)

`run_adversarial_training` runs the full iterative loop:

1. Train the ensemble on the (optionally augmented) training set.
2. Generate FGSM adversarial examples using the freshly trained scorer.
3. Mix them into the training set for the next epoch.
4. Report clean AUC-ROC and adversarial AUC-ROC after each epoch.

**Acceptance criterion (Issue #191):** Clean test accuracy must not degrade
by more than **3 percentage points** vs. the baseline trained without
adversarial augmentation.

### Enabling Adversarial Training

Adversarial training is opt-in. Enable it via:

```bash
# Environment variable (takes effect for any training run)
ADV_TRAINING_ENABLED=true python -m detection.model_training --data-path ...

# Or CLI flag
python -m detection.model_training --data-path ... --adv-training
```

| Variable | Default | Description |
|---|---|---|
| `ADV_TRAINING_ENABLED` | `false` | Enable the FGSM adversarial training loop |
| `ADV_TRAINING_EPOCHS` | `3` | Number of adversarial training epochs |
| `ADV_TRAINING_EPSILON` | `0.1` | L-inf budget (standardised units) |
| `ADV_TRAINING_RATIO` | `0.5` | Fraction of wash-trade rows to augment per epoch |

A JSON report (`reports/adversarial_training_<timestamp>.json`) is written
after each run with per-epoch clean and adversarial AUC-ROC values.

### Accuracy–Robustness Trade-off

Adversarial training imposes a trade-off: the model becomes more robust to
perturbations at the cost of slightly lower clean accuracy. Empirically,
`epsilon=0.1` and `adv_ratio=0.5` keeps clean accuracy degradation well within
the 3 pp tolerance. Increase `epsilon` or `adv_ratio` for stronger robustness
guarantees at the cost of higher clean degradation.

### References

- Goodfellow, I. et al. (2015) "Explaining and Harnessing Adversarial Examples"
  https://arxiv.org/abs/1412.6572
- Madry, A. et al. (2018) "Towards Deep Learning Models Resistant to Adversarial Attacks"
  https://arxiv.org/abs/1706.06083
