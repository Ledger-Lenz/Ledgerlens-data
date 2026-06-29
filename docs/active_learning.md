# Active Learning Pipeline

LedgerLens uses an active learning (AL) pipeline to maximise detection
improvement per analyst-hour. Rather than retraining on all data periodically,
the pipeline selects the wallets that will teach the model the most, routes
them to an analyst for labelling, and incrementally updates the ensemble.

## Overview

```
Unscored wallet pool
        │
        ▼
  Query Strategy            ← selects N most informative wallets
        │
        ▼
  Annotation Queue          ← persists selection; analyst works through it
        │
        ▼
  scripts/annotate.py       ← terminal annotation loop
        │
        ▼
  IncrementalTrainer        ← warm-start or full retrain; rollback if AUC drops
        │
        ▼
  Updated model artifacts
```

## Query Strategies

All strategies live in `detection/active_learning/query_strategies.py`.
Each implements `select(pool, n_query, model=None) -> list[str]`.

| Strategy | Key idea | Best when |
|---|---|---|
| `least_confidence` | Lowest max predicted probability | Quick single-model baseline |
| `margin` | Smallest gap between top-2 class probs | Near-boundary wallets |
| `entropy` | Highest Shannon entropy over class probs | More nuanced uncertainty |
| `coreset` | Greedy k-center in feature space | Maximising coverage of unlabelled space |
| `badge` | k-means++ in (prob × feature) space | Combining uncertainty + diversity |
| `committee_disagreement` | Variance of RF/XGB/LightGBM probability estimates | **Default; best overall** |

Select with `--strategy <name>` in `run_active_learning.py` or set
`AL_QUERY_STRATEGY` in `.env`.

### CommitteeDisagreement efficiency

`CommitteeDisagreement` is the recommended default because it exploits the
three-model ensemble already present in LedgerLens. Wallets where all three
models disagree are those the ensemble is most uncertain about — labelling
them yields maximum information gain. This is equivalent to Query by Committee
(QBC) with KL-divergence-like disagreement measured via variance of
class-1 probability estimates.

**Statistical requirement**: `CommitteeDisagreement`-selected wallets must
have significantly higher score variance across models than randomly selected
wallets (t-test, p < 0.05). This is verified in `tests/test_query_strategies.py`.

## Annotation Workflow

### 1. Populate the queue

```bash
python -m scripts.run_active_learning \
    --pool data/unscored_wallets.parquet \
    --strategy committee_disagreement \
    --batch-size 20
```

This writes wallet IDs to `data/annotation_queue.json` with `status: pending`.

### 2. Annotate

```bash
python -m scripts.annotate --annotator-id yourname
```

For each wallet the CLI shows:

```
================================================================
Wallet : GABCD...
Score  : 87
Strategy: committee_disagreement
Asset Pair: XLM/USDC
SHAP top-3 features:
  benford_chi_square_24h=18.3  (↑ wash, contribution=+0.34)
  round_trip_frequency=0.94    (↑ wash, contribution=+0.28)
  order_cancellation_rate=0.71 (↑ wash, contribution=+0.12)

Label [w=wash, c=clean, s=skip, q=quit]:
```

Labels: `w` = wash trading (1), `c` = clean (0), `s` = skip, `q` = quit.

**Replay mode** — re-annotate previously skipped wallets:

```bash
python -m scripts.annotate --annotator-id yourname --replay
```

**Export** — write annotated rows to parquet for downstream use:

```bash
python -m scripts.annotate --export data/annotated.parquet
```

### 3. Incremental model update

```bash
python -m scripts.run_active_learning \
    --pool data/unscored_wallets.parquet \
    --update data/annotated.parquet \
    --historical data/synthetic_dataset.parquet
```

## Incremental Update Policy

`IncrementalTrainer.update(new_labelled, model_dir)` chooses one of two paths:

| Condition | Action |
|---|---|
| `len(new_labelled) < AL_RETRAIN_THRESHOLD` | **Warm-start**: re-fit XGBoost + LightGBM on new data only using the existing booster as a starting point. RandomForest unchanged. |
| `len(new_labelled) >= AL_RETRAIN_THRESHOLD` | **Full retrain**: combine historical + new data and train from scratch. |

After either path, AUC-ROC is evaluated on a held-out validation split.

**Rollback**: if AUC-ROC drops by more than `AL_ROLLBACK_AUC_DROP` (default 0.01),
the update is rejected, the original model artifacts are restored from `.bak`
copies, and their SHA-256 hashes are re-verified before serving. A rollback
event is logged and recorded in the AL update report.

Update reports are written to `reports/al_update_{timestamp}.json`:

```json
{
  "updated_at": "2026-06-20T12:00:00+00:00",
  "strategy": "warm_start",
  "n_new_samples": 18,
  "auc_before": 0.921,
  "auc_after": 0.934,
  "auc_delta": 0.013,
  "rolled_back": false
}
```

## Annotation Queue Integrity

Each annotation in `data/annotation_queue.json` is protected by an
HMAC-SHA256 computed over `wallet|label|annotator_id|annotated_at`, keyed
by `ANNOTATION_HMAC_SECRET`. Tampered annotations are rejected at export
time before they can influence a training run.

- `annotator_id` must be non-empty (accountability requirement).
- The queue file is written atomically (write to temp file, then `os.rename`).
- The queue file is created with permissions `0o600` (owner read/write only).

## Scheduled Execution

The AL loop runs weekly via `.github/workflows/active_learning.yml`.
Maintainers can also trigger it manually via `workflow_dispatch`.

## Configuration

All settings are controlled via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `AL_QUERY_STRATEGY` | `committee_disagreement` | Query strategy to use |
| `AL_BATCH_SIZE` | `20` | Number of wallets to select per run |
| `AL_RETRAIN_THRESHOLD` | `50` | Min new labels to trigger full retrain |
| `AL_ROLLBACK_AUC_DROP` | `0.01` | Max allowed AUC drop before rollback |
| `AL_QUEUE_PATH` | `data/annotation_queue.json` | Path to queue file |

## Running Tests

```bash
make test     # includes test_query_strategies, test_annotation_queue, test_incremental_trainer
make lint
```

---

## Label Quality Estimation

### Motivation

Human annotators make errors, particularly on borderline wallets where wash-trading
patterns are ambiguous.  Noisy labels cause model underfitting and reduce detection
performance.  `LabelQualityEstimator` (in
`detection/active_learning/label_quality_estimator.py`) runs **confident learning**
([Northcutt et al., 2021](https://jair.org/index.php/jair/article/view/12125)) on
each new annotation batch before it reaches the training set.

### How It Works

1. The **production model** generates out-of-sample predicted probabilities for
   each sample in the batch.
2. `cleanlab.filter.find_label_issues` computes a per-sample **noise score** using
   class-conditional confidence matrices.  Class-conditional estimation handles
   severe class imbalance (typical in wash-trade datasets).
3. The top `LABEL_QUALITY_NOISE_THRESHOLD` percent (default **10 %**) of flagged
   samples are **quarantined** — they are withheld from the training set and logged
   for re-annotation.
4. Per-annotator noise rates are tracked cumulatively; when an annotator's rate
   exceeds `ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD` (default **20 %**) a structured
   `WARNING` log is emitted.

```
New annotation batch
        │
        ▼
LabelQualityEstimator.evaluate_batch()
        │
        ├─ cleanlab: find_label_issues()
        │
        ├─ top 10% noisy → quarantined   ──► data/label_quality_quarantine.ndjson
        │
        └─ remainder → clean_indices     ──► IncrementalTrainer
```

### Configuration

| Environment variable | Default | Description |
|---|---|---|
| `LABEL_QUALITY_NOISE_THRESHOLD` | `0.10` | Fraction of batch to quarantine |
| `ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD` | `0.20` | Alert threshold for per-annotator noise rate |

### Re-annotation Workflow

Quarantined items are written to `data/label_quality_quarantine.ndjson` as NDJSON
records:

```json
{
  "quarantined_at": "2026-06-27T12:00:00+00:00",
  "batch_index": 94,
  "label": 0,
  "noise_score": 0.87,
  "annotator_id": "analyst_alice",
  "wallet": "GABC...",
  "status": "quarantined"
}
```

Operators must review quarantined items and re-annotate them via the normal
annotation workflow.  Quarantined items are **never silently deleted** — the audit
log is the source of truth.

### Bootstrapping Problem

`LabelQualityEstimator` needs a trained model to generate predicted probabilities,
but the model was trained on potentially noisy labels.  Two mitigations:

1. **Use the production model** (not the model-under-training) to generate
   probabilities.  The production model was trained on a larger, presumably cleaner
   dataset.
2. **Iterative refinement**: after the first clean-label retrain, re-run the
   estimator with the new model to catch any remaining noise.

### Usage Example

```python
from detection.active_learning.label_quality_estimator import LabelQualityEstimator
from detection.model_inference import RiskScorer

scorer = RiskScorer()  # production model used for out-of-sample probs

estimator = LabelQualityEstimator(
    model=scorer.models["xgb"],  # or any model with predict_proba
    noise_threshold=0.10,
    annotator_alert_threshold=0.20,
)

result = estimator.evaluate_batch(
    features=feature_matrix,
    labels=labels,
    annotator_ids=annotator_ids,
    wallet_ids=wallet_ids,
)

# Only add clean samples to the training set
clean_X = feature_matrix.iloc[result["clean_indices"]]
clean_y = labels[result["clean_indices"]]
```

### Testing

```bash
pytest tests/test_label_quality_estimator.py -v
```

Tests cover:
- 7-of-10 mislabelled detection rate on a synthetic noisy batch
- Per-annotator noise rate alert fires at > 20 %
- Quarantine log contains `noise_score` and `annotator_id` for every quarantined item
- Quarantined items are never silently deleted
