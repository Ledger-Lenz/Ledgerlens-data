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
| `coreset_hybrid` | `alpha × uncertainty + (1-alpha) × coreset_distance` | **Diversity + uncertainty balance** |

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

### Core-Set Selection and the Hybrid Scorer (Issue #253)

**Why pure uncertainty sampling fails for batches**

Uncertainty-only strategies can select batches where every wallet comes from
the same wash-trading ring. Labelling one effectively labels all, so the
analyst's effort is wasted on near-duplicates.

**Core-set selection** (Sener & Savarese, 2018) solves this by treating
batch selection as a *k-centre* coverage problem: iteratively pick the
candidate furthest from all already-labelled and already-selected points in
the embedding space. This guarantees each new annotation covers a distinct
region of feature space.

**Implementation** (`detection/active_learning/coreset_selector.py`)

- `CoresetSelector.select(candidate_embeddings, n_select, labelled_embeddings)`
  runs the greedy k-centre loop.
- An `hnswlib` approximate nearest-neighbour index answers furthest-point
  queries in O(log N) instead of O(N), scaling to 10 000+ candidates.
- **Cold-start**: when no labelled samples exist, the first batch falls back
  to random selection.
- **Embedding privacy**: vectors are computed in-memory for selection only
  and are never stored, logged, or exported.

**Hybrid scorer** (`coreset_hybrid` strategy)

The hybrid scorer combines both signals:

```
priority(x) = alpha × uncertainty(x) + (1 - alpha) × coreset_distance(x)
```

- `alpha = 1.0` → pure uncertainty (same as `entropy`)
- `alpha = 0.0` → pure diversity (same as `coreset`)
- `alpha = 0.5` → default balanced mode

`alpha` is set via `ACTIVE_LEARNING_ALPHA` (default `0.5`) and can be
overridden at selection time.

**Minimum distance constraint**

Samples closer than `CORESET_MIN_DISTANCE` (default `0.1` L2) to any
already-selected or labelled point are skipped. This prevents near-duplicate
batches even when alpha is set high.

**Tuning alpha**

| Signal | When to increase alpha |
|---|---|
| High uncertainty, low batch diversity | Decrease alpha → more diversity |
| Diverse but all low-confidence | Increase alpha → more uncertainty |
| New pair with no labelled data | Set alpha=0 for first batch (cold-start) |

**Configuration**

| Variable | Default | Description |
|---|---|---|
| `ACTIVE_LEARNING_ALPHA` | `0.5` | Blend weight between uncertainty and diversity |
| `CORESET_MIN_DISTANCE` | `0.1` | Minimum L2 distance between selected samples |

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
| `ACTIVE_LEARNING_ALPHA` | `0.5` | Hybrid scorer blend weight (0=coreset, 1=uncertainty) |
| `CORESET_MIN_DISTANCE` | `0.1` | Min L2 distance between selected samples in coreset_hybrid |
| `ACTIVE_LEARNING_EER_THRESHOLD` | `0.001` | EER below which stopping criterion fires |
| `ACTIVE_LEARNING_CONVERGENCE_WINDOW` | `5` | Rounds used for rolling AUC improvement check |

## Active Learning Stopping Criterion (Issue #256)

The annotation loop should not run indefinitely.  `StoppingCriterion`
detects when additional annotations are unlikely to improve the model and
halts the loop automatically.

### Convergence definition

Convergence is declared when **either** condition is true at the end of
an annotation batch:

1. **EER (Expected Error Reduction)** — the maximum EER across all
   unlabelled candidates falls below `ACTIVE_LEARNING_EER_THRESHOLD`
   (default `0.001`).  EER is approximated as `1 − max_class_probability`
   for the most uncertain unlabelled sample, computed with the current
   production model (no separate model trained).

2. **Rolling AUC plateau** — the mean AUC-ROC improvement over the last
   `ACTIVE_LEARNING_CONVERGENCE_WINDOW` (default `5`) rounds is below
   `0.005`.  This triggers when the model has been effectively converging
   for five consecutive annotation batches.

### Usage

```python
from detection.active_learning.annotation_queue import StoppingCriterion

criterion = StoppingCriterion()
# After each annotation batch:
criterion.record_round_auc(current_auc)
if criterion.should_stop(model=production_model, unlabelled_pool=pool_df):
    criterion.emit_convergence_report(queue_path="data/annotation_queue.json")
    break
```

`run_active_learning.py` checks the criterion automatically before selecting
each batch.

### --force-continue

For research experiments where you want to collect annotations beyond the
convergence point:

```bash
python -m scripts.run_active_learning --pool ... --force-continue
```

This flag suppresses the stopping criterion entirely.  It is intended for
research use only — in production, allow the criterion to halt the loop.

### Convergence report

When the criterion fires, a report is written to
`reports/al_convergence_{timestamp}.json`:

```json
{
  "converged_at": "2026-06-28T20:00:00+00:00",
  "rounds_completed": 7,
  "total_annotations": 140,
  "annotator_counts": {"alice": 80, "bob": 60},
  "auc_history": [0.82, 0.83, 0.837, 0.841, 0.843, 0.844, 0.844]
}
```

The report deliberately excludes raw label values (only annotator IDs and
counts are recorded) to protect annotation privacy.

## Running Tests

```bash
make test     # includes test_query_strategies, test_annotation_queue, test_incremental_trainer, test_issues_253_254_255_256
make lint
```
