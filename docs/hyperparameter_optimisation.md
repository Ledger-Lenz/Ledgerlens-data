# Hyperparameter Optimisation

LedgerLens uses **BOHB-style automated hyperparameter optimisation** (Bayesian
Optimisation + HyperBand) via [Optuna](https://optuna.readthedocs.io/) to find
near-optimal hyperparameters for the three ensemble classifiers — Random Forest,
XGBoost, and LightGBM — faster and more reliably than manual tuning or grid search.

## Algorithm

### Sampler — TPE (single-objective) / NSGA-II (multi-objective)

**Tree-structured Parzen Estimator (TPE)** is functionally equivalent to the
Bayesian surrogate component of BOHB.  It models the distribution of
high-performing trials separately from low-performing trials and proposes new
candidates from the high-performing region:

```
p(x | good) / p(x | bad)   →   maximise EI(x)
```

For multi-objective runs (AUC-ROC + latency), Optuna's **NSGA-II sampler**
evolves a population of trials toward the Pareto front.

### Pruner — MedianPruner (HyperBand early stopping)

Each trial reports its intermediate AUC after training.  `MedianPruner` halts
trials whose performance falls below the median of all trials completed so far,
implementing the HyperBand successive-halving discipline without requiring a
multi-fidelity training schedule.

### No-improvement early stopping

After the first 10 completed trials, a custom callback stops the study if no
improvement in AUC-ROC is seen for `HPARAM_NO_IMPROVEMENT_PATIENCE` consecutive
trials (default 30).  This prevents wasting compute when the search has converged.

## Search Spaces

All bounds are validated before being passed to model constructors — see
[Security](#security) below.

### Random Forest

| Parameter | Type | Range | Notes |
|---|---|---|---|
| `n_estimators` | int | 50 – 500 | Number of trees |
| `max_depth` | int | 2 – 30 | Max tree depth; ≥ 1 required |
| `min_samples_split` | int | 2 – 20 | Min samples to split a node |
| `min_samples_leaf` | int | 1 – 10 | Min samples at a leaf |
| `max_features` | float | 0.1 – 1.0 | Fraction of features per split |

### XGBoost

| Parameter | Type | Range | Notes |
|---|---|---|---|
| `n_estimators` | int | 50 – 500 | Boosting rounds |
| `max_depth` | int | 2 – 10 | Max tree depth |
| `learning_rate` | float (log) | 0.001 – 0.3 | Step size shrinkage |
| `subsample` | float | 0.5 – 1.0 | Row subsampling fraction |
| `colsample_bytree` | float | 0.5 – 1.0 | Column subsampling per tree |
| `min_child_weight` | int | 1 – 10 | Minimum sum of instance weight |
| `gamma` | float | 0.0 – 5.0 | Min split loss reduction |

### LightGBM

| Parameter | Type | Range | Notes |
|---|---|---|---|
| `n_estimators` | int | 50 – 500 | Boosting rounds |
| `max_depth` | int | 2 – 15 | Max tree depth |
| `learning_rate` | float (log) | 0.001 – 0.3 | Step size shrinkage |
| `subsample` | float | 0.5 – 1.0 | Row subsampling fraction |
| `colsample_bytree` | float | 0.5 – 1.0 | Column subsampling per tree |
| `num_leaves` | int | 15 – 255 | Max leaves per tree (LightGBM-specific) |
| `min_child_samples` | int | 5 – 100 | Min data in a leaf |

Log-uniform sampling is used for `learning_rate` so the search spends equal
effort on orders of magnitude (0.001–0.01 vs. 0.1–0.3).

## Multi-Objective Pareto Front (AUC vs. Latency)

Pass `--multiobjective` to `scripts/optimize_hyperparams.py` to additionally
run an NSGA-II search that simultaneously maximises AUC-ROC and minimises
per-sample inference latency (ms):

```
maximise  AUC-ROC(params)
minimise  latency_ms(params)
```

The resulting Pareto front is saved to `models/pareto_front_{model}.json`.

### Selecting an operating point

Use `select_pareto_point(pareto_front, min_auc, max_latency_ms)` to pick the
fastest Pareto point that still satisfies your precision floor:

```python
from detection.hyperparameter_search import (
    run_multiobjective_study,
    select_pareto_point,
)

pareto = run_multiobjective_study("xgboost", n_trials=100, ...)
chosen = select_pareto_point(pareto, min_auc=0.82, max_latency_ms=3.0)
if chosen:
    print(f"AUC={chosen['auc']:.3f}  latency={chosen['latency_ms']:.2f} ms")
    print(chosen["params"])
```

**Trade-off intuition:** shallow trees (low `max_depth`, low `num_leaves`) are
faster at inference but may sacrifice a few AUC points.  The Pareto front lets
operators pick the exact trade-off that fits their SLA.

## Running the Optimiser

```bash
# Optimise all three ensemble models (100 trials, 2 h timeout each):
python -m scripts.optimize_hyperparams \
    --data-path data/synthetic_dataset.parquet

# Single model, 50 trials, with Pareto front:
python -m scripts.optimize_hyperparams \
    --data-path data/synthetic_dataset.parquet \
    --model xgboost --n-trials 50 --multiobjective

# Override via environment variables:
HPARAM_SEARCH_TRIALS=200 HPARAM_SEARCH_TIMEOUT_HOURS=4 \
python -m scripts.optimize_hyperparams --data-path ...
```

Results are written to:

- `models/best_params_{model}.json` — per-model best single-objective params
- `models/best_hyperparams.json` — unified file for all three ensemble models
- `models/pareto_front_{model}.json` — Pareto front (multi-objective only)
- `models/optuna_studies.db` — Optuna SQLite backend (resume-able)

## Integration with `model_training.py`

`detection/model_training.py::train_models()` automatically loads
`models/best_hyperparams.json` (or per-model files) when they exist.
Resolution order:

1. `models/best_params_{model}.json` (per-model, highest priority)
2. `models/best_hyperparams.json` (unified file)
3. Model defaults (if neither file exists)

All loaded parameters are validated against hard bounds before being passed to
the model constructor; invalid params fall back to defaults with a warning.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HPARAM_SEARCH_TRIALS` | `100` | Trials per model |
| `HPARAM_SEARCH_TIMEOUT_HOURS` | `2` | Wall-clock timeout per model (hours) |
| `HPARAM_RANDOM_STATE` | `42` | RNG seed — fixes results for reproducibility |
| `HPARAM_NO_IMPROVEMENT_PATIENCE` | `30` | Early-stop after N non-improving trials |

## Reproducibility

Set `HPARAM_RANDOM_STATE` to a fixed integer before any run.  The TPE sampler
and NSGA-II sampler are both seeded from this value.  Re-running with the same
seed, data, and number of trials produces identical parameter suggestions.

## Security

`validate_hyperparams(model_name, params)` enforces hard bounds on every
parameter before it reaches a model constructor.  The following classes of
degenerate values are rejected:

- `max_depth ≤ 0` — causes silent failure in scikit-learn RF
- `n_estimators ≤ 0` — produces an unusable empty ensemble
- `learning_rate ≤ 0` or negative — divergent training or no learning
- `subsample` / `colsample_bytree` outside (0, 1] — invalid fractions
- `num_leaves < 2` — degenerate single-leaf LightGBM tree

Any parameter value outside its safe range raises `ValueError` and the
offending params are discarded; the pipeline falls back to model defaults and
logs a warning.  This prevents a tampered `best_hyperparams.json` from causing
silent model degradation.
