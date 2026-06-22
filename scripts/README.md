# scripts/

## `generate_synthetic_dataset.py`

Generates a synthetic labelled feature matrix for local training, demos,
and tests, without needing live Stellar Horizon data.

The output schema matches `detection/feature_engineering.py::build_feature_matrix`
(`wallet` + all Benford / trade-pattern / volume-timing / wallet-graph
feature columns), plus a `label` column (`1` = wash-trading-like, `0` =
legitimate). Roughly half the rows are generated with "legitimate"
distributions and half with "wash-trading-like" distributions, then
shuffled.

### Usage

```bash
python -m scripts.generate_synthetic_dataset \
    --n-wallets 500 \
    --seed 42 \
    --output data/synthetic_dataset.parquet
```

| Flag | Default | Description |
|---|---|---|
| `--n-wallets` | `500` | Number of synthetic wallet rows to generate |
| `--seed` | `42` | Random seed (controls both data generation and the final shuffle) |
| `--output` | `data/synthetic_dataset.parquet` | Output parquet path |

### Training on the generated dataset

```bash
python -m detection.model_training --data-path data/synthetic_dataset.parquet
```

This trains every model in `MODEL_REGISTRY` (Random Forest, XGBoost,
LightGBM) with SMOTE-balanced training data, writes the fitted models to
`config.MODEL_DIR`, and writes `metrics.json` (AUC-ROC / PR-AUC / F1 per
model) alongside them.

## `run_adversarial_eval.py`

Generates an adversarial robustness report for the wash-trade ensemble
(`detection/adversarial/`). It attacks strongly-flagged wash wallets with
PGD (or FGSM) and measures how much an adversary must perturb features to
slip below the alert threshold.

The report (JSON) contains:

- **Attack success rate** — fraction of wash wallets pushed from `80+` to
  below the target score (default `40`) within the step budget, plus the
  mean steps-to-success.
- **Minimum epsilon per feature** — the smallest single-feature L-infinity
  perturbation (in per-feature std units) that fools the model.
- **Most vulnerable features** — features ranked by ascending min epsilon
  (cheapest to perturb first).
- **Adversarial augmentation gain** — baseline vs. adversarially-retrained
  AUC-ROC on a perturbed test set (target: `>= +0.05`).

### Usage

```bash
python -m scripts.run_adversarial_eval \
    --data-path data/synthetic_dataset.parquet \
    --output reports/adversarial_robustness.json
```

| Flag | Default | Description |
|---|---|---|
| `--data-path` | *(required)* | Labelled feature matrix (parquet) with a `label` column |
| `--model-dir` | *(train on data)* | Directory of trained models; if absent, an ensemble is trained on `--data-path` |
| `--attack` | `pgd` | `pgd` or `fgsm` |
| `--epsilon` | `2.0` | L-infinity budget in per-feature scale (std) units |
| `--steps` | `40` | PGD iterations |
| `--step-size` | `0.2` | PGD step size (scale units) |
| `--target-score` | `40.0` | Score an attack must drop below to "succeed" |
| `--clean-threshold` | `80.0` | Only attack wash wallets scoring at/above this |
| `--max-wallets` | `50` | Cap on wallets attacked (finite differences are costly) |
| `--skip-augmentation` | `false` | Skip the (slower) adversarial-retraining step |
| `--output` | `reports/adversarial_robustness.json` | Output JSON report path |

> Epsilon, step size, and the finite-difference probe are all expressed in
> **per-feature scale (std) units**, so the L-infinity budget is comparable
> across features with very different magnitudes (e.g. a Benford MAD vs. a
> volume ratio).
