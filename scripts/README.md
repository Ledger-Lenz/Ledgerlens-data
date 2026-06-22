# scripts/

## `stream.py` — Real-time streaming pipeline

Streams trades from the Stellar Horizon SSE API, maintains a rolling feature
buffer per wallet, and dispatches risk alerts within one ledger close (~5 s)
of a wallet crossing the risk threshold.

### Usage

```bash
# Alert to stdout (local dev default)
python -m scripts.stream

# Webhook delivery
ALERT_WEBHOOK_URL=https://hooks.example.com/alert \
python -m scripts.stream --alert-channel webhook

# WebSocket broadcast (starts ws server on 127.0.0.1:8765)
python -m scripts.stream --alert-channel websocket

# Skip WebSocket server but still use websocket channel via custom ws_client
python -m scripts.stream --alert-channel websocket --no-ws

# Custom dedup window and warmup threshold
python -m scripts.stream --cooldown-seconds 1800 --min-trades 50
```

| Flag | Default | Description |
|---|---|---|
| `--alert-channel` | `stdout` | Alert delivery: `stdout`, `webhook`, or `websocket` |
| `--cooldown-seconds` | `3600` | Per-wallet alert dedup window (seconds) |
| `--min-trades` | `20` | Minimum buffered trades before a wallet is scored |
| `--no-ws` | off | Disable the WebSocket broadcast server |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WATCHED_ASSET_PAIRS` | — | **Required** — comma-separated `CODE:ISSUER` pairs |
| `ALERT_CHANNEL` | `stdout` | Overrides `--alert-channel` flag |
| `ALERT_WEBHOOK_URL` | — | HTTPS webhook endpoint (required for webhook channel) |
| `ALERT_COOLDOWN_SECONDS` | `3600` | Overrides `--cooldown-seconds` flag |
| `WS_PORT` | `8765` | WebSocket server port |
| `WS_BIND_HOST` | `127.0.0.1` | WebSocket bind address |
| `WS_ALLOW_EXTERNAL` | — | Set to `1` to bind to `0.0.0.0` |

### Stdout alert format

```
[ALERT] wallet=G… pair=USDC:…/XLM:native score=83 benford=True ml=True confidence=76
```

See [docs/streaming_architecture.md](../docs/streaming_architecture.md) for the
full pipeline diagram and threading model.

---

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

# Use a specific attacker profile
python -m scripts.generate_synthetic_dataset \
    --profile RingAttacker \
    --n-wallets 20 \
    --output data/ring_dataset.parquet

# Run the full adversarial training loop
python -m scripts.generate_synthetic_dataset \
    --profile AdaptiveAttacker \
    --gan-rounds 5
```

| Flag | Default | Description |
|---|---|---|
| `--n-wallets` | `500` | Number of synthetic wallet rows to generate |
| `--seed` | `42` | Random seed (controls both data generation and the final shuffle) |
| `--output` | `data/synthetic_dataset.parquet` | Output parquet path |
| `--profile` | `NaiveAttacker` | Attacker profile: `NaiveAttacker`, `TimingJitterAttacker`, `AmountConformanceAttacker`, `RingAttacker`, `LayeringAttacker`, `CrossPairAttacker`, `AdaptiveAttacker` |
| `--gan-rounds` | `0` | Run N rounds of adversarial training (0 = skip). Requires `--profile AdaptiveAttacker` |
| `--model-path` | — | Path to trained model `.joblib` file for `AdaptiveAttacker` |

---

## `wash_trade_simulator.py`

The Wash Trade Simulation Engine (WTSE) implements 7 attacker strategy
profiles for generating realistic trade-level data.

### Profiles

| Profile | Description |
|---|---|
| `NaiveAttacker` | Fixed amounts, regular intervals — baseline |
| `TimingJitterAttacker` | Poisson-distributed trade intervals |
| `AmountConformanceAttacker` | Benford-conforming amounts via log-uniform sampling |
| `RingAttacker` | N-wallet ring where each wallet trades with neighbours |
| `LayeringAttacker` | Interleaves wash trades with noise trades (3:1 ratio) |
| `CrossPairAttacker` | Rotates wash volume across K asset pairs |
| `AdaptiveAttacker` | Reads model feature importances and down-weights top features |

### Programmatic usage

```python
from scripts.wash_trade_simulator import NaiveAttacker, trades_to_feature_matrix

profile = NaiveAttacker(n_wallets=10, trades_per_wallet=50)
trades = profile.generate_trades()
features = trades_to_feature_matrix(trades)
```

---

## `adversarial_training_loop.py`

Runs a GAN-style adversarial training loop: Round 0 uses `NaiveAttacker`,
subsequent rounds use `AdaptiveAttacker` (which reads the previous round's
model feature importances). Per-round metrics are written to
`reports/adversarial_loop_{timestamp}.json`.

### Usage

```bash
python -m scripts.adversarial_training_loop \
    --gan-rounds 5 \
    --n-wallets 50
```

| Flag | Default | Description |
|---|---|---|
| `--gan-rounds` | `5` | Number of adversarial rounds |
| `--n-wallets` | `50` | Wallets per generated dataset |
| `--trades-per-wallet` | `100` | Trades per wallet |
| `--output-dir` | `reports` | Directory for output JSON |
| `--seed` | `42` | Random seed |

---

## `evaluate_simulator_realism.py`

Computes realism metrics for the simulator: Fréchet Feature Distance (FFD)
and discriminator accuracy between simulated and real labelled data.

### Usage

```bash
python -m scripts.evaluate_simulator_realism \
    --simulated data/synthetic_dataset.parquet \
    --real data/labelled_dataset.parquet
```

| Flag | Default | Description |
|---|---|---|
| `--simulated` | `data/synthetic_dataset.parquet` | Path to simulated feature matrix |
| `--real` | `data/labelled_dataset.parquet` | Path to real labelled dataset |
| `--output-dir` | `reports` | Directory for output JSON |
| `--seed` | `42` | Random seed |

---

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
