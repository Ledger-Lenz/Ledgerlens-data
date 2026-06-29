# Historical Backtesting Framework

## Overview

The Historical Backtesting Framework (`scripts/backtest.py`) replays Stellar Horizon trade
history, scores wallets using time-appropriate model versions, and evaluates detection
performance against a hand-curated ground truth of known market manipulation events.

This answers the question: *"Would LedgerLens have caught the XYZ wash-trade campaign
of March 2024?"*

## Replay Architecture

```
                    ┌─────────────────────────────┐
                    │  BacktestEngine.replay()     │
                    │                              │
  ┌─────────────┐   │  For each timestep (24h):    │   ┌──────────────────┐
  │ Horizon API  │───▶  1. Load trades up to t     │──▶│  RiskScorer       │
  │ (paginated)  │   │  2. Build feature matrix    │   │  (score_matrix)   │
  └─────────────┘   │  3. Score each wallet        │   └────────┬─────────┘
                    │  4. Record (wallet, t, score) │            │
                    └─────────────────────────────┘            ▼
                                         ┌──────────────────────────┐
                                         │  compute_detection_lag() │
                                         │  compute_temporal_auc()  │
                                         └──────────────────────────┘
```

### Time Stepping

The engine walks from `start_date` to `end_date` in fixed intervals
(``step_hours``, default 24). At each timestep `t`:

1. All trades with `ledger_close_time <= t` are loaded for each asset pair
   appearing in the ground truth.
2. `build_feature_matrix` computes wallet-level features from the cumulative
   trade data.
3. `RiskScorer.score_matrix` produces a risk score (0–100) per wallet.
4. The result row `(wallet, t, risk_score, features)` is appended to the
   output DataFrame.

This produces a **time series of scores** per wallet, which is then compared
against ground-truth campaign intervals.

## Caching Strategy

### Cache Format

Trades fetched from Horizon are cached as Parquet files in
`data/backtest_cache/{asset_pair}_{date}.parquet`.

### Integrity Verification

Each parquet file has a SHA-256 sidecar file
(`{asset_pair}_{date}.parquet.sha256`) containing the hex digest. Before loading
from cache, the engine verifies:

1. The parquet file exists.
2. The `.sha256` file exists.
3. `SHA256(parquet) == sidecar contents`.

If any check fails, the cache is treated as invalid and the data is re-fetched
from Horizon.

### Cache Invalidation

| Condition | Behaviour |
|---|---|
| `--force-refresh` flag | Ignore cache, re-fetch all data |
| Missing `.sha256` file | Re-fetch |
| SHA-256 mismatch | Re-fetch (data corruption detected) |
| Schema validation failure | Re-fetch |

## Sliding Window Evaluation

`BacktestEngine.sliding_window_eval()` implements a walk-forward validation
scheme to measure model decay between retraining cycles:

```
Window 1:  Train [Jan 1 – Jan 30]  →  Eval [Jan 30 – Feb 6]
Window 2:  Train [Jan 8 – Feb 6]   →  Eval [Feb 6 – Feb 13]
Window 3:  Train [Jan 15 – Feb 13] →  Eval [Feb 13 – Feb 20]
...
```

### No Data Leakage

The training window ends strictly before the evaluation window begins. The
`eval_start` timestamp equals `train_end`, ensuring no temporal overlap.

### Metrics per Window

| Metric | Description |
|---|---|
| AUC-ROC | Area under the receiver operating characteristic curve |
| Precision@10% | Precision among the top 10% of scored wallets |
| Recall@10% | Recall among the top 10% of scored wallets |

The time series of AUC-ROC values reveals model decay: if AUC drops
significantly across windows, the model is not generalising well to new data
and retraining is overdue.

## Ground Truth Dataset

`data/known_manipulation_events.csv` contains 25 curated market manipulation
events on Stellar Mainnet with the following columns:

| Column | Type | Description |
|---|---|---|
| `wallet` | string | Stellar account ID (G...) |
| `asset_pair` | string | Asset pair identifier (CODE:ISSUER/CODE:ISSUER) |
| `campaign_start` | ISO datetime | Start of manipulation campaign |
| `campaign_end` | ISO datetime | End of manipulation campaign |
| `label_source` | URL (HTTPS only) | Public source documenting the event |
| `label_confidence` | int (1–3) | Confidence in the label (3 = highest) |
| `description` | string | Description of the manipulation pattern |

### Source Requirements

- All `label_source` URLs must use **HTTPS**. HTTP sources are rejected with
  a `ValueError` during `load_ground_truth` to prevent MITM attacks on ground
  truth provenance.
- Sources include: DEX Explorer anomaly flags, community reports, public wash
  trade analysis reports, and academic publications.

## Detection Lag

`compute_detection_lag` measures how quickly LedgerLens would have detected
each campaign:

```
lag_hours = first_detection_timestep - campaign_start
```

- **lag = 0**: flagged at the first timestep (ideal — instant detection)
- **lag > 0**: detection delayed by `lag_hours` hours
- **lag = inf**: never crossed the threshold (missed campaign)

The mean detection lag is compared against a **random baseline** (uniform
random scores at each timestep) to demonstrate that early detection is
non-trivial.

## Time-Averaged AUC

`compute_temporal_auc` computes AUC-ROC at each timestep by treating wallets
with active campaigns as positive class and others as negative. The scores are
then averaged across all timesteps in the replay window.

## Report Format

Reports are written as JSON to `reports/backtest_{timestamp}.json` with a
corresponding Markdown version at `reports/backtest_{timestamp}.md`.

### Key Metrics

| Metric | Target | Description |
|---|---|---|
| `time_averaged_auc` | ≥ 0.75 | Minimum bar for production readiness |
| `mean_detection_lag_hours` | — | Average hours from campaign start to first alert |
| `campaigns_detected` | — | Number of campaigns flagged before campaign end |

## CLI Usage

```bash
# Basic replay
python -m scripts.backtest \\
    --start 2024-01-01 \\
    --end 2024-06-30

# Full options
python -m scripts.backtest \\
    --start 2024-01-01 \\
    --end 2024-06-30 \\
    --model-path ./models \\
    --ground-truth data/known_manipulation_events.csv \\
    --output reports/backtest_h1_2024.json \\
    --step-hours 24 \\
    --threshold 70

# With sliding window evaluation
python -m scripts.backtest \\
    --start 2024-01-01 \\
    --end 2024-06-30 \\
    --sliding-window \\
    --window-days 30 \\
    --step-days 7

# Force cache refresh
python -m scripts.backtest \\
    --start 2024-01-01 \\
    --end 2024-06-30 \\
    --force-refresh
```

### All CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--start` | required | Start date (ISO format) |
| `--end` | required | End date (ISO format) |
| `--model-path` | `./models` | Path to model directory |
| `--ground-truth` | `data/known_manipulation_events.csv` | Path to ground truth CSV |
| `--output` | `reports/backtest_{start}_{end}.json` | Output report path |
| `--threshold` | `70` | Risk score threshold for detection |
| `--step-hours` | `24` | Hours per replay timestep |
| `--force-refresh` | off | Ignore cache and re-fetch from Horizon |
| `--sliding-window` | off | Run sliding window evaluation |
| `--window-days` | `30` | Training window size in days |
| `--step-days` | `7` | Step size for sliding window in days |
| `--random-baseline` | off | Compute random baseline detection lag |
| `--random-baseline-simulations` | `100` | Simulations for random baseline |

## Known Limitations

1. **Horizon rate limits**: Historical data fetching from Horizon may be
   throttled. The `utils/retry.py` backoff decorator provides resilience, but
   large replay windows ( > 6 months) may require multiple runs with caching.
2. **Data availability before 2021**: Stellar Horizon has limited trade data
   before late 2021 on some asset pairs. Replays spanning pre-2021 periods
   may return incomplete results.
3. **Feature completeness**: The backtest engine uses the same feature pipeline
   as production (`build_feature_matrix`), but without orderbook events or
   GNN embeddings (which are optional dependencies). Features dependent on
   these data sources will default to 0.
4. **Model versioning**: The framework loads a single model path. In production,
   models are retrained periodically. For strict backtesting, the model used
   should predate the evaluation window.
5. **Ground truth completeness**: The 25 curated events are a sample of known
   manipulation. Missed events in the ground truth will cause the framework to
   overstate detection performance.

## Testing

```bash
# Unit tests (no Horizon calls)
pytest tests/test_backtest.py -v

# Integration tests (requires LEDGERLENS_INTEGRATION_TESTS=1)
LEDGERLENS_INTEGRATION_TESTS=1 pytest tests/test_backtest.py -v -m "" -k integration
```
