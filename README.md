# LedgerLens Data 🔍

[![CI](https://github.com/Ledger-Lenz/Ledgerlens-data/actions/workflows/ci.yml/badge.svg)](https://github.com/Ledger-Lenz/Ledgerlens-data/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Built on Stellar](https://img.shields.io/badge/Built%20on-Stellar-blue?logo=stellar)](https://stellar.org)
[![Soroban Smart Contracts](https://img.shields.io/badge/Smart%20Contracts-Soroban-purple)](https://soroban.stellar.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

Data ingestion, fraud-detection engine, and feature pipeline for **LedgerLens** — a hybrid on-chain fraud detection system for the Stellar DEX that combines **Benford's Law digit analysis** with **ensemble machine learning** to detect wash trading and artificial volume.

> *"On a transparent ledger, every transaction is visible. LedgerLens makes them legible."*

## Overview

This repository holds the data and detection layer of LedgerLens: the pipelines that pull trade data from the Stellar Horizon API, compute Benford's Law anomaly metrics, extract on-chain ML features, train and run the ensemble classifiers, and produce the **LedgerLens Risk Score (0–100)** consumed by the API, dashboard, and Soroban contract layer.

## The Problem

Wash trading — simultaneously buying and selling the same asset to artificially inflate volume — is one of the most pervasive forms of market manipulation in DeFi. Stellar's 3–5 second settlement finality and sub-cent fees make wash trading cheap to execute at scale, while the sheer volume of on-chain activity makes manual detection impossible. No production-grade, open-source detection system exists for the Stellar DEX — LedgerLens fills that gap.

## What This Repo Does

- **Ingests** — Streams and bulk-loads trade history from both the Stellar SDEX (order book) and AMM liquidity pools, plus order book events and account activity from the Stellar Horizon API
- **Detects** — Computes Benford's Law anomaly metrics (chi-square, per-digit Z-scores, MAD) per wallet, per asset, and per trading pair across rolling time windows; detects **cross-venue coordination** between SDEX and AMM pool activity
- **Scores** — Extracts 37+ on-chain features (including 7 cross-venue coordination features) and runs ensemble ML classifiers (Random Forest, XGBoost, LightGBM) to produce a 0–100 risk score per wallet/asset pair
- **Explains** — Generates SHAP-based interpretability output so every risk score is auditable

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LAYER 1: DATA INGESTION                  │
│                                                               │
│  Stellar Horizon API → Trade history, order book events,     │
│  account activity, asset metadata, payment paths             │
│  Streamed continuously via SSE or polled per ledger close    │
└──────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 2: DETECTION ENGINE                   │
│                                                               │
│  ┌─────────────────────┐   ┌──────────────────────────┐     │
│  │  Benford's Law       │   │  Ensemble ML Models       │    │
│  │  Anomaly Engine      │   │  (RF, XGBoost, LightGBM)  │    │
│  │                       │   │                           │    │
│  │  • Chi-square stat   │   │  • 30+ on-chain features  │    │
│  │  • Z-score per digit │   │  • Trained on labelled     │    │
│  │  • MAD score         │   │    wash trade patterns     │    │
│  │  • Per asset, per    │   │  • SHAP interpretability    │    │
│  │    wallet, per pair  │   │  • Continuous retraining    │    │
│  └──────────┬──────────┘   └──────────────┬────────────┘     │
│             │                              │                   │
│             └──────────┬───────────────────┘                  │
│                         ▼                                      │
│              LedgerLens Risk Score (0–100)                    │
└──────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
            Consumed by the API, dashboard, and the
            ledgerlens-score Soroban contract
```

## Benford's Law on the Blockchain

Benford's Law states that in many naturally occurring numerical datasets, the leading digit 1 appears ~30.1% of the time, declining to ~4.6% for digit 9. Genuine trading activity produces a wide, unbiased spread of transaction sizes that conforms to this distribution. Wash trading — typically driven by bots using fixed lot sizes, round numbers, or algorithmically generated amounts — deviates systematically from it.

LedgerLens applies three Benford metrics over rolling time windows (1h, 4h, 24h, 7d, 30d):

| Metric | What it measures |
|---|---|
| **Chi-square statistic** | Whether the overall digit distribution deviates significantly from Benford's expected distribution |
| **Z-score (per digit)** | Whether any individual digit (1–9) appears with significantly higher or lower frequency than expected |
| **Mean Absolute Deviation (MAD)** | A composite measure of distributional divergence; values above 0.015 indicate non-conformity |

Benford signals alone aren't definitive — legitimate high-frequency market makers can also produce non-Benford distributions — which is why they're combined with the ML layer below.

## Machine Learning Layer

### Features (36+, grouped by category)

**Benford Features (15)**
- Chi-square, Z-score, and MAD for transaction amounts across 5 rolling time windows

**Trade Pattern Features**
- Counterparty concentration ratio (fraction of volume with a single counterparty)
- Round-trip trade frequency (trades returning assets to the originating wallet within N ledgers)
- Self-matching rate (buy/sell orders from wallets sharing funding sources)
- Order cancellation rate and timing patterns

**Volume and Timing Features**
- Volume-to-unique-counterparty ratio
- Intra-minute trade clustering coefficient
- Off-hours activity ratio
- Volume spike frequency relative to rolling baseline

**Wallet Graph Features**
- Funding source similarity score
- Network centrality within trading cluster graphs
- Account age at time of trading activity

**Cross-Asset Coordination Features (6)**
- **Cross-pair trade synchrony** (0–1): Fraction of trades where the wallet simultaneously trades on other pairs within a configurable synchrony window (default: 30 seconds). High values indicate coordinated multi-pair activity — a strong wash-trading signal.
- **Net asset flow deviation** (0–1+): Maximum absolute net asset flow normalized by total volume. Values close to 0 indicate closed-cycle trading (suspect); large values indicate genuine inventory management.
- **Cross-pair counterparty overlap** (0–1): Jaccard similarity of the wallet's counterparty sets across different pairs. Wash traders reuse the same sock-puppet wallets across pairs; legitimate market makers have pair-specific counterparties.
- **Cross-pair volume correlation** (-1 to 1): Pearson correlation of trade volumes across pairs, bucketed by minute. Coordinated wash traders' volumes spike together; legitimate traders' pair-specific volumes are uncorrelated.
- **Pair diversity score** (0–1): Shannon entropy of volume distribution across pairs, normalized. High entropy = volume spread across many pairs (market maker); low entropy = concentrated on one or two pairs.
- **Cross-pair MAD consistency**: Standard deviation of Benford MAD scores across pairs. Low values mean all pairs have similar Benford conformity (consistent wash-trading pattern). High values indicate mixed conformity (concentrated on specific pairs).

### Models

| Model | Role |
|---|---|
| **Random Forest** | Stable baseline; handles missing features gracefully |
| **XGBoost** | Primary classifier; strongest performance on tabular on-chain data |
| **LightGBM** | High-speed inference for real-time scoring |

Models are trained with **SMOTE** to handle class imbalance and evaluated using **AUC-ROC**, **Precision-Recall AUC**, and **F1-score**. SHAP values provide interpretable explanations for every risk score.

## Repository Structure

```
ledgerlens-data/
│
├── README.md                         ← This file
├── requirements.txt                  ← Python dependencies
├── pyproject.toml                    ← Lint/format/test config (ruff, black, mypy, pytest)
├── Makefile                          ← make install / lint / format / test / run
├── Dockerfile                        ← Container image (entrypoint: run_pipeline.py)
├── run_pipeline.py                   ← Full detection pipeline entry point
│
├── ingestion/
│   ├── horizon_streamer.py           ← Real-time trade data from Horizon API
│   ├── historical_loader.py          ← Bulk historical trade ingestion
│   ├── orderbook_loader.py           ← Order-book event ingestion (cancellation rate)
│   ├── account_activity_loader.py    ← Account creation/funding event ingestion (funding graph)
│   └── data_models.py                ← Pydantic schemas for trade records
│
├── detection/
│   ├── benford_engine.py             ← Benford's Law feature computation
│   ├── feature_engineering.py        ← 30+ feature builder
│   ├── wallet_graph.py               ← Funding-graph similarity/centrality features
│   ├── model_training.py             ← Train ensemble classifiers (CLI)
│   ├── model_inference.py            ← Real-time risk scoring
│   ├── drift_monitor.py              ← PSI-based feature drift detection
│   ├── shap_explainer.py             ← SHAP interpretability layer
│   ├── persistence.py                ← SQLAlchemy RiskScore model + engine
│   └── risk_score_store.py           ← RiskScore upsert/read repository
│
├── integrations/
│   └── contract_client.py            ← ledgerlens-score Soroban contract client
│
├── streaming/
│   ├── feature_buffer.py             ← FeatureBuffer (rolling per-wallet trade buffer)
│   ├── alert_dispatcher.py           ← AlertDispatcher (threshold, dedup, delivery)
│   ├── ws_server.py                  ← asyncio WebSocket server for dashboard push
│   └── pipeline.py                   ← StreamingPipeline orchestrator
│
├── scripts/
│   ├── stream.py                     ← Real-time pipeline CLI (python -m scripts.stream)
│   ├── generate_synthetic_dataset.py ← Synthetic labelled dataset for local training/demo
│   ├── retrain_if_drifted.py         ← Automated drift detection + retraining trigger
│   └── list_model_versions.py        ← List archived model versions with metrics
│
├── docs/
│   ├── streaming_architecture.md     ← Real-time pipeline diagram and component docs
│   └── drift_detection.md           ← PSI drift detection methodology and retraining docs
│
├── utils/
│   ├── logging.py                    ← Shared logger setup
│   └── retry.py                      ← Retry/backoff decorator for Horizon calls
│
├── tests/
│   ├── test_benford.py
│   ├── test_features.py
│   ├── test_orderbook.py
│   ├── test_wallet_graph.py
│   ├── test_persistence.py
│   ├── test_contract_client.py
│   ├── test_model_training.py
│   ├── test_inference_shap.py
│   ├── test_drift_monitor.py
│   └── test_retrain_trigger.py
```

## Quick Start

```bash
# Install dependencies
make install
# (equivalent to: pip install -r requirements.txt)

# Generate a synthetic labelled dataset and train the ensemble
python -m scripts.generate_synthetic_dataset --output data/synthetic_dataset.parquet
python -m detection.model_training --data-path data/synthetic_dataset.parquet

# Run the full detection pipeline
python run_pipeline.py

# Score a single wallet on-demand (targeted investigation)
python -m scripts.score_wallet --wallet <G...> --pair "USDC:<G...>/XLM:native"
```

### Real-time streaming (`scripts/stream.py`)

After training models, `scripts/stream.py` scores wallets in real time as
trades land on the Stellar ledger and delivers alerts within one ledger close
(~5 seconds) of a wallet crossing the risk threshold.

```bash
# Stdout alerts (local dev)
python -m scripts.stream

# HTTP webhook delivery (must be https://)
ALERT_WEBHOOK_URL=https://hooks.example.com/alert \
python -m scripts.stream --alert-channel webhook

# WebSocket broadcast to dashboard subscribers
python -m scripts.stream --alert-channel websocket
```

| Flag | Default | Description |
|---|---|---|
| `--alert-channel` | `stdout` | `stdout`, `webhook`, or `websocket` |
| `--cooldown-seconds` | `3600` | Per-wallet alert dedup window |
| `--min-trades` | `20` | Minimum trades before a wallet is eligible for scoring |
| `--no-ws` | off | Disable the WebSocket broadcast server |

**New environment variables for streaming:**

| Variable | Default | Description |
|---|---|---|
| `ALERT_CHANNEL` | `stdout` | Alert delivery channel |
| `ALERT_WEBHOOK_URL` | — | Required when `ALERT_CHANNEL=webhook`; must be `https://` |
| `ALERT_COOLDOWN_SECONDS` | `3600` | Per-wallet alert dedup window (seconds) |
| `WS_PORT` | `8765` | WebSocket server port |
| `WS_BIND_HOST` | `127.0.0.1` | WebSocket bind address (loopback by default) |
| `WS_ALLOW_EXTERNAL` | — | Set to `1` to allow non-loopback WebSocket binding |
| `CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS` | `30` | Time window (seconds) for detecting simultaneous trades across pairs |

See [docs/streaming_architecture.md](docs/streaming_architecture.md) for the
full pipeline diagram, threading model, and latency budget.

#### Kafka deployment option (`STREAMING_BACKEND=kafka`)

The default `sse` backend runs one thread per pair in a single process. For
scale-out, durability, event replay, and backpressure, set
`STREAMING_BACKEND=kafka` to route trades through Apache Kafka instead. The
`sse` backend remains the default and is unchanged — operators without Kafka
need do nothing.

```bash
# Bring up Zookeeper, Kafka, the producer, 3 scorer replicas, Prometheus + Grafana
docker-compose up --scale ledgerlens-scorer=3
```

Architecture: a `HorizonKafkaProducer` serialises each Horizon SSE trade to Avro
(`data/trade_avro_schema.json`) and produces it to
`ledgerlens.trades.{asset_pair}`, keyed by `wallet_id` so per-wallet ordering is
preserved. `KafkaWorker` replicas in the shared `ledgerlens-scorer` consumer
group consume via a wildcard subscription, score wallets, dispatch alerts, and
commit offsets only after dispatch (at-least-once). Serialisation failures go to
a dead-letter queue (`ledgerlens.trades.dlq`); they are never auto-retried.

| Variable | Default | Description |
|---|---|---|
| `STREAMING_BACKEND` | `sse` | `sse` (threaded) or `kafka` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Broker list |
| `KAFKA_SASL_USERNAME` | — | SASL username (read from env only) |
| `KAFKA_SASL_PASSWORD` | — | SASL password (read from env only) |
| `KAFKA_LAG_ALERT_THRESHOLD` | `500` | Consumer lag (messages) that triggers a CRITICAL log |

Each scorer exposes Prometheus metrics (`kafka_lag_by_partition`,
`scoring_latency_ms`, `alerts_dispatched_total`, `kafka_messages_consumed_total`)
on `KAFKA_METRICS_PORT` (default `9100`); Grafana ships a pre-configured Kafka
lag + scoring-latency dashboard. See
[docs/streaming_architecture.md](docs/streaming_architecture.md#kafka-streaming-backend-issue-36)
for the Kafka topology, partition strategy, and at-least-once semantics.

### `run_pipeline.py` flags

| Flag | Effect |
|---|---|
| `--since <ISO date>` | Only load trades from this date onward (default: all available) |
| `--no-persist` | Skip writing scored wallets to `RISK_SCORE_DB_URL` |
| `--no-orderbook` | Skip loading order-book events (faster; `order_cancellation_rate` stays `0`) |
| `--no-graph` | Skip loading account activity and building the wallet funding graph (faster; `funding_source_similarity` and `network_centrality` stay `0`) |
| `--submit-onchain` | Submit flagged wallets' `RiskScore` to the `ledgerlens-score` contract via `integrations/contract_client.py` |
| `--dry-run` | Run all pipeline stages but skip every write — no DB persistence and no on-chain submission (implies `--no-persist`; silently skips `--submit-onchain`). Flagged wallets are still printed. |

## Model Artifacts

Trained models and their associated metadata are stored in `config.MODEL_DIR` (default: `./models`).

### `model_metadata.json`

Every training run produces a `model_metadata.json` sidecar file. This is used by the `RiskScorer` at load time to ensure that the current feature schema matches the schema the model was trained on, preventing silent scoring errors due to feature drift.

**Schema:**
```json
{
  "trained_at": "2026-06-16T12:00:00Z",
  "data_path": "data/synthetic_dataset.parquet",
  "n_training_rows": 400,
  "n_test_rows": 100,
  "feature_columns": ["benford_chi_square_1h", "benford_mad_1h", "..."],
  "feature_schema_hash": "sha256:<hash-of-sorted-feature-column-list>",
  "model_names": ["random_forest", "xgboost", "lightgbm"],
  "python_version": "3.11.9",
  "ledgerlens_version": "0.2.0"
}
```

If the `feature_schema_hash` computed from the input feature row does not match the hash in the metadata, `RiskScorer.score()` will raise a `RuntimeError` detailing the mismatched columns.

## Continuous Retraining Pipeline

LedgerLens includes an automated retraining pipeline that detects feature drift
using the **Population Stability Index (PSI)** and safely promotes new models
without disrupting production.

### Drift Detection

The `DriftMonitor` class (`detection/drift_monitor.py`) computes PSI for every
feature column by comparing the current production distribution (from recent
Horizon data) against the training-time reference distribution stored in
`model_metadata.json`. PSI ≥ 0.25 triggers automatic retraining.

### Retraining Workflow

The `scripts/retrain_if_drifted.py` script:

1. Loads the reference distribution from `model_metadata.json`
2. Builds a feature matrix from recent Horizon data (`--lookback-days`)
3. Computes PSI drift via `DriftMonitor`
4. If drift is detected: archives old models, trains new ones, evaluates
   against old metrics, and promotes only if all models meet AUC-ROC/F1
   tolerance (≥ old - 0.01)
5. Writes detailed reports to `reports/`

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No drift — no action |
| 2 | Retrained and promoted |
| 3 | Retrained but not promoted (regression) |
| 1 | Fatal error |

### Scheduled Execution

A GitHub Actions workflow (`.github/workflows/retrain.yml`) runs weekly
(Monday 02:00 UTC) or on-demand via `workflow_dispatch`. It uses OIDC for
artifact store authentication — no long-lived secrets.

See [docs/drift_detection.md](docs/drift_detection.md) for full methodology,
thresholds, and architecture diagrams.

### `scripts/`

See [`scripts/README.md`](scripts/README.md) for detailed usage of:
- `generate_synthetic_dataset.py` — synthetic labelled feature matrix for
  local training/demo/tests without live Horizon data
- `retrain_if_drifted.py` — automated drift detection and retraining trigger
- `list_model_versions.py` — list archived models with training dates and metrics

## Active Learning

LedgerLens includes an active learning pipeline that intelligently selects the most informative
wallets for analyst annotation, minimising labelling effort while maximising model improvement.

```bash
# Populate the annotation queue (selects 20 wallets by committee disagreement):
python -m scripts.run_active_learning --pool data/unscored_wallets.parquet

# Annotate wallets interactively:
python -m scripts.annotate --annotator-id yourname

# Export annotations and update models:
python -m scripts.annotate --export data/annotated.parquet
python -m scripts.run_active_learning \
    --pool data/unscored_wallets.parquet \
    --update data/annotated.parquet
```

The pipeline runs automatically every Monday at 08:00 UTC via
`.github/workflows/active_learning.yml`. See [`docs/active_learning.md`](docs/active_learning.md)
for the full query strategy comparison, annotation workflow, and incremental update policy.

| Variable | Default | Description |
|---|---|---|
| `AL_QUERY_STRATEGY` | `committee_disagreement` | Query strategy |
| `AL_BATCH_SIZE` | `20` | Wallets selected per run |
| `AL_RETRAIN_THRESHOLD` | `50` | Min labels for full retrain |
| `AL_ROLLBACK_AUC_DROP` | `0.01` | Max AUC drop before rollback |

## Development

```bash
make install   # pip install -r requirements.txt
make lint      # ruff check .
make format    # black .
make test      # pytest
make run       # python run_pipeline.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full dev setup and PR process.

## Organization Map

This section orients agents and contributors working across the LedgerLens
organization. It describes how the six repos fit together, what each one
owns, and the shared contracts/interfaces that let them be developed
independently while staying integrated.

### The Six Repositories

| Repo | Role | Primary stack |
|---|---|---|
| **`.github`** | Org-wide config: shared GitHub Actions workflows, issue/PR templates, CODEOWNERS, security policy | YAML / Actions |
| **`ledgerlens-data`** *(this repo)* | Ingestion + detection: pulls Stellar DEX trade data, computes Benford's Law metrics and ML features, trains/runs the ensemble, produces the **LedgerLens Risk Score** | Python (pandas, scikit-learn, XGBoost, LightGBM, SHAP) |
| **`ledgerlens-core`** | Shared types, schemas, and SDK code used by `api`, `dashboard`, and `contract` (e.g. `RiskScore`, asset pair identifiers, scoring constants) | TypeScript / Python (shared package) |
| **`ledgerlens-api`** | Public REST API — serves risk scores, alerts, asset rankings; reads from this repo's output store and the on-chain contract | TypeScript (FastAPI/Express-style) |
| **`ledgerlens-dashboard`** | Web dashboard — visualizes risk scores, flagged wallets/pairs, SHAP explanations | React/Vite |
| **`ledgerlens-contract`** (`ledgerlens-score`) | Soroban smart contract — on-chain truth layer for risk scores (`submit_score`, `get_score`) | Rust / Soroban SDK |

### End-to-End Data Flow

```
 Stellar Horizon API
        │
        ▼
┌───────────────────┐
│  ledgerlens-data   │  ingestion/ → detection/ → RiskScore records
│  (this repo)       │
└─────────┬──────────┘
          │ writes scored records (DB / object store / queue)
          │
          ├──────────────────────────────┐
          ▼                               ▼
┌───────────────────┐          ┌────────────────────────┐
│  ledgerlens-contract│ ◄────── │   ledgerlens-api        │
│  submit_score()     │  calls  │   reads RiskScore store │
│  get_score()        │         │   reads on-chain scores │
└─────────┬──────────┘          └───────────┬────────────┘
          │                                  │
          │ get_score() (composable,        │ REST
          │ used by other Soroban contracts) │
          ▼                                  ▼
   Other Soroban protocols          ┌────────────────────┐
   (AMMs, lending, aggregators)     │ ledgerlens-dashboard│
                                     └────────────────────┘

ledgerlens-core: shared types/SDK imported by api, dashboard, contract clients
.github: CI/CD workflows + policies shared by all of the above
```

**Summary of the handoff points:**

1. **`ledgerlens-data` → `ledgerlens-api`**: this repo's pipeline output (the
   `RiskScore` feature/score records from `detection/model_inference.py`)
   is written to a shared store (`RISK_SCORE_DB_URL`). `ledgerlens-api`
   reads from that store to serve `/score/{wallet}/{pair}`,
   `/alerts/recent`, and `/assets/risk-ranking`.
2. **`ledgerlens-data` / `ledgerlens-api` → `ledgerlens-contract`**: an
   authorized service account calls `submit_score(wallet, asset_pair,
   score, timestamp)` to register scores on-chain. Any contract can then
   call `get_score(wallet, asset_pair)` permissionlessly.
3. **`ledgerlens-core` → everyone**: defines the canonical `RiskScore`
   shape, asset-pair identifier format, and risk thresholds so all repos
   agree on field names and units without copy-pasting types.
4. **`.github` → everyone**: shared CI workflows (lint/test/build per
   language), required status checks, and release automation.

### Shared Contracts

These are the data shapes and conventions every repo must agree on. If you
change one of these in `ledgerlens-data`, update `ledgerlens-core` and open
linked issues/PRs in the consuming repos.

#### `RiskScore` (on-chain + API shape)

```rust
pub struct RiskScore {
    pub score: u32,          // 0-100; higher = more suspicious
    pub benford_flag: bool,  // True if Benford anomaly detected
    pub ml_flag: bool,       // True if ML classifier flagged
    pub timestamp: u64,      // Ledger timestamp of last update
    pub confidence: u32,     // Model confidence 0-100
}
```

Produced in this repo by `detection/model_inference.py::RiskScorer.score()`.
Mirrors `ledgerlens-contract`'s `submit_score` payload and
`ledgerlens-api`'s `/score/{wallet}/{pair}` response.

#### Asset pair identifier

Format: `CODE:ISSUER/CODE:ISSUER` (e.g. `USDC:GA5Z.../XLM:native`). The
pipeline enforces this format as the canonical `pair_id` string in every
`RiskScore` record — one record per `(wallet, pair_id)` tuple. See
`ingestion/data_models.py::Asset.pair_id()`. Used consistently across the
API path parameters, contract `Symbol` arguments, and dashboard routing.

#### Risk thresholds

- `RISK_SCORE_FLAG_THRESHOLD` (default `70`) — score at or above this is
  surfaced as "flagged" in the API and dashboard.
- `MAD_NONCONFORMITY_THRESHOLD` (`0.015`) — Benford MAD above this sets
  `benford_flag = true` (Nigrini, 2012).

#### Feature schema

The 36+ feature columns produced by
`detection/feature_engineering.py::build_feature_matrix` are the training
input for `detection/model_training.py`. Any new feature column must be
added to both the training pipeline and `model_inference.py`'s
`FEATURE_COLUMNS_EXCLUDE`-aware scoring path, and documented in
`ledgerlens-core` if other repos need to display feature attributions
(SHAP output from `detection/shap_explainer.py`). Cross-asset coordination
features require that `all_pairs_df` be passed to `build_feature_matrix`;
if not provided, these features default to 0.0.

### This Repo (`ledgerlens-data`) — Current State

```
ledgerlens-data/
├── config.py                  ← env-driven configuration
├── run_pipeline.py            ← pipeline entry point (ingest → features → score → persist/submit)
├── ingestion/
│   ├── data_models.py          ← Trade / OrderBookEvent / AccountActivity (shared schema)
│   ├── horizon_streamer.py      ← live trade stream (auto-reconnect)
│   ├── historical_loader.py    ← bulk historical trade load (retry/backoff)
│   └── orderbook_loader.py     ← order-book event ingestion (cancellation rate)
├── detection/
│   ├── benford_engine.py        ← chi-square / Z-score / MAD (done)
│   ├── feature_engineering.py  ← 30+ feature builder (done)
│   ├── wallet_graph.py          ← funding-graph similarity/centrality
│   ├── model_training.py        ← ensemble training CLI
│   ├── model_inference.py       ← RiskScorer ensemble scoring
│   ├── shap_explainer.py        ← per-wallet + ensemble SHAP attributions
│   ├── persistence.py           ← SQLAlchemy RiskScore model + engine
│   └── risk_score_store.py      ← RiskScore upsert/read repository
├── integrations/
│   └── contract_client.py      ← ledgerlens-score Soroban contract client
├── scripts/
│   └── generate_synthetic_dataset.py ← synthetic labelled dataset generator
└── tests/ (8 modules, see Repository Structure above)
```

#### Known gaps / TODOs

- `model_training.py` trains on `scripts/generate_synthetic_dataset.py`'s
  synthetic data by default; the real labelled wash-trade dataset is still
  the "Open dataset release" roadmap item.
- `--submit-onchain` assumes a deployed `ledgerlens-score` contract and a
  funded `LEDGERLENS_SUBMITTER_SECRET`; it has unit test coverage via mocks
  but hasn't been exercised against a live Soroban network from this repo.

When picking up one of these, check whether `ledgerlens-core` already
defines the relevant shared type before inventing a new one.

## Roadmap

### Phase 1 — Foundation
- [ ] Stellar Horizon API ingestion pipeline (historical + streaming)
- [ ] Benford's Law engine for on-chain transaction amounts
- [ ] Initial feature engineering from SDEX trade data
- [ ] Baseline ML model training on historical wash trade patterns
- [ ] Internal testing on Stellar Testnet

### Phase 2 — Core Product
- [ ] Full ensemble model training and evaluation
- [ ] SHAP interpretability integration
- [ ] Public REST API (v1) with rate limiting
- [ ] Web dashboard (beta)

### Phase 3 — Ecosystem Integration
- [ ] Mainnet deployment
- [ ] SDK for protocol integrations (Python + JavaScript)
- [x] Open dataset release: labelled SDEX wash trade patterns — see [`data/dataset_card.md`](data/dataset_card.md)

## Security

LedgerLens includes a hardened inference stack to protect against adversarial attacks on the model layer itself. See [`docs/security.md`](docs/security.md) for full details.

### Artifact Integrity (Ed25519 Trust Chain)

Every trained model artifact is verified through a four-step chain before loading:

1. SHA-256 of the `.joblib` file matches the value recorded in `metrics.json`
2. `metrics.json` carries a valid Ed25519 detached signature (`metrics.json.sig`)
3. The signing key fingerprint matches `TRUSTED_SIGNING_KEY_FINGERPRINT`
4. The training dataset SHA-256 matches the recorded provenance (optional)

`ModelIntegrityError` is raised on any failure. A CI grep check enforces that every `joblib.load` in `detection/` is immediately followed by `verify_chain`.

### Byzantine-Fault-Tolerant Ensemble Voting

The three models (RF, XGBoost, LightGBM) vote using a **trimmed mean / median** scheme. If the spread across model scores exceeds `BFT_SCORE_DIVERGENCE_THRESHOLD` (default 30 points), the outlier scores are trimmed and the median is used — ensuring a single compromised model cannot shift the final score by more than ~17 points. Divergence events are logged, counted in a Prometheus counter (`bft_divergence_detected_total`), and surfaced in the score response as `bft_divergence: true`.

### Label Poisoning Detection

Each training run records the SHA-256 of the input dataset and the label distribution. If the wash-trade ratio has shifted more than `POISON_LABEL_RATIO_THRESHOLD` (default 15%) from the stored baseline, training is aborted and an alert is written to `reports/poisoning_alert_{timestamp}.json`.

### Annotation Queue Integrity

Each annotation in `data/annotation_queue.json` is protected by an HMAC-SHA256 computed over `wallet|label|annotator_id|annotated_at`, keyed by `ANNOTATION_HMAC_SECRET`. Tampered annotations are rejected before they can influence a training run.

## Why This Matters

A DEX where volume figures cannot be trusted is one that institutional participants and serious traders will avoid. LedgerLens is an **open-source public good** — its scores, methodology, and training data are fully transparent and auditable, and will always be free to query.

## Compliance & Forensic Reporting

LedgerLens risk scores are designed to support regulatory compliance workflows
including FATF Travel Rule reviews, SEC market-manipulation investigations, and
FinCEN Suspicious Activity Report (SAR) filings.

Every risk score can be accompanied by a **tamper-evident forensic report** that
documents exactly how the score was computed:

- The 20 most anomalous trades with direct Horizon URLs for independent verification.
- SHAP feature attributions with plain-English descriptions of each risk factor.
- Benford's Law analysis across five time windows.
- A SHA-256 fingerprint of the entire report, optionally anchored to the Stellar
  ledger via Soroban for a non-repudiable timestamp.

**Generating a report:**

```bash
# Markdown report for human review
python -m scripts.score_wallet \
  --wallet G... \
  --pair "USDC:GA5Z.../XLM:native" \
  --report --report-format markdown

# JSON report anchored on-chain
python -m scripts.score_wallet ... --report --anchor

# Bulk report generation from a CSV of wallets
python -m scripts.generate_reports --input wallets.csv --anchor
```

Reports are written to `reports/forensic/` with mode `0o600` (owner-readable
only). See [`docs/forensic_reporting.md`](docs/forensic_reporting.md) for the
full schema, anchoring workflow, and the three-step regulator verification guide.

## Contributing

We're actively looking for collaborators with experience in:

- Python backend development and ML pipeline engineering
- On-chain data analysis and blockchain forensics
- Stellar / Soroban smart contract development (Rust)
- DeFi protocol integration

## References

- Benford, F. (1938) 'The law of anomalous numbers', *Proceedings of the American Philosophical Society*, 78(4), pp. 551–572.
- Al Ali, A. et al. (2023) 'A powerful predicting model for financial statement fraud based on optimized XGBoost ensemble learning technique', *Applied Sciences*, 13(4).
- Antonio, G.R. (2023) 'Numbers don't lie: Decoding financial error and fraud through Benford's law', *Journal of Entrepreneurship*.
- Nti, I.K. and Somanathan, A.R. (2024) 'A scalable RF-XGBoost framework for financial fraud mitigation', *IEEE Transactions on Computational Social Systems*, 11(2), pp. 410–422.
- Yadavalli, R. and Polisetti, R. (2025) 'Optimized financial fraud detection using SMOTE-enhanced ensemble learning with CatBoost and LightGBM', *ICVADV 2025*.
- Harea, R. and Mihailă, S. (2025) 'Benford's law: Applicability in accounting and financial anomaly detection', *Challenges of Accounting for Young Researchers*, 3(1).
- Stellar Development Foundation (2024) *Horizon API Documentation*. Available at: https://developers.stellar.org/api/horizon
- Stellar Development Foundation (2024) *Soroban Smart Contract Documentation*. Available at: https://soroban.stellar.org/docs

## License

MIT

---

<div align="center">

**LedgerLens** — Making the Stellar ledger legible.

*Built for the Stellar ecosystem. Open source. Community owned.*

</div>
