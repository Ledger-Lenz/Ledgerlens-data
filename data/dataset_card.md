# Dataset Card — LedgerLens Labelled Stellar SDEX Wash-Trade Dataset

## Dataset Summary

A labelled feature dataset for detecting wash trading on the Stellar Decentralised Exchange
(SDEX). Each row represents one wallet observed within a defined data window and carries
the full 30+ feature vector produced by `detection/feature_engineering.py::build_feature_matrix`,
plus ground-truth labels and provenance metadata.

## Dataset Details

| Field | Value |
|---|---|
| **Version** | 1.0.0 |
| **Release date** | 2026-06-19 |
| **Licence** | MIT (same as the LedgerLens project) |
| **Data source** | Stellar Horizon public API |
| **Data window** | 2024-01-01 to 2024-06-30 |
| **Asset pairs covered** | USDC/XLM, BTC/XLM, AQUA/XLM |
| **Format** | Apache Parquet |
| **Path** | `data/labelled_dataset.parquet` |

## Schema

### Feature columns (from `build_feature_matrix`)

| Column | Type | Description |
|---|---|---|
| `wallet` | str | Stellar public key |
| `benford_chi_square_{1,4,24,168,720}h` | float | Benford chi-square statistic per rolling window |
| `benford_mad_{1,4,24,168,720}h` | float | Benford Mean Absolute Deviation per rolling window |
| `benford_z_max_{1,4,24,168,720}h` | float | Max per-digit Z-score per rolling window |
| `counterparty_concentration_ratio` | float | Fraction of volume with single counterparty |
| `round_trip_frequency` | float | Proportion of trades that are round-trips |
| `self_matching_rate` | float | Rate of self-matched buy/sell orders |
| `order_cancellation_rate` | float | Fraction of manage-offer ops that were cancellations |
| `counterparty_variance` | float | Normalised variance of per-counterparty volume (CV²); range [0, 1]; higher = more suspicious |
| `volume_per_counterparty_ratio` | float | Total volume / number of unique counterparties |
| `intra_minute_clustering` | float | Fraction of minute buckets with > 1 trade |
| `off_hours_activity_ratio` | float | Proportion of trades in UTC 00:00–05:00 |
| `volume_spike_frequency` | float | Fraction of trades exceeding 3× rolling mean volume |
| `funding_source_similarity` | float | Max Jaccard similarity of funding ancestors _(legacy scalar — kept for model backwards compat)_ |
| `network_centrality` | float | Degree centrality in the funding graph _(legacy scalar — kept for model backwards compat)_ |
| `account_age_days` | float | Account age at the time of last trade in window |
| `inter_arrival_cv` | float | Coefficient of variation of inter-trade intervals |
| `entropy_of_amounts` | float | Shannon entropy of the trade amount distribution |
| `cross_wallet_volume_corr` | float | Pearson correlation of per-minute volumes across top-2 counterparties |

### Cross-venue coordination features (from `compute_cross_venue_features`)

These features are populated when AMM pool data is available (via `WATCHED_AMM_POOLS`).
When AMM data is unavailable, all cross-venue features default to `0.0`.

| Column | Type | Description |
|---|---|---|
| `venue_trade_ratio` | float | Ratio of SDEX to AMM trade count; balanced ratios indicate wash trading |
| `cross_venue_volume_correlation` | float | Pearson correlation of 1-hour SDEX and AMM trade volumes |
| `cross_venue_timing_synchrony` | float | Fraction of AMM trades occurring within 10 s of a paired SDEX trade |
| `cross_venue_net_flow` | float | Absolute net XLM flow across SDEX and AMM venues (near-zero = wash) |
| `counterparty_venue_overlap` | float | Fraction of SDEX counterparties also seen as AMM liquidity providers |
| `simultaneous_order_pair` | float | Binary: 1.0 if wallet has overlapping SDEX and AMM activity windows |
| `cross_venue_cluster_score` | float | Centrality within Louvain cross-venue coordination cluster |

### GNN Embedding Features

When the GNN encoder artifact (`models/gnn_encoder.pt`) is present, the feature
vector is extended by **`GNN_EMBEDDING_DIM`** additional columns (default 32):

| Column | Type | Description |
|---|---|---|
| `gnn_0` … `gnn_31` | float32 | GraphSAGE embedding dimensions for the wallet node |

These embeddings encode multi-hop structural patterns in the wallet funding and
co-trade graph (see `docs/gnn_architecture.md`).  When the encoder artifact is
absent (e.g., before the first training run), all `gnn_*` columns default to
`0.0`.

**Total feature vector length:** 30+ legacy features + `GNN_EMBEDDING_DIM`
(default 32) GNN embedding features = **62+ features** when the GNN encoder is
enabled.

### Label and provenance columns

| Column | Type | Description |
|---|---|---|
| `label` | int (0 or 1) | 1 = wash trading, 0 = legitimate |
| `labelling_signal` | str | `roundtrip_and_graph` / `roundtrip_only` / `graph_only` / `clean` / `manual` |
| `review_notes` | str | Human reviewer rationale (empty string if not manually reviewed) |
| `data_window_start` | str (ISO datetime) | Start of the data window used to compute features |
| `data_window_end` | str (ISO datetime) | End of the data window used to compute features |
| `n_trades` | int | Number of trades used to build the feature row |

## Class Balance

| Class | Count | Fraction |
|---|---|---|
| Wash trading (label = 1) | ≥ 200 | ≥ 40% |
| Legitimate (label = 0) | ≥ 300 | ≥ 60% |
| Excluded (label = NaN) | variable | excluded from released file |

> **Note:** Grey-zone wallets (flagged by only one signal or with insufficient trade history)
> are excluded from the released Parquet (`label = NaN` rows are dropped before writing).

## Labelling Methodology

See `data/labelling_notes.md` for the full methodology, signal descriptions, and manual
review notes.

The conservative two-signal rule minimises false positives:
- **Positive (label = 1):** flagged by BOTH round-trip detection AND funding-graph clustering.
- **Negative (label = 0):** no flags from either signal, > 50 trades, > 5 distinct counterparties.

## Reproducibility

The build pipeline is fully deterministic given the same Horizon data window and parameters.
See `data/build_config.json` for the exact configuration used to produce this release.

```bash
# Reproduce the dataset
python -m scripts.build_labelled_dataset \
    --trades data/raw_trades.parquet \
    --output data/labelled_dataset.parquet \
    --config data/build_config.json
```

## Simulation Engine — Wash Trade Simulation Engine (WTSE)

The dataset can also be generated synthetically using the Wash Trade Simulation
Engine in `scripts/wash_trade_simulator.py`. The WTSE implements 7 attacker
strategy profiles, each modelling a different wash-trading behaviour pattern.

| Profile | Description | Parameters | Wash / Legitimate Ratio |
|---|---|---|---|
| `NaiveAttacker` | Fixed amounts, regular intervals — baseline | `fixed_amount=500.0`, `interval_seconds=60` | 100% wash (label=1) |
| `TimingJitterAttacker` | Poisson-distributed trade intervals | `lambda_seconds=60.0` | 100% wash |
| `AmountConformanceAttacker` | Benford-conforming amounts via log-uniform sampling | `min_amount=50.0`, `max_amount=5000.0` | 100% wash |
| `RingAttacker` | N-wallet ring where each wallet trades with its neighbour | `fixed_amount=500.0` | 100% wash |
| `LayeringAttacker` | Interleaves wash trades with noise trades at 3:1 ratio | `wash_to_noise_ratio=3` | 25% wash / 75% noise |
| `CrossPairAttacker` | Rotates wash volume across K asset pairs | `n_pairs=3` | 100% wash |
| `AdaptiveAttacker` | Reads model feature importances and down-weights top features | `model_path`, `top_k=3` | 100% wash |

### Distribution Plot Description

When the CI notebook job runs, it generates per-profile distribution plots
showing the feature value distributions for each attacker profile overlaid
with the legitimate-trader distribution from the real labelled dataset.
These plots validate that each profile produces realistic feature separations.
The generated plots are available in the `reports/` directory.

### Usage

```bash
# Generate using a specific profile
python -m scripts.generate_synthetic_dataset \
    --profile RingAttacker \
    --n-wallets 20 \
    --output data/ring_dataset.parquet

# Run the full adversarial loop (5 rounds by default)
python -m scripts.generate_synthetic_dataset \
    --profile AdaptiveAttacker \
    --gan-rounds 5
```

---

## Benford Baseline Calibration (issue #279)

Benford's Law baselines are **asset-class-specific** rather than universal. The dataset contains three asset classes:

| Asset class | Examples | Baseline distribution |
|---|---|---|
| `stablecoin` | USDC, USDT, USDX | Empirical — elevated digit-1 frequency (round-number convention) |
| `native` | XLM | Theoretical Benford distribution |
| `volatile` | BTC, AQUA | Theoretical Benford distribution |

### Calibration Methodology

Stablecoin pairs systematically deviate from the standard Benford distribution because stablecoin trades cluster around round dollar amounts by convention (e.g., 100 USDC, 1000 USDC, 10000 USDC). This causes elevated digit-1 frequency that is **not** indicative of manipulation.

The `AssetClassifier` in `detection/benford_engine.py`:
1. Loads the stablecoin asset list from `data/build_config.json` (the `stablecoins` field).
2. Uses the static stablecoin digit distribution from `build_config.json` as the initial baseline.
3. After each `generate_synthetic_dataset` run, recomputes empirical baselines from clean-labelled (label=0) trade amounts via `recompute_asset_class_baselines()`.
4. Falls back to the theoretical Benford distribution for unknown asset classes.

The feature columns `benford_chi_square_*h`, `benford_mad_*h`, and `benford_z_max_*h` in this dataset are computed against the **per-class baseline**, not the universal theoretical distribution.  This reduces false positives on stablecoin pairs by approximately 40–60% in backtesting on the 2024-H1 data window.

## Known Biases and Limitations

1. Three asset pairs only — does not cover all SDEX activity.
2. Round-trip window of 100 ledgers may miss slow wash-trading rings.
3. Funding-graph features require account-activity data; when unavailable, graph signal defaults
   to 0 and wallets fall into the grey zone.
4. Temporal coverage: 2024-01-01 to 2024-06-30 only.
5. Legitimate wallets are identified by absence of flags + trade volume thresholds, not
   by positive evidence of legitimacy.

## Ethics Statement

All data is sourced exclusively from the **public Stellar Horizon API**
(https://developers.stellar.org/api/horizon). No private or off-chain data was used.

Wallet addresses in this dataset are **Stellar public keys on a permissionless blockchain**,
which are inherently public by design. Publishing these keys does not reveal personal
identifying information. The Stellar protocol is designed to be transparent and auditable.

This dataset is released as an open-source public good under the MIT licence. Its purpose
is to improve the quality of wash-trade detection for the benefit of the Stellar ecosystem.

## Citation

If you use this dataset, please cite:

```
LedgerLens Data (2026). Labelled Stellar SDEX Wash-Trade Dataset v1.0.0.
https://github.com/Ledger-Lenz/Ledgerlens-data
```

## Licence

MIT — see [LICENSE](../LICENSE)
