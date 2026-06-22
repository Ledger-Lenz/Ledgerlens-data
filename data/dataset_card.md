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
| `volume_per_counterparty_ratio` | float | Total volume / number of unique counterparties |
| `intra_minute_clustering` | float | Fraction of minute buckets with > 1 trade |
| `off_hours_activity_ratio` | float | Proportion of trades in UTC 00:00–05:00 |
| `volume_spike_frequency` | float | Fraction of trades exceeding 3× rolling mean volume |
| `funding_source_similarity` | float | Max Jaccard similarity of funding ancestors |
| `network_centrality` | float | Degree centrality in the funding graph |
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
