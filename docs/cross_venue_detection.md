# Cross-DEX Coordination Detection

## Overview

Sophisticated wash traders do not operate within a single exchange. This module detects coordinated wash trading campaigns that span both the **Stellar SDEX** (Central Limit Order Book) and **Stellar AMM liquidity pools** by ingesting trade signals from both venues and identifying temporally and volumetrically correlated activity.

## Venue Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STELLAR LEDGER                                   │
│                                                                       │
│  ┌───────────────────────────┐   ┌──────────────────────────────┐   │
│  │      SDEX (Order Book)    │   │    AMM Liquidity Pools        │   │
│  │                            │   │                              │   │
│  │  Horizon endpoint:         │   │  Horizon endpoint:            │   │
│  │  GET /trades               │   │  GET /liquidity_pools/{id}/  │   │
│  │  ?base_asset=...           │   │  trades                      │   │
│  │  &counter_asset=...        │   │                              │   │
│  │                            │   │  Pool IDs: 64-char hex       │   │
│  │  Trade model: Trade        │   │  Trade model: Trade          │   │
│  └────────────┬───────────────┘   └──────────────┬───────────────┘   │
│               │                                   │                   │
│               └─────────────────┬─────────────────┘                   │
│                                  │                                     │
│                     FeatureBuffer (keyed by wallet)                   │
└──────────────────────────────────┼─────────────────────────────────────┘
                                   │
                                   ▼
                     CrossVenueFeatureSet + CoordinationGraph
```

Stellar **Protocol 18** introduced native AMM liquidity pools alongside the existing SDEX order book. Trades on these two venues are tracked via different Horizon API endpoints and would be invisible to a single-venue detector. This module bridges that gap.

## Data Ingestion

### AMM Pool Trade Loader (`ingestion/amm_pool_loader.py`)

| Function | Description |
|---|---|
| `load_amm_pool_trades(pool_id, since, until)` | Bulk-load historical AMM pool trades for a date range |
| `stream_amm_pool_trades(pool_id)` | Real-time SSE stream of AMM trade events |
| `list_active_pools(asset_code, asset_issuer)` | Discover active pool IDs for an asset |

**Security:** Pool IDs are validated as 64-character lowercase hex strings before use in any API call to prevent injection attacks.

**Deduplication:** All AMM trade records are deduplicated by `paging_token` before joining with SDEX data to prevent double-counting.

**Error handling:** HTTP 404 from Horizon raises `PoolNotFoundError`, not a generic exception.

## Cross-Venue Features (`detection/cross_venue_features.py`)

Seven features are computed per wallet from combined SDEX + AMM trade data:

| Feature | Description | Wash trader signal |
|---|---|---|
| `venue_trade_ratio` | SDEX trade count / AMM trade count | Balanced ratio (≈1.0) |
| `cross_venue_volume_correlation` | Pearson correlation of hourly SDEX vs AMM volumes | High positive correlation |
| `cross_venue_timing_synchrony` | Fraction of AMM trades within 10 s of a SDEX trade | > 0.5 |
| `cross_venue_net_flow` | Absolute net XLM flow across venues | Near 0 (closed cycle) |
| `counterparty_venue_overlap` | Fraction of SDEX counterparties also in AMM activity | High overlap |
| `simultaneous_order_pair` | Binary: overlapping SDEX and AMM activity windows | 1.0 |
| `cross_venue_cluster_score` | Centrality in Louvain cross-venue cluster | High score |

All features fall back to `0.0` gracefully when AMM data is unavailable.

## Coordination Graph Construction

The **coordination graph** models temporal co-occurrence of wallet activity across venues.

```
build_coordination_graph(sdex_trades, amm_trades, window_seconds=10) → nx.DiGraph
```

### Algorithm

1. Extract `(timestamp, wallet_A, wallet_B)` events for each trade on each venue.
2. Sort events by timestamp in O(n log n).
3. Use a **sorted sliding window** (binary search via `bisect`) to find all pairs of wallets that both appear in trades within `window_seconds` of each other.
4. Add a directed edge `(wallet_A → wallet_B, venue=sdex|amm)` for each such pair.

**Performance:** Tested to complete in < 30 s for 10,000 wallets × 100,000 trades.

### Graph properties

- **Nodes:** wallet addresses
- **Edges:** `(wallet_A, wallet_B)` with attributes `venue` (sdex/amm) and `weight` (co-occurrence count)
- **Directionality:** captures who appeared in trades first within the window

## Louvain Community Detection

```
detect_coordinated_clusters(graph) → list[set[str]]
```

Applies `networkx.algorithms.community.louvain_communities` to the undirected projection of the coordination graph to find tightly coupled wallet clusters. Returns a **partition** — each wallet appears in exactly one cluster.

**Performance:** Tested to complete in < 10 s for 10,000 nodes.

### Cluster score

```
cross_venue_cluster_score(wallet, clusters, graph) → float ∈ [0, 1]
```

Combines two signals:
- **Degree centrality** of the wallet within its cluster subgraph.
- **Cross-venue ratio**: 1.0 if the cluster contains edges from both SDEX and AMM venues, 0.0 otherwise.

## Streaming Pipeline Extension

The `StreamingPipeline` now accepts `amm_pools: list[str]` (or reads from `config.WATCHED_AMM_POOLS`) and starts one daemon thread per AMM pool alongside the existing SDEX threads. All events feed into the same `FeatureBuffer` keyed by wallet.

### Configuration

```env
# .env
WATCHED_AMM_POOLS=<64-char-pool-id-1>,<64-char-pool-id-2>
```

Pool IDs are validated at config load time. An invalid hex string raises `ValueError` immediately.

## Backfill Script

```bash
python -m scripts.backfill_amm_trades \
    --pool-ids <pool_id_1> <pool_id_2> \
    --since 2024-01-01 \
    --until 2024-06-30 \
    --sdex-trades data/raw_trades.parquet \
    --output data/labelled_with_cross_venue.parquet
```

The script:
1. Loads historical AMM trades for the specified pools and date range.
2. Loads existing SDEX historical trades from `--sdex-trades` (optional).
3. Builds the coordination graph and runs Louvain community detection.
4. Computes all 7 cross-venue features for every wallet in the combined dataset.
5. Writes results to `data/labelled_with_cross_venue.parquet`.

## Testing

```bash
make test      # runs all unit tests including test_amm_loader.py and test_cross_venue_features.py
make lint      # ruff + black
```

Integration tests (require live Testnet access):
```bash
LEDGERLENS_INTEGRATION_TESTS=1 pytest tests/test_cross_venue_features.py -k integration
```
