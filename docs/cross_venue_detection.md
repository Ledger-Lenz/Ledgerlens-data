# Cross-DEX Coordination Detection

## Overview

Sophisticated wash traders do not operate within a single exchange. This module detects coordinated wash trading campaigns that span both the **Stellar SDEX** (Central Limit Order Book) and **Stellar AMM liquidity pools** by ingesting trade signals from both venues and identifying temporally and volumetrically correlated activity.

Additionally, this module detects **cross-chain coordination** with the **Solana blockchain** through the **Wormhole bridge**, identifying Stellar wallets that have linked Solana addresses used for coordinated wash trading across chains.

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
└──────────────────────────────────┼──────────────────────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │                     │
                        │  Identity Graph     │
                        │  (Cross-chain)      │
                        │                     │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
        ┌────────────────────┐       ┌────────────────────┐
        │  Stellar Wallets   │       │  Solana Wallets    │
        │  (Risk Scores)     │       │  (via Wormhole)    │
        └────────────────────┘       └────────────────────┘
```

Stellar **Protocol 18** introduced native AMM liquidity pools alongside the existing SDEX order book. Trades on these two venues are tracked via different Horizon API endpoints and would be invisible to a single-venue detector. This module bridges that gap.

Additionally, the **Wormhole bridge** enables wrapped asset trading across chains. A Stellar wallet might bridge USDC to Solana, engage in wash trading on Solana, then bridge back to Stellar. This module detects such cross-chain linkages.

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

### Solana Cross-Chain Resolver (`detection/cross_chain/solana_resolver.py`)

**NEW**: Solana address resolution for Wormhole-linked wallets.

| Function | Description |
|---|---|
| `validate_solana_address(address)` | Validate base58-encoded Solana public key (32-44 chars) |
| `parse_wormhole_vaa_payload(tx_data)` | Extract destination address from Wormhole VAA payload |
| `extract_stellar_address_from_vaa(vaa_data)` | Decode embedded Stellar address from VAA |
| `resolve_stellar_to_solana(stellar_addr, rpc_client)` | Query Solana RPC to find linked Solana addresses |
| `SolanaRPCClient.get_signatures_for_address(addr)` | Query Solana RPC with 1-hour caching |
| `SolanaRPCClient.get_transaction(signature)` | Fetch transaction data (cached) |

**Security:**
- Solana addresses validated before RPC calls to prevent injection attacks
- Wormhole VAA signatures require verification (see Signature Verification section below)
- RPC endpoint configurable via `SOLANA_RPC_URL` environment variable (defaults to public endpoint; private RPC recommended for production)

**Caching:**
- Solana RPC responses cached with **1-hour TTL** to avoid rate limiting
- Cache size: 1000 entries (configurable)

**Rate Limiting:**
- Solana public RPC: ~100 requests/sec limit
- Recommended: use private RPC endpoint with higher limits
- Backoff logic: exponential retry on rate limit errors (future enhancement)

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

### Cross-Chain Features

**NEW**: One feature from Solana cross-chain linkage:

| Feature | Description | Wash trader signal |
|---|---|---|
| `solana_linked_wash_score` | Risk score of linked Solana address (via Wormhole) | High score (0-100) from linked Solana wallet |

This feature queries the identity graph to find Solana addresses linked to a Stellar wallet via Wormhole bridge transactions. If a cached risk score is available for the Solana address (from external Solana chain analysis), this signal is surfaced. Value: [0, 100], or 0 if no link found.

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

# Solana RPC endpoint (NEW)
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com  # or private RPC for production
```

Pool IDs are validated at config load time. An invalid hex string raises `ValueError` immediately.

## Wormhole VAA Signature Verification

When parsing Wormhole VAA payloads, the embedded Stellar destination address must be verified before trusting it. The VAA (Verified Action Approval) is signed by multiple Wormhole Guardians.

### Guardian Signature Verification

```python
from detection.cross_chain.solana_resolver import parse_wormhole_vaa_payload

vaa_data = parse_wormhole_vaa_payload(transaction_data)

# Full VAA verification requires:
# 1. Reconstruct the VAA hash from core fields
# 2. Verify each signature against the Guardian set (threshold-based, e.g., 2/3)
# 3. Compare signature count against current Guardian set size

# Implementation: Use Wormhole SDK or custom signature verification
```

**Current Implementation:** Basic structure validation only (see `parse_wormhole_vaa_payload`). Full signature verification requires:
- Access to Wormhole Guardian set state (updated weekly)
- ECDSA signature verification library
- Threshold signature scheme (2/3 or configurable)

**Recommendation:** For production use, integrate [Wormhole TypeScript SDK](https://github.com/wormhole-foundation/wormhole) or implement Guardian set verification with caching.

### Data Freshness Limitations

1. **Bridge Latency:** Wormhole Guardians require ~15 minutes to finalize cross-chain transfers. VAAs are not immediately available.
2. **Observability Lag:** Solana RPC queries reflect on-chain state, which lags by 1-2 blocks (~1 second).
3. **Cache TTL:** Solana RPC responses cached for 1 hour to avoid rate limits. Fresh lookups require cache invalidation.

**Implication:** Cross-chain linkages detected with 15-60 minute latency, suitable for retrospective analysis but not real-time alerts.

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
