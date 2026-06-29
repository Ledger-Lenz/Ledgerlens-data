# Feature Store Architecture

The feature store caches precomputed wallet feature vectors to eliminate redundant computation and reduce scoring latency.

## Cache Structure

Features are stored in Redis Hash keys:

- Key format: `feat:{hashed_wallet_address}:{asset_pair}`
- Serialization: MessagePack-encoded dictionaries
- TTL: `FEATURE_STORE_TTL_SECONDS` (default 300, configurable)

## Schema Versioning

Each cached feature vector includes a `schema_version` field. On version mismatch, the cache is invalidated and features are recomputed.

## API

### get_or_compute

Returns cached features if fresh, else calls `compute_fn`, stores the result, and returns it:

```python
store.get_or_compute(wallet, pair, compute_fn)
```

### prefetch

Bulk fetch features for multiple wallet-pair combinations using a Redis pipeline:

```python
store.prefetch([(wallet1, pair1), (wallet2, pair2)])
```

## TTL Rationale

The 5-minute TTL balances cache hit rate with feature freshness. Wallets scored repeatedly within this window benefit from caching, while stale data expires automatically.

## Security

Wallet addresses are SHA-256 hashed before use in Redis keys to avoid exposing addresses in cache infrastructure logs.

## Fallback Behavior

On Redis timeout or error, the system falls back to computing features and logs a warning, ensuring availability even when the cache is unavailable.
