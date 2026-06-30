# Feature Store Architecture

## OHLCV-derived candle features

Computed from individual trade events by aggregating into OHLCV candles at
multiple time resolutions.

### Resampling logic

For each requested resolution (validated against an allowlist: `1m`, `5m`,
`15m`, `1h`, `4h`), trades are bucketed using pandas `resample` anchored to
the trade timestamps.

For each candle window, these features are computed:

- **`price_range_ratio_{res}`**: 
  
  
  \(\frac{high-low}{open}\)

- **`vwap_deviation_{res}`**:
  
  \(\frac{close - VWAP}{VWAP}\)

- **`candle_body_ratio_{res}`**:
  
  \(\frac{close-open}{high-low}\)

- **`volume_spike_ratio_{res}`**:
  
  \(\frac{candle\_volume}{rolling\_avg\_{20}(candle\_volume)}\)

  Where `rolling_avg_20` is computed over the previous 20 candles **excluding
  the current candle** (anchored to the current candle via a `.shift(1)`).

### NaN handling for sparse candles

If a candle window contains fewer than 2 trades, all candle-based features
(`price_range_ratio`, `vwap_deviation`, `candle_body_ratio`, and
`volume_spike_ratio`) are set to `NaN`.


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
