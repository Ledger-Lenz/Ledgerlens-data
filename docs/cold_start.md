# Cold-Start Scoring with Neural Process Meta-Learning

> Last verified against code: 2026-08-28. `NP_COLD_START_THRESHOLD` and the
> blend formula below were checked against `detection/neural_process.py`.

## Overview

When a new asset pair is first listed on the Stellar DEX, the system has too few
trades to compute reliable Benford statistics or ML features.  The standard
ensemble would fall back to the global prior (average statistics across all
pairs), producing poorly calibrated scores for that pair.

The Neural Process (NP) meta-learning layer addresses this by learning **how to
adapt** from a small context set rather than relying on a fixed global fallback.

## Architecture

The implementation in `detection/neural_process.py` uses a **Conditional Neural
Process (CNP)**:

- **Encoder** — a two-layer MLP that maps each `(features, label)` context
  trade to a fixed-dimensional latent vector, then aggregates variable-size
  context sets via **mean pooling**.  This makes the encoder permutation-
  invariant and compatible with any context size from 1 to 50 trades.
- **Decoder** — a two-layer MLP that concatenates the pooled context embedding
  with a query feature vector and outputs a wash-trade probability.

The CNP design was chosen over a Latent NP (which adds a stochastic latent
variable) because calibration accuracy — not uncertainty quantification — is the
primary goal in the cold-start path.

## Cold-Start Threshold and Blending

```
NP_COLD_START_THRESHOLD = 50  # trades
```

When a pair has `trade_count < 50` labelled trades, the scorer blends the NP
score with the ensemble score **linearly**:

```
blend_weight = 1.0 - trade_count / threshold
blended_score = blend_weight * np_score + (1 - blend_weight) * ensemble_score
```

- At `trade_count = 0` → pure NP score (blend_weight = 1.0)
- At `trade_count = 25` → 50 / 50 mix
- At `trade_count ≥ 50` → pure ensemble score (blend_weight = 0.0)

This transition avoids a hard cutover and produces smooth score evolution as
trade history accumulates.

## Usage

`detection/neural_process.py` is not currently wired into
`detection.model_inference.RiskScorer` — there is no `score_cold_start`
method on the main scorer. The module is used directly today (see
`detection/certified_robustness.py` and `scripts/run_adversarial_eval.py`).
The public API is the `NeuralProcess` class plus the blending helpers:

```python
from detection.neural_process import NeuralProcess, cold_start_blend_weight, blend_scores
import numpy as np

np_model = NeuralProcess(feature_dim=32)

# context_features: (n_context, feature_dim) array of seed trades
# context_labels:   binary wash-trade labels for context trades
np_score = np_model.predict_score(
    context_features=np.array([...]),
    context_labels=[0, 1, 0, 1, 0],
    query_feature_row=feature_row,
)

blend_weight = cold_start_blend_weight(trade_count=5)  # == 0.9
blended = blend_scores(np_score, ensemble_score, trade_count=5)
```

Wiring this into `RiskScorer` so the ensemble path picks it up automatically
is tracked as a separate follow-up, not covered by this doc.

## Testing

There is no dedicated `tests/test_neural_process.py` yet; `NeuralProcess` is
currently exercised indirectly through `detection/certified_robustness.py`
and `scripts/run_adversarial_eval.py`. Adding direct unit tests (consistency
of predictions for identical context/query sets, cold-start regression on
known wash-trade pairs) is tracked as a separate follow-up.

---

# Cold-Start Feature Cache Warming Strategy

*Issue #969 — `detection/feature_cache.py` CacheWarmer*

## Problem

`detection.feature_cache.FeatureCache` uses a TTL + LRU cache to avoid
re-computing feature matrices on every re-score of the same wallet. After every
deployment the cache is empty (cold). The first wave of requests must rebuild
every feature matrix from scratch — typically 50–200 ms per wallet — causing a
**latency spike** for the first 1–2 minutes of production traffic after deployment.

## Solution

`CacheWarmer` (in `detection/feature_cache.py`) pre-populates the cache with the
*hot set* — the wallets most likely to be scored in the first scoring cycle after
deployment — **before** the new instance is marked ready to receive traffic.

### Latency improvement (measured on synthetic dataset)

| Scenario | p50 score latency | p99 score latency |
|---|---|---|
| Cold start (no warming) | ~180 ms | ~420 ms |
| After 100-wallet warm-up | ~8 ms (cache hit) | ~22 ms |
| **Reduction** | **~96%** | **~95%** |

The spike window reduces from ~90 seconds (time to organically fill the cache)
to the warm-up duration (~2–8 seconds for 100 wallets).

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `CACHE_WARM_HOT_SET_SIZE` | `100` | Number of wallets to pre-warm |
| `CACHE_WARM_TIMEOUT_SECONDS` | `30` | Maximum seconds for warm-up phase |

## Usage

```python
from detection.feature_cache import CacheWarmer, FeatureCache
from detection.feature_engineering import build_feature_vector

cache = FeatureCache()
warmer = CacheWarmer(cache, hot_set_size=100, timeout_seconds=30.0)

# Warm from the risk-score store (most recently scored wallets)
warmer.warm_from_store(
    risk_store,
    build_features_fn=build_feature_vector,
    trades_df=recent_trades_df,
    lookback_hours=24,
)

# Signal readiness only after warming
if warmer.is_warm:
    mark_instance_ready()
```

### Readiness probe integration

```python
@app.get("/ready")
def readiness():
    if not warmer.is_warm:
        return JSONResponse({"ready": False}, status_code=503)
    return {"ready": True, "cache_info": warmer.describe()}
```

## Operator checklist

- [ ] Set `CACHE_WARM_HOT_SET_SIZE` based on expected request volume.
- [ ] Set `CACHE_WARM_TIMEOUT_SECONDS` to a value that fits your deployment SLA.
- [ ] Hook `warmer.is_warm` into your readiness probe before routing traffic.
- [ ] Monitor `cache_warm_entries_total` Prometheus gauge post-deployment.

See `tests/test_cache_warmer.py` for unit and latency-spike simulation tests.
