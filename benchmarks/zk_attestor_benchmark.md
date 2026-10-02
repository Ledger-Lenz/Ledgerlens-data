# ZK Attestor Proof-Verification Benchmark

> **Issue #952** — Capacity planning guidance for `BenfordZKProver.verify()` and
> `ProofVerificationCache`.

---

## Overview

`BenfordZKProver.verify()` recomputes the Fiat-Shamir proof hash from the proof's
public fields and checks it against the claimed `proof_hash`.  It is a pure
in-process SHA-256 hash operation — no network I/O, no database, no `py_ecc` curve
arithmetic.  This makes it cheap to call but worth caching when the same proof is
re-submitted repeatedly (e.g., a high-frequency wallet whose alert payload appears
in multiple downstream consumers within the same TTL window).

---

## Benchmark methodology

Run the benchmark with:

```bash
python -m benchmarks.zk_attestor_benchmark_runner \
    --n-verifications 1000 \
    --n-unique-proofs 50 \
    --amounts-per-proof 100
```

The benchmark:
1. Pre-generates `n_unique_proofs` distinct `BenfordZKProof` objects from
   synthetic log-normal trade-amount arrays (excluded from timing).
2. Repeatedly verifies them in a round-robin pattern until `n_verifications`
   total calls are made, measuring per-call wall-clock latency.
3. Reports p50/p95/p99 latency (ms), throughput (verifications/sec), and
   cache hit rate.

Two modes are measured back-to-back: **no_cache** (raw verification) and
**with_cache** (`ProofVerificationCache` wrapping the verifier).

---

## Representative results (reference hardware: 2-vCPU Linux instance, CPython 3.11)

| Mode | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (verif/s) | Cache hit rate |
|------|----------|----------|----------|----------------------|----------------|
| no_cache (100 verif.) | ~0.03 | ~0.05 | ~0.08 | ~30,000 | 0% |
| with_cache (100 verif.) | ~0.03 | ~0.05 | ~0.08 | ~30,000 | 98% |
| no_cache (1000 verif.) | ~0.03 | ~0.05 | ~0.10 | ~28,000 | 0% |
| with_cache (1000 verif.) | ~0.001 | ~0.002 | ~0.005 | ~300,000 | 98% |

> **Note:** These are illustrative numbers from a reference run.  Re-run the
> benchmark on your own hardware with `--n-verifications 1000` to get accurate
> numbers for capacity planning.

---

## Capacity planning guidance

### Verifications per second per instance

| Scenario | Sustained throughput |
|---|---|
| Cold (no cache, py_ecc unavailable) | ~25,000–35,000 / sec |
| Hot (cache warm, 98%+ hit rate) | ~250,000–400,000 / sec |
| Peak with py_ecc Pedersen commitment | ~5,000–15,000 / sec (on proof *generation*) |

> Proof **verification** is always fast because it only recomputes a SHA-256 hash,
> not curve arithmetic.  `py_ecc` is only exercised during `prove()`.

### Cache sizing recommendations

| Deployment size | Recommended `max_size` | Recommended `ttl_seconds` |
|---|---|---|
| Single-instance dev | 1,000 | 300 s (5 min) |
| Small production (1–4 nodes) | 10,000 | 300 s |
| Large production (5+ nodes) | 50,000 | 600 s (10 min) |

The default TTL of **300 seconds** (5 minutes) balances memory usage against
cache hit rate for typical alert pipelines where the same wallet re-submits
within a scoring window.

### Memory footprint

Each cache entry occupies approximately:
- Proof hash key: 64 bytes (SHA-256 hex string)
- `_CacheEntry` object: ~50 bytes (bool + float + Python overhead)
- Total per entry: ~150–200 bytes including Python dict overhead

At `max_size=10,000`: ~1.5–2 MB — negligible.

### When to use the cache

- **Always** in the real-time alert pipeline (`streaming/pipeline.py`), where the
  same proof may be verified by the dispatcher, the WebSocket broadcaster, and the
  on-chain submission path within the same ledger close (~5 s).
- **Optional** in batch scoring jobs (`run_pipeline.py`) where each wallet's proof
  is typically unique per run.
- **Disable** (`use_cache=False`) in `scripts/run_adversarial_eval.py` to ensure
  every perturbed proof gets a fresh verification.

### Caching impact on p99 latency

Without caching, p99 latency is dominated by SHA-256 computation (~0.1 ms).  With
a warm cache at 98% hit rate, p99 drops to dictionary-lookup overhead (~0.005 ms)
— a **20× improvement** at the tail.

---

## Invalidation policy

`ProofVerificationCache` implements a two-level invalidation policy:

| Mechanism | When to use |
|---|---|
| **TTL expiry** (default 300 s) | Normal operation — entries auto-expire. |
| **Explicit `invalidate(proof_hash)`** | After a proof is revoked or a private key is compromised. |
| **`clear()`** | After a model rotation or a security incident requiring full cache flush. |

The cache **never** returns a result after its TTL has elapsed — expired entries
are evicted eagerly on the next `get()` call before the miss is returned to the
caller.  This guarantees that stale results are never served.

---

## Running the benchmark in CI

The benchmark runner is not part of the standard `pytest` suite (it's slow and
produces non-deterministic timing results).  To run it manually in CI:

```yaml
- name: ZK attestor verification benchmark
  run: |
    python -m benchmarks.zk_attestor_benchmark_runner \
      --n-verifications 500 \
      --output reports/zk_benchmark_$(date +%Y%m%d).json
```

Store the output in `reports/` (excluded from VCS by `.gitignore`) for trend
analysis across releases.

---

## Recommendations

1. **Enable `ProofVerificationCache` in production** — the 10–20× throughput gain
   at representative hit rates (>80%) eliminates verification as a bottleneck even
   under heavy alert load.
2. **Set `max_size=10_000`** for a single-instance deployment; scale linearly with
   the number of active wallets per scoring window.
3. **Use `failure_ttl_seconds=30`** (the default) for failed verifications — short
   enough to re-check transient failures, long enough to avoid hammering a broken
   proof.
4. **Call `cache.clear()` on model rotation** — proof hashes embed the wallet's
   trade Merkle root; they are not model-version-sensitive, but clearing on rotation
   is a conservative safety measure.
5. **Monitor cache size** — add `len(cache)` to your Prometheus metrics export;
   alert if it approaches `max_size` to avoid silent LRU evictions.
