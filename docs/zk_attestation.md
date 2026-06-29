# LedgerLens ZK Attestation

This document describes the zero-knowledge proof attestation system used to
submit wash-trade evidence to the on-chain LedgerLens governance contract in
a privacy-preserving, cost-efficient way.

---

## Overview

Each flagged wallet produces a SNARK proof attesting that:

- the wallet's risk score exceeds the flag threshold,
- the score was derived from a valid feature vector, and
- the prover knows the witness (trade history) without revealing it on-chain.

Verifying N proofs individually costs O(N) Soroban compute budget.
**Proof aggregation reduces this to O(1)** regardless of N, making bulk
submission economically viable.

---

## Aggregation Scheme

LedgerLens uses a **SnarkPack-style** inner-product argument over the
BLS12-381 pairing group.  The scheme is:

1. **Individual proofs** — each wallet attestation produces a Groth16 proof
   `π = (π_A, π_B, π_C)` with 128 bytes on the wire.

2. **Fiat-Shamir challenge** — the aggregator hashes all proof commitments in
   order to derive a pseudorandom challenge `r`:

   ```
   r = SHA-256(wallet_0 || π_0 || inputs_0 || wallet_1 || …)
   ```

3. **Linear combination** — the N proofs are combined under `r` into a single
   64-byte aggregate `Σ`:

   ```
   Σ = Σ_i  r^i · π_i    (inner-product argument)
   ```

4. **Public-input commitment** — the Merkle root of all public input vectors
   (32 bytes) is appended, giving a 96-byte aggregate proof.

5. **On-chain verification** — the Soroban contract performs two fixed pairing
   checks regardless of N.  See `integrations/soroban/verify_aggregate_proof.rs`
   for the contract stub.

### Size comparison

| N wallets | Individual proofs | Aggregate proof | Saving |
|-----------|------------------|-----------------|--------|
| 1  | 128 B  | 96 B  | 25 %   |
| 5  | 640 B  | 96 B  | 85 %   |
| 10 | 1 280 B | 96 B | 93 %   |
| 50 | 6 400 B | 96 B | 98.5 % |

---

## Batching Controller

`BatchingController` (in `integrations/zk_attestor.py`) accumulates proofs
and flushes a batch when either condition is met:

| Trigger | Default | Env var |
|---------|---------|---------|
| Batch full | 10 proofs | `ZK_BATCH_SIZE` |
| Timeout | 300 s (5 min) | `ZK_BATCH_TIMEOUT_SECONDS` |

### Trade-offs

| Property | Detail |
|----------|--------|
| **Latency** | A wallet proved at second 0 may not be submitted until the timeout fires if fewer than `ZK_BATCH_SIZE` proofs arrive.  Reduce `ZK_BATCH_TIMEOUT_SECONDS` for lower latency at the cost of smaller batches. |
| **Cost efficiency** | Larger batches amortise on-chain verification cost more aggressively.  The default of 10 gives a 93 % compute-budget saving. |
| **Fault isolation** | Any single invalid proof causes the entire batch to be rejected.  Each proof is individually validated before joining the batch so corrupt proofs are rejected immediately at `add_proof()`. |
| **Idempotency** | Duplicate `wallet_id` within an unflushed batch is silently dropped.  Cross-batch uniqueness is enforced by the on-chain contract's unique constraint on `(wallet_id, pair_hash)`. |

---

## On-Chain Verification Cost Savings

Soroban charges compute units (CUs) per instruction.  Approximate costs:

| Operation | CUs |
|-----------|-----|
| BLS12-381 pairing check (single) | ~2 000 000 |
| Verifying N individual Groth16 proofs | N × ~2 000 000 |
| Verifying 1 aggregate proof (SnarkPack) | ~2 × 2 000 000 = ~4 000 000 |

For N=10 this is a **5× reduction** in on-chain compute cost, falling comfortably
within the Soroban per-transaction CU budget.

---

## Module Reference

| Symbol | Location | Description |
|--------|----------|-------------|
| `SNARKProof` | `integrations/zk_attestor.py` | Individual wallet proof |
| `AggregateProof` | `integrations/zk_attestor.py` | Batched aggregate proof |
| `aggregate_proofs(proofs)` | `integrations/zk_attestor.py` | Aggregates N proofs |
| `verify_aggregate_proof(agg, proofs)` | `integrations/zk_attestor.py` | Off-chain verification |
| `BatchingController` | `integrations/zk_attestor.py` | Accumulation + submission |
| `make_valid_proof(...)` | `integrations/zk_attestor.py` | Test helper |
| `verify_and_record(...)` | `integrations/soroban/verify_aggregate_proof.rs` | On-chain Soroban contract |
