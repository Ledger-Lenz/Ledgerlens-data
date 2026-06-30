# ZK Attestation Design

This repository supports two attestation tiers for on-chain score submissions.

## V1 — Hash Commitment

The attestor computes three public values from the wallet submission:

1. `trade_data_hash` — SHA-256 of the canonical serialisation of the public trade set.
2. `model_version_hash` — SHA-256 of the model parameters committed at deployment.
3. `commitment = SHA-256(wallet, trade_data_hash, model_version_hash, score)`.

The contract client submits the `RiskScore` fields plus the commitment metadata.
Re-running the same computation over the same trade set yields the same receipt
(deterministic because trade rows and columns are canonicalised before hashing).

## V2 — Benford MAD ZK Proof (Issue #194)

High-risk scores (> 70) include a `BenfordZKProof` in the alert payload.
The proof attests — without revealing trade amounts — that the claimed MAD against
Benford's expected distribution was computed correctly.

### Circuit Statement

> Given trade amounts x₁, …, xₙ (committed to Merkle root R), the MAD of
> leading-digit frequencies from Benford's expected distribution equals the
> claimed value v, within ±0.001 tolerance.

### Proof System

The guest should emit the public receipt values above and a proof artifact that Soroban can verify before accepting the attested score.
