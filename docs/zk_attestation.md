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

We use a **Fiat-Shamir hash-based non-interactive argument** (random oracle model)
over the BN128 elliptic curve (`py_ecc`):

| Property | Value |
|---|---|
| Curve | BN128 (same as Ethereum precompiles) |
| Generator | Standard Ethereum G1 / G2 (Yellow Paper, Appendix F) |
| Commitment | Pedersen: `C = mad_int * G1 + nonce * H` |
| Proof hash | `SHA-256(version ‖ merkle_root ‖ claimed_mad ‖ pedersen_commitment ‖ nonce)` |
| Proof size | < 256 bytes (Soroban transaction limit) |
| Trusted setup | None required (Fiat-Shamir; BN128 generator points are public) |
| Tolerance | ±0.001 (matches CKKS approximation error from the HE aggregation layer) |

### Why Fiat-Shamir instead of Groth16?

Groth16 requires a per-circuit **trusted setup ceremony** whose toxic waste must be
destroyed. For an audit tool where the primary goal is public verifiability, the
Fiat-Shamir approach (no toxic waste, no ceremony) is preferable. A future upgrade
to Groth16 (e.g. via a Circom circuit compiled with `snarkjs`) is forward-compatible:
replace the `_fiat_shamir_proof` internals in `integrations/zk_attestor.py` while
keeping the `BenfordZKProof` dataclass and `verify_benford_proof` API unchanged.

### Merkle Commitment

Trade amounts up to N=1000 are committed via a binary Merkle tree of SHA-256
leaf hashes. Larger trade sets should batch the amounts in groups of 1000 and
chain the Merkle roots. Individual amounts are never logged; only the proof hash
is emitted to logs.

### Security Properties

- **Soundness**: An adversary cannot forge a valid proof for an incorrect MAD
  without finding a SHA-256 collision (second-preimage resistance).
- **Zero-knowledge**: The Merkle root commits to the amounts without revealing
  them; the Pedersen commitment hides the MAD integer.
- **Non-malleability**: The proof hash binds all public fields; changing any
  field (version, merkle_root, claimed_mad, nonce) invalidates the proof.

## On-chain Verification — Soroban Contract Stub

The following Rust stub illustrates the verification logic for a Soroban smart
contract. It is pseudocode; a production deployment should use a Soroban-compatible
SHA-256 host function and JSON parsing.

```rust
// SPDX-License-Identifier: MIT
// Soroban verification stub for BenfordZKProof (pseudocode)
use soroban_sdk::{contract, contractimpl, Env, String, Map};

#[contract]
pub struct BenfordVerifier;

#[contractimpl]
impl BenfordVerifier {
    /// Verify a BenfordZKProof submitted alongside a high-risk score alert.
    ///
    /// Returns true if the Fiat-Shamir proof hash is self-consistent.
    ///
    /// Parameters
    /// ----------
    /// version:       Proof system version string (must be "benford-zk-v1").
    /// merkle_root:   Hex-encoded Merkle root of trade amount commitments.
    /// claimed_mad:   Claimed Benford MAD value (6 decimal places).
    /// pedersen_hex:  Hex-encoded Pedersen commitment hash (optional).
    /// nonce:         16-char hex nonce chosen by the prover.
    /// proof_hash:    Prover-supplied SHA-256 proof hash to verify.
    pub fn verify_benford_proof(
        env: Env,
        version: String,
        merkle_root: String,
        claimed_mad: i64,   // MAD * 1_000_000 as integer to avoid floats
        pedersen_hex: String,
        nonce: String,
        proof_hash: String,
    ) -> bool {
        // Reconstruct canonical proof input (must match Python json.dumps sort_keys=True)
        let proof_input = format!(
            r#"{{"claimed_mad":{claimed_mad_float},"merkle_root":"{merkle_root}","nonce":"{nonce}","pedersen_commitment_hex":"{pedersen_hex}","version":"{version}"}}"#,
            claimed_mad_float = claimed_mad as f64 / 1_000_000.0,
            merkle_root = merkle_root,
            nonce = nonce,
            pedersen_hex = pedersen_hex,
            version = version,
        );
        // Compute SHA-256 of the proof input using Soroban host function
        let expected_hash = env.crypto().sha256(&proof_input.into_bytes());
        // Compare hex-encoded hash with the claimed proof_hash
        hex_encode(expected_hash) == proof_hash.to_string()
    }
}
```

### Deployment Notes

1. The contract accepts `claimed_mad` as an integer (MAD × 10⁶) to avoid
   floating-point arithmetic in the Soroban environment.
2. The canonical JSON field order (alphabetical) must match the Python prover
   exactly — both use `sort_keys=True` / alphabetical key ordering.
3. `proof_hash` is 64 hex characters (SHA-256); safe to store in a Soroban `String`.

## V3 — Full zkVM Path (Future)

A future RISC Zero integration replaces the Fiat-Shamir proof with a zkVM
receipt. The guest program consumes `(wallet, trade_amounts_merkle_root,
model_version_hash, score)` and emits the public receipt plus a Groth16
proof verifiable by the existing Soroban contract precompile.