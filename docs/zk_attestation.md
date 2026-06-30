# zk Attestation Design

This repository now supports a V1 hash-commitment flow for on-chain score submissions.

## V1

The attestor computes three public values from the wallet submission:

1. `trade_data_hash` from a canonical serialization of the public trade set.
2. `model_version_hash` from the model parameters committed at deployment time.
3. `commitment = SHA-256(wallet, trade_data_hash, model_version_hash, score)`.

The contract client can submit the usual `RiskScore` fields plus the commitment metadata. The raw `submit_score` method remains available as a fallback when attestation is not required.

## Reproducibility

The commitment is deterministic because trade rows and columns are canonicalized before hashing. Re-running the same score computation over the same trade set yields the same receipt.

## V2 zkVM path

Future Risc Zero integration should keep the same public receipt shape, but replace the hash-commitment builder with a guest program that consumes:

- `wallet`
- `trade_data_hash`
- `model_version_hash`
- `score`

The guest should emit the public receipt values above and a proof artifact that Soroban can verify before accepting the attested score.
