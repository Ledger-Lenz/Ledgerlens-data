# LedgerLens Security

See [`docs/security_threat_model.md`](security_threat_model.md) for a comprehensive STRIDE-based threat model covering data ingestion, model inference, persistence, SHAP interpretability, and on-chain integration.

## Threat Model: Model Poisoning

LedgerLens is a fraud-detection system — making it a high-value target for adversaries who want their wash trading to go undetected. Three attack vectors are in scope:

| Threat | Description |
|---|---|
| **Artifact substitution** | A compromised CI pipeline or model storage replaces a legitimate `.joblib` with a backdoored one. |
| **Training data poisoning** | An adversary injects fraudulent wash-trade labels into the annotation queue, causing the retrained model to develop a blind spot. |
| **Ensemble manipulation** | If one of RF/XGBoost/LightGBM is compromised, a naive average gives the poisoned model equal weight, potentially reducing the final score by ~33 points. |

## Artifact Integrity Verification

Every model artifact goes through a four-step trust chain enforced by `ModelArtifact.verify_chain()` in `detection/persistence.py`:

1. **SHA-256 match** — the `.joblib` file's SHA-256 must match the `artifact_sha256` field recorded in `metrics.json` at training time.
2. **Ed25519 signature** — `metrics.json` must be accompanied by `metrics.json.sig`, a detached Ed25519 signature produced by the authorised signing key.
3. **Key fingerprint** — the SHA-256 fingerprint of the public key used for verification must match `TRUSTED_SIGNING_KEY_FINGERPRINT` in config.
4. **Training data SHA-256** — (optional, supplied at call site) the SHA-256 of the training dataset recorded in `metrics.json` must match the caller's expectation.

A `ModelIntegrityError` with a specific failure reason is raised on any step failure. `RiskScorer._load_models()` calls `verify_chain` immediately after every `joblib.load`; a CI grep check enforces this invariant.

### Generating a Signing Key

```bash
# Generate an Ed25519 private key (PEM format)
openssl genpkey -algorithm ed25519 -out signing_key.pem

# Extract the corresponding public key
openssl pkey -in signing_key.pem -pubout -out signing_key_pub.pem
```

Set `MODEL_SIGNING_PRIVATE_KEY_PATH=./signing_key.pem` in your environment (not in `.env` committed to git).

### Computing the Trusted Fingerprint

```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import hashlib

with open("signing_key_pub.pem", "rb") as f:
    pub = serialization.load_pem_public_key(f.read())

raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
fingerprint = hashlib.sha256(raw).hexdigest()
print(fingerprint)
```

Set `TRUSTED_SIGNING_KEY_FINGERPRINT=<output>` in your environment.

### Signing Key Rotation

When rotating the Ed25519 signing key:

1. Generate the new key pair (see above).
2. Update `MODEL_SIGNING_PRIVATE_KEY_PATH` in CI secrets / deployment environment.
3. Update `TRUSTED_SIGNING_KEY_FINGERPRINT` to the new public key's fingerprint.
4. Re-run `python -m detection.model_training --data-path ...` to produce freshly signed artifacts.
5. Deploy the new artifacts. Old artifacts signed with the previous key will fail `verify_chain` and must not be loaded.
6. Revoke access to the old private key and delete it from all locations.

**Never commit `signing_key.pem` or its contents to version control.**

## Byzantine-Fault-Tolerant Ensemble Voting

The inference stack uses a trimmed-mean / median voting scheme so that a single compromised model cannot materially change the final score.

### Algorithm

1. Collect the raw 0–100 scores from RF, XGBoost, and LightGBM.
2. If `|max - min| > BFT_SCORE_DIVERGENCE_THRESHOLD` (default 30): log a WARNING with all three raw scores, increment the `bft_divergence_detected_total` Prometheus counter, and set `bft_divergence: true` in the response. Use the **median** as the final score (for 3 models this is the trimmed mean with the extremes dropped).
3. If fewer than `BFT_MIN_CONSENSUS` (default 2) models agree within 10 points: return `score=100`, `confidence=0`, `consensus_failure=true`.

### Tuning

| Config var | Default | Effect |
|---|---|---|
| `BFT_SCORE_DIVERGENCE_THRESHOLD` | 30 | Minimum score span that triggers trimmed-mean fallback |
| `BFT_MIN_CONSENSUS` | 2 | Minimum number of models required to be within 10 points of each other |

## Training Data Provenance

`detection/model_training.py` records the following for every training run in `metrics.json`:

- `training_data_sha256` — SHA-256 of the row-sorted input parquet (deterministic).
- `label_distribution` — `{0: N, 1: M}` counts; a sudden shift in the 1:0 ratio is a poisoning signal.

### Label Poisoning Detection

`detect_label_poisoning()` compares the current wash-trade label ratio against a baseline stored in `models/label_distribution_baseline.json`. If the ratio has shifted by more than `POISON_LABEL_RATIO_THRESHOLD` (default 15%), training is aborted and an alert is written to `reports/poisoning_alert_{timestamp}.json`.

## Supply Chain Security: Model Artifact Transparency Log

### Overview

`ModelArtifactVerifier` in `detection/persistence.py` extends the existing trust chain with a third independent check: every artifact's SHA-256 must appear in an append-only **transparency log** stored in the risk score database.  A coordinated attack that replaces the artifact **and** tampers with `metrics.json` will still fail unless the attacker also corrupts the transparency log, which is separately backed up.

### Verification Flow

```
Download artifact
      │
      ▼
1. SHA-256 hash                  — fast, no model parsing
      │
      ▼
2. Ed25519 signature on          — verifies metrics.json
   metrics.json
      │
      ▼
3. Transparency log lookup       — append-only, separately backed up
      │
      ▼
   ✅ Load model  /  ❌ ModelIntegrityError → refuse to start
```

Any of the three checks failing raises `ModelIntegrityError` and the scorer refuses to start.

### Publishing a New Artifact

```bash
python -m scripts.publish_model_artifact \
    --model-name rf \
    --model-dir ./models \
    --private-key-path /secrets/signing_key.pem \
    --db-url sqlite:///ledgerlens.db
```

This script:
1. Computes the SHA-256 of the `.joblib` file.
2. Records the hash in `metrics.json` and re-signs it.
3. Appends the hash to the `transparency_log` DB table.

### Transparency Log Format

```sql
CREATE TABLE transparency_log (
    id            INTEGER PRIMARY KEY,
    model_name    TEXT    NOT NULL,
    artifact_sha256 TEXT  NOT NULL UNIQUE,  -- 64-char lowercase hex
    registered_at DATETIME NOT NULL
);
```

Rows are never updated or deleted.  The table supports public auditability: export the full `artifact_sha256` column to a public append-only ledger (e.g. Sigstore Rekor, a public blockchain, or a signed NDJSON file) to allow external parties to verify artifact provenance without access to the internal database.

### Security Requirements

| Requirement | Detail |
|---|---|
| **Signing key storage** | Store the Ed25519 private key in an HSM or encrypted secrets manager (AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager). Never write it to disk unencrypted in production. |
| **Transparency log backup** | Back up the `transparency_log` table separately from the model artifact store. A coordinated attacker who modifies both the artifact and the log would otherwise bypass the check. |
| **Log immutability** | The application layer exposes no UPDATE or DELETE path for `transparency_log`. Implement DB-level row-security policies to enforce this in production. |

## Annotation Queue Integrity

Each entry in `data/annotation_queue.json` carries an `annotation_hmac` field: HMAC-SHA256 of `wallet|label|annotator_id|annotated_at` keyed by `ANNOTATION_HMAC_SECRET`. `export_labelled()` verifies every HMAC before including an annotation; tampered entries are logged as WARNING and excluded.

**Set `ANNOTATION_HMAC_SECRET` to a cryptographically random value (≥ 32 bytes hex) and never commit it.**

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Field-Level Encryption for Forensic Reports

Forensic reports (`detection/forensic_report.py`) contain raw Stellar wallet
addresses that identify subjects under investigation.  To prevent premature
exposure, wallet addresses should be encrypted at rest using AES-256-GCM
field-level encryption provided by `utils/field_encryption.py`.

### Encryption Scheme

- **Algorithm**: AES-256-GCM (authenticated encryption with 128-bit tag).
- **IV**: A fresh 12-byte random IV is generated per field per encryption call.
  This guarantees that encrypting the same wallet address twice produces
  different ciphertexts (IV-reuse attack prevention).
- **Stored format**: `iv (12 bytes) || ciphertext || GCM tag (16 bytes)` as a
  binary blob.
- **Authentication**: Decryption raises `cryptography.exceptions.InvalidTag`
  on any authentication failure.  Silent corruption is impossible.

### Key Management

Set `FORENSIC_REPORT_ENCRYPTION_KEY` to a 64-character lowercase hex string
representing 32 random bytes:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

- The key must be **exactly 32 bytes**; a shorter key raises a startup error.
- If the variable is absent, a warning is emitted and plaintext is stored
  (development only — never acceptable in production).
- Store the key in a secrets manager (Vault, AWS Secrets Manager, etc.),
  never in `.env` committed to version control.

### Key Rotation

To re-encrypt existing forensic reports when the key changes:

1. Decrypt all stored blobs with the **old** key.
2. Re-encrypt each field value with the **new** key.
3. Atomically swap the updated blobs in the database.
4. Revoke the old key.

Run this as a migration script with the old key available as a temporary
environment variable (e.g. `OLD_FORENSIC_KEY`), then set
`FORENSIC_REPORT_ENCRYPTION_KEY` to the new key.

### Export Authorisation

Before generating a PDF that includes decrypted wallet addresses, call:

```python
from utils.field_encryption import check_export_permission

check_export_permission(current_user.permissions)  # raises PermissionError if not authorised
```

Only users with the `forensic_export` permission may trigger exports with
decrypted addresses.
