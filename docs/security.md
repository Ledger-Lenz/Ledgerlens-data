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

## Model Inversion Attack Defence

A model inversion attack allows an adversary to reconstruct features used in
scoring by querying the model repeatedly and analyzing the gradients (or in
discrete settings, the score deltas). In LedgerLens, an attacker could query
the score for a wallet, remove one trade, re-query, and observe the delta to
identify which individual trade was most anomalous. Repeated queries across
variants reconstruct the full anomaly profile.

### Output Perturbation (Laplace Mechanism)

`detection/model_inference.py::score()` applies **Laplace output perturbation**
per external API call to defend against this attack. The mechanism:

1. **Per-query random seed**: derived from `(caller_id, timestamp_bucket)` so
   identical repeated queries produce different scores, preventing averaging.
2. **Laplace noise**: drawn with scale `σ = sensitivity / ε`, where sensitivity
   is the max score change from one trade (bounded at 100) and `ε` is the privacy
   budget (default 1.0).
3. **Clipping + rounding**: perturbed scores clipped to [0, 100] and rounded to
   `SCORE_ROUNDING_GRANULARITY` (default 1 point).

**Internal pipeline calls** (e.g. `run_pipeline.py` → model training) use
`caller_id="internal"` and are **not perturbed**, preserving model quality for
batch operations.

### Query Rate-Limiting

`detection/risk_score_store.py::RiskScoreStore` tracks queries per
`(caller_id, wallet_id)` pair. Once a caller exceeds `MODEL_INVERSION_QUERY_LIMIT`
queries (default 100) on a wallet, subsequent API requests for that wallet return
**429 Too Many Requests**. This prevents sustained gradient-based attacks.

| Config Variable | Default | Description |
|---|---|---|
| `MODEL_INVERSION_QUERY_LIMIT` | `100` | Max queries per (caller, wallet) before 429 |
| `MODEL_INVERSION_DP_EPSILON` | `1.0` | Privacy budget ε for Laplace mechanism |
| `SCORE_ROUNDING_GRANULARITY` | `1` | Quantize scores to this granularity |

### Seeding Prevents Averaging

The seed derivation `hash(caller_id + timestamp_bucket)` ensures:
- **Different results for identical queries**: Attacker's repeated score queries
  get different noisy values, so averaging them doesn't converge to the truth.
- **Reproducibility within a time window**: For testing / audit, the same
  `(caller_id, timestamp)` always produces the same noise.

### Example: Attacker Scenario

```
Attacker's intended attack:
  Query 1: GET /score/G...wallet.../USDC:G.../XLM:native → score = 75
  Remove last trade, re-query
  Query 2: GET /score/G...wallet.../USDC:G.../XLM:native → score = 72
  Delta = 3 points → identifies last trade as moderately anomalous

With model inversion defence:
  Query 1: score = 75 + Laplace(0, 100/ε) → 68 (noisy)
  Query 2: score = 72 + Laplace(0, 100/ε) → 79 (different noise seed)
  Delta = 11 points (includes noise) → cannot invert
  After 100 queries: 429 Too Many Requests → rate limited
```

---

## Federated Learning Auditability (Issue #227)

### Overview

LedgerLens uses a federated learning architecture in which multiple participants
train on local data and submit gradient updates to a central coordinator.  To
satisfy regulatory requirements and enable post-incident investigation of
Byzantine participants, every training round is recorded in a tamper-evident
audit trail.

### Information Recorded per Round

Each row in the `federated_audit_trail` database table captures:

| Field | Type | Description |
|---|---|---|
| `round_id` | SHA-256 hex | Deterministic hash of `(timestamp, sorted fingerprints, model_version)` — prevents sequential-ID manipulation |
| `round_timestamp` | ISO-8601 UTC | When the round was recorded |
| `participant_fingerprints` | JSON array | SHA-256 certificate fingerprints of every contributing participant |
| `gradient_norms` | JSON object | `{"participant_id": l2_norm}` — scalar norms only, **never raw tensors** |
| `aggregation_algorithm` | string | E.g. `"fedavg"`, `"staleness_weighted_fedavg"` |
| `aggregate_model_hash` | SHA-256 hex | Hash of the post-aggregation global model weights |
| `round_outcome` | string | `"success"` or `"abort"` |
| `model_version` | integer | Monotonically increasing version counter after aggregation |
| `participant_count` | integer | Number of contributors in this round |
| `prev_hash` | SHA-256 hex | Hash of the previous record's canonical JSON — forms a Merkle chain |
| `recorded_at` | datetime | DB insertion timestamp |

### Security Invariants

| Invariant | Enforcement |
|---|---|
| **No raw gradients** | `FederatedAuditTrail.record_round()` raises `TypeError` if any `gradient_norms` value is not a scalar. Raw tensors are rejected before reaching the DB. |
| **Append-only records** | The application layer exposes no UPDATE or DELETE path for `federated_audit_trail`. Enforce DB-level row-security policies in production (e.g. PostgreSQL `GRANT INSERT` only). |
| **Deterministic round ID** | `round_id = SHA-256(timestamp + sorted(fingerprints) + model_version)` — an attacker cannot insert a fake round without the hash changing. |
| **Merkle chain** | `prev_hash` chains each record to its predecessor. Retroactive insertion or deletion breaks the chain and is detectable. |

### Enabling the Audit Trail

Attach a `FederatedAuditTrail` instance to `AsyncFederatedCoordinator`:

```python
from detection.federated.coordinator import AsyncFederatedCoordinator, FederatedAuditTrail

audit = FederatedAuditTrail()          # uses RISK_SCORE_DB_URL by default
coord = AsyncFederatedCoordinator(weight_dim=256)
coord._audit = audit

# Optionally register real certificate fingerprints (hex SHA-256 of cert DER)
audit.set_participant_fingerprints({
    "participant_alice": "aabbccddeeff...",
    "participant_bob":   "11223344...",
})
```

If `coord._audit` is `None` (the default), no audit records are written.

### Query Tool

`scripts/query_federated_audit.py` provides a CLI interface to the audit trail.

```bash
# Look up a specific round by its deterministic hash
python -m scripts.query_federated_audit --round-id <round_id_hex>

# List all rounds a participant contributed to
python -m scripts.query_federated_audit --participant <fingerprint_hex>

# Find the round that produced a specific model version
python -m scripts.query_federated_audit --model-hash <sha256_hex>

# List the 50 most recent rounds (default)
python -m scripts.query_federated_audit --list

# List with pagination
python -m scripts.query_federated_audit --list --limit 20 --offset 40

# Machine-readable NDJSON output
python -m scripts.query_federated_audit --list --json

# Full detail for each record
python -m scripts.query_federated_audit --round-id <round_id_hex> --detail

# Use a custom database URL
python -m scripts.query_federated_audit \
    --db-url postgresql://user:pass@host/dbname \
    --list
```

### Regulatory Compliance Notes

Gradient norms (L2 scalar values) are recorded to enable anomaly detection
(unusually large or small norms may indicate a Byzantine participant) without
exposing any model parameters or training data.  The combination of:

- Deterministic, content-addressed round IDs
- Certificate fingerprints linking each round to specific participants
- Merkle hash chain over all records
- Aggregate model hash per round

…provides a complete, verifiable training lineage that satisfies common
regulatory requirements (e.g. EU AI Act audit trail obligations, MAS TRM
guidelines on model governance) and allows investigators to pinpoint exactly
which participant's update corrupted a model version.
