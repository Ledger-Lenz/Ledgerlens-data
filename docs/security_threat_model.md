# LedgerLens Security Threat Model (STRIDE)

## System Boundary Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL INTEGRATIONS                         │
│  Stellar Horizon API  │  Soroban RPC  │  External Webhook Alerts    │
└────────────┬──────────────┬──────────────┬──────────────────────────┘
             │              │              │
┌────────────▼──────────────▼──────────────▼──────────────────────────┐
│                   LEDGERLENS DATA BOUNDARY                           │
│                                                                      │
│  ┌──────────────────────┐  ┌─────────────────────────────────────┐ │
│  │  INGESTION LAYER     │  │  DETECTION & SCORING LAYER          │ │
│  │  (ingestion/)        │  │  (detection/)                       │ │
│  │                      │  │                                     │ │
│  │  • horizon_streamer  │  │  • benford_engine                   │ │
│  │  • kafka_producer    │  │  • feature_engineering              │ │
│  │  • historical_loader │  │  • model_inference (RiskScorer)     │ │
│  │  • orderbook_loader  │  │  • shap_explainer                   │ │
│  └──────────┬───────────┘  │  • adversarial/robustness           │ │
│             │              │  • differential_privacy              │ │
│             │              └──────────────┬──────────────────────┘ │
│             │                             │                        │
│  ┌──────────▼──────────────────────────────▼──────────────────────┐ │
│  │            STREAMING & PERSISTENCE LAYER                        │ │
│  │  (streaming/, detection/persistence.py, detection/risk_score_store.py) │
│  │                                                                 │ │
│  │  • feature_buffer          • risk_score_store (SQLAlchemy)    │ │
│  │  • streaming_scorer        • ShapQueryCount (DP budget)       │ │
│  │  • alert_dispatcher        • RiskScoreRecord                  │ │
│  │  • ws_server (WebSocket)   • ModelArtifact (integrity)        │ │
│  │  • kafka_worker            • audit_trail                      │ │
│  └──────────┬──────────────────────────────┬───────────────────┘ │
│             │                              │                     │
└─────────────┼──────────────────────────────┼─────────────────────┘
              │                              │
     ┌────────▼────────┐           ┌────────▼────────┐
     │  SQLite / PostgreSQL        │  Soroban Smart  │
     │  (RISK_SCORE_DB_URL)        │  Contract       │
     │                             │  (submit_score) │
     └─────────────────┘           └─────────────────┘

TRUST BOUNDARIES:
  [1] Horizon API (untrusted external data source)
  [2] Model artifacts on disk (must verify integrity)
  [3] Database connection (SQL injection risk)
  [4] WebSocket client connections (auth bypass, DoS)
  [5] Kafka broker (message tampering if not using TLS/SASL)
  [6] Soroban contract RPC endpoint (network latency, RPC spoofing)
```

---

## STRIDE Analysis by Component

### 1. API & WebSocket Server (streaming/ws_server.py, streaming/streaming_scorer.py)

| Category | Threat | Impact | Current Mitigation | Strength | Recommendation |
|----------|--------|--------|---------------------|----------|-----------------|
| **S** (Spoofing) | Unauthenticated WebSocket connections | Attacker subscribes to risk scores meant for internal use | JWT token validation via `ws_auth.py::validate_jwt_token()` | Medium | Implement rate-limiting per JWT sub; rotate JWT keys monthly; log all auth failures |
| **T** (Tampering) | Risk score query inversion attack | Adversary queries same wallet 100x, observes score deltas, reconstructs feature vector | [NEW] Query-rate limiting + Laplace noise (see #264) | Medium→High | Implement per-caller budget tracking in `risk_score_store.py::ShapQueryCount`; add exponential backoff |
| **R** (Repudiation) | API caller denies requesting a specific wallet's score | Regulatory audit unable to prove who queried what | [NEW] Query audit trail in `audit_trail.py` | Low | Implement immutable append-only query log; timestamp and sign each entry |
| **I** (Information Disclosure) | SHAP explanations leak feature values via differentials | Adversary observes `explain_private()` results across two wallet states | Gaussian DP mechanism in `detection/differential_privacy.py` (ε=1.0, δ=1e-5) | Medium | Increase ε validation; cap total queries per wallet at 100 (see config.DP_RENYI_QUERY_THRESHOLD) |
| **D** (Denial of Service) | WebSocket server flooded with connections | Service unavailable; alert latency breaches SLA | WS_MAX_CLIENTS (200), per-client queue depth (100) in `config.py` | Medium | Implement connection rate-limiting; add Prometheus metrics for queue depth alerts |
| **E** (Elevation of Privilege) | Attacker forges JWT to impersonate admin role | Access to internal audit endpoints; can manually override scores | No admin/privilege distinction in JWT; all tokens treated equally | Low | Add `role` claim to JWT; check role before exposing internal endpoints |

**File References:**
- `streaming/ws_server.py` — WebSocket server, auth validation
- `streaming/ws_auth.py` — JWT validation
- `detection/differential_privacy.py` — Gaussian DP mechanism
- `detection/audit_trail.py` — Query audit logging
- `detection/risk_score_store.py` — Query tracking (ShapQueryCount model)

---

### 2. Model Inference & Risk Scoring (detection/model_inference.py, detection/ensemble_calibrator.py)

| Category | Threat | Impact | Current Mitigation | Strength | Recommendation |
|----------|--------|--------|---------------------|----------|-----------------|
| **S** (Spoofing) | Attacker substitutes a backdoored model artifact | Risk scores systematically lowered for attacker's wallet | Ed25519 signature verification chain (see `detection/persistence.py::ModelArtifact.verify_chain()`) | High | Implement GPG key pinning; archive all signed artifacts with deployment metadata |
| **T** (Tampering) | Model weights corrupted on disk (bit flip, ransomware) | Incorrect scores or crash on inference | SHA-256 integrity check in `ModelArtifact.verify_chain()` | High | Store models in immutable object store (S3 with versioning); enable MFA delete |
| **R** (Repudiation) | Training team denies responsibility for model regression | Score quality degraded but no audit trail | `model_metadata.json` records training date, dataset SHA-256, commit hash | Medium | Sign `model_metadata.json` with trainer's Ed25519 key; require approval workflow for promotion |
| **I** (Information Disclosure) | Model feature sensitivities leak via SHAP values | Attacker learns which features drive scores | Per-feature SHAP sensitivities with Gaussian noise in `detection/shap_explainer.py::ShapExplainer.explain_private()` | Medium | Validate that noise scale ≥ 0.05 (DP_DEFAULT_SENSITIVITY); audit explain calls in logs |
| **D** (Denial of Service) | Adversary queries ensemble with malformed feature rows | RiskScorer.score() crashes; pipeline halts | Feature schema validation + try/except in batch scorer | Low | Add circuit breaker; timeout per-batch; quarantine bad feature rows to DLQ |
| **E** (Elevation of Privilege) | Unauthorized model retraining triggered | Attacker injects poisoned training data | Retraining script requires Git branch protection + CI approval | Medium | Require signed Git commits from authorized developers; audit `scripts/retrain_if_drifted.py` |

**File References:**
- `detection/model_inference.py` — RiskScorer.score(), ensemble voting
- `detection/ensemble_calibrator.py` — Multi-objective calibration
- `detection/persistence.py` — ModelArtifact integrity chain
- `detection/shap_explainer.py` — Differentially-private SHAP explanations

---

### 3. Model Training & Feature Engineering (detection/model_training.py, detection/feature_engineering.py)

| Category | Threat | Impact | Current Mitigation | Strength | Recommendation |
|----------|--------|--------|---------------------|----------|-----------------|
| **S** (Spoofing) | Attacker substitutes labelled training dataset | Model learns false positives → no wash trades detected | Training data SHA-256 recorded in `model_metadata.json` | Medium | Implement cryptographic attestation for dataset provenance; store checksums in contract |
| **T** (Tampering) | Poisoned labels injected into annotation queue | Model trained on mislabelled data | HMAC-SHA256 verification on `data/annotation_queue.json` entries (see `detection/active_learning/annotation_queue.py`) | Medium | Require all annotations signed by annotator's Ed25519 key; audit trail in database |
| **R** (Repudiation) | Annotator claims they didn't label a wallet as wash trade | False accusation / regulatory dispute | Annotation HMAC includes annotator_id and timestamp | Low | Store signed annotations in immutable ledger; add annotator signature |
| **I** (Information Disclosure) | Training dataset with sensitive wallet activity exposed | Privacy violation; GDPR/CCPA liability | No explicit redaction; relies on access control to training data | Low | Implement differential privacy in training (DP-SGD via Opacus in `detection/privacy/dp_training.py`) |
| **D** (Denial of Service) | Retraining process consumes 100% CPU/memory | Scoring pipeline starved of resources | Separate retraining script; can be run off-peak | Medium | Implement resource quotas; add monitoring for training job hangs; timeout after 1 hour |
| **E** (Elevation of Privilege) | Unauthorized script execution in CI | Attacker runs arbitrary training code | GitHub Actions branch protection; OIDC for AWS artifact store | High | Enforce signed commits; require explicit PR approval before retraining; restrict secrets scope |

**File References:**
- `detection/model_training.py` — Training entry point, label validation
- `detection/feature_engineering.py` — 36+ feature computation
- `detection/active_learning/annotation_queue.py` — Annotation integrity
- `detection/privacy/dp_training.py` — Differentially-private training
- `scripts/retrain_if_drifted.py` — Automated retraining trigger

---

### 4. Data Ingestion (ingestion/horizon_streamer.py, ingestion/kafka_producer.py, ingestion/historical_loader.py)

| Category | Threat | Impact | Current Mitigation | Strength | Recommendation |
|----------|--------|--------|---------------------|----------|-----------------|
| **S** (Spoofing) | Attacker spoof Horizon API responses | Fake trade data fed into feature pipeline | SSL/TLS pinning to Horizon domain; certificate validation | High | Implement DNS pinning; validate Ledger timestamp in response; detect timestamp gaps |
| **T** (Tampering) | MITM intercepts Stellar trade stream | Trade amounts altered before Benford analysis | TLS 1.3 required for Horizon connection | High | Implement message authentication codes (HMAC-SHA256) over trade payload; log any TLS downgrades |
| **R** (Repudiation) | Horizon claims they didn't serve certain trades | Scoring inconsistency / audit mismatch | Ingestion logs include cursor position and ledger hash | Medium | Record full Horizon response body (encrypted) for 30 days; hash response and store offchain |
| **I** (Information Disclosure) | Trade stream includes sensitive wallet metadata | PII exposed if not filtered | Ingestion fetches only trade amounts, counterparties, timestamps — no email/identity | Medium | Implement column-level access control; audit which fields are stored |
| **D** (Denial of Service) | Horizon rate limit exhausted (10 req/sec default) | Feature buffer backlog grows unbounded | Exponential backoff + circuit breaker in `utils/retry.py` | Medium | Implement token bucket; detect rate-limit headers; scale ingestion workers dynamically |
| **E** (Elevation of Privilege) | Attacker modifies ingestion credentials to exfiltrate data | API keys compromised; data stolen | HORIZON_URL / credentials in env vars only (not committed) | Medium | Rotate API keys monthly; use IAM roles if using Horizon SDK; audit credential access logs |

**File References:**
- `ingestion/horizon_streamer.py` — SSE stream handler
- `ingestion/kafka_producer.py` — Kafka producer serialization
- `ingestion/historical_loader.py` — Bulk historical data loader
- `utils/retry.py` — Backoff/circuit breaker

---

### 5. Persistence & Database (detection/persistence.py, detection/risk_score_store.py)

| Category | Threat | Impact | Current Mitigation | Strength | Recommendation |
|----------|--------|--------|---------------------|----------|-----------------|
| **S** (Spoofing) | SQL connection string intercepted | Attacker connects to database as app | RISK_SCORE_DB_URL in env var; no hardcoded credentials | Medium | Enforce TLS for DB connection; use IAM auth (AWS RDS) instead of passwords |
| **T** (Tampering) | SQL injection in RiskScoreStore queries | Risk scores modified after insertion; query results forged | SQLAlchemy ORM (parameterized queries); no string formatting | High | Implement query audit logging; store risk_scores in append-only ledger for forensics |
| **R** (Repudiation) | DBA modifies risk_scores directly | False risk score; regulator cannot verify authenticity | No audit trail of schema modifications | Low | Enable PostgreSQL audit logging (pg_audit); track all DML operations with timestamps |
| **I** (Information Disclosure) | Backup database copied to attacker workstation | All risk scores, wallet analysis exposed | Database backups encrypted at rest; access control via IAM | Medium | Implement row-level security (RLS) in PostgreSQL; encrypt PII columns; add masking |
| **D** (Denial of Service) | Attacker performs full table scan via `SELECT *` | Database CPU saturated; scoring latency increases | Connection pooling (5–10 connections); query timeout (30 sec) | Medium | Add query rate limiting per connection; implement index on (wallet, asset_pair); add slow-query logging |
| **E** (Elevation of Privilege) | Attacker escalates to DB admin via weak password | Full database compromise | App uses restricted IAM role (no ALTER TABLE, no DROP) | Medium | Enforce strong password policy; separate read-only role for API; require MFA for admin access |

**File References:**
- `detection/persistence.py` — SQLAlchemy models, engine creation
- `detection/risk_score_store.py` — RiskScoreStore CRUD
- `config.py` — DB_POOL_SIZE, DB_POOL_TIMEOUT

---

### 6. Kafka Pipeline (ingestion/kafka_producer.py, streaming/kafka_worker.py)

| Category | Threat | Impact | Current Mitigation | Strength | Recommendation |
|----------|--------|--------|---------------------|----------|-----------------|
| **S** (Spoofing) | Attacker publishes trades to `ledgerlens.trades.*` topics | Fraudulent trades ingested; scores corrupted | Kafka SASL/SSL authentication (KAFKA_SASL_USERNAME/PASSWORD in env) | Medium | Require SASL/SCRAM with TLS; validate broker certificate; use service account with least privilege |
| **T** (Tampering) | Message intercepted and modified in-flight | Trade amount changed; feature calculation skewed | TLS encryption in transit (enforce TLS_REQUIRED) | High | Implement HMAC-SHA256 over message payload; validate in consumer; reject if verification fails |
| **R** (Repudiation) | Producer denies sending a specific trade | Audit trail broken; cannot identify source | Kafka broker stores producer ID + timestamp per message | Low | Sign each message with producer key; store signed message in Soroban contract (anchor) |
| **I** (Information Disclosure) | Kafka broker unencrypted; network traffic sniffed | Trade data exposed in plaintext | TLS in transit; no at-rest encryption on broker | Medium | Enable Kafka broker-side encryption (KMS if AWS MSK); implement network segmentation (VPC) |
| **D** (Denial of Service) | Attacker floods topic with 1M messages/sec | Consumer lag grows unbounded; scoring latency unacceptable | Consumer group auto-scales; lag threshold (500 messages) triggers alert | Medium | Implement producer rate limiting; partition strategy to distribute load; scale consumers dynamically |
| **E** (Elevation of Privilege) | Attacker gains broker credentials | Can consume any topic, modify replicas, delete data | KAFKA_SASL_PASSWORD stored in GitHub Actions secret | Medium | Rotate credentials monthly; use OIDC + temporary tokens instead of static password; audit broker access logs |

**File References:**
- `ingestion/kafka_producer.py` — Message serialization, producer config
- `streaming/kafka_worker.py` — Consumer group, offset management
- `config.py` — KAFKA_BOOTSTRAP_SERVERS, KAFKA_SASL_*

---

### 7. Soroban Smart Contract Integration (integrations/contract_client.py)

| Category | Threat | Impact | Current Mitigation | Strength | Recommendation |
|----------|--------|--------|---------------------|----------|-----------------|
| **S** (Spoofing) | Attacker forges contract invocation | False risk scores published on-chain | Soroban contract validates caller signature (via Stellar auth) | High | Verify contract ID matches LEDGERLENS_CONTRACT_ID in config; implement contract version check |
| **T** (Tampering) | Network partition delays score submission | On-chain score stale; outdated by 5+ minutes | No retry mechanism; relies on Soroban RPC availability | Low | Implement exponential backoff + replay logic; store pending submissions in local queue |
| **R** (Repudiation) | Contract denies receiving score submission | Risk score disputes | Soroban logs transaction on-chain; immutable proof | High | Store submission transaction hash in local DB; implement audit report from contract state |
| **I** (Information Disclosure) | RPC endpoint observes all submitted risk scores | Privacy violation; attacker learns scoring patterns | Soroban RPC endpoint in config; no TLS pinning | Medium | Require TLS for RPC connection; use private RPC endpoint; implement rate limiting at RPC level |
| **D** (Denial of Service) | Soroban RPC endpoint unresponsive | Score submissions hang indefinitely | Timeout (10 sec) in contract_client.py; exceptions bubbled to caller | Medium | Implement circuit breaker; queue submissions if RPC down; async batch submission with retry |
| **E** (Elevation of Privilege) | LEDGERLENS_SUBMITTER_SECRET exposed in GitHub Actions logs | Attacker uses secret to submit arbitrary scores | Secret stored in GitHub Actions (masked in logs); rotation manual | Medium | Use OIDC to generate short-lived tokens instead of long-lived secrets; rotate secret quarterly |

**File References:**
- `integrations/contract_client.py` — Contract invocation, error handling
- `config.py` — LEDGERLENS_CONTRACT_ID, SOROBAN_RPC_URL, LEDGERLENS_SUBMITTER_SECRET

---

## Mitigations Table (Consolidated)

| Threat ID | Component | Threat | Mitigation File(s) | Status | Effort |
|-----------|-----------|--------|-------------------|--------|--------|
| T1 | API/WebSocket | Query inversion attack | `detection/risk_score_store.py` (query limit), `detection/model_inference.py` (Laplace noise) | **Implementing** (#264) | High |
| T2 | Model Inference | Artifact substitution | `detection/persistence.py::ModelArtifact.verify_chain()` | ✅ Implemented | Medium |
| T3 | Training | Data poisoning | `detection/active_learning/annotation_queue.py` (HMAC) | ✅ Implemented | Low |
| T4 | Ingestion | Horizon API spoofing | TLS pinning in `ingestion/horizon_streamer.py` | ⚠️ Partial | Medium |
| T5 | Database | SQL injection | SQLAlchemy ORM in `detection/persistence.py` | ✅ Implemented | Low |
| T6 | Kafka | Message tampering | TLS in transit; SASL auth in config | ⚠️ Partial | Medium |
| T7 | WebSocket | Privilege escalation | `streaming/ws_auth.py` JWT validation | ⚠️ Partial | Low |
| T8 | Contract | RPC endpoint spoofing | TLS + timeout in `integrations/contract_client.py` | ⚠️ Partial | Medium |

---

## Residual Risk Register

### Accepted Risks (documented rationale)

| Risk | Reason | Mitigation Accepted | Review Date |
|------|--------|-------------------|-------------|
| **Horizon API MITM** | Assumes Stellar's CDN infrastructure is trustworthy. DNS spoofing / BGP hijacking mitigated by TLS but not pinned. | Accept: Pinning would require manual rotation; TLS sufficient for most threat actors. Revisit if regulatory requirement. | 2026-07-15 |
| **Database backup exposure** | Backups to S3 encrypted; attacker must compromise AWS IAM. Residual: misconfiguration could leave backups unencrypted. | Accept: Standard AWS security best practice (versioning + MFA delete). Quarterly access audit sufficient. | 2026-08-01 |
| **Soroban RPC rate limiting** | RPC node is external; no backpressure control. High-volume submission storms could cause timeouts. | Accept: Circuit breaker in `contract_client.py` prevents cascade failures. Fallback to local queue. | 2026-09-01 |
| **Feature privacy in DP-SHAP** | Gaussian noise with ε=1.0 is conservative; adversary may still infer features with 1000+ queries. | Accept: Rényi composition scales noise for high-frequency users (100+ queries → 3× noise). Operational limit enforced. | 2026-10-15 |
| **Training data labelling bias** | Annotations may reflect annotator bias (e.g., over-flagging specific market venues). | Accept: Monitor label distribution shift (PSI); retrain if drift exceeds 0.25. Audit trail of annotators. | 2026-09-30 |

---

## Attack Surface Summary

### High-Risk Entry Points (require hardened dev practices)

1. **API Risk Score Query Endpoint** (`streaming/streaming_scorer.py`)
   - **Risk**: Model inversion via 100+ queries per wallet
   - **Entry Control**: [NEW] Query-rate limiting + Laplace noise (#264)
   - **Dev Guidance**: Any endpoint returning continuous risk scores must apply output perturbation

2. **Model Artifact Loading** (`detection/persistence.py::ModelArtifact.verify_chain()`)
   - **Risk**: Backdoored `.joblib` files bypass scoring
   - **Entry Control**: Ed25519 signature + SHA-256 verification (mandatory)
   - **Dev Guidance**: NEVER load a model without calling `verify_chain()` first; CI grep enforces invariant

3. **Training Data Ingestion** (`detection/active_learning/annotation_queue.py`)
   - **Risk**: Poisoned labels cause model to miss wash trades
   - **Entry Control**: HMAC verification of all annotations
   - **Dev Guidance**: All labels must originate from the annotation queue with valid HMAC; audit trail required

4. **WebSocket Auth** (`streaming/ws_auth.py`)
   - **Risk**: Unauthenticated clients subscribe to real-time alerts
   - **Entry Control**: JWT token validation; token rotation policy
   - **Dev Guidance**: All WebSocket connections must validate JWT before subscription; no exception

5. **Database Persistence** (`detection/persistence.py`)
   - **Risk**: SQL injection; unauthorized modifications
   - **Entry Control**: SQLAlchemy ORM (parameterized); IAM role access control
   - **Dev Guidance**: Never construct SQL strings; use ORM exclusively; enable audit logging on production DB

6. **Kafka Message Production** (`ingestion/kafka_producer.py`)
   - **Risk**: Unauthenticated producers inject fake trades
   - **Entry Control**: SASL/SCRAM authentication; TLS in transit
   - **Dev Guidance**: Enforce TLS and SASL for all brokers; sign messages with HMAC

---

## Links & References

- **From `docs/security.md`**: See [Security Threat Model](security_threat_model.md) for comprehensive STRIDE analysis and attack surface.
- **From `CONTRIBUTING.md`**: Developers should review [Security Threat Model](../docs/security_threat_model.md) before implementing features that touch API endpoints, model loading, training data, or database persistence. High-risk components require security architect review.

---

## Implementation Checklist

- [x] System boundary diagram with trust boundaries
- [x] STRIDE analysis: 7 components × 6 categories
- [x] Mitigations table with file references
- [x] Residual risk register
- [x] Attack surface entry points
- [x] Links from security.md and CONTRIBUTING.md
- [ ] CI validation: markdown-link-check (to be run by CI)
- [ ] Security review sign-off (pending maintainer review)

---

**Last Updated**: 2026-06-29  
**Threat Model Version**: 1.0  
**Status**: Draft (pending review)
