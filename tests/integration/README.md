# Integration Tests — Testnet

This directory contains live integration tests that run against the Stellar
Testnet. They are **not** executed by `make test` and require explicit opt-in.

---

## Available Tests

| Test | File | Purpose |
|---|---|---|
| **E2E Pipeline Test** | `test_full_pipeline_e2e.py` | Full-stack test: trades from Horizon SSE → ingestion → features → scoring → DB |
| **Contract Client Live** | `test_contract_client_live.py` | Soroban contract interaction (deploy, submit, retrieve scores) |
| **Kafka Streaming** | `test_kafka_streaming.py` | Kafka producer and consumer integration |

---

## Prerequisites

| Requirement | How to satisfy |
|---|---|
| Funded Testnet keypair | Run `scripts/testnet_setup.py` (funds via Friendbot) |
| Deployed `ledgerlens-score` contract | Run `scripts/testnet_setup.py --wasm-path ledgerlens_score.wasm` |
| `LEDGERLENS_INTEGRATION_TESTS=1` env var | Set in shell or CI |

---

## Environment variables

| Variable | Required | Default |
|---|---|---|
| `LEDGERLENS_INTEGRATION_TESTS` | Yes (must be `1`) | — |
| `LEDGERLENS_CONTRACT_ID` | Yes | — |
| `LEDGERLENS_SUBMITTER_SECRET` | Yes | — |
| `SOROBAN_RPC_URL` | No | `https://soroban-testnet.stellar.org` |
| `HORIZON_URL` | No | `https://horizon-testnet.stellar.org` |
| `RISK_SCORE_DB_URL` | No | `sqlite:///./testnet_scores.db` |

Populate them by sourcing `.env.testnet` (written by `scripts/testnet_setup.py`):

```bash
export $(grep -v '^#' .env.testnet | xargs)
export LEDGERLENS_INTEGRATION_TESTS=1
```

---

## Running locally

```bash
# 1. Deploy contract (once per testnet reset or keypair rotation)
python -m scripts.testnet_setup \
    --wasm-path ledgerlens_score.wasm \
    --wasm-sha256 <sha256-from-release> \
    --salt ci-testnet

# 2. Source the generated env file
export $(grep -v '^#' .env.testnet | xargs)
export LEDGERLENS_INTEGRATION_TESTS=1

# 3. Run all integration tests
pytest tests/integration/ -v --timeout=120

# 3b. Or run just the e2e test
make test-e2e
```

---

## End-to-End Test Expectations

The E2E test (`test_full_pipeline_e2e.py`) validates the full pipeline:

1. **Setup:** Generates known trade patterns using test factories
2. **Stream:** Ingests trades from Horizon Testnet SSE
3. **Process:** Computes Benford metrics, extracts features, scores with ensemble
4. **Store:** Persists risk scores to the database
5. **Validate:** Asserts scores are within expected ranges for the trade pattern

### Test Patterns

| Pattern | Trade behavior | Expected score range | Risk flag |
|---|---|---|---|
| **Clean** | Random amounts, diverse counterparties | 0–30 | ✗ (legitimate) |
| **Round-trip** | Buy then sell same asset back | 60–100 | ✓ (wash trade) |
| **Same-amount** | Repeated fixed-amount trades | 60–100 | ✓ (wash trade) |

### Timeout & Assertions

- Each pattern generates ≥3 wallets with ≥20 trades
- Pipeline must compute scores within 60 seconds
- If scores don't appear, test fails with descriptive message
- DB is cleaned before and after test to avoid contamination

---

## WASM artifact

The `ledgerlens-score` WASM is built from the
[`ledgerlens-contract`](https://github.com/Ledger-Lenz/ledgerlens-contract)
repository. Always verify the SHA-256 hash before deploying.

| Field | Value |
|---|---|
| **Repository** | `Ledger-Lenz/ledgerlens-contract` |
| **WASM file** | `ledgerlens_score.wasm` |
| **Version / tag** | `v0.1.0` (update this when the contract is released) |
| **Commit hash** | _To be filled in after first release_ |
| **SHA-256** | _To be filled in after first release — verify with `sha256sum ledgerlens_score.wasm`_ |

The CI workflow downloads the WASM via:
```bash
gh release download v0.1.0 \
  --repo Ledger-Lenz/ledgerlens-contract \
  --pattern 'ledgerlens_score.wasm'
```

---

## Testnet fee model

| Operation | Approximate cost |
|---|---|
| `submit_score` (each call) | ~0.00001–0.0001 XLM (Soroban fee simulation) |
| Full pipeline run (10 flagged wallets) | ~0.001 XLM total |
| Friendbot funding | Free (Testnet only) |

Fees are estimated by `ContractClient.invoke`'s simulation step before signing.
The exact fee is resource-dependent; the figures above are for a typical
`submit_score` invocation with 7 parameters on Testnet as of Soroban Protocol 21.

---

## Soroban transaction details

### Parameter XDR encoding (`submit_score`)

| Parameter | Soroban type | Python encoder |
|---|---|---|
| `wallet` | `Address` | `scval.to_address(wallet)` |
| `asset_pair` | `String` | `scval.to_string(asset_pair)` |
| `score` | `Uint32` | `scval.to_uint32(score)` |
| `benford_flag` | `Bool` | `scval.to_bool(benford_flag)` |
| `ml_flag` | `Bool` | `scval.to_bool(ml_flag)` |
| `timestamp` | `Uint64` | `scval.to_uint64(timestamp)` |
| `confidence` | `Uint32` | `scval.to_uint32(confidence)` |

### Authorization model

`submit_score` requires the submitter's account to be authorized. Authorization
is handled via standard Soroban invoker auth: the transaction is signed by the
submitter keypair (`LEDGERLENS_SUBMITTER_SECRET`) before submission. There is no
separate admin-call requirement — the contract constructor bakes in the
authorized submitter address at deploy time (or the contract uses invoker
authorization only). Consult the `ledgerlens-contract` source for the exact
model.

---

## CI isolation

Integration tests are completely separate from the main `ci.yml` workflow:

- `make test` → runs `pytest tests/` → **skips** `tests/integration/` unless
  `LEDGERLENS_INTEGRATION_TESTS=1` is set (enforced via `conftest.py`).
- `testnet-integration.yml` → triggers on `workflow_dispatch` and weekly schedule
  only; never blocks PRs.
- Each CI run deploys with `--salt ci-testnet` for a deterministic contract ID
  that can be identified and cleaned up.

---

## Troubleshooting

- **`LEDGERLENS_INTEGRATION_TESTS not set`** — set `export LEDGERLENS_INTEGRATION_TESTS=1`.
- **`LEDGERLENS_CONTRACT_ID not set`** — run `scripts/testnet_setup.py` first.
- **`Friendbot rate limit (429)`** — the setup script retries 3 times with a
  5-second delay automatically.
- **Test timeout** — each test has a 120-second timeout enforced by
  `pytest-timeout`. Transient RPC failures are retried up to 3 times with a
  5-second delay before the test fails.
- **WASM hash mismatch** — pass the correct `--wasm-sha256` or use
  `--skip-hash-check` only for local development.
- **Database locked** — if the integration test DB is locked after a crash,
  delete `testnet_scores.db` and try again.

