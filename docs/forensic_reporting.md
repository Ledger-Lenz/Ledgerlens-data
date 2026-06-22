# LedgerLens Forensic Reporting

This document describes the Forensic Reporting Engine: the report schema,
the on-chain anchoring workflow, and the step-by-step verification guide
a regulator can use to independently validate any LedgerLens report.

---

## Why Forensic Reports?

Blockchain analytics tools must produce auditable evidence, not just scores.
In a FATF Travel Rule review, SEC market-manipulation investigation, or FinCEN
SAR filing, "an AI flagged it" is insufficient.  A forensic report documents:

- The exact on-chain trades that contributed to the score, with Horizon URLs.
- Which statistical and ML features crossed which thresholds, with plain-English
  descriptions.
- SHAP values explaining each feature's contribution to the final score.
- The model version, training dataset hash, and feature schema used.
- A SHA-256 fingerprint of the entire report, anchored to the Stellar ledger.

---

## Report Schema

Every forensic report is a JSON object with the following top-level fields.

| Field | Type | Description |
|---|---|---|
| `report_id` | string (UUID v4) | Globally unique identifier for this report. |
| `generated_at` | string (ISO 8601 UTC) | Timestamp the report was created. |
| `wallet` | string | The Stellar account ID (G…) being assessed. |
| `asset_pair` | string | The asset pair in `CODE:ISSUER/CODE:ISSUER` format. |
| `risk_score` | integer 0–100 | The LedgerLens ensemble risk score. |
| `score_lower` | integer 0–100 | Lower bound of the conformal prediction interval. |
| `score_upper` | integer 0–100 | Upper bound of the conformal prediction interval. |
| `verdict` | `"clean"` \| `"suspicious"` \| `"wash_trade"` | Human-readable classification. |
| `top_shap_features` | array of objects | Top 10 SHAP attributions (see below). |
| `benford_analysis` | object | Per-window Benford metrics (see below). |
| `trade_evidence` | array of `TradeEvidence` objects | Up to 20 most anomalous trades. |
| `model_metadata` | object | Model name, version, dataset hash, schema version. |
| `report_sha256` | string | SHA-256 fingerprint of all other fields. |
| `soroban_anchor_tx` | string \| null | Stellar transaction hash of the on-chain anchor. |

### SHAP Feature Attribution Entry

```json
{
  "feature": "benford_mad_24h",
  "description": "Mean Absolute Deviation between observed and expected Benford digit frequencies over the trailing 24-hour window.",
  "value": 0.047,
  "contribution": 0.34
}
```

`contribution` is the SHAP value: positive increases risk score, negative decreases it.

### Benford Analysis Entry

```json
{
  "24": {
    "chi_square": 18.4,
    "mad": 0.021,
    "mad_nonconforming": true,
    "z_scores": {"1": 2.1, "2": 0.4, ...},
    "sample_size": 312
  }
}
```

Keys are window sizes in hours (matching `config.BENFORD_WINDOWS_HOURS`).

### TradeEvidence Entry

```json
{
  "trade_id": "abc123",
  "ledger": 49123456,
  "base_account": "GABC…",
  "counter_account": "GDEF…",
  "base_amount": 5000.0,
  "counter_amount": 5001.2,
  "asset_pair": "XLM:native/USDC:GA5Z…",
  "horizon_url": "https://horizon.stellar.org/trades/abc123"
}
```

`horizon_url` is always constructed from `config.HORIZON_URL` — it is never
derived from user input, preventing SSRF.

---

## Verdict Thresholds

| Verdict | Score Range |
|---|---|
| `clean` | 0 – 69 |
| `suspicious` | 70 – 79 (configurable via `RISK_SCORE_FLAG_THRESHOLD`) |
| `wash_trade` | 80 – 100 |

---

## On-Chain Anchoring Workflow

```
Report generated (JSON)
        │
        ▼
SHA-256(to_dict minus sha256 field) ──► stored in report.report_sha256
        │
        ▼  (--anchor flag)
anchor_report(report_id, report_sha256)
        │
        ▼
Soroban ledgerlens-score contract
anchor_report(report_id: String, sha256: String)
        │
        ▼
Stellar ledger records tx at objective ledger close time
        │
        ▼
tx_hash stored in report.soroban_anchor_tx
```

The anchor transaction is visible to anyone via:

```
GET https://horizon.stellar.org/transactions/{tx_hash}
```

The embedded `sha256` in the transaction must match `report.report_sha256`
for the report to be considered valid.

---

## Generating a Report

### Single wallet (CLI)

```bash
python -m scripts.score_wallet \
  --wallet G... \
  --pair "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native" \
  --report \
  --report-format markdown

# With on-chain anchor:
python -m scripts.score_wallet ... --report --anchor
```

Output is written to `reports/forensic/{wallet[:12]}_{timestamp}.{ext}` with
permissions `0o600` (owner-readable only).

### Bulk job (CSV)

```bash
python -m scripts.generate_reports \
  --input wallets.csv \
  --pair "XLM:native" \
  --anchor

# wallets.csv format:
# wallet,pair
# GABC...,XLM:native/USDC:issuer
# GDEF...,
```

The bulk job uses `config.REPORT_CONCURRENCY` (default: 4) parallel workers
and shows a `tqdm` progress bar.

---

## Regulator Verification Guide

A regulator or compliance officer can independently verify any LedgerLens
forensic report in three steps.

### Step 1 — Verify the report's internal integrity

The `report_sha256` field must equal the SHA-256 of the report with that
field removed:

```python
import hashlib, json

with open("report.json") as f:
    data = json.load(f)

stored_hash = data.pop("report_sha256")
computed = hashlib.sha256(
    json.dumps(data, sort_keys=True).encode()
).hexdigest()

if computed == stored_hash:
    print("✓ Report integrity verified")
else:
    print("✗ Report has been tampered with!")
    print(f"  Stored:   {stored_hash}")
    print(f"  Computed: {computed}")
```

### Step 2 — Verify the on-chain anchor timestamp

If `soroban_anchor_tx` is non-null, fetch the transaction from Horizon:

```
GET https://horizon.stellar.org/transactions/{soroban_anchor_tx}
```

1. Note the `created_at` field — this is the objective, non-repudiable
   timestamp of the report's existence.
2. Locate the `INVOKE_HOST_FUNCTION` operation in the transaction envelope.
3. Confirm the `anchor_report` invocation parameters include the `report_id`
   and `report_sha256` from the JSON report.
4. Cross-check the SHA-256 against the locally computed hash from Step 1.

### Step 3 — Verify individual trades on Horizon

Each entry in `trade_evidence` contains a `horizon_url`.  Open any URL in a
browser or `curl` it to retrieve the raw on-chain trade record:

```
GET https://horizon.stellar.org/trades/abc123
```

Confirm that `base_account`, `counter_account`, `base_amount`, and
`counter_amount` match the values in the report.

---

## Security Properties

| Property | Mechanism |
|---|---|
| Tamper-evidence | SHA-256 covers all fields; any change produces a different hash. |
| Non-repudiation | Soroban anchor records hash + timestamp immutably on the Stellar ledger. |
| SSRF prevention | `horizon_url` constructed only from `config.HORIZON_URL`. |
| Data confidentiality | Report files written with mode `0o600` (owner-readable only). |
| Audit trail | `report_id` is a UUID v4; `generated_at` is UTC ISO 8601. |

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `HORIZON_URL` | `https://horizon.stellar.org` | Base URL for Horizon API and trade links. |
| `REPORT_CONCURRENCY` | `4` | Number of parallel workers for bulk report generation. |
| `RISK_SCORE_FLAG_THRESHOLD` | `70` | Score at or above which verdict is `suspicious`. |
| `SOROBAN_RPC_URL` | `https://soroban-testnet.stellar.org` | Soroban RPC endpoint for on-chain anchoring. |
| `LEDGERLENS_CONTRACT_ID` | _(required for anchoring)_ | Contract ID of the `ledgerlens-score` Soroban contract. |
| `LEDGERLENS_SUBMITTER_SECRET` | _(required for anchoring)_ | Secret key of the service account authorised to anchor reports. |
