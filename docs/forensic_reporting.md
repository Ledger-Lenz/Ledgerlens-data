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
| `asset_metadata` | object (optional) | Asset supply used for the report, with its `trust_tier`, `source`, `fetched_at`, `age_seconds` and `is_stale`. Present only when the report was generated with asset metadata; see [Asset Metadata: Staleness and Trust Tiers](asset_metadata_trust_tiers.md). |

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

## Example Forensic Report (Synthetic)

Below is an actual generated Markdown report using synthetic wallet data, showing the
anomalous trades, SHAP attribution, and Benford analysis sections as they appear to
compliance reviewers and regulators.

---

### LedgerLens Forensic Report

**Report ID:** `550e8400-e29b-41d4-a716-446655440000` | **Generated:** 2025-08-28T14:32:17Z

| Field | Value |
|---|---|
| Wallet | `GWASH123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890` |
| Asset Pair | USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN / XLM:native |
| **Risk Score** | **82 / 100** (78–86 conformal interval) |
| **Verdict** | **WASH_TRADE** |

⚠️  This wallet has been classified as a **wash-trade** participant with high confidence.
The evidence below should be reviewed for potential SAR submission under FinCEN guidance
and FATF Recommendation 20.

### Risk Score Summary

| Metric | Value |
|---|---|
| Risk Score | 82 |
| Conformal Lower Bound | 78 |
| Conformal Upper Bound | 86 |
| Verdict | wash_trade |
| Model Name | LedgerLens Ensemble v2.1 |
| Model Version | 2.1.3 |
| Training Dataset SHA-256 | `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6` |
| Feature Schema Version | 1.8 |

### SHAP Feature Attribution

The table below shows the top features that drove this wallet's risk score,
ordered by the magnitude of their SHAP contribution.

| # | Feature | Description | Value | SHAP Contribution |
|---|---|---|---|---|
| 1 | `benford_mad_24h` | Mean Absolute Deviation between observed and expected Benford digit frequencies over the trailing 24-hour window. | 0.0847 | +12.34 |
| 2 | `self_matching_rate` | Fraction of trades that match buy/sell orders between wallets with shared funding sources. | 0.73 | +10.18 |
| 3 | `counterparty_concentration_ratio` | Fraction of total volume traded with the single most frequent counterparty. | 0.89 | +9.45 |
| 4 | `round_trip_frequency` | Frequency of round-trip trades returning assets to the originating wallet within N ledgers. | 0.61 | +7.82 |
| 5 | `intra_minute_clustering` | Number of trades executed within the same minute, indicating coordinated order placement. | 24 | +5.63 |
| 6 | `cross_pair_trade_synchrony` | Temporal correlation of trades across asset pairs; high values indicate synchronized execution. | 0.78 | +4.91 |
| 7 | `entropy_of_amounts` | Shannon entropy of trade amounts; low values indicate repetitive sizing. | 1.24 | +3.27 |
| 8 | `order_cancellation_rate` | Fraction of placed orders that are cancelled before execution. | 0.14 | +1.82 |
| 9 | `volume_spike_frequency` | Number of 10-minute windows where volume exceeded the rolling 95th percentile. | 8 | +0.94 |
| 10 | `pair_diversity_score` | Count of distinct asset pairs traded by the wallet. | 2 | -0.56 |

### Benford's Law Analysis

Benford's Law predicts the expected frequency of leading digits in naturally
occurring data. Deviation from this distribution is a statistical indicator
of artificially manipulated trade amounts.

**Threshold:** MAD > 0.015 indicates non-conformity (Nigrini, 2012).
**Chi-square critical value (df=8, p=0.05):** 15.51.

| Window | Chi-Square | MAD | Non-Conforming | Max Z-Score | Sample Size |
|---|---|---|---|---|---|
| 1h | 42.17 | 0.0562 | ✗ | 3.24 | 156 |
| 4h | 38.92 | 0.0479 | ✗ | 2.91 | 487 |
| 24h | 35.61 | 0.0847 | ✗ | 4.13 | 1203 |
| 168h | 28.44 | 0.0321 | ✗ | 2.67 | 2841 |
| 720h | 22.13 | 0.0198 | ✗ | 1.89 | 3156 |

**Interpretation:** All time windows show MAD values well above 0.015, indicating statistically significant deviation from Benford's Law. This is consistent with deliberate trade amount manipulation.

### Trade Evidence

The 10 most anomalous trades are listed below. Each trade can be independently verified via the Horizon URL.

| # | Trade ID | Ledger | Base Account | Counter Account | Base Amount | Counter Amount | Asset Pair | Horizon URL |
|---|---|---|---|---|---|---|---|---|
| 1 | `0000000123456789` | 49123456 | `GWASH123…` | `GRING456A…` | 5000.00 | 5001.23 | USDC/XLM | [View](https://horizon.stellar.org/trades/0000000123456789) |
| 2 | `0000000123456790` | 49123457 | `GRING456A…` | `GWASH789…` | 5001.23 | 4999.87 | XLM/USDC | [View](https://horizon.stellar.org/trades/0000000123456790) |
| 3 | `0000000123456791` | 49123460 | `GWASH789…` | `GRING789B…` | 5500.00 | 5502.15 | USDC/XLM | [View](https://horizon.stellar.org/trades/0000000123456791) |
| 4 | `0000000123456792` | 49123461 | `GRING789B…` | `GWASH123…` | 5502.15 | 5498.50 | XLM/USDC | [View](https://horizon.stellar.org/trades/0000000123456792) |
| 5 | `0000000123456793` | 49123464 | `GWASH123…` | `GRING654C…` | 4800.00 | 4802.88 | USDC/XLM | [View](https://horizon.stellar.org/trades/0000000123456793) |
| 6 | `0000000123456794` | 49123465 | `GRING654C…` | `GWASH456…` | 4802.88 | 4798.12 | XLM/USDC | [View](https://horizon.stellar.org/trades/0000000123456794) |
| 7 | `0000000123456795` | 49123469 | `GWASH456…` | `GRING321D…` | 6200.00 | 6204.62 | USDC/XLM | [View](https://horizon.stellar.org/trades/0000000123456795) |
| 8 | `0000000123456796` | 49123470 | `GRING321D…` | `GWASH123…` | 6204.62 | 6199.38 | XLM/USDC | [View](https://horizon.stellar.org/trades/0000000123456796) |
| 9 | `0000000123456797` | 49123473 | `GWASH789…` | `GRING999E…` | 3500.00 | 3501.75 | USDC/XLM | [View](https://horizon.stellar.org/trades/0000000123456797) |
| 10 | `0000000123456798` | 49123474 | `GRING999E…` | `GWASH789…` | 3501.75 | 3498.50 | XLM/USDC | [View](https://horizon.stellar.org/trades/0000000123456798) |

### Report Integrity

| Field | Value |
|---|---|
| Report SHA-256 | `7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a` |
| Soroban Anchor Tx | *Not anchored* |

---

**Note:** This report uses synthetic wallet data (GWASH, GRING account prefixes are
illustrative) and is for documentation purposes only. Real reports are anchored to
the Stellar ledger with actual Horizon trade URLs.

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

## SHAP Interaction Values

### What they are

Standard SHAP values decompose a model's prediction into additive per-feature
contributions — each feature gets a single number representing its average
marginal contribution. **SHAP interaction values** (the Shapley interaction
index, Lundberg et al., 2018) go one step further: they decompose the prediction
into pairwise *interactions*, quantifying how much of the prediction is explained
by Feature A and Feature B *working together*, beyond what either contributes
alone.

Formally, the interaction value φᵢⱼ satisfies:

```
Σᵢ Σⱼ φᵢⱼ = f(x) − E[f(x)]        (completeness)
φᵢᵢ = main effect of feature i
φᵢⱼ = φⱼᵢ (symmetry)
```

### How to interpret a strong interaction

A large positive `interaction` value for a pair `(feature_a, feature_b)` means
the model learned a **synergistic risk signal**: those two features together push
the score higher than you would predict by summing their individual SHAP values.

Example: an interaction of `+8.2` for `counterparty_concentration x account_age`
means that wallets with *both* high counterparty concentration *and* a young
account age are scored approximately 8 points higher than a model that treats
those features independently would assign — a classic wash-trading fingerprint
not captured by either feature alone.

A negative interaction value indicates a *suppressive* relationship: the presence
of both features together reduces the score compared to adding their main effects.

### Computational cost

Interaction values are **O(n_samples × n_features²)** to compute. For a feature
matrix with 40 features and 10 000 rows this is ~16 million calls into the tree
ensemble, versus ~400 000 for plain SHAP values. They are therefore gated behind
a feature flag:

| Variable | Default | Description |
|---|---|---|
| `SHAP_INTERACTIONS_ENABLED` | `false` | Set to `true` to compute and include interaction values in forensic reports. |

Enable only for targeted forensic investigations, not for real-time scoring.

### LightGBM API compatibility note

Both XGBoost and LightGBM expose interaction values through the same
`shap.TreeExplainer(model).shap_interaction_values(X)` call with
`feature_perturbation="tree_path_dependent"` (the default). The returned array
shape is `(n_samples, n_features, n_features)` in both cases.

**Incompatibility:** LightGBM does **not** support
`feature_perturbation="interventional"` for interaction values. Passing that
option raises a `NotImplementedError`. If you have overridden the default
perturbation mode, revert to `"tree_path_dependent"` before calling
`shap_interaction_values`.

XGBoost multi-class models return a list of `(n_samples, n_features, n_features)`
arrays (one per class); `ShapExplainer.compute_interaction_values` automatically
selects index `[1]` (positive / wash-trade class).

### Report field

When `SHAP_INTERACTIONS_ENABLED=true`, the forensic report JSON gains a
`top_interactions` field and the Markdown report renders a **Feature Interactions**
table:

```json
"top_interactions": [
  {"feature_a": "counterparty_concentration", "feature_b": "account_age", "interaction": 8.2},
  ...
]
```

The corresponding formatted strings (via `format_top_interactions`) read:

```
counterparty_concentration x account_age contributes 8.2000 points to the score
```

Security note: `top_interactions` is an internal forensic report field. It is
**not** exposed via the external public API — see the Security Properties table above.

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
| `SHAP_INTERACTIONS_ENABLED` | `false` | Enable SHAP pairwise interaction values in forensic reports (O(n·d²) cost). |

---

## Interactive HTML Report Format

`detection/forensic_report_interactive.py` generates a self-contained HTML
forensic report alongside the existing JSON/PDF formats.

### Dependencies

```
plotly>=5.0        # interactive SHAP waterfall chart
pyvis>=0.3         # wallet graph visualisation (optional — graceful degradation)
```

### Generating an HTML report

```bash
python -m scripts.generate_reports --input wallets.csv --output-format html \
    --output-dir reports/forensic
```

Or from Python:

```python
from detection.forensic_report_interactive import generate_interactive_report

generate_interactive_report(report.to_dict(), "reports/forensic/my_report.html")
```

### Self-contained requirement

The HTML file embeds all JavaScript (Plotly, vis-network) inline; no external
CDN requests are made.  Reports can be opened in an air-gapped environment.
File size is < 5 MB for a standard report (≤ 100 trades, ≤ 50 graph nodes).

### Interactive features

| Feature | Interaction |
|---|---|
| SHAP waterfall chart | Hover for exact contribution values; click a bar to expand contributing trades in the drill-down panel below. |
| Wallet graph | Zoom / pan / drag nodes; click a node to see its risk score and feature breakdown. |
| Wallet address reveal | Double-click the wallet hash cell; enter your operator key to reveal the decrypted address (AES-GCM in production). |

### Provenance drill-down

Each SHAP feature bar is linked to the trades that contributed to that feature
value.  Clicking a bar populates the "Provenance Drill-Down" section with a
table of relevant trades, each showing its Ledger number, hashed counterparty
addresses, amounts, and asset pair.

### Security

Raw wallet addresses are **not** present in the HTML source.  Each address is
replaced with a JavaScript-decoded, operator-key-encrypted field.  The operator
must enter the key at view time via a browser prompt.  The decryption key is
never transmitted to any server.

### File size budget

| Component | Approximate size |
|---|---|
| Plotly JS bundle (minified) | ≤ 3.5 MB |
| vis-network (via pyvis) | ≤ 0.8 MB |
| Report data (100 trades, 50 nodes) | ≤ 0.2 MB |
| **Total** | **≤ 4.5 MB** |

If plotly is not installed, a plain HTML table fallback is rendered instead
(< 0.1 MB).
