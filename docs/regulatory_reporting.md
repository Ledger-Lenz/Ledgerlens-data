# Regulatory Reporting: FATF Travel Rule / IVMS101 Export

## Overview

Regulatory bodies and partner VASPs increasingly require wallet risk data in standardised formats.  The FATF Travel Rule (Recommendation 16) mandates that virtual asset service providers share originator and beneficiary information on transfers above prescribed thresholds.

`reporting/fatf_exporter.py` maps LedgerLens detection output to the **IVMS101 1.0** (Inter-VASP Messaging Standard) data standard, producing JSON-LD documents that exchange compliance teams can submit directly to regulators and partner VASPs without manual reformatting.

```python
from reporting.fatf_exporter import export_ivms101, export_batch_ivms101

# Single report
doc = export_ivms101(forensic_report.to_dict())

# Batch — filters below FATF_EXPORT_THRESHOLD automatically
docs = export_batch_ivms101([r.to_dict() for r in high_risk_reports])
```

---

## IVMS101 Field Mapping

The table below documents how LedgerLens forensic report fields map to IVMS101 data elements.  Fields with no detection equivalent are populated with a **data-unavailable sentinel** (see below) rather than being omitted, ensuring downstream parsers receive structurally complete documents.

| IVMS101 Field | LedgerLens Source | Notes |
|---|---|---|
| `payloadMetadata.reportId` | `report_id` | UUID v4 from the forensic report |
| `payloadMetadata.generatedAt` | `generated_at` | ISO 8601 UTC timestamp |
| `payloadMetadata.reportingEntity` | Constant `"LedgerLens"` | |
| `payloadMetadata.schemaVersion` | Constant `"1.0"` | |
| `payloadMetadata.sourceReportSha256` | `report_sha256` | Tamper-evidence chain |
| `originator.accountNumber[0]` | `wallet` | Masked or revealed (see address policy) |
| `originator.originatorPersons[].naturalPerson.customerIdentification` | `wallet` | Same masking policy |
| `originator.originatorPersons[].naturalPerson.name` | — | Data unavailable |
| `originator.originatorPersons[].naturalPerson.geographicAddress` | — | Data unavailable |
| `originator.originatorPersons[].naturalPerson.nationalIdentification` | — | Data unavailable |
| `originator.originatorPersons[].naturalPerson.dateAndPlaceOfBirth` | — | Data unavailable |
| `originator.originatorPersons[].naturalPerson.countryOfResidence` | — | Data unavailable |
| `beneficiary` | — | All sub-fields data unavailable (see note) |
| `beneficiaryVASP` | — | Data unavailable |
| `originatingVASP.legalPerson.name` | Constant `"LedgerLens"` | |
| `riskIndicators[].code` | `verdict` + SHAP features | FATF VA-xxx codes |
| `riskIndicators[].severity` | Derived from code taxonomy | |
| `transactionReference.reportId` | `report_id` | |
| `transactionReference.assetPair` | `asset_pair` | Unavailable sentinel if empty |
| `transactionReference.riskScore` | `risk_score / 100` | Normalised to [0, 1] |
| `transactionReference.verdict` | `verdict` | "clean" / "suspicious" / "wash_trade" |
| `transactionReference.scoreConfidenceInterval.lower` | `score_lower / 100` | |
| `transactionReference.scoreConfidenceInterval.upper` | `score_upper / 100` | |

> **Note on beneficiary**: LedgerLens detection operates on flagged originator wallets.  In wash-trading ring scenarios, many counterparties may be involved; there is no single designated "beneficiary" in the Travel Rule sense.  All beneficiary sub-fields use the data-unavailable sentinel.

---

## Data-Unavailable Sentinel

Fields that have no corresponding value in the detection output are not omitted; instead they are populated with:

```json
{
  "value": null,
  "dataUnavailable": true,
  "fieldName": "<ivms101-field-name>"
}
```

This ensures:
- Downstream parsers always receive structurally complete documents.
- Auditors can distinguish "field not collected" from "field intentionally absent".
- The IVMS101 JSON Schema validates the document without `required` violations.

The `fieldName` annotation is included for machine-readable disambiguation and may be omitted in minimal sentinel values (e.g. for `beneficiaryVASP` at the top level).

---

## Risk Code Taxonomy

FATF risk indicator codes (`VA-xxx`) are defined in `reporting/fatf_risk_codes.py` and derived from the FATF Guidance for a Risk-Based Approach to Virtual Assets (October 2021).

### Code Registry

| Code | Severity | Description | FATF Reference |
|------|----------|-------------|----------------|
| VA-001 | HIGH | Structuring: amounts split to avoid reporting thresholds | §5.2 Red Flag A1 |
| VA-002 | CRITICAL | Wash trading: artificial volume between related wallets | §6.1 Red Flag C3 |
| VA-003 | HIGH | Layering: funds routed through intermediate hops | §5.4 Red Flag B2 |
| VA-004 | MEDIUM | Statistical anomaly: Benford's Law deviation | §5.3 Red Flag A3 |
| VA-005 | HIGH | Round-trip cycling: assets returned to originator | §6.2 Red Flag C1 |
| VA-006 | MEDIUM | Counterparty concentration: dominant single partner | §5.5 Red Flag A5 |
| VA-007 | HIGH | Network cluster: co-located with flagged entities | §7.1 Red Flag D2 |
| VA-008 | MEDIUM | Velocity anomaly: spike in frequency or volume | §5.1 Red Flag A2 |
| VA-009 | CRITICAL | Self-matching: coordinated orders via shared funding | §6.3 Red Flag C4 |

### Mapping Logic

Codes are assigned by `map_to_risk_codes(report: dict) -> list[RiskCode]`:

1. **Verdict-based primary codes**:
   - `"wash_trade"` → VA-002, VA-009
   - `"suspicious"` → VA-001

2. **SHAP-feature supplementary codes** (only for features with positive contribution):
   - `benford_mad_*` → VA-004
   - `round_trip_frequency` → VA-005
   - `counterparty_concentration_ratio` → VA-006
   - `self_matching_rate` → VA-009
   - `velocity*` → VA-008
   - `cross_pair*` → VA-003

3. **Deduplication**: each code appears at most once per export.

4. **Sort order**: highest severity first (CRITICAL > HIGH > MEDIUM > LOW).

---

## Wallet Address Policy

Raw on-chain wallet addresses are **pseudonymised by default** to prevent unnecessary exposure in transit documents.

The masking function:
```
masked = "REDACTED-" + sha256(address)[:12]
```

Properties:
- The original address is not recoverable from the masked token.
- The same wallet always produces the same token, enabling correlation across reports from the same reporting period without exposing the raw address.

### Revealing Addresses

Raw addresses can be included by setting `reveal_addresses=True` **and** the `FATF_ADMIN_TOKEN` environment variable:

```python
# Only succeeds when FATF_ADMIN_TOKEN is set
doc = export_ivms101(report, reveal_addresses=True)
```

```bash
# CLI usage (example integration)
FATF_ADMIN_TOKEN=<token> python -m scripts.export_fatf --reveal-addresses
```

If `reveal_addresses=True` is passed without `FATF_ADMIN_TOKEN`, a `PermissionError` is raised.

---

## Export Threshold

Only reports with a calibrated risk score at or above the configured threshold are included in batch exports:

```
risk_score / 100 >= FATF_EXPORT_THRESHOLD
```

| Setting | Default | Environment variable |
|---------|---------|---------------------|
| `FATF_EXPORT_THRESHOLD` | `0.85` | `FATF_EXPORT_THRESHOLD` |

Reports below the threshold are silently excluded from `export_batch_ivms101`.

---

## Schema Validation

Every export is validated against the bundled JSON Schema at `reporting/schemas/ivms101.json` before being returned.  The schema enforces:

- Required top-level fields: `@context`, `@type`, `payloadMetadata`, `originator`, `beneficiary`, `originatingVASP`, `beneficiaryVASP`, `riskIndicators`, `transactionReference`
- Risk indicator codes must match `^VA-\d{3}$`
- Risk indicator severity must be one of `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- `transactionReference.reportId` must be a non-empty string
- `payloadMetadata` required fields must all be present and non-empty strings
- Optional fields may be a typed value OR a data-unavailable sentinel object

A schema violation raises `ExportValidationError` with the failing path and message.

---

## Compliance Use Case

The typical integration workflow for exchange compliance teams:

```
1. LedgerLens scores wallets in real time
2. Flagged reports (risk_score >= 85) are collected
3. export_batch_ivms101(reports) produces IVMS101 JSON-LD documents
4. Documents are forwarded to:
   a. Regulators via VASP-to-regulator Travel Rule channel
   b. Partner VASPs via VASP-to-VASP messaging (e.g. TRP, OpenVASP, Sygna Bridge)
   c. Internal compliance database for SAR/STR filing support
```

The `sourceReportSha256` field in `payloadMetadata` chains the IVMS101 document back to the original tamper-evident LedgerLens forensic report, providing an auditable link between the regulatory submission and the underlying detection evidence.
