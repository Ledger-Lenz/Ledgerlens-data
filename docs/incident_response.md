# LedgerLens Incident Response Runbook

This document is the authoritative runbook for LedgerLens security incidents.
Automated steps are executed by `monitoring/incident_responder.py`; manual
escalation steps follow.

---

## Automated Steps (IncidentResponder)

The `IncidentResponder` class subscribes to the alert dispatcher and, for any
alert that meets the **high-severity threshold**, automatically executes the
playbook defined in `data/playbooks/high_risk_wallet.yaml`.

### High-severity thresholds

| Condition | Threshold |
|---|---|
| Risk score | > 90 |
| Benford MAD | > 0.05 |
| Alert type | `high_risk_wallet` or `emergency_drift` |

### Step 1 — Snapshot risk score history

`IncidentResponder` calls `RiskScoreStore.get_history(wallet, days=30)` and
stores the result in the incident record.  This preserves the pre-investigation
state of the wallet's score curve in case subsequent trades alter it.

### Step 2 — Generate preliminary forensic report

`ForensicReportGenerator.generate()` is called with cached features.  The
summary (report ID, risk score, verdict, top 5 SHAP features) is stored in the
incident record.  The full report is available in `reports/forensic/`.

**SLA**: must complete within 60 seconds of the alert firing.

### Step 3 — Create DB incident record

An `IncidentRecord` is persisted to the incident store with the following
fields: `incident_id`, `wallet_hash` (SHA-256 of the raw address), `alert_fingerprint`,
`alert_type`, `risk_score`, `created_at`, `status`.

**Idempotency**: re-triggering the same `(wallet_hash, alert_fingerprint)` pair
within the deduplication window (default 3600 s) is a no-op; no second record
is created.

### Step 4 — Send webhook notification

A JSON payload is POSTed to the URL in `INCIDENT_WEBHOOK_URL` (env var).
The raw wallet address is **never** included; only the SHA-256 hash appears.

```json
{
  "incident_id": "<uuid>",
  "wallet_hash": "<sha256[:16]>",
  "alert_type": "high_risk_wallet",
  "risk_score": 95,
  "created_at": "2026-06-29T10:00:00+00:00",
  "status": "open",
  "report_summary": {
    "report_id": "<uuid>",
    "risk_score": 95,
    "verdict": "wash_trade",
    "top_features": ["benford_mad_24h", "counterparty_concentration_ratio"]
  }
}
```

Supported webhook targets: Slack incoming webhooks, PagerDuty Events API v2,
or any generic HTTPS endpoint.

---

## Simulating the Playbook

To verify the playbook or generate a test incident without a live alert:

```bash
python -m scripts.generate_reports --simulate --wallet GABC1234... \
    --output-dir reports/forensic --output-format json
```

The `--simulate` flag runs identical code to a live alert execution (with the
same mocked data sources), so the output can be used to regression-test the
report format.

---

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `INCIDENT_WEBHOOK_URL` | HTTPS URL to POST notifications to | (none — notifications skipped) |
| Playbook: `deduplication.window_seconds` | Dedup window | 3600 s |
| Playbook: `severity_triggers.risk_score_threshold` | Minimum score for high-severity | 90 |

---

## Manual Escalation Steps

After the automated steps complete, an analyst must:

1. **Acknowledge the incident** — update `status` to `investigating` in the
   incident store or your ticketing system.
2. **Review the full forensic report** — open the JSON report in
   `reports/forensic/` and cross-reference the SHAP features with on-chain
   evidence via Horizon URLs in `trade_evidence`.
3. **Check for related wallets** — use `scripts/score_wallet.py --propagate`
   to identify wallets with elevated risk propagation scores connected to this
   wallet.
4. **Escalate to compliance** — if `verdict == "wash_trade"` and risk score
   > 95, file a SAR or exchange notification per your jurisdiction's regulatory
   requirements.
5. **Close or re-open the incident** — update `status` to `closed` or
   `escalated` and attach any external ticket reference.

---

## Security considerations

- The webhook URL is treated as a secret.  Set it via the `INCIDENT_WEBHOOK_URL`
  environment variable only.  Never log it or include it in error messages.
- Webhook payloads contain hashed wallet identifiers, not raw Stellar account IDs.
- Incident records store only the SHA-256 hash of the wallet address.
