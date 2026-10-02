# Governance Audit Process

**Related issue:** #950  
**Related modules:** `integrations/governance_audit_listener.py`, `detection/governance_audit_trail.py`

---

## Purpose

Every change executed by the LedgerLens on-chain governance contract
(`integrations/governance_contract.rs`) must be accompanied by an off-chain
written justification.  This requirement exists because:

1. **Auditability.** Regulators, auditors, and community members must be able
   to understand *why* a scoring threshold or other risk parameter was changed,
   not just *that* it was changed.
2. **Accountability.** A written justification signed by a responsible party
   creates a non-repudiable record linking the on-chain action to a human
   decision.
3. **Incident reconstruction.** If a governance change later turns out to be
   incorrect or to have been made under duress, the justification provides the
   starting point for an investigation.

---

## How it works

```
On-chain governance event
        │ (Soroban RPC getEvents)
        ▼
GovernanceAuditListener.on_governance_event()
        │
        ▼
AuditTrail.record_governance_event()
        │ (stores event_id, event_type, payload, timestamp)
        │ (justification = None)
        ▼
── grace period timer starts (default: 5 minutes) ──────────────────────────
        │
        │ Within grace period:
        ▼
Change author calls:
GovernanceAuditListener.submit_justification(event_id, justification, submitter)
        │
        ▼
AuditTrail.add_justification()
        │ (links justification, submitter, correlated_at timestamp)
        │ (entry now correlated)
        ▼
── run_once() / check_missing_justifications() ──────────────────────────────
        │
        │ If correlated before grace period expired:
        │   No alert. Audit trail is complete.
        │
        │ If NOT correlated after grace period:
        ▼
alert_fn(mismatch_details) → PagerDuty / Slack / email
```

---

## Who is responsible

| Role | Responsibility |
|---|---|
| **Governance keyholder** who proposes a change | Must submit a justification within the grace period after their proposal is approved on-chain. |
| **Governance keyholder** who approves a change | Should co-sign or annotate the justification if they had additional context informing their vote. |
| **On-call engineer** | Responds to missing-justification alerts during the grace period and follows up with the proposer. |
| **Compliance lead** | Reviews the audit trail periodically and flags entries with insufficient justifications. |

---

## How to submit a justification

### Via the CLI

```bash
python -m scripts.submit_governance_justification \
  --event-id "gov-20260928-001" \
  --submitter "alice@example.com" \
  --justification "Increased RISK_SCORE_FLAG_THRESHOLD from 70 to 75.
    Rationale: Q3 2026 precision/recall analysis (reports/q3_2026_threshold_review.pdf)
    shows that threshold=70 produces an 8% false positive rate on the testnet
    backtest set.  Raising to 75 reduces FPR to 3.5% with negligible AUC impact
    (ΔAUC < 0.01).  Approved by security review on 2026-09-28."
```

### Via the Python API

```python
from integrations.governance_audit_listener import GovernanceAuditListener
from detection.governance_audit_trail import AuditTrail

store = AuditTrail(db_url="sqlite:///governance_audit.db")
listener = GovernanceAuditListener(audit_store=store, alert_fn=my_alert_fn)

listener.submit_justification(
    event_id="gov-20260928-001",
    justification="...",
    submitter="alice@example.com",
)
```

---

## What makes a good justification

A justification should answer:

1. **What changed?**  Name the parameter and its old and new values.
2. **Why was the change necessary?**  Reference supporting data (backtest
   results, precision/recall curves, incident reports, regulatory guidance).
3. **Who authorised it?**  Name the keyholder(s) and the internal approval
   process they followed (e.g., security review, governance vote).
4. **What is the expected impact?**  State the expected effect on FPR, FNR,
   alert volume, or other relevant metrics.
5. **When is it expected to be reviewed?**  Governance changes should have
   a review date to ensure they remain appropriate.

**Minimum acceptable length:** ~50 words.  Terse entries like "changed for
performance" will be flagged during compliance review.

---

## Grace period

The default grace period is **300 seconds (5 minutes)** after the on-chain
event is observed.  This is intentionally short to encourage timely
documentation.

The grace period can be adjusted via the ``GOVERNANCE_JUSTIFICATION_GRACE_PERIOD``
environment variable or by passing ``grace_period_seconds=N`` to
``GovernanceAuditListener``.

**Note:** The grace period starts when the event is *observed by the listener*,
not when it is submitted to the chain.  Network delays between chain finality
and event observation are bounded by the Soroban event polling interval.

---

## Querying the audit trail

### List all governance events

```bash
python -m scripts.list_governance_audit_entries
```

### List uncorrelated (un-justified) events

```bash
python -m scripts.list_governance_audit_entries --uncorrelated
```

### Python API

```python
from detection.governance_audit_trail import AuditTrail

store = AuditTrail(db_url="sqlite:///governance_audit.db")

# All entries (newest first)
entries = store.list_entries(limit=50)

# Un-justified entries past the grace period
overdue = store.get_uncorrelated_events(older_than_seconds=300)
```

---

## Alert response

If a missing-justification alert fires, the on-call engineer should:

1. **Identify the change author** — look up the governance contract event via
   Stellar Explorer using the `event_id`.
2. **Contact the proposer** — reach out immediately to request a justification.
3. **Acknowledge the alert** in the alerting system with a note: _"Contacted
   <name>, justification expected within 1 hour."_
4. **Escalate** if no justification is received within 1 hour:
   - Notify the compliance lead.
   - Consider whether the governance change should be rolled back pending
     documentation.

If the alert fires due to a pipeline issue (e.g., the listener was down during
the event), the engineer should:

1. Replay governance events for the affected time window.
2. Contact the change author for a retrospective justification.
3. Note in the justification that it was submitted retrospectively due to a
   monitoring gap.

---

## Compliance review

The compliance lead should review the governance audit trail at least **monthly**
to verify:

- All events have been correlated with justifications.
- Justification quality meets the minimum standard above.
- No governance changes were made outside the documented keyholder set.

Compliance reports can be generated with:

```bash
python -m scripts.generate_governance_compliance_report \
  --since 2026-09-01 \
  --output reports/governance_compliance_2026_09.json
```

---

## Related documentation

- `integrations/governance_contract.rs` — on-chain governance contract
- `integrations/contract_client.py` — `propose_threshold_change` / `approve_threshold_change`
- `docs/governance.md` — governance model overview
- `docs/runbooks/pause_monitor_mismatch.md` — emergency pause runbook
- Issue #238 (multi-sig governance), #950 (audit log correlation)
