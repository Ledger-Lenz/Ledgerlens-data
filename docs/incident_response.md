# LedgerLens Incident Response — Emergency Pause Runbook

## Purpose

This runbook covers the procedure for activating the **emergency pause**
on the `ledgerlens-score` Soroban oracle contract when a critical defect in
the scoring pipeline is detected (e.g. scores are wildly incorrect, the
pipeline has been compromised, or downstream applications are being harmed
by bad data).

## When to Use the Emergency Pause

Activate the pause when **all** of the following are true:

- The scoring pipeline is producing incorrect or harmful scores.
- The defect cannot be fixed within the current Stellar ledger cycle.
- Downstream applications are acting on the corrupted scores.

Do **not** use the emergency pause as a routine maintenance tool.

## Emergency Keyholders

| Role | Custody |
|---|---|
| Emergency Key 1 | Security lead (HSM) |
| Emergency Key 2 | On-call engineer (hardware wallet) |
| Emergency Key 3 | CISO delegate (hardware wallet) |

Two of the three keyholders must independently sign approval transactions for
the pause to take effect.

## Pause Procedure

### Step 1 — Detect the anomaly

The `EmergencyWatchdog` (`monitoring/emergency_watchdog.py`) automatically
proposes a pause when > 90% of scores in a 60-second window exceed 95.  If
the watchdog is offline or the anomaly does not meet that threshold, a
keyholder can manually initiate.

### Step 2 — Initiate the pause (Keyholder 1)

```python
from integrations.contract_client import LedgerLensContractClient

client = LedgerLensContractClient(
    contract_id="<score_contract_id>",
    rpc_url="https://soroban-testnet.stellar.org",
)
proposal_id = client.initiate_emergency_pause(
    pause_contract_id="<pause_contract_id>",
    reason="Scoring pipeline producing anomalous scores — see incident #XYZ",
    signing_key="<keyholder_1_secret>",   # sign on HSM/hardware wallet
)
print(f"Pause proposed: proposal_id={proposal_id}")
```

Share `proposal_id` with Keyholder 2 via the secure channel.

### Step 3 — Approve the pause (Keyholder 2)

```python
applied = client.approve_emergency_pause(
    pause_contract_id="<pause_contract_id>",
    proposal_id=proposal_id,
    signing_key="<keyholder_2_secret>",
)
print(f"Paused: {applied}")
```

### Step 4 — Verify the pause

The Soroban event listener (`integrations/soroban_event_listener.py`) receives
the `contract_paused` event and halts the local scoring pipeline within two
Stellar ledger closes (10–15 seconds).  Verify in the monitoring dashboard
that score submissions have stopped.

### Step 5 — Investigate and fix

- Root-cause the defect.
- Deploy a fix in a staging environment.
- Obtain sign-off from the security lead.

### Step 6 — Unpause

After the fix is deployed and verified:

```python
client._client.invoke(
    "unpause",
    [scval.to_address(keyholder_1_pubkey), scval.to_uint64(proposal_id)],
    source=keyholder_1_pubkey,
    signer=keyholder_1_signer,
).sign_and_submit()
```

Verify that the event listener resumes the pipeline.

## Proposal Expiry

Pause proposals expire after **15 minutes** (≈180 ledger closes).  A stale
proposal (e.g. one submitted but not yet approved when the incident was
resolved) cannot be used to re-pause the contract after expiry.

## DoS Prevention

A compromised keyholder cannot unilaterally pause the contract — they can
only propose, not approve.  To prevent an attacker from spamming proposals
and cluttering the contract storage, proposals are keyed by a monotonically
incrementing ID and the contract rejects approval of an expired proposal.

## Key Custody

Emergency keyholder secrets are stored in an HSM or hardware wallet.  They
are **never** written to disk, environment variables, or CI secrets.  The
`signing_key` parameter to `initiate_emergency_pause` and
`approve_emergency_pause` is expected to be passed by the keyholder operator
at the time of the incident, not pre-loaded.

## Post-Incident Checklist

- [ ] Document root cause in incident log.
- [ ] Confirm fix deployed and verified in staging.
- [ ] Unpause and confirm pipeline resumes.
- [ ] Run full test suite post-unpause.
- [ ] Schedule post-mortem within 48 hours.
- [ ] Update this runbook if procedures changed.
