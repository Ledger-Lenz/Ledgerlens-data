# LedgerLens On-Chain Governance

## Overview

`RISK_SCORE_FLAG_THRESHOLD` — the boundary between "clean" and "suspicious"
verdicts — is protected by a **2-of-3 multi-signature governance contract**
(`integrations/governance_contract.rs`) deployed on Stellar as a Soroban
smart contract.  A single compromised key cannot change the threshold; at
least two of the three registered keyholders must sign independent approval
transactions.

## Keyholders

| Role | Custody |
|---|---|
| Keyholder A | Security lead (HSM-backed) |
| Keyholder B | Engineering lead (hardware wallet) |
| Keyholder C | Compliance officer (hardware wallet) |

Private keys **never touch the Python client**.  Keyholders sign Stellar
transactions on their HSM or hardware wallet and pass the resulting signed
XDR to `LedgerLensContractClient.propose_threshold_change` /
`approve_threshold_change`.

## Changing the Threshold

1. **Propose** — any one keyholder calls `propose_threshold_change(new_threshold)`.
   This opens a proposal and records the proposer's approval automatically.
   The proposal expires after **7 days** if quorum is not reached.

2. **Approve** — a second keyholder independently reviews the proposal and
   calls `approve_threshold_change(proposal_id)`.  On the second approval the
   contract applies the new threshold and emits a `threshold_changed` event.

3. **Local update** — `integrations/soroban_event_listener.py` receives the
   `threshold_changed` event and updates `config.RISK_SCORE_FLAG_THRESHOLD`
   at runtime without requiring a service restart.

```python
from integrations.contract_client import LedgerLensContractClient

client = LedgerLensContractClient(...)

# Keyholder A proposes
proposal_id = client.propose_threshold_change(
    governance_contract_id="C...",
    new_threshold=75,
    proposer_secret="S...",   # signed by HSM/hardware wallet
)

# Keyholder B approves — threshold applied when this returns True
applied = client.approve_threshold_change(
    governance_contract_id="C...",
    proposal_id=proposal_id,
    approver_secret="S...",
)
```

## Keyholder Rotation

To rotate a keyholder:

1. Generate a new keypair on the replacement's HSM.
2. All **three** current keyholders approve a special `rotate_keyholder`
   governance proposal (requires M=3 for this operation — unanimity).
3. The contract replaces the old address with the new one in its keyholder list.
4. Revoke/destroy the old private key.
5. Update this document with the new custodian name.

## Proposal Expiry

A proposal that does not reach 2 approvals within 7 days is automatically
expired (the contract checks `env.ledger().sequence() > expiry_ledger` on
every `approve_threshold_change` call and panics with `"proposal expired"`).
Expired proposals cannot be reactivated; a new proposal must be submitted.

## Emergency Single-Key Override

**Risk: use only as a last resort.**

If two keyholders are simultaneously unavailable (e.g. during a coordinated
incident) and the threshold urgently needs changing, an operator may:

1. Deploy a temporary replacement contract that accepts a single signature.
2. Apply the change.
3. Rotate that contract back to the 2-of-3 scheme within 24 hours.

This override bypasses the governance scheme's security guarantees.  It must
be documented in the incident log, approved by the CISO, and reviewed in the
post-mortem.

## Time-Sensitive Threats

If a known attack is underway and waiting for M-of-N signatures would allow
it to proceed, the **emergency pause mechanism** (see `docs/incident_response.md`)
should be used to halt scoring immediately.  Threshold changes can then be
made once the pipeline is paused and the threat is contained.
