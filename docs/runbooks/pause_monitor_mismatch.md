# Runbook: Pause Monitor Mismatch Alert

**Severity:** CRITICAL  
**Alert source:** `integrations/pause_monitor.py` — `PauseMonitor`  
**Issue:** #949  

---

## What this alert means

A **PauseMonitor mismatch** fires when the off-chain scoring pipeline's
operational state disagrees with the on-chain emergency-pause contract state.
This means one of two things:

| Scenario | `onchain_paused` | `pipeline_paused` | Risk |
|---|---|---|---|
| **A** — On-chain paused, pipeline still running | `true` | `false` | **CRITICAL** — Pipeline is actively producing and submitting scores while the chain says it should be halted. Scores submitted during a pause may be fraudulent or based on stale data. |
| **B** — On-chain running, pipeline is halted | `false` | `true` | **HIGH** — Legitimate trades are not being scored. Detection gap opens. |

---

## Immediate response (< 5 minutes)

### Scenario A — On-chain paused, pipeline still running

1. **Verify the alert is genuine.**  Log into the Soroban RPC endpoint or
   Stellar Explorer and confirm `is_paused()` returns `true` for the pause
   contract.

2. **Immediately halt the scoring pipeline.**
   ```bash
   # If running as a systemd service:
   sudo systemctl stop ledgerlens-pipeline

   # If running in Docker:
   docker stop ledgerlens-pipeline

   # If running in Kubernetes:
   kubectl scale deployment ledgerlens-pipeline --replicas=0 -n ledgerlens
   ```

3. **Stop any pending on-chain score submissions.**  Ensure no `submit_score`
   transactions are in flight.  Check the Horizon account for any pending
   transactions from `LEDGERLENS_SUBMITTER_SECRET`.

4. **Notify the incident channel.**  Post to `#ledgerlens-incidents` with:
   - Time alert fired
   - On-chain pause proposal ID (from the pause contract event)
   - Whether the pipeline was successfully halted

5. **Investigate why the event listener missed the pause event.** Check:
   ```bash
   grep "c_paused\|emergency pause" /var/log/ledgerlens/pipeline.log | tail -50
   grep "SorobanEventListener\|PauseMonitor" /var/log/ledgerlens/pipeline.log | tail -50
   ```
   Common causes:
   - Soroban RPC network error during event polling
   - `SorobanEventListener` polling interval longer than the pause-to-detection window
   - `EVENT_HMAC_SECRET` mismatch causing event parsing to fail silently

---

### Scenario B — On-chain running, pipeline halted

1. **Verify the alert is genuine.**  Confirm `is_paused()` returns `false` for
   the pause contract on-chain.

2. **Check whether the pipeline was intentionally paused** (e.g., maintenance
   window, operator halt):
   ```bash
   grep "halted\|HALT\|pipeline.*paused\|stop" /var/log/ledgerlens/pipeline.log | tail -20
   ```

3. **If intentional:** The monitoring team should have pre-acknowledged this
   state.  If no acknowledgement exists, treat it as an incident and proceed.

4. **If unintentional:** Restart the scoring pipeline.
   ```bash
   sudo systemctl start ledgerlens-pipeline
   # or
   kubectl scale deployment ledgerlens-pipeline --replicas=1 -n ledgerlens
   ```

5. **Verify pipeline comes up cleanly** and begins scoring:
   ```bash
   kubectl logs -f deployment/ledgerlens-pipeline -n ledgerlens | grep "scoring"
   ```

---

## Root cause investigation (< 30 minutes)

### Check the SorobanEventListener

The `SorobanEventListener` should have received a `c_paused` or `c_unpaused`
event from the pause contract.  If it didn't:

```bash
# Check the on-chain event watermark — has it advanced recently?
python -m scripts.inspect_watermark --contract "$PAUSE_CONTRACT_ID"

# Replay events from the last 200 ledgers
python -m scripts.replay_events --contract "$PAUSE_CONTRACT_ID" --last-ledgers 200
```

### Check the PauseMonitor logs

```bash
grep "PauseMonitor" /var/log/ledgerlens/pipeline.log | tail -100
```

Expected log lines when healthy:
- `PauseMonitor: states in parity (onchain_paused=False, pipeline_running=True)` — every `poll_interval_seconds`

If absent, the PauseMonitor loop may have crashed.  Check:
```bash
grep "PauseMonitor.*error\|PauseMonitor.*exception" /var/log/ledgerlens/pipeline.log
```

### Check Soroban RPC health

```bash
curl -s "$SOROBAN_RPC_URL" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getLatestLedger"}' | jq .
```

If the RPC is returning errors, the event listener will have been unable to
poll for pause events.  Check the Stellar network status page and your RPC
provider's status.

---

## Escalation

If the mismatch cannot be resolved within 30 minutes, escalate to the
**LedgerLens on-call engineer** via PagerDuty policy `ledgerlens-oncall`.

Include in the escalation:
- The exact `mismatch_details` dict from the alert
- Output of `scripts/inspect_watermark`
- Last 200 lines of pipeline log
- Stellar Explorer link to the pause contract

---

## Post-incident actions

1. **Write a post-mortem** within 48 hours documenting:
   - Root cause of the mismatch
   - Why the event listener did not catch it
   - Proposed improvements (e.g., reduce polling interval, add redundant
     event delivery)

2. **Update the `PAUSE_MONITOR_POLL_INTERVAL_SECONDS`** env var if the current
   30 s window is too coarse for the detected SLA.

3. **File a follow-up issue** if any scoring data was emitted during an
   invalid window and needs to be retracted.

---

## Reference

- `integrations/pause_monitor.py` — implementation
- `integrations/emergency_pause_contract.rs` — on-chain contract
- `integrations/contract_client.py` — `initiate_emergency_pause` / `approve_emergency_pause`
- `docs/governance.md` — governance and pause process overview
- Issue #949, #241
