# Post-Recovery Consistency Verification

Issue #920. Implementation: `pipeline/recovery.py`
(`RecoveryManager.complete_recovery`, `verify_recovery`, `RecoveryReport`).

## Flow

1. The recovery run replays its stages through `RecoveryManager.stage()`.
2. Recovery then calls `complete_recovery(run_id, pair_id, expected, actual,
   recovered_range)`:
   - `expected` holds a `StageSnapshot` (record count plus checksum) per stage,
     taken from the source of truth for the recovered range.
   - `actual` holds the same snapshot taken from the recovered output.
3. Every stage that appears in either mapping is compared. A stage fails if:
   - its record counts differ,
   - its checksums differ (`compute_checksum` does not depend on row order), or
   - it appears on only one side.
4. **PASS:** normal processing resumes automatically.
   **FAIL:** every later `stage()` call for that `(run_id, pair_id)` raises
   `RecoveryBlockedError` until someone does a manual review. Other pairs are
   not affected.

## Report format

`RecoveryReport.render()` output, which is also logged at ERROR level when a
check fails:

```
=== LedgerLens post-recovery consistency report ===
Run ID:           <run_id>
Pair:             <pair_id>
Generated at:     <ISO-8601 UTC timestamp>
Recovered range:  <start> → <end>
Recovered stages: ingest, features
Result:           FAIL

Stage checks:
  [PASS] ingest: expected=1200 actual=1200
  [FAIL] features: expected=340 actual=338
         - record count mismatch: expected 340, actual 338

Automatic resumption: BLOCKED — manual review required. ...
```

| Field | Meaning |
|-------|---------|
| Recovered range | Time or ledger range that recovery covered, as passed by the caller. |
| Recovered stages | Stages that re-ran and completed during this recovery. |
| Result | `PASS` only if every stage check passed. |
| Stage checks | One line per stage, followed by one line per issue found. |
| Final line | Whether automatic resumption is `ALLOWED` or `BLOCKED`. |

`RecoveryReport.to_dict()` returns the same content as JSON, for attaching to
tickets or feeding log pipelines.

## On-call actions for a FAIL report

1. Find the failed stage(s) and check whether the mismatch is in the record
   count or the checksum.
   - A count mismatch usually means rows were dropped or duplicated during
     replay.
   - A checksum mismatch with equal counts usually means rows were changed
     (for example by a non-deterministic transform or a partial write).
2. Fix the cause, then either re-run recovery (a new `complete_recovery`
   call replaces the report), or accept the state with
   `RecoveryManager.approve_resume(run_id, pair_id, reviewer="<you>")`.
   Approvals are logged at WARNING level together with the reviewer.
