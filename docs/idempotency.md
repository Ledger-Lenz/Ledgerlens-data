# Idempotent Job Execution

## Overview

Pipeline jobs get re-invoked for reasons outside the pipeline's control:
at-least-once Kafka delivery, a cron/Celery retry, or an operator re-running
a script after a partial failure. Without an idempotency layer, replays
either duplicate side effects (double alerts, double writes to
`risk_score_store`) or — worse — silently return a cached result computed
for the *wrong* input.

`utils/idempotency.py` provides `IdempotencyLedger`, a small SQLite-backed
contract for exactly-once job completion:

- **Cached completion** — once a key's job succeeds, replay returns the
  cached result without re-running the body.
- **Key-reuse detection** — reusing a key with a different input payload
  raises `IdempotencyConflictError` instead of returning a stale result.
- **Lease-based concurrency control** — a job already `PENDING` is assumed
  to be executing elsewhere and a concurrent duplicate raises
  `ConcurrentExecutionError`, unless the lease has expired (the previous
  attempt crashed), in which case it's reclaimed automatically.

## Contract

- The `key` UNIQUE constraint in SQLite is the actual mutual-exclusion
  mechanism (`INSERT OR IGNORE`), not an application-level
  check-then-insert race.
- `input_payload` is hashed (SHA-256 over canonical JSON) and stored
  alongside the key; a mismatch on replay is treated as a caller bug and
  raises rather than silently proceeding.
- `lease_seconds` bounds how long a `PENDING` job blocks a duplicate before
  it's considered dead and reclaimed — tune this to comfortably exceed the
  job's expected runtime.

## Usage

```python
from utils.idempotency import IdempotencyLedger, idempotent

ledger = IdempotencyLedger("idempotency.db")

result = ledger.run(
    key=f"score_wallet:{wallet_id}:{ledger_close_time}",
    fn=lambda: score_wallet(wallet_id),
    input_payload={"wallet_id": wallet_id, "ledger_close_time": ledger_close_time},
)

# Or as a decorator:
@idempotent(ledger, key_fn=lambda wallet_id, **_: f"score_wallet:{wallet_id}")
def score_wallet(wallet_id: str) -> dict: ...
```

## Validation

```
pytest tests/test_idempotency.py -v
```

Covers: single execution with cached replay, key-reuse conflict detection,
rejection of concurrent in-flight duplicates, reclaiming an expired lease,
retrying a previously-failed job, `reset()`, the decorator form, and
independent keys not interfering with each other.

## Pipeline idempotency-key TTL and eviction

`pipeline/idempotency.py::CheckpointStore` keeps one idempotency key per
`(run_id, pair_id, stage)` so that re-running the batch pipeline with the same
inputs skips stages that already completed. Without a bound, a long-running
deployment would accumulate one key per stage per run for as long as it runs.

### Policy

- `IDEMPOTENCY_TTL_HOURS` (default `48`) sets how long a key is honoured. A key
  older than the TTL no longer blocks reprocessing, so the stage re-runs.
- Expired keys are evicted from the store. `CheckpointStore.evict_expired()`
  deletes every entry last updated before `now - IDEMPOTENCY_TTL_HOURS`,
  whether it is completed, failed, or a `running` stage that never finished.
  It runs when a store is opened and then at most once an hour as stages
  start, so no separate cron job is needed.
- The store is therefore bounded to about one TTL window of keys:
  `pairs × stages × runs per TTL window`. For example, 50 pairs × 7 stages ×
  hourly runs × 48 h is about 17k rows.

### Recommended TTL: 48 hours

The TTL has to cover the longest realistic gap between a run and its
re-invocation with the same inputs, because that is the window in which a
replay must be skipped:

- At-least-once redelivery and cron or worker retries re-invoke a job within
  minutes to a few hours.
- The longest routine gap is an operator re-running a failed overnight job
  the next working day, up to about 24 hours later.

48 hours is twice that daily cycle, so a job that failed and was re-run the
next day is still covered even if the re-run itself slips. Going longer buys
little. Past the TTL, a replay only recomputes stages: risk scores are
upserted by `(wallet, asset_pair)` in `RiskScoreStore`, and
`idempotent_upsert` also skips unchanged scores, so an expired key costs
compute and never creates duplicate rows. Raise the TTL only if re-runs routinely happen more than a day
after the original run, for example when failures over a weekend are only
retried on Monday. In that case use 72 to 96 hours and expect the store to
grow in proportion.

### Monitoring

The store exports `ledgerlens_idempotency_store_entries{status}` (gauge) and
`ledgerlens_idempotency_store_evicted_total` (counter), shown on the
"LedgerLens — Idempotency Key Store" Grafana dashboard
(`monitoring/grafana/dashboards/idempotency_store.json`). Once a full TTL
window of keys is stored, size should plateau and the growth rate should
hover around zero. Sustained growth means eviction is not running.

### Validation

```
pytest tests/test_idempotency_ttl.py -v
```

Covers eviction of keys older than the TTL (they no longer block
reprocessing), retention of keys inside the TTL, eviction of stale `running`
and `failed` entries, eviction on open and on stage start, and the exported
metrics.

## Design tradeoffs / follow-ups

- SQLite (WAL mode) was chosen over a JSON file (as used in
  [[checkpointing]]) specifically because the UNIQUE constraint gives real
  atomic claim semantics across processes, which a JSON store cannot
  provide without an external lock.
- Lease expiry is polled at call time (no background reaper); a key whose
  owner crashed and is never retried stays `PENDING` until something calls
  `run` or `reset` again. Acceptable for batch/worker-triggered jobs; a
  background sweeper could be added for long-idle queues.
- Results must be JSON-serialisable. Non-serialisable results should be
  persisted by the caller (e.g. to `risk_score_store`) with only a
  reference/ID passed through the ledger.
