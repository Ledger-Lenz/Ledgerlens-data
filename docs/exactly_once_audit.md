# Exactly-Once Audit

## Overview

`pipeline/exactly_once.py` gives each pipeline stage a two-phase
`STAGED → COMMITTED` dedup protocol (see
`docs/adr/0001-unified-idempotency-finality.md`), but each stage builds its
own `ExactlyOnceStore` from its own configuration. A misconfigured stage
raises nothing. It quietly degrades that boundary to at-least-once
(duplicate side effects) or at-most-once (dropped records).

`pipeline/exactly_once_audit.py` checks the whole pipeline end-to-end.
`audit_pipeline` traces a sample of synthetic records through every stage
boundary in pipeline order and verifies the exactly-once invariants at each
one. A violation at one stage does not stop the trace, so every broken
boundary is reported in a single run.

## Stage boundaries

`scripts/audit_exactly_once.py` builds each boundary from the same
configuration the stage reads:

| Boundary | Stage code | Backend | Dedup source | TTL setting |
|---|---|---|---|---|
| `ingestion` | `ingestion/trade_deduplicator.py` | Redis (`TRADE_DEDUP_CACHE_KEY_PREFIX`) | `horizon_trade:*` | `TRADE_DEDUP_TTL_SECONDS` |
| `feature_scoring` | `streaming/kafka_worker.py` | Redis (`ledgerlens:kafka_dedup:`) | `kafka_trade` | `KAFKA_DEDUP_TTL_SECONDS` |
| `alerting` | `streaming/alert_ledger.py` | SQL (`RISK_SCORE_DB_URL`) | `alert_delivery` | store default (24h) |

The Kafka worker updates features and scores a trade inside one dedup
boundary: a message is committed only after both have completed. The
feature and scoring stages therefore share one boundary.

## Invariants

For each sampled record, the audit replays the delivery sequence a real
record goes through at every boundary:

| Invariant | Check | Violation means |
|---|---|---|
| `backend_available` | The backend passes its health check and answers every call. | The stage cannot tell duplicates from new records. Fail-closed stages stop processing; anything that swallows the error is running without dedup. |
| `first_delivery_processed` | First delivery of a never-seen record returns `NEW`. | New records are treated as already seen and skipped. The boundary is **at-most-once**: data is silently dropped. |
| `inflight_redelivery_redone` | Redelivery before commit returns `STAGED`. | `NEW`: the staged claim is not durable, so a redelivery during processing runs the record twice at once. `COMMITTED`: an attempt that crashed mid-flight is treated as done and its record is dropped. |
| `committed_redelivery_skipped` | After commit, redelivery returns `COMMITTED`. | Committed records are not recognised as duplicates, so side effects (feature updates, scores, alerts) repeat on every redelivery. The boundary is **at-least-once**. |
| `ttl_covers_redelivery_window` | The stage's dedup TTL is at least `--min-ttl-seconds` (default 3600). | Keys expire while redeliveries can still arrive, and a late redelivery is processed again (at-least-once). |

Audit records use `external_id` values of the form
`__eo_audit__:<run_id>:<n>`, so they never collide with real traffic. Each is
released with `mark_failed` after it has been probed: deleted on Redis, left
in the `FAILED` state on SQL.

## Running it

```bash
python -m scripts.audit_exactly_once \
    --redis-url redis://staging-redis:6379/0 \
    --db-url postgresql://user:pass@staging-db/ledgerlens \
    --sample-size 5 --min-ttl-seconds 3600 \
    --output reports/exactly_once_audit.json
```

The JSON report lists every violation with its stage, invariant, record and
detail. The command exits `0` when every invariant holds and `1` otherwise.
`--redis-url` and `--db-url` default to `REDIS_URL` and `RISK_SCORE_DB_URL`.

## Scheduled job

`.github/workflows/exactly-once-audit.yml` runs the audit against staging
daily at 04:17 UTC, and on demand via *Run workflow*. It uses the `staging`
GitHub environment, which must define:

- secrets `STAGING_REDIS_URL` and `STAGING_RISK_SCORE_DB_URL`
- optionally, variables `TRADE_DEDUP_TTL_SECONDS` and `KAFKA_DEDUP_TTL_SECONDS`
  when staging's values differ from the defaults

The job fails if these secrets are missing or if any violation is found. The
JSON report is uploaded as the `exactly-once-audit` artifact for 30 days.

## Responding to a violation

1. Download the `exactly-once-audit` artifact and note which stage and
   invariant failed.
2. `backend_available`: follow the `DedupBackendUnavailableError` section of
   `docs/dedup_idempotency_incident_runbook.md`.
3. `ttl_covers_redelivery_window`: raise the stage's TTL setting (table
   above) to cover the redelivery window, which must include consumer lag and
   replay windows.
4. Any other invariant: the stage's dedup store is not behaving as a
   two-phase store. Check the stage's backend and key-prefix configuration
   against the table above. Until it is fixed, treat that stage's output
   since the last passing audit as possibly duplicated (at-least-once) or
   incomplete (at-most-once), and reconcile it with
   `validation.reconciliation`.

## Validation

```
pytest tests/test_exactly_once_audit.py -v
```

Covers a correctly configured pipeline passing cleanly, and a fail-open
stage, a TTL shorter than the redelivery window, and an unreachable backend
each being flagged at the right stage, plus the CLI exit codes and report.
