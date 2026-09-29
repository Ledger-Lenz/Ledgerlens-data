# Ingestion

## Distributed rate limiting for Horizon REST calls

When multiple ingestion workers run in parallel, each making independent
Horizon REST calls, their combined request rate can exceed Horizon's
per-IP limit (100 req/s) and trigger 429 responses with dropped data.
`ingestion/rate_limiter.py` enforces one global cap shared across all
workers.

### Architecture

```
 worker 1 ─┐
 worker 2 ─┼─► TokenBucketLimiter.acquire() ─► Redis (shared token bucket) ─► Horizon
 worker N ─┘                                         │
                                         WATCH/MULTI optimistic transaction
                                         guarantees atomic token decrement
```

* `TokenBucketLimiter` stores `tokens` / `updated_at` in a single Redis hash
  and mutates it through a `WATCH`/`MULTI` optimistic transaction, so two
  workers can never decrement the same token.
* `HORIZON_MAX_RPS` (config.py, default 80, capped at 100) is the global
  budget; the bucket refills continuously at that rate up to its capacity.
* `ingestion/horizon_fetcher.fetch()` wraps a Horizon call: it acquires a
  token first, then on a 429 response backs off exponentially (1s, 2s, 4s,
  ... capped at 60s) with +/-20% jitter and retries up to
  `HORIZON_MAX_RETRIES` (default 5) times. Any other 4xx (e.g. 403) is
  raised immediately without retry.
* `ingestion/historical_loader.py`'s `_fetch_page` routes through
  `horizon_fetcher.fetch()` as the reference integration point for paginated
  REST calls.

### Redis dependency

Requires a reachable Redis instance via `REDIS_URL` (default
`redis://localhost:6379/0`).

### Degraded mode

If Redis is unreachable -- at startup or on any later call -- the limiter
logs a warning once and grants every request immediately rather than
blocking ingestion on a rate-limiter outage. This sacrifices the global cap
under Redis downtime in favor of not stalling data collection.

## Resumable chunked backfill

Multi-hour (or multi-day) historical backfills go through
`ingestion.historical_loader.backfill_trades`, which processes an asset
pair's trade history in fixed-size chunks and persists progress after each
one, so an interrupted run resumes from the last completed chunk instead of
starting over.

```python
from ingestion.historical_loader import backfill_trades, parquet_chunk_sink

backfill_trades(
    base_asset,
    counter_asset,
    checkpoint_path="checkpoints/usdc_xlm_backfill.json",
    sink=parquet_chunk_sink("data/backfill/usdc_xlm"),
    chunk_size=10_000,
)
```

* A chunk is `chunk_size` raw Horizon records. Its valid trades are handed to
  the sink, and only after the sink returns is the chunk recorded in the
  checkpoint together with the paging token of its last record. The
  checkpoint is a JSON file written atomically via `utils/checkpoint.py`
  (temp file + fsync + `os.replace`), so a hard kill never corrupts it.
* On restart, `backfill_trades` reads the last completed chunk and resumes
  paging from its cursor. At most the in-flight chunk is re-fetched.
* A sink must be idempotent per chunk index: a crash after the sink writes
  but before the checkpoint is saved re-delivers that chunk.
  `parquet_chunk_sink` writes `chunk-NNNNNN.parquet` and simply overwrites it
  on re-delivery, so there are no gaps or duplicate rows.
* A sink failure is recorded against the chunk (error and attempt count) and
  re-raised; the next run retries the same chunk.
* Re-running a finished backfill later resumes from the last cursor and only
  picks up trades that arrived since.
* The checkpoint is bound to the pair and `start_time`. Resuming with
  different values raises `CheckpointMismatchError`. Pass `fresh=True` to
  discard it and start over.

### Monitoring progress

Operators can inspect a running (or interrupted) backfill without
disturbing it:

```bash
python -m cli.main backfill-status --checkpoint-file checkpoints/usdc_xlm_backfill.json
python -m cli.main backfill-status --checkpoint-file checkpoints/usdc_xlm_backfill.json --json
```

The command reports chunks completed, trades loaded, raw records scanned,
the latest ledger close time reached, the resume cursor, start/update
timestamps, and any failed chunks with their error and attempt count. It
exits `1` if the checkpoint is missing or is not a backfill checkpoint.

### Choosing a chunk size

A chunk is the unit of lost work on interruption and the unit of
checkpoint I/O. At Horizon's default page size of 200 records and roughly
500 ms per page (about 400 records/s), the trade-offs are:

| `chunk_size` | Pages per chunk | Work lost on interruption | Checkpoint writes per 1M records |
|---|---|---|---|
| 2,000 | 10 | ~5 s | 500 |
| 10,000 (default) | 50 | ~25 s | 100 |
| 50,000 | 250 | ~2 min | 20 |

Recommendations:

* Keep `chunk_size` a multiple of `limit_per_page` so chunks line up with
  Horizon pages and no page is fetched twice on resume.
* The default of 10,000 suits most backfills. Checkpoint overhead is
  negligible, and a restart repeats well under a minute of fetching.
* Use 2,000 to 5,000 when the Horizon connection is unreliable or heavily
  rate-limited (many 429s or failovers), where interruptions are frequent and
  re-fetching is expensive.
* Use up to 50,000 for very large multi-day backfills on a stable connection
  to keep the number of chunk files and checkpoint writes down. Going higher
  mainly increases sink memory (one chunk of trades is held in memory) and
  the work repeated after a crash.

## Error handling

Ingestion and validation failures raise typed exceptions rather than bare
`ValueError` / `KeyError`, so callers can distinguish a malformed upstream
record from a bad argument from an unavailable source, and so every failure
carries enough context to triage without reproducing it.

### Hierarchy

```
LedgerLensError                     utils/exceptions.py
└── IngestionError                  ingestion/exceptions.py
    ├── InvalidInputError           (also a ValueError)
    ├── RecordValidationError
    │   └── SchemaValidationError
    └── SourceUnavailableError
        ├── HorizonRateLimitExceeded    ingestion/horizon_fetcher.py
        └── PoolNotFoundError           ingestion/amm_pool_loader.py
```

The base lives in `utils/exceptions.py` and the domain taxonomy in
`ingestion/exceptions.py` because `utils/` is the repo's home for
cross-cutting infrastructure with no upward dependencies: other packages can
adopt `LedgerLensError` later without importing `ingestion/`.

* `InvalidInputError` -- a caller-supplied argument failed validation before
  any I/O was attempted. Also inherits `ValueError` so existing
  `except ValueError` handlers and tests keep working.
* `RecordValidationError` -- an upstream record could not be turned into a
  typed model (missing field, wrong type, pydantic failure). Deliberately
  **not** a `KeyError`: pydantic-originated failures would otherwise satisfy
  unrelated `except KeyError` control flow, such as
  `ingestion/sketches.py`'s hot-path wallet-lock lookup.
* `SchemaValidationError` -- a record failed Avro schema validation.
* `SourceUnavailableError` -- an upstream source was unavailable or exhausted
  its retry budget.

### Context convention

`IngestionError` carries three optional fields, mirroring the two conventions
already in the codebase -- the dead-letter envelope written by
`ingestion/kafka_producer.py::_produce_to_dlq` (`reason` plus a best-effort
`raw` payload), and the `component` attribute on
`utils/circuit_breaker.py::CircuitOpenError` (which `source` plays the role
of):

| Field | Meaning |
|---|---|
| `source` | Module/function that raised it, e.g. `horizon_streamer._to_trade` |
| `reason` | Underlying cause, typically `str(original_exception)` |
| `raw` | The offending record, scrubbed to JSON-safe values |

`raw` is scrubbed by `ingestion.exceptions.safe_raw`, which mirrors
`kafka_producer._safe_raw` so an exception and a DLQ envelope describe the
same failed record identically. All three are also collected into
`exc.context` for logging.

`ingestion.exceptions.record_context` is the shared helper that translates raw
construction failures at a boundary:

```python
with record_context("horizon_streamer._to_trade", record):
    return Trade(...)
```

It wraps `KeyError`, `TypeError`, `ValueError` (which covers pydantic's
`ValidationError`) and `ZeroDivisionError`, and passes anything already in
this hierarchy through unchanged so a more specific type is never downgraded.

### Logging context

Pass the context through as a logging `extra` so it survives into structured
output:

```python
logger.error("%s", exc, extra={"context": exc.context})
```

`utils/logging.py` emits these fields when `LOG_FORMAT=json` (via
`python-json-logger`). The plain-text fallback formatter does **not** render
`extra` fields -- they are attached to the `LogRecord` but invisible in
dev-mode text logs. This is pre-existing behaviour of `utils/logging.py`, not
something the typed exceptions changed; be aware of it when debugging
locally with the default formatter.

### Raise vs degrade

Several ingestion paths degrade deliberately rather than failing. Typing the
exceptions did **not** change whether any path degrades -- only what type
propagates when one does raise.

These keep degrading and are unchanged:

* `rate_limiter.py` -- if Redis is unreachable the limiter grants every
  request rather than stalling ingestion (see *Degraded mode* above).
* `account_activity_loader.py::load_accounts_activity` -- per-item failures
  are logged and skipped so the rest of the batch is unaffected. The
  single-item `load_account_activity` raises `RecordValidationError`; the
  batch caller still swallows it, along with network errors.
* `asset_metadata_fetcher.py` -- Redis cache read/write failures are
  swallowed, and a failed Horizon fetch returns `None`.
* `amm_pool_loader.py::stream_amm_pool_trades` -- the SSE loop logs and
  reconnects. `_amm_record_to_trade`'s price fallback to `0.0` on an
  unparseable price is also preserved.
* `horizon_streamer.py::stream_trades` -- the reconnect loop still retries
  connection errors up to `max_reconnect_attempts`.
* `kafka_producer.py::produce_trade` -- serialisation failures still route to
  the DLQ. `SchemaValidationError` is an ordinary `Exception`, so the
  existing broad catch is unaffected.
* `trade_deduplicator.py` -- every Redis failure is handled internally; the
  cache never raises to its caller.

These now raise a typed exception where they previously raised a bare stdlib
one. **They still propagate** -- a malformed record continues to terminate the
stream or bulk load it occurs in, exactly as before:

| Location | Was | Now |
|---|---|---|
| `horizon_streamer._to_trade` | `KeyError` / pydantic | `RecordValidationError` |
| `orderbook_loader._to_orderbook_event` | `KeyError` / pydantic | `RecordValidationError` |
| `amm_pool_loader._amm_record_to_trade` | `KeyError` / pydantic | `RecordValidationError` |
| `avro_codec.record_to_trade` | `KeyError` / pydantic | `RecordValidationError` |
| `account_activity_loader.load_account_activity` | `KeyError` / pydantic | `RecordValidationError` |
| `avro_codec.serialize` / `validate` | `fastavro` `ValidationError` | `SchemaValidationError` |
| `avro_codec.SchemaRegistry` fingerprint lookup | `KeyError` | `InvalidInputError` |
| `horizon_streamer._validate_urls` | `ValueError` | `InvalidInputError` |
| `horizon_streamer.stream_all_watched_pairs` | `ValueError` | `InvalidInputError` |
| `amm_pool_loader._validate_pool_id` | `ValueError` | `InvalidInputError` |
| `kafka_producer._to_canonical_pair_id` | `ValueError` | `InvalidInputError` |
| `payment_path_analyzer.reconstruct_path_flow` (bad op type) | `ValueError` | `InvalidInputError` |

Whether the four malformed-record paths above should *skip and continue*
instead of terminating is a separate decision: dropping records silently
changes the input distribution the Benford engine and feature pipeline see,
so it was left alone here.

One behaviour change is intentional and visible to callers:
`payment_path_analyzer.reconstruct_path_flow` raised `KeyError` for missing
required fields and now raises `RecordValidationError`, which is **not** a
`KeyError` subclass, for the reason given under *Hierarchy* above.

### Known gap

Pydantic models are wrapped at the five boundary call sites that build them
from untrusted upstream dicts (`horizon_streamer._to_trade`,
`orderbook_loader._to_orderbook_event`,
`amm_pool_loader._amm_record_to_trade`, `avro_codec.record_to_trade`,
`account_activity_loader.load_account_activity`) rather than inside
`ingestion/data_models.py`, which is untouched. Wrapping in the models
themselves would route every construction -- including tests and trusted
internal re-validation -- through the wrapper, which is neither idiomatic
pydantic nor needed here.

The accepted consequence: **a future call site that constructs these models
from raw external data will not get typed wrapping automatically.** New
ingestion boundaries must opt in with `record_context`.
