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
