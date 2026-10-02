# Streaming resilience (Issues #897–#900)

## Model hot-reload — `StreamingScorer` (#897)

- `swap_model(model_dir)` builds the new `RiskScorer` fully, then replaces the
  active reference in one assignment (double-buffered). `swap_model_async()`
  does the load on a background thread so the stream never pauses.
- `score_wallet` pins the scorer reference once at the start of the request, so
  an in-flight request finishes on the version it started with — no
  mixed-version scoring, no dropped requests.
- `rollback_model()` swaps back to the previously active model.
- Swap latency (promotion request → new model active, including artifact load)
  is returned by `swap_model` and exposed as `last_swap_latency_ms`; it is also
  logged as `swap_latency_ms`. The swap itself is a single reference
  assignment (sub-microsecond); latency is dominated by artifact load time.

## RL threshold safety — `ThresholdController` (#898)

- Hard bounds: `min_threshold` / `max_threshold` (default 40 / 95). Every
  policy output — including out-of-range or non-finite actions — is clamped.
- Circuit breaker: trips when `alerts_fired_last_hour` leaves
  `[min_alerts_per_hour, max_alerts_per_hour]` or `analyst_tp_rate_last_24h`
  drops below `min_tp_rate` (env: `RL_MAX_ALERTS_PER_HOUR`,
  `RL_MIN_ALERTS_PER_HOUR`, `RL_MIN_TP_RATE`). While open, every asset uses
  `safe_threshold` until `reset_circuit_breaker()`.
- Operator override: `pin_threshold()` / `release_override()` in code, or
  `python scripts/threshold_override.py pin 80 [--asset PAIR]` /
  `release` / `show` (file: `RL_THRESHOLD_OVERRIDE_PATH`, watched when the
  controller is built with `override_path`). Precedence:
  override > circuit breaker > RL policy.

## PubSub dead-letter — `PubSubRouter` (#899)

- `deliver(client_id, channel, message, handler)` retries up to
  `max_delivery_attempts` (env `PUBSUB_MAX_DELIVERY_ATTEMPTS`, default 3), then
  appends to the dead-letter store (`PUBSUB_DEAD_LETTER_PATH`) with `error`,
  `retry_count`, `first_failure_at`, `dead_lettered_at`.
- Inspect / replay: `python scripts/pubsub_dead_letter.py list|show ID|replay [ID ...]`.
  Successful replays are removed; still-failing ones are re-dead-lettered.

## Idempotent alert delivery — `AlertDispatcher` (#900)

Each alert gets a deterministic idempotency key
(`sha256(wallet|pair|score|cooldown-window)`). An `in_flight` intent is written
to `AlertDeliveryLedger` **before** the external call; the terminal outcome is
written after. `reconcile_on_startup()` resolves intents with no terminal
outcome before delivery resumes.

| Channel     | Idempotency key usage                          | Reconciliation of in-flight-at-crash |
|-------------|------------------------------------------------|--------------------------------------|
| `webhook`   | `Idempotency-Key` HTTP header + `idempotency_key` body field | Re-sent with the same key; receiver dedupes |
| `websocket` | `idempotency_key` field in the JSON frame (no server-side dedup) | Not re-sent; closed as `reconciled_skipped` (at-most-once) |
| `stdout`    | Recorded in the ledger only                    | Not re-sent; closed as `reconciled_skipped` |
