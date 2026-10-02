# LedgerLens Real-Time Streaming Architecture

This document describes the end-to-end real-time detection pipeline introduced
in Issues #012 (Phase 1), #013 (Phase 2), and #014 (Phase 3 — Kafka partitioning).
It covers every component, the data flow between them, threading model, alert 
delivery channels, and the security constraints applied to the WebSocket server.

---

## Architecture Overview (Phase 3: Kafka Partitioning)

### Problem
The original architecture (Phase 1–2) processed all trades through a single
consumer thread per pair. For Benford analysis at scale, this was inefficient:
- Cross-pair Benford metrics require all events in the same process
- Per-pair Benford metrics are fully independent

### Solution
**Kafka topic partitioned by asset_pair_id**: each partition handles trades for
a single asset pair independently. Independent workers consume partitions in
parallel, enabling near-linear throughput scaling.

```
Stellar Horizon SSE
        │
        │  (historical or Kafka producer)
        ▼
Kafka Producer (ingestion/kafka_producer.py)
  Partition Key: canonical asset_pair_id (sorted alphabetically)
        │
        ▼
    Kafka Topic (e.g., "trades")
  ┌─────┬─────┬─────┬─────┐
  │ P:0 │ P:1 │ P:2 │ P:3 │  (4 partitions = 4 independent asset pairs)
  └──┬──┴──┬──┴──┬──┴──┬──┘
     │     │     │     │
     ▼     ▼     ▼     ▼
  Worker Worker Worker Worker
  (KafkaWorker threads)
     │     │     │     │
     ├─ FeatureBuffer (per-worker)
     ├─ StreamingScorer
     ├─ AlertDispatcher
     └─ Benford state (per-pair)
     │
     ├─── stdout
     ├─── webhook
     └─── WebSocket

CrossVenueAggregator (separate consumer group)
  Reads from all partitions for cross-pair analysis
        │
        ▼
  Cross-pair feature cache
```

---

## Phase 1–2: SSE-based Streaming

```
Stellar Horizon SSE
  (one stream per pair)
        │
        │  Trade objects (Pydantic)
        ▼
  ┌─────────────┐
  │ FeatureBuffer│  Phase 1 — streaming/feature_buffer.py
  │  (per wallet)│  Thread-safe rolling trade buffer.
  └──────┬──────┘  update(trade) adds to base_account AND
         │         counter_account buffers.
         │  wallet_trade_count / get_wallet_df
         ▼
  ┌────────────────┐
  │ StreamingScorer │  Phase 1 — streaming/feature_buffer.py
  │                 │  Wraps RiskScorer + FeatureBuffer.
  │ score_wallet()  │  Returns None until min_trades reached.
  └───────┬─────────┘  Calls build_feature_vector → RiskScorer.score().
          │
          │  RiskScore dict {score, benford_flag, ml_flag, confidence}
          ▼
  ┌──────────────────┐
  │ AlertDispatcher   │  Phase 2 — streaming/alert_dispatcher.py
  │                   │  Threshold check + per-wallet cooldown (Lock-protected).
  │ dispatch()        │  Delivers once per cooldown window per wallet.
  └───────┬───────────┘
          │
          ├─── stdout ──────────────────────── [ALERT] wallet=… score=…
          │
          ├─── HTTP POST ───────────────────── ALERT_WEBHOOK_URL (https:// only)
          │
          └─── ws_client.send() ────────────► ws_server.py
                                               (asyncio, loopback-only by default)
                                               Broadcasts to all connected clients.

StreamingPipeline    Phase 2 — streaming/pipeline.py
  One daemon Thread per WATCHED_ASSET_PAIR
  Each thread: stream_trades() → buffer.update() → scorer.score_wallet()
               → dispatcher.dispatch()

scripts/stream.py    Phase 2 CLI
  python -m scripts.stream [flags]
```

---

## Phase 3: Kafka-based Partitioning

### New Components

#### `ingestion/kafka_producer.py`

**Function: `_to_canonical_pair_id(code_a, issuer_a, code_b, issuer_b)`**
- Generates deterministic partition key from asset pair
- Format: `CODE1:ISSUER1/CODE2:ISSUER2` (alphabetically sorted)
- Example: `USDC:GA.../XLM:native` → `USDC:GA.../XLM:native`
- If reversed: `XLM:native/USDC:GA...` → same result
- Validation: code (1-12 alphanumeric), issuer ("native" or 56-char Stellar ID)

**Class: `KafkaTradeProducer`**

| Method | Purpose |
|--------|---------|
| `produce_trade(trade: Trade)` | Send trade to Kafka with canonical pair key; invalid pairs → DLQ |
| `flush()` | Flush pending messages |
| `close()` | Close producer |

**Dead-Letter Queue:**
- Topic: `{topic}-dlq` (default: `trades-dlq`)
- Invalid pairs routed here with error reason
- Enables audit and remediation

#### `streaming/kafka_worker.py`

**Class: `KafkaWorker`**

Actual constructor signature (`scorer` and `dispatcher` are required
positional arguments; everything else is optional/keyword — this table
previously documented a `partitions`/`commit_interval_seconds` shape that
never existed in the shipped code):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `scorer` | — | `StreamingScorer` (required, positional) |
| `dispatcher` | — | `AlertDispatcher` (required, positional) |
| `buffer` | new `FeatureBuffer()` | Per-worker trade buffer |
| `watchdog` | `None` | Optional `EmergencyWatchdog` |
| `consumer` | new `confluent_kafka.Consumer` | Inject a pre-built consumer (used by tests) |
| `bootstrap_servers` | `config.KAFKA_BOOTSTRAP_SERVERS` | Kafka brokers |
| `group_id` | `config.KAFKA_CONSUMER_GROUP` | Consumer group |
| `topic_pattern` | `config.KAFKA_TOPIC_PATTERN` | Regex subscription |
| `dlq_topic` | `config.KAFKA_DLQ_TOPIC` | Dead-letter topic to never process |
| `lag_threshold` | `config.KAFKA_LAG_ALERT_THRESHOLD` | Lag before a CRITICAL log fires |
| `enable_backpressure` | `True` | Wire up `BackPressureController` |
| `dedup_cache` | new `DeduplicationCache()` | Inject an exactly-once dedup cache (used by tests) |

| Method | Purpose |
|--------|---------|
| `run()` | Poll-and-process loop; blocks until `stop()` is called |
| `process_message(msg)` / `_process_correlated_message(msg)` | Process one message: decode → dedup stage → buffer → score → dispatch → dedup commit → offset commit |
| `_check_lag(msg)` | Publish per-partition consumer lag, CRITICAL-log if over threshold |
| `stop()` / `close()` | Signal shutdown / release the consumer and flush the DLT producer |

**Rebalancing:** `confluent_kafka.Consumer` (`enable.auto.commit=False`)
manages partition assignment itself; this worker never calls a manual
rebalance/revocation hook. Because offsets are only committed after a
message's dedup key is durably committed (see "Exactly-once dedup" below),
a rebalance mid-processing simply leaves the in-flight message's offset
uncommitted — the next assignee redelivers and reprocesses it safely.

#### `detection/cross_venue_features.py`

**Class: `CrossVenueAggregator`**

| Method | Purpose |
|--------|---------|
| `collect_trades(max_batches)` | Consume and buffer trades from all partitions |
| `_buffer_trade(payload)` | Add trade to wallet/pair buffers |
| `get_cross_pair_features(wallet)` | Compute cross-pair stats for wallet |
| `get_pair_cross_venue_features(pair_id)` | Compute pair-specific stats |
| `clear_buffers()` | Clear buffers after aggregation |
| `close()` | Close consumer |

**Features Computed:**
- `n_distinct_pairs`: number of asset pairs wallet traded on
- `cross_pair_volume_concentration`: max pair volume / total volume
- `venue_diversity_score`: (1 - concentration) / n_pairs

#### `scripts/kafka_workers.py`

**Usage:**
```bash
make scale-workers N=4
python -m scripts.kafka_workers --num-workers 4 --topic trades --group ledgerlens-workers
```

**Behavior:**
1. Spawn N worker threads
2. Each worker subscribes to the same topic and group
3. Kafka automatically assigns partition subsets to each worker
4. Workers process partitions in parallel
5. On shutdown (Ctrl+C), gracefully stop all workers and commit offsets

**Configuration:**
- `ALERT_CHANNEL` (env var): `stdout`, `webhook`, or `websocket`
- `ALERT_WEBHOOK_URL` (env var): HTTPS endpoint
- `ALERT_COOLDOWN_SECONDS` (env var): per-wallet dedup window

---

## Partition Key Scheme

**Canonical Format**
```
CODE1:ISSUER1/CODE2:ISSUER2
```

**Sorting Rule**
- Lexicographic sort by `CODE:ISSUER`
- Examples:
  - `BTC:native, XLM:native` → `BTC:native/XLM:native`
  - `USDC:GA.../XLM:native` → `USDC:GA.../XLM:native` (USDC < XLM)
  - `XLM:native, USDC:GA...` → `USDC:GA.../XLM:native` (same result)

**Guarantees**
- **Deterministic**: same pair always maps to same partition
- **Stable**: invocation order doesn't matter
- **Validated**: invalid assets rejected before send (routed to DLQ)

**Validation Rules**
- Asset code: 1-12 alphanumeric characters
- Issuer: either `"native"` or 56-character Stellar account ID

---

## Threading Model (Phase 3)

```
Main thread (scripts/kafka_workers.py)
│  installs SIGTERM/SIGINT → stop_event.set()
│  spawns N worker threads
│
├── Thread: worker-0 (daemon)
│     KafkaWorker.run()
│     ├─ FeatureBuffer + StreamingScorer (per-worker state)
│     ├─ for message in consumer.poll():
│     │    dedup_cache.check_and_stage() → buffer.update(trade)
│     │    score_wallet(wallet) → dispatch() → dedup_cache.commit()
│     └─ Commits the Kafka offset per-message, after the dedup commit above
│        (see "Exactly-once effects, at-least-once delivery" below)
│
├── Thread: worker-1
│     (same as worker-0, different partitions via Kafka assignment)
│
└── Thread: worker-N
```

All workers access `dispatcher` (shared AlertDispatcher with Lock-protected cooldowns).

---

## Deployment Scenarios

### Scenario 1: 1 Worker, All Partitions (Default SSE Compatibility)
```bash
make scale-workers N=1
```
- Single worker handles all partitions
- Equivalent to Phase 1–2 behavior
- Use for backward compatibility or single-pair testing

### Scenario 2: 4 Workers, 4 Partitions (1 Pair per Worker)
```bash
make scale-workers N=4
```
- Each worker handles 1 partition (1 asset pair)
- Maximum parallelism for 4 monitored pairs
- Linear throughput scaling: 4× vs. 1 worker

### Scenario 3: 2 Workers, 8 Partitions (4 Pairs per Worker)
```bash
make scale-workers N=2
```
- Each worker handles 4 partitions
- Reduces resource overhead (fewer threads, less memory)
- Good balance for moderate traffic

### Scenario 4: Cross-Venue Aggregation
```bash
# Terminal 1: start 4 workers
make scale-workers N=4

# Terminal 2: start aggregator (reads from all partitions in separate consumer group)
python -c "from detection.cross_venue_features import CrossVenueAggregator; \
  agg = CrossVenueAggregator('trades', group_id='ledgerlens-aggregator'); \
  agg.collect_trades(max_batches=1000)"
```

---

## Latency Budget

| Stage | Typical latency |
|---|---|
| Ledger close → Horizon SSE event | ~1–2 s |
| SSE event → Kafka producer (optional) | < 100 ms |
| Producer → Kafka broker (ack) | < 50 ms |
| Kafka broker → Worker poll | < 100 ms |
| Worker: buffer.update() + score | < 50 ms |
| dispatch() stdout/webhook | < 5 s (webhook timeout) |
| **Total ledger close → alert** | **< 10 s** |

---

## Troubleshooting

Common local failures are usually visible in the same places the runtime already exposes telemetry; these checks are the quickest way to narrow the problem.

### 1) WebSocket server refuses to bind
- Symptom: `OSError: [Errno 98] Address already in use` or no dashboard updates on `--alert-channel websocket`.
- Likely cause: another process is already listening on `WS_PORT` (default `8765`), or `WS_BIND_HOST` is set to a non-loopback interface without `WS_ALLOW_EXTERNAL=1`.
- Check: `WS_PORT`, `WS_BIND_HOST`, `WS_ALLOW_EXTERNAL`, and the startup log line from `streaming/ws_server.py` that shows the bind address.

### 2) Kafka consumer lag climbs aggressively
- Symptom: score throughput drops while the backlog grows; Prometheus shows `kafka_lag_by_partition` rising.
- Likely cause: the consumer is slower than the producer, a partition is hot, or the worker is stuck in a slow scoring path.
- Check: `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_LAG_ALERT_THRESHOLD`, the `kafka_lag_by_partition` gauge, and the worker log lines around offset commits / poll loops.

### 3) No trades are scored and alerts never fire
- Symptom: the stream is running but the dashboard stays silent and `alerts_dispatched_total` stays flat.
- Likely cause: `WATCHED_ASSET_PAIRS` is empty, the worker never receives messages, or the wallet has not reached the configured `--min-trades` threshold.
- Check: the startup config for `WATCHED_ASSET_PAIRS`, the `--min-trades` value, the `scoring_latency_ms` and `kafka_messages_consumed_total` metrics, and any log line showing a wallet buffer not yet meeting the threshold.

### 4) Webhook delivery fails silently or is rejected
- Symptom: `ALERT_CHANNEL=webhook` but no HTTP POSTs are received, or the service logs a warning and keeps running.
- Likely cause: `ALERT_WEBHOOK_URL` is missing, not `https://`, or the endpoint is returning an error code / timeout.
- Check: `ALERT_WEBHOOK_URL`, `ALERT_CHANNEL`, the `AlertDispatcher` warning about invalid webhook URLs, and the `alerts_dispatched_total` metric to confirm the event was emitted before transport.

### 5) The pipeline appears healthy but no data is in the worker buffers
- Symptom: Kafka topic is populated but no scoring is happening; the worker accepts messages but does nothing.
- Likely cause: a malformed trade or a consumer/group mismatch; the producer may be sending to a different topic or partition than the worker is subscribed to.
- Check: `KAFKA_TOPIC`, `KAFKA_GROUP_ID`, `KAFKA_BOOTSTRAP_SERVERS`, and the worker log line that records the last processed message or partition assignment.

### 6) Local runs stop after a reconnect storm
- Symptom: repeated reconnect warnings or stalled streamers on a laptop or local VM.
- Likely cause: the upstream Horizon or Kafka endpoint is unavailable, the socket is timing out, or the worker is retrying with a stale configuration.
- Check: the `stream_trades()` reconnect warnings, `KAFKA_BOOTSTRAP_SERVERS`, `STREAMING_BACKEND`, and the per-partition lag metrics before/after the reconnect window.

These are the same surfaces the code already exposes: Prometheus metrics (`kafka_lag_by_partition`, `scoring_latency_ms`, `alerts_dispatched_total`), worker logs, and the streaming env vars above.

---

## Security Notes

- **Partition keys**: validated against canonical format before production
  - Invalid pairs rejected at source (no invalid data in Kafka)
  - Malformed pairs → dead-letter queue for audit
- **Offset commits**: manual commit only after a message's exactly-once
  dedup key is durably committed (see "Exactly-once dedup" below) —
  Kafka delivery itself is still at-least-once (redelivery on crash/rebalance
  is expected and normal), but a message's *effects* (feature update, score,
  alert dispatch) are applied exactly once per dedup key
- **Webhook**: HTTPS-only (http:// rejected at AlertDispatcher init)
- **WebSocket**: bound to `127.0.0.1` by default (loopback-only)

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker addresses |
| `KAFKA_TOPIC` | `trades` | Topic name |
| `KAFKA_GROUP_ID` | `ledgerlens-workers` | Consumer group |
| `ALERT_CHANNEL` | `stdout` | `stdout`, `webhook`, or `websocket` |
| `ALERT_WEBHOOK_URL` | — | HTTPS endpoint for webhooks |
| `ALERT_COOLDOWN_SECONDS` | `3600` | Per-wallet dedup window |
| `WS_PORT` | `8765` | WebSocket server port |
| `WS_BIND_HOST` | `127.0.0.1` | WebSocket bind address |
| `WS_ALLOW_EXTERNAL` | — | Set to `1` to allow external connections |

---

## Testing

### Unit Tests
```bash
pytest tests/test_kafka_partitioning.py -v
```
- Partition key generation (deterministic, alphabetic sorting)
- Asset pair validation
- Dead-letter queue routing

### Integration Tests
```bash
pytest tests/test_kafka_integration.py -v
```
- Producer → consumer flow (mocked Kafka)
- Worker message processing
- Cross-venue aggregator

### Manual Testing
```bash
# Start Kafka locally (Docker Compose)
docker-compose up -d

# Run unit tests
make test

# Start 2 workers
make scale-workers N=2

# In another terminal: produce test trades
python scripts/generate_synthetic_dataset.py | python -m ingestion.kafka_producer

# Monitor alerts
tail -f /tmp/ledgerlens.log | grep ALERT
```

---

## Pipeline Overview

```
Stellar Horizon SSE
  (one stream per pair)
        │
        │  Trade objects (Pydantic)
        ▼
  ┌─────────────┐
  │ FeatureBuffer│  Phase 1 — streaming/feature_buffer.py
  │  (per wallet)│  Thread-safe rolling trade buffer.
  └──────┬──────┘  update(trade) adds to base_account AND
         │         counter_account buffers.
         │  wallet_trade_count / get_wallet_df
         ▼
  ┌────────────────┐
  │ StreamingScorer │  Phase 1 — streaming/feature_buffer.py
  │                 │  Wraps RiskScorer + FeatureBuffer.
  │ score_wallet()  │  Returns None until min_trades reached.
  └───────┬─────────┘  Calls build_feature_vector → RiskScorer.score().
          │
          │  RiskScore dict {score, benford_flag, ml_flag, confidence}
          ▼
  ┌──────────────────┐
  │ AlertDispatcher   │  Phase 2 — streaming/alert_dispatcher.py
  │                   │  Threshold check + per-wallet cooldown (Lock-protected).
  │ dispatch()        │  Delivers once per cooldown window per wallet.
  └───────┬───────────┘
          │
          ├─── stdout ──────────────────────── [ALERT] wallet=… score=…
          │
          ├─── HTTP POST ───────────────────── ALERT_WEBHOOK_URL (https:// only)
          │
          └─── ws_client.send() ────────────► ws_server.py
                                               (asyncio, loopback-only by default)
                                               Broadcasts to all connected clients.

StreamingPipeline    Phase 2 — streaming/pipeline.py
  One daemon Thread per WATCHED_ASSET_PAIR
  Each thread: stream_trades() → buffer.update() → scorer.score_wallet()
               → dispatcher.dispatch()

scripts/stream.py    Phase 2 CLI
  python -m scripts.stream [flags]
```

---

## Components

### `streaming/feature_buffer.py` — Phase 1

#### `FeatureBuffer`

| Method | Description |
|---|---|
| `update(trade: Trade)` | Appends a trade record to the rolling buffer for both `trade.base_account` and `trade.counter_account`. Protected by `threading.Lock`. |
| `get_wallet_df(wallet)` | Returns a `pd.DataFrame` of all buffered trades for the wallet. |
| `wallet_trade_count(wallet)` | Returns the number of buffered trades (used to gate scoring). |

The buffer caps each wallet at `max_trades_per_wallet` (default 5 000) most-recent trades, trimming old entries on each `update()`.

#### `StreamingScorer`

Wraps a trained `RiskScorer` and a `FeatureBuffer`.  `score_wallet(wallet)` returns `None` until `wallet_trade_count >= min_trades` (default 20), then builds a full feature vector via `detection.feature_engineering.build_feature_vector` and calls `RiskScorer.score()`.

---

### `streaming/alert_dispatcher.py` — Phase 2

#### `AlertDispatcher`

| Parameter | Default | Description |
|---|---|---|
| `channel` | `"stdout"` | Delivery channel: `stdout`, `webhook`, or `websocket` |
| `webhook_url` | `None` | Falls back to `ALERT_WEBHOOK_URL` env var |
| `ws_client` | `None` | Object with `.send(str)` method; injected for testability |
| `alert_cooldown_seconds` | `3600` | Per-wallet dedup window |
| `threshold` | `RISK_SCORE_FLAG_THRESHOLD` | Minimum score to fire an alert |

**Deduplication**: `{wallet: expiry_timestamp}` dict, guarded by `threading.Lock`.  A wallet is suppressed while `time.time() < expiry`.

**Stdout format**:
```
[ALERT] wallet=G… pair=USDC:…/XLM:native score=83 benford=True ml=True confidence=76
```

**Webhook**: `POST` with 5-second timeout.  `http://` URLs are rejected at construction with `ValueError`.  HTTP errors are logged as `WARNING` and do not crash the pipeline.  The URL is never logged.

**WebSocket**: calls `ws_client.send(json.dumps(payload))` where `payload` is the `RiskScore` dict plus `wallet` and `pair_id`.

---

### `streaming/ws_server.py` — Phase 2

A minimal asyncio WebSocket server.

| Symbol | Description |
|---|---|
| `run_ws_server(host, port)` | Async coroutine that starts the server and runs until cancelled. |
| `send_alert(payload)` | Async broadcast to all connected clients (runs inside the server loop). |
| `push_alert_sync(payload)` | Thread-safe: schedules `send_alert` on the server loop from any thread. |
| `start_ws_server_thread(host, port)` | Starts the server in a daemon thread; returns when the loop is ready. |
| `_WsClientAdapter` | Adapts `ws_client.send(msg)` → `push_alert_sync(json.loads(msg))`. |

**Security**:
- Default bind: `127.0.0.1` (loopback).
- `WS_BIND_HOST=0.0.0.0` raises `ValueError` unless `WS_ALLOW_EXTERNAL=1` is also set.
- `_clients` is only mutated from inside the asyncio event loop (`_handler`, `send_alert`).

---

### `streaming/pipeline.py` — Phase 2

#### `StreamingPipeline`

| Parameter | Default | Description |
|---|---|---|
| `buffer` | — | `FeatureBuffer` instance |
| `scorer` | — | `StreamingScorer` instance |
| `dispatcher` | — | `AlertDispatcher` instance |
| `pairs` | `config.WATCHED_ASSET_PAIRS` | Optional override for testing |

`run()` converts each `(code, issuer)` pair to a `SdkAsset`, starts one daemon thread per pair running `_stream_pair()`, then blocks in a `while not stop_event.is_set()` loop.

When called from the main thread, `run()` installs a `SIGINT` handler that sets the stop event.  It also catches `KeyboardInterrupt` in case the signal arrives while blocked.  On exit, all worker threads are joined with a 5-second timeout.

`_stream_pair()` wraps `stream_trades()` in a `try/except` so that after `stream_trades` exhausts its own internal reconnect attempts, `_stream_pair` logs a warning and restarts the generator.

---

### `scripts/stream.py` — Phase 2

CLI entrypoint: `python -m scripts.stream`.

```
usage: python -m scripts.stream [--alert-channel {stdout,webhook,websocket}]
                                 [--cooldown-seconds N]
                                 [--min-trades N]
                                 [--no-ws]
```

**Startup sequence**:
1. Validate `WATCHED_ASSET_PAIRS` is set.
2. Load `RiskScorer`; exit 1 if no models found.
3. Start WebSocket server thread (if `channel=websocket` and not `--no-ws`).
4. Instantiate `FeatureBuffer`, `StreamingScorer`, `AlertDispatcher`, `StreamingPipeline`.
5. Log startup banner (pair count, channel, WS address if active).
6. Call `pipeline.run()`.

---

## Threading Model

```
Main thread (scripts/stream.py)
│  installs SIGINT → _stop_event.set()
│  runs pipeline.run() — blocks on _stop_event
│
├── Thread: ws-server (daemon)
│     asyncio event loop running run_ws_server()
│
├── Thread: pair-0 (daemon)  → _stream_pair(USDC/XLM)
│     for trade in stream_trades():
│         buffer.update(trade)          # Lock-protected
│         scorer.score_wallet(base)     # reads buffer
│         dispatcher.dispatch(base, …)  # Lock-protected dedup
│         scorer.score_wallet(counter)
│         dispatcher.dispatch(counter, …)
│
└── Thread: pair-N (daemon)  → _stream_pair(…)
```

All threads are `daemon=True` so they are automatically killed if the main process exits.  The 5-second `join()` timeout in `run()` gives in-flight scoring a chance to flush before process exit.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WATCHED_ASSET_PAIRS` | — | Comma-separated `CODE:ISSUER` pairs to stream |
| `ALERT_CHANNEL` | `stdout` | `stdout`, `webhook`, or `websocket` |
| `ALERT_WEBHOOK_URL` | — | HTTPS endpoint; required when channel is `webhook` |
| `ALERT_COOLDOWN_SECONDS` | `3600` | Per-wallet alert dedup window (seconds) |
| `WS_PORT` | `8765` | WebSocket server port |
| `WS_BIND_HOST` | `127.0.0.1` | WebSocket server bind address |
| `WS_ALLOW_EXTERNAL` | — | Set to `1` to allow non-loopback binding |

---

## Latency Budget

| Stage | Typical latency |
|---|---|
| Ledger close → Horizon SSE event | ~1–2 s |
| SSE event → buffer.update() | < 1 ms |
| score_wallet() (feature build + 3-model inference) | < 50 ms |
| dispatch() stdout/webhook | < 5 s (webhook timeout) |
| **Total ledger close → alert** | **< 10 s** |

---

## Security Notes

- `ALERT_WEBHOOK_URL` must use `https://`; `http://` is rejected at startup.
- The URL is never written to logs.
- The WebSocket server binds to `127.0.0.1` by default; opt-in is required for external binding.
- `_clients` is mutated only inside the asyncio event loop, preventing data races.

---

## Sliding Window Benford Aggregator (Issue #254)

The DB-based Benford computation in `benford_engine.py` scans all trades in a
time window from the database on every call.  For the 30-day window on active
pairs this takes several seconds and scans millions of rows.

`SlidingWindowBenfordAggregator` (`detection/sliding_window_benford.py`)
maintains per-digit counts in memory, updated incrementally as trades arrive
and expire.

### Design

```
add_trade(amount, timestamp)
  └── _lazy_expire(timestamp)          ← drain expired entries from heap
  └── digit = leading_digit(amount)
  └── digit_counts[digit-1] += 1
  └── heappush(heap, (timestamp, digit))

chi_square()  ← O(9) arithmetic over digit_counts; no DB access
mad()         ← O(9) arithmetic
z_scores()    ← O(9) arithmetic
```

**Lazy expiry**: a min-heap keyed by timestamp drains expired entries on each
`add_trade` call.  No background thread or timer is required.

**Concurrency**: all mutations are guarded by `asyncio.Lock`, making the
aggregator safe for concurrent use from multiple scoring coroutines within the
same event loop.

### Tolerance guarantee

Running chi-square matches the batch-computed value (from `benford_engine.py`)
within **1e-4 absolute tolerance** on synthetic data.  Verified in
`tests/test_issues_253_254_255_256.py`.

### Backward clock / NTP correction

If the system clock jumps backward (e.g. an NTP step correction), some trades
will appear to have timestamps in the future relative to the new system time.
`_lazy_expire` uses the *current trade's timestamp* as the reference for
expiry — not the system clock — so a backward step does not immediately
un-expire in-window trades.  Trades with future-relative timestamps will stay
in the window longer than their nominal window size, but will eventually expire
correctly as real trades arrive.  For deployments sensitive to this edge case,
use a monotonic clock for `add_trade` timestamps.

### Performance

10 000 `add_trade` calls complete in < 100ms on a single asyncio event loop
(verified in the performance test).

### Integration

For real-time (streaming) scoring, instantiate a `SlidingWindowBenfordAggregator`
per wallet / per window size and call `add_trade` as each new trade arrives.
Call `to_metrics()` to get a `BenfordMetrics` object compatible with the rest
of the feature pipeline.  The DB-based `compute_benford_metrics_for_windows`
in `benford_engine.py` remains in use for batch scoring runs.

### Environment variables

No new variables are introduced.  The aggregator window width is determined
by `BENFORD_WINDOWS_HOURS` (default `1,4,24,168,720`).

---

## Kafka Streaming Backend (Issue #36)

The default `sse` backend runs one thread per pair inside a single process — it
cannot scale beyond one machine, replay missed events, or apply backpressure.
Setting `STREAMING_BACKEND=kafka` swaps the transport for an Apache Kafka log
that decouples ingestion from scoring and allows horizontal scale-out. The
`sse` backend remains the default and is unchanged.

### Topology

```
Horizon SSE (one producer thread per pair)
      │  Trade → Avro (data/trade_avro_schema.json)
      ▼
HorizonKafkaProducer  (ingestion/kafka_producer.py)
      │  key = wallet_id (base_account)
      ▼
Kafka topics: ledgerlens.trades.{asset_pair_sanitised}     (+ ledgerlens.trades.dlq)
      │  regex subscription ^ledgerlens\.trades\..*
      ▼
KafkaWorker × N replicas   group.id = "ledgerlens-scorer"   (streaming/kafka_worker.py)
      │  FeatureBuffer → StreamingScorer → AlertDispatcher
      ▼
Alerts (stdout / webhook / websocket)  +  Prometheus /metrics
```

### Partition strategy

Messages are keyed by **`wallet_id` (the base account)**. Kafka hashes the key
to a partition, so every trade for a given wallet lands in the same partition
and is therefore consumed in order by exactly one worker. This preserves the
per-wallet ordering that feature computation depends on, while still spreading
distinct wallets across partitions for parallelism. New per-pair topics are
picked up automatically by the workers' regex subscription — no restart needed.

### Exactly-once effects, at-least-once delivery (Issue #670)

Kafka delivery is at-least-once — the same message can be redelivered after a
crash, a rebalance, or an uncommitted offset. Prior to Issue #670,
`KafkaWorker` did not correctly handle this: its `DeduplicationCache` marked a
message's dedup key as "seen" *before* processing it, so a crash partway
through processing (e.g. `AlertDispatcher.dispatch` raising for the second of
two wallets in a trade) left the dedup key set but the second wallet never
scored — on redelivery the message was misclassified as a duplicate and its
offset committed **without reprocessing**, silently dropping the wallet. See
`docs/adr/0001-unified-idempotency-finality.md` for the full analysis.

The fix is a two-phase staged/committed dedup protocol
(`pipeline.exactly_once.ExactlyOnceStore`), ordered so a crash at any point
results in "redo the tail", never "silently drop":

```
consumer.poll() → msg
       │
       ▼
dedup_cache.check_and_stage(ledger_seq, trade_id)   ← atomic Redis SET NX
       │
       ├── COMMITTED ──────────────────────────────► consumer.commit(msg)  [skip reprocessing]
       │
       └── NEW or STAGED (a prior attempt crashed
           before committing — "redo this")
                   │
                   ▼
           buffer.update(trade)              ← idempotent per (wallet, trade_id);
                   │                            safe to redo, never double-counts
                   ▼
           for wallet in (base, counter):
               score_wallet() → dispatcher.dispatch()
                   │                            (if dispatch raises here, neither
                   │                             commit below is reached — offset
                   │                             stays uncommitted, dedup key stays
                   │                             STAGED, redelivery redoes the message)
                   ▼
           dedup_cache.commit(ledger_seq, trade_id)   ← durable "done" marker
                   │
                   ▼
           consumer.commit(msg)                       ← Kafka offset, committed LAST
```

* Consumers run with `enable.auto.commit=false`.
* If the dedup backend (Redis) is unreachable, `check_and_stage`/`commit`
  raise `DedupBackendUnavailableError` — the worker leaves the offset
  uncommitted and does **not** fall back to "not a duplicate" (fail closed,
  not fail open). This is exposed via the `dedup_backend_available` gauge and
  `kafka_dedup_backend_degraded_total` counter.
* `FeatureBuffer.update()` is idempotent per `(wallet, trade_id)`, which is
  what makes redoing a `STAGED`-but-not-`COMMITTED` message safe — it can
  never double-count a trade into feature state, whether that message is
  redelivered within the same process or after a full restart.
* Duplicate alerts from a redo within a live cooldown window are still
  absorbed by `AlertDispatcher`'s per-wallet cooldown, as before.

### Avro schema & validation

The wire format is schemaless Avro binary encoding of the `Trade` record in
`data/trade_avro_schema.json`. The producer validates every record **before**
serialisation; the worker validates again **after** decode. Records that are
missing fields or have wrong-typed values never reach the scorer:

* On the **producer**, a serialisation/validation failure routes the raw
  payload plus a `reason` to the dead-letter queue `ledgerlens.trades.dlq`.
* On the **consumer**, a decode/validation failure (a poison pill) is logged,
  counted (`kafka_poison_messages_total`), and its offset committed (skipped) so
  one bad record cannot wedge a partition.

DLQ messages are **never** retried automatically — the worker's regex
subscription explicitly skips the DLQ topic, and triage is a human task.

### Backpressure & lag alerting

Per-partition lag (high watermark − committed offset) is published as the
Prometheus gauge `kafka_lag_by_partition`. When lag exceeds
`KAFKA_LAG_ALERT_THRESHOLD` (default 500) the worker emits a **CRITICAL** log
and keeps running. Scaling `ledgerlens-scorer` replicas adds consumers to the
`ledgerlens-scorer` group, redistributing partitions to drain the backlog.

### Security

* Broker credentials are read from `KAFKA_SASL_USERNAME` / `KAFKA_SASL_PASSWORD`
  **environment variables only**; when both are set the clients use
  `SASL_SSL` / `PLAIN`. They are never logged or committed.
* The producer enables idempotence (`enable.idempotence=true`, `acks=all`).

### Prometheus metrics (exposed by each worker on `KAFKA_METRICS_PORT`)

| Metric | Type | Description |
|---|---|---|
| `ledgerlens_ingestion_trades_produced_total` | Counter (`topic`) | Number of trades successfully produced to Kafka |
| `ledgerlens_ingestion_trades_failed_total` | Counter (`reason`) | Number of trades failed during ingestion |
| `kafka_messages_consumed_total` | Counter | Trade messages fully processed |
| `kafka_lag_by_partition` | Gauge (`topic`, `partition`) | Consumer lag |
| `scoring_latency_ms` | Histogram | Per-wallet scoring latency |
| `alerts_dispatched_total` | Counter | Alerts dispatched |
| `kafka_poison_messages_total` | Counter | Decode/validation failures dropped |

### Deployment

```bash
docker-compose up --scale ledgerlens-scorer=3
```

Brings up Zookeeper, Kafka, one `ledgerlens-producer`, three `ledgerlens-scorer`
replicas, Prometheus (`:9090`), and Grafana (`:3000`, dashboard
"LedgerLens Kafka Streaming").

### Kafka environment variables

| Variable | Default | Description |
|---|---|---|
| `STREAMING_BACKEND` | `sse` | `sse` (threaded) or `kafka` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Broker list |
| `KAFKA_SASL_USERNAME` | — | SASL username (env only) |
| `KAFKA_SASL_PASSWORD` | — | SASL password (env only) |
| `KAFKA_CONSUMER_GROUP` | `ledgerlens-scorer` | Worker consumer group |
| `KAFKA_TOPIC_PREFIX` | `ledgerlens.trades` | Per-pair topic prefix |
| `KAFKA_DLQ_TOPIC` | `ledgerlens.trades.dlq` | Dead-letter topic |
| `KAFKA_TOPIC_PATTERN` | `^ledgerlens\.trades\..*` | Worker regex subscription |
| `KAFKA_LAG_ALERT_THRESHOLD` | `500` | Lag (messages) for CRITICAL log |
| `KAFKA_METRICS_PORT` | `9100` | Prometheus scrape port |

---

## Adaptive Micro-Batch Sizing (Issue #243)

### Problem

The streaming scorer processes trade events in micro-batches. A fixed batch size is suboptimal: small batches waste per-batch inference overhead during quiet periods; large batches introduce queueing latency during spikes.

### PID Controller

`AdaptiveBatchController` (in `streaming/streaming_scorer.py`) adjusts batch size in real time using a **Proportional-Integral-Derivative (PID)** controller.

```
error = observed_p95_latency - target_p95_latency
Δbatch = -(Kp × error + Ki × ∫error + Kd × Δerror)
batch_size = clamp(batch_size + Δbatch, min_batch, max_batch)
```

- **P term** — responds immediately to the current latency error; dominant during fast changes.
- **I term** — integrates accumulated error to correct steady-state offset (e.g. consistently high latency at the current batch size).
- **D term** — damps oscillation by reacting to the rate of change in error.
- **Anti-windup** — the integral is clamped to `±50` so that sustained overload does not build an unbounded correction term that overshoots when conditions improve.

### Default Gains

| Gain | Default | Notes |
|---|---|---|
| `Kp` | 0.5 | Scale down for pipelines with high natural latency variance. |
| `Ki` | 0.1 | Increase if steady-state offset persists after many seconds. |
| `Kd` | 0.05 | Increase if the batch size oscillates. |

The defaults target a **p95 latency of 2 seconds** on typical LedgerLens workloads.

### Tuning for Different Deployment Sizes

- **High-throughput (>10 k trades/s)**: reduce `Kp` to 0.2–0.3 and increase `max_batch` to 1000.
- **Low-throughput (<100 trades/s)**: the controller naturally settles near `max_batch` since latency is always below target.
- **Bursty workloads (exchange listings)**: increase `Kd` to dampen oscillation during volume spikes.

Batch-size adjustments are logged at `DEBUG` level (`streaming.streaming_scorer`). Enable debug logging to produce a PID trace:

```bash
LOG_LEVEL=DEBUG python -m scripts.stream
```

### Disabling Adaptive Sizing

Pass `--fixed-batch-size N` to `scripts/stream.py` to pin the batch size for debugging:

```bash
python -m scripts.stream --fixed-batch-size 64
```

### Prometheus Gauges

| Metric | Type | Description |
|---|---|---|
| `ledgerlens_adaptive_batch_size` | Gauge | Current batch size chosen by the PID controller |
| `ledgerlens_batch_target_latency_seconds` | Gauge | Configured p95 latency target |

### Configuration

| Variable | Default | Description |
|---|---|---|
| `STREAM_TARGET_P95_LATENCY_SECONDS` | `2.0` | PID target latency |
| `STREAM_MIN_BATCH_SIZE` | `1` | Minimum allowed batch size |
| `STREAM_MAX_BATCH_SIZE` | `500` | Maximum allowed batch size |
| `STREAM_PID_KP` | `0.5` | Proportional gain |
| `STREAM_PID_KI` | `0.1` | Integral gain |
| `STREAM_PID_KD` | `0.05` | Derivative gain |

---

## Unified Exactly-Once Dedup, Finality, and Reconciliation (Issue #670)

Full design rationale: `docs/adr/0001-unified-idempotency-finality.md`.

### `pipeline/exactly_once.py`

Single canonical dedup/idempotency key (`DedupKey(source, external_id,
tenant_id)`) and two-phase `STAGED → COMMITTED` protocol
(`ExactlyOnceStore`), backed by either `RedisExactlyOnceBackend` (fail-closed
— raises `DedupBackendUnavailableError` on any Redis error, never silently
treats an outage as "not a duplicate") or `SqlExactlyOnceBackend` (durable,
for batch/offline use). `ingestion/trade_deduplicator.py` and
`streaming/kafka_worker.py`'s `DeduplicationCache` are both built on this.

### Finality

`RiskScoreRecord.finality` (migration `0006`) is `"provisional"` (default;
written by the continuous streaming/SSE path, which has no window-close
event) or `"final"` (written by a completed batch pipeline run or completed
stream-replay run over a closed, bounded time window). Set via
`RiskScoreStore.upsert(..., finality="final")`.

### Alert-delivery reconciliation

`streaming/alert_ledger.py::AlertDeliveryLedger` durably records the terminal
outcome of every alert-dispatch attempt that clears the threshold —
`delivered`, `dead_lettered`, or `suppressed_cooldown` — keyed through the
same `DedupKey` scheme (`source="alert_delivery"`). Wired into
`AlertDispatcher` via an optional `delivery_ledger` constructor argument
(passed by `scripts/stream.py` and `scripts/kafka_workers.py`; omitted by
default so existing callers see no behavior change).
`validation/reconciliation.py::reconcile_alert_delivery` traces every scored
wallet at or above threshold to a ledger entry, flagging scores with no
recorded outcome as a hard error — the "silently dropped alert" failure mode.

### Audit-trail restart safety

`detection/audit_trail.py::AuditMerkleChain` now persists leaf content
(`content_hash`, `prev_merkle_root` — migration `0005`) alongside the Merkle
root, and rehydrates `self._entries` from durable storage on every
construction. A routine process restart no longer makes `verify_chain()`
indistinguishable from real tampering.

## FeatureBuffer memory bound and eviction (Issue #901)

`streaming.FeatureBuffer` holds at most `FEATURE_BUFFER_MAX_WALLETS × max_trades`
trade records (default `100000 × 1000`; `FEATURE_BUFFER_MAX_WALLETS=0` disables the wallet cap).

**Eviction policy:** each wallet's deque drops its oldest trade once full. When a
new wallet arrives and the wallet cap is reached, the **least-recently-used**
wallet (by last `update`/`get_feature_row`) is evicted along with its dedup
index and Benford sketches.

**Metrics** (published by `FeatureBuffer.stats()`, called by `buffer_stats` logger):

| Metric | Meaning |
|---|---|
| `ledgerlens_feature_buffer_evictions_total` | LRU wallet evictions (use `rate()` for eviction rate) |
| `ledgerlens_feature_buffer_wallets` | Wallets currently buffered |
| `ledgerlens_feature_buffer_trades` | Trades currently buffered |
| `ledgerlens_feature_buffer_occupancy_ratio` | wallets / max_wallets |

**Tradeoffs:**
- *Tightening* the bound lowers memory and OOM risk, but evicted wallets lose
  history and Benford state — their next score is computed from a cold start and
  the dedup index no longer protects against redelivery of their old trades.
- *Loosening* the bound keeps more wallets warm (better feature quality) at the
  cost of memory proportional to `max_wallets × max_trades`.
- Sustained non-zero eviction rate with occupancy ≈ 1.0 means the working set
  exceeds the cap; raise it (with memory headroom) or shard by wallet.
