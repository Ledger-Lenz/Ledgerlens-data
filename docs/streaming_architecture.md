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

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `topic` | — | Kafka topic |
| `group_id` | — | Consumer group (e.g., `ledgerlens-workers`) |
| `bootstrap_servers` | `localhost:9092` | Kafka brokers |
| `buffer` | new FeatureBuffer() | Per-worker trade buffer |
| `scorer` | — | StreamingScorer (required) |
| `dispatcher` | — | AlertDispatcher (required) |
| `partitions` | None | Explicit partition list (optional; uses group assignment if None) |
| `commit_interval_seconds` | 30 | Offset commit frequency |

| Method | Purpose |
|--------|---------|
| `run()` | Start consuming; blocks until SIGTERM/SIGINT |
| `_process_batch(messages_by_partition)` | Process batch of messages |
| `_process_message(payload)` | Process single trade, score wallets |
| `_commit_offsets()` | Manually commit offsets |
| `stop()` | Signal worker to stop |

**Rebalancing:**
- Kafka's consumer group protocol handles partition reassignment
- On revocation: `_commit_offsets()` called to preserve progress
- New worker resumes from committed offset
- No data loss or duplication (exactly-once semantics)

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
│     │    buffer.update(trade)
│     │    score_wallet(wallet) → dispatch()
│     └─ Commits offsets every 30s or on rebalance
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

## Security Notes

- **Partition keys**: validated against canonical format before production
  - Invalid pairs rejected at source (no invalid data in Kafka)
  - Malformed pairs → dead-letter queue for audit
- **Offset commits**: manual commit after successful message processing
  - On rebalance: offsets committed before partition revocation
  - Exactly-once semantics maintained
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

### At-least-once semantics

* Consumers run with `enable.auto.commit=false`.
* `KafkaWorker.process_message` commits a message's offset **only after** the
  scorer and `AlertDispatcher.dispatch` have completed for that message.
* If `dispatch` raises, the offset is left uncommitted; the message is
  redelivered after the next restart/rebalance. Duplicate alerts are absorbed
  by the dispatcher's per-wallet cooldown.

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

## Streaming Account Metadata Join (Phase 4)

### Overview

Trade events carry wallet IDs but not account metadata (account age, trust
lines, signer configuration). Wallet-graph features that depend on account age
or trust-line configuration were previously computed from a static snapshot
fetched at pipeline startup. When account state changes between snapshot
refreshes — for example, when a trust line is added to enable trading a new
asset pair — those features silently lag behind reality until the next snapshot.

Phase 4 closes this gap with a **streaming metadata join**: the pipeline
subscribes to the Stellar Horizon
`/accounts/{wallet}/effects` SSE endpoint for every wallet it sees trading.
Incoming effect events are validated and fed into a `MetadataJoinState` store.
Each trade event then enriches the involved wallets' features from the latest
in-window metadata entry before scoring.

### Architecture

```
Stellar Horizon
  /accounts/{wallet}/effects (SSE)
          │
          │ raw effect records
          ▼
  ┌─────────────────────────────────┐
  │  AccountMetadataStream           │  streaming/account_metadata_stream.py
  │  (one daemon thread per wallet)  │
  │                                  │
  │  _stream_effects()               │
  │   → filter to interesting types  │
  │   → validate_metadata_event()    │  Pydantic validation / discard
  │   → on_update callback           │  (or Kafka produce)
  └──────────────┬──────────────────┘
                 │ AccountMetadataUpdate
                 ▼
  ┌─────────────────────────────────┐
  │  MetadataJoinState               │  streaming/pipeline.py
  │  (thread-safe, bounded)          │
  │                                  │
  │  apply_update()                  │  stores update in _active or _pending
  │  get_metadata(wallet)            │  returns metadata if within join window
  │  evict_inactive_wallets()        │  background housekeeping (every 5 min)
  └──────────────┬──────────────────┘
                 │
  ┌──────────────▼──────────────────┐
  │  StreamingPipeline               │
  │  _stream_pair() / AMM loop       │
  │                                  │
  │  for each trade event:           │
  │    buffer.update(trade)          │
  │    _enrich_from_metadata(wallet) │ ← get_metadata() + buffer.apply_metadata()
  │    scorer.score_wallet(wallet)   │
  │    dispatcher.dispatch(…)        │
  └─────────────────────────────────┘
```

### Join State

`MetadataJoinState` is an in-memory, thread-safe dictionary of the latest
`AccountMetadataUpdate` per wallet:

```python
# Apply an incoming Horizon effect to join state:
join_state.apply_update(update)

# Retrieve metadata for a wallet when a trade arrives:
metadata = join_state.get_metadata(wallet)
if metadata:
    buffer.apply_metadata(wallet, metadata)
```

#### Join Window

A metadata entry enriches trade events for `METADATA_JOIN_WINDOW_SECONDS`
(default `3600`, configurable) after the update arrives. When the window
expires the entry is evicted from active state; the next metadata update for
the wallet is admitted as a new entry.

**Why a join window?**  Account state can change at any time; keeping metadata
indefinitely would mean scoring wallets with arbitrarily stale entries.  A 1-
hour window strikes a balance between freshness and the cost of re-subscribing
wallets after a Horizon reconnect.  For wallets that trade continuously, new
metadata updates typically arrive more frequently than the window, so there is
no coverage gap in practice.

#### Late-Arrival Handling

A metadata update that arrives *while no active entry exists* for a wallet (the
wallet is new, or its previous entry expired) is stored in `_pending_updates`
rather than being discarded. On the next `get_metadata()` call for that wallet
— which is triggered by the next trade event involving the wallet — the pending
update is **promoted** to active state and its lag is recorded in the
`metadata_join_lag_seconds` Prometheus histogram. This ensures no metadata
update is ever silently dropped.

```
Timeline:                                  is_active?  pending?
───────────────────────────────────────────────────────────────
t=0   trade arrives for wallet W           no           no
      get_metadata(W) → None, records W in last_trade_at

t=10  Horizon effect for W arrives         no           no
      apply_update(update)                 → queued as pending
                                           no           yes

t=15  trade arrives for W (next cycle)
      get_metadata(W):
        • pending entry found → promoted   yes          no
        • METADATA_JOIN_LAG.observe(5s)
        → returns update

t=3615 join window expires
        get_metadata(W) → None (evicted)   no           no
```

#### Memory Bounding

Join state is bounded at O(active wallets). A background daemon thread
(`_run_housekeeping`) runs every 300 seconds and calls
`evict_inactive_wallets()`. Any wallet whose last trade event was more than
`METADATA_ACTIVE_WALLET_TTL_SECONDS` ago (default `86400` / 24 hours) is
removed from both `_active` and `_pending_updates`. High-frequency wallets
(those with a trade in the last 24 hours) are never evicted prematurely — their
`_last_trade_at` timestamp is refreshed on every `get_metadata()` call.

This mirrors the same eviction strategy as Kafka Streams' [windowed state
stores](https://kafka.apache.org/documentation/streams/developer-guide/dsl-api.html#windowing)
but is implemented entirely in Python using a monotonic clock and a periodic
sweep, avoiding a JVM dependency. The trade-off versus time-to-live on
individual entries is that a brief spike in unique wallets (e.g., a burst of
new accounts during a token launch) can temporarily increase memory usage
before the next housekeeping pass clears them. Operators can reduce
`METADATA_ACTIVE_WALLET_TTL_SECONDS` to reclaim memory more aggressively at
the cost of re-fetching metadata for returning wallets.

### Security

All effect records from Horizon are validated through
`validate_metadata_event()` before being admitted to join state:

* `account_id` must match the Stellar account ID format (`G` followed by 55
  base-32 characters).
* `effect_type` must be a non-empty string.
* Any `ValidationError` raised by Pydantic causes the record to be **discarded
  and logged at WARNING level**. Malformed records never reach `MetadataJoinState`
  or the scorer.

This follows the same defense-in-depth pattern as the Kafka dead-letter queue
for trade messages: untrusted data is validated at the ingestion boundary and
silently dropped on failure so a single malformed Horizon response cannot
corrupt join state or trigger a pipeline crash.

### Prometheus Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `metadata_join_lag_seconds` | Histogram | — | Seconds between a metadata update arriving and the affected wallet being re-scored with the new metadata. Observed when a pending update is promoted to active state. |

Typical lag distribution:
- **< 5 s**: metadata arrived and was immediately promoted on the next trade event.
- **5–60 s**: trade activity is moderate (1 trade per minute).
- **> 60 s**: wallet is trading infrequently; the pending update sits until the
  next trade arrives.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `METADATA_TOPIC` | `ledgerlens.account_metadata` | Kafka topic for account metadata update events (used when `STREAMING_BACKEND=kafka`) |
| `METADATA_JOIN_WINDOW_SECONDS` | `3600` | How long (seconds) a metadata entry enriches trade events before expiring |
| `METADATA_ACTIVE_WALLET_TTL_SECONDS` | `86400` | Wallets inactive for this many seconds are evicted from join state |

### Files

| File | Purpose |
|---|---|
| `streaming/account_metadata_stream.py` | Horizon effects subscriber; validates events; produces to Kafka in `kafka` mode |
| `streaming/pipeline.py` — `MetadataJoinState` | Stateful join store with join window, late-arrival pending queue, and housekeeping eviction |
| `streaming/pipeline.py` — `StreamingPipeline` | Wires `_enrich_from_metadata`, `_subscribe_wallet`, `_trigger_rescore_from_metadata`, `_run_housekeeping` |
| `tests/test_metadata_join.py` | Unit tests: validation, join state, late-arrival, window expiry, eviction, histogram, pipeline enrichment |

### Threading Model with Metadata Join

```
Main thread (scripts/stream.py)
│  installs SIGINT → _stop_event.set()
│
├── Thread: pair-0 (daemon)  → _stream_pair(USDC/XLM)
│     for trade in stream_trades():
│         buffer.update(trade)
│         _enrich_from_metadata(base)   ← looks up MetadataJoinState
│         _subscribe_wallet(base)        ← adds to AccountMetadataStream
│         scorer.score_wallet(base)
│         dispatcher.dispatch(base, …)
│         _enrich_from_metadata(counter)
│         …
│
├── Thread: metadata-housekeeping (daemon)
│     every 300 s: MetadataJoinState.evict_inactive_wallets()
│
├── Thread: metadata-<WALLET_A[:8]> (daemon, per wallet)
│     AccountMetadataStream._subscribe(WALLET_A)
│       → Horizon SSE /accounts/WALLET_A/effects
│       → validate_metadata_event(record)
│       → on_update(AccountMetadataUpdate) → MetadataJoinState.apply_update()
│
└── Thread: metadata-<WALLET_B[:8]> (daemon, per wallet)
      AccountMetadataStream._subscribe(WALLET_B)
      …
```

### Join Window Trade-offs

| Window duration | Freshness | Memory pressure | Re-subscription cost |
|---|---|---|---|
| **Short (< 300 s)** | High — stale entries evicted quickly | Low | High — effects stream re-opened frequently for wallets with low trade rates |
| **Medium (3600 s, default)** | Good — covers most intra-hour account changes | Moderate — one entry per active wallet | Low — one SSE connection per wallet, reconnects only on failure |
| **Long (> 86400 s)** | Low — stale entries persist for days | High — proportional to all wallets ever seen | Minimal | 

The default 1-hour window is chosen to match typical DEX trading sessions
while remaining well within Stellar's 3–5 second ledger close cycle. Operators
running high-frequency monitoring against thin markets (where individual trades
are minutes apart) should increase `METADATA_JOIN_WINDOW_SECONDS` to match
their expected inter-trade interval so wallets are not prematurely evicted
between trades.
