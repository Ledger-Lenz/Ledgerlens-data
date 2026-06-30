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

## Distributed Tracing with OpenTelemetry (Issue #198)

### Overview

LedgerLens instruments the full trade event pipeline with OpenTelemetry (OTel)
spans so a single trade can be traced end-to-end from Horizon ingestion through
to the stored risk score in Grafana Tempo or Jaeger.

### Architecture

```
Stellar Horizon SSE
       │
       │  [span: trade_event.received]   ingestion/horizon_streamer.py
       ▼
Kafka message (W3C TraceContext headers injected)
       │
       │  [span: features.computed]      detection/feature_engineering.py
       ▼
       │  [span: benford.computed]       detection/benford_engine.py
       ▼
       │  [span: model.scored]           detection/model_inference.py
       ▼
       │  [span: score.stored]           detection/risk_score_store.py
       ▼
   risk_scores DB
```

All spans belong to the same trace thanks to W3C TraceContext propagation
across the Kafka message boundary (see next section).

### Kafka trace context propagation

The producer side injects trace headers via `utils.tracing.inject_trace_context`:

```python
from utils.tracing import inject_trace_context
headers: dict[str, str] = {}
inject_trace_context(headers)
producer.send(topic, value=payload, headers=list(headers.items()))
```

The consumer side extracts the context and uses it as the parent span:

```python
from utils.tracing import extract_trace_context, get_tracer
ctx = extract_trace_context(dict(msg.headers))
tracer = get_tracer(__name__)
with tracer.start_as_current_span("features.computed", context=ctx):
    ...
```

### Span attributes

| Span | Key attributes |
|---|---|
| `trade_event.received` | `trade.id` (hashed), `trade.base_asset`, `trade.counter_asset` |
| `features.computed` | `wallet.id` (hashed), `trade.count` |
| `benford.computed` | `benford.windows_count`, `benford.sample_size` |
| `model.scored` | `wallet.id` (hashed), `model.score` |
| `score.stored` | `wallet.id` (hashed), `score.value` |

**Security**: raw wallet addresses and trade amounts are never included in span
attributes.  All wallet/trade identifiers are passed through
`utils.tracing.hash_span_id()` (SHA-256, truncated to 16 hex chars) before
being set as span attributes.

### Sampling strategy

The default sampling rate is **10 %** (`OTEL_SAMPLING_RATE=0.1`), set via
`ParentBased(TraceIdRatioBased(rate))`.  This means:

- A sampled parent span causes all child spans to be sampled (parent-based).
- 10 % of root spans are selected by ratio sampling.

For high-throughput production deployments, reduce to 1 % (`OTEL_SAMPLING_RATE=0.01`).
For debugging, set to 100 % (`OTEL_SAMPLING_RATE=1.0`).

### Querying traces in Grafana Tempo

1. Open Grafana → Explore → select the Tempo data source.
2. Search by `service.name = ledgerlens` and filter by span name or wallet hash.
3. Use the trace ID returned in API responses (if propagated to HTTP headers)
   to jump directly to the full trace waterfall.

Alternatively, use Jaeger UI at `http://localhost:16686` (when running the
local docker-compose stack with `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317`).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP gRPC endpoint |
| `OTEL_SAMPLING_RATE` | `0.1` | Fraction of traces sampled (0.0–1.0) |
