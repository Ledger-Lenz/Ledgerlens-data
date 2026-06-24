# LedgerLens Real-Time Streaming Architecture

This document describes the end-to-end real-time detection pipeline introduced
in Issues #012 (Phase 1) and #013 (Phase 2).  It covers every component, the
data flow between them, threading model, alert delivery channels, and the
security constraints applied to the WebSocket server.

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
