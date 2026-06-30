# Load Testing the LedgerLens Streaming Pipeline

This document explains how to run the load test harness, interpret its output,
and understand the pass/fail thresholds.

---

## Background

The Stellar DEX processed approximately **5 million trades in its busiest
24-hour period** to date. A 10× peak scenario translates to roughly
**580 trades/second sustained**, with bursts up to **5 000 trades/second**.
The load test harness verifies the streaming pipeline can handle these rates
without violating the latency and memory constraints required for real-time
fraud detection.

---

## Architecture of the Load Driver

The driver is implemented as a custom **asyncio coroutine loop** rather than
Locust or k6, for the following reasons:

| Tool | Why not chosen |
|---|---|
| Locust | HTTP-oriented; no native Avro/Kafka support |
| k6 | Node.js runtime; cannot instrument Python pipeline internals directly |
| Custom asyncio | Shares the same runtime as the pipeline, enables direct `FeatureBuffer` measurement, trivially unit-testable `TokenBucket` |

The `TokenBucket` rate limiter enforces the target throughput with sub-millisecond
precision. A linear ramp phase prevents burst-start artifacts from inflating
early-latency percentiles.

---

## Running the Load Test

### Prerequisites

```bash
# Install dependencies (mutmut and psutil are already in requirements.txt)
make install

# Optional: ensure psutil is available for memory sampling
pip install psutil
```

### Quick smoke test (no Kafka broker needed)

```bash
python scripts/load_test_pipeline.py \
    --rate 500 \
    --duration 60 \
    --ramp-time 10 \
    --no-kafka
```

### Full test against a live Kafka broker

```bash
# Start a local broker (Docker):
# docker run -p 9092:9092 apache/kafka:3.7.0

python scripts/load_test_pipeline.py \
    --rate 500 \
    --duration 120 \
    --ramp-time 30 \
    --bootstrap-servers localhost:9092
```

### All supported rate targets

```bash
for RATE in 10 100 500 1000 5000; do
  python scripts/load_test_pipeline.py \
      --rate $RATE \
      --duration 60 \
      --ramp-time 15 \
      --no-kafka \
      --output "reports/load_${RATE}tps.json"
done
```

### Makefile shortcut

```bash
# Default (500 tps, 120s, in-process):
make load-test

# Custom rate and Kafka broker:
make load-test LOAD_RATE=1000 LOAD_KAFKA=1

# CI mode — fail if thresholds not met:
RUN_LOAD_TESTS=1 make load-test
```

The `load-test` target is **disabled by default** in CI. Set `RUN_LOAD_TESTS=1`
in the environment or as a Make variable to enable it.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--rate` | `500` | Target trades/second (sustained after ramp) |
| `--duration` | `120` | Total test duration in seconds (including ramp) |
| `--ramp-time` | `30` | Linear ramp from 1 tps → target over this many seconds |
| `--no-kafka` | off | Drive in-process `FeatureBuffer` only; no broker required |
| `--bootstrap-servers` | `localhost:9092` | Kafka bootstrap servers |
| `--topic-prefix` | `ledgerlens.trades` | Kafka topic prefix |
| `--seed` | `42` | RNG seed for reproducible synthetic data |
| `--output` | `reports/load_test_results.json` | Path for the JSON report |
| `--fail-on-threshold` | off | Exit 1 if pass/fail criteria are not met |

---

## Interpreting the Results

A results file is written to `--output` after every run. Example:

```json
{
  "results": {
    "summary": {
      "total_sent": 56432,
      "total_acked": 56432,
      "throughput_tps": 470.27,
      "benford_throughput_tps": 470.27
    },
    "latency_s": {
      "p50":  0.0003,
      "p95":  0.0011,
      "p99":  0.0024,
      "p999": 0.0089,
      "max":  0.0143
    },
    "memory_mb": { "p50": 142.3, "p99": 158.7, "max": 162.1 }
  }
}
```

### Latency

| Percentile | Meaning |
|---|---|
| p50 | Median processing time per trade |
| p95 | 95% of trades processed faster than this |
| p99 | Pass/fail threshold (must be < 10 s at ≥ 500 tps) |
| p99.9 | Tail latency — occasional Benford window recomputation spikes |

In `--no-kafka` mode, latency is measured from `FeatureBuffer.update()` call to
`StreamingScorer.score_wallet()` return. In Kafka mode, it is measured from
`Producer.produce()` call to the delivery callback confirming the broker acked
the message.

### Memory

Sampled from the current process RSS every 2 seconds using `psutil`.
Values are in megabytes. The threshold is **1 024 MB per worker process**.

### Kafka consumer lag

Polled from the broker admin API every 5 seconds when `--no-kafka` is not set.
High lag (above `KAFKA_LAG_ALERT_THRESHOLD = 500`) indicates the worker is
falling behind the producer and will cause growing end-to-end latency.

### Benford throughput

Approximated as `total_acked / duration`. This represents the number of trades
per second flowing through the `FeatureBuffer` → `StreamingBenfordSketch` path.
At 500 tps with 5 Benford windows, this is ~2 500 window-updates per second.

---

## Pass/Fail Thresholds

| Criterion | Threshold | Enforced at |
|---|---|---|
| p99 end-to-end latency | < 10 seconds | ≥ 500 tps |
| Worker process RSS | < 1 024 MB | all rates |

### Rationale

**10 s p99 latency at 500 tps:**
Stellar ledger close time is ~5 seconds. A fraud alert must arrive before the
next ledger close to be actionable (e.g., to submit an on-chain score before
the next transaction settles). A 10 s p99 provides a 2× safety margin over
the median ledger interval. Below 500 tps the latency check is informational
only because the pipeline is not under meaningful load at those rates.

**1 GB memory per worker:**
A single `FeatureBuffer` holding 1 000 trades × 10 000 wallets with all
Benford sketches is estimated at ~300 MB. The 1 GB limit provides a 3×
headroom for GNN embeddings, model inference caches, and OS overhead, while
remaining within standard container resource limits (2 GB per pod).

---

## Comparing Results Across Runs

The JSON report includes a `meta.parameters` block with all CLI flags so runs
are fully reproducible. To compare two runs:

```bash
python - <<'EOF'
import json, sys

a = json.load(open("reports/load_500tps_before.json"))
b = json.load(open("reports/load_500tps_after.json"))

for pct in ("p50", "p95", "p99"):
    before = a["results"]["latency_s"][pct]
    after  = b["results"]["latency_s"][pct]
    delta  = (after - before) / before * 100
    print(f"  latency {pct}: {before:.4f}s → {after:.4f}s  ({delta:+.1f}%)")
EOF
```

A > 20% regression in p99 at the same rate warrants investigation before merge.

---

## Security

- All wallet addresses use the `GLOAD` prefix and are generated deterministically
  from `hashlib.sha256(f"load-test-wallet-{idx}")`. They are not valid Stellar
  accounts and will never appear in production data.
- Trade amounts are log-uniform random numbers with no connection to real order
  sizes or account balances.
- The load test never reads from, writes to, or queries any production Kafka
  topic, database, or Horizon endpoint.
- No real wallet activity is ever exposed in test infrastructure.
