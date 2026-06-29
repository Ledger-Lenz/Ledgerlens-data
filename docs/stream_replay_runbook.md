# Stream Replay Runbook

## Overview

Stream replay enables backfilling risk scores for historical trade events when a new model version is deployed. The replay capability processes past Kafka events through the current model, updating stored risk scores and allowing consistent comparisons between model versions.

**Key Features:**
- Resume-capable: Interruptions don't lose progress (tracked via consumer group)
- Alert-safe: Replay mode suppresses alert dispatcher (no live alerts)
- Dry-run mode: Process and log scores without DB writes
- Explicit confirmation: Prevents accidental large-scale overwrites
- High throughput: 10x+ real-time processing speed

---

## When to Use Stream Replay

### Use Cases

1. **Model Version Deployment**
   - New ensemble model trained
   - Want consistent risk scores across all historical data
   - Compare new scores to old model predictions

2. **Model Parameter Tuning**
   - Adjusted feature thresholds or weights
   - Need to re-evaluate past wallets
   - Validate improvement in detection accuracy

3. **Bug Fix**
   - Fixed feature engineering bug
   - Need to correct scores for affected wallets
   - Ensure DB consistency with new code

4. **Data Correction**
   - Discovered issue in historical data
   - Re-scoring specific time window
   - Update derived metrics (risk propagation, rings)

### When NOT to Use

- **Real-time scoring**: Use live streaming pipeline
- **One-off wallet score**: Use `detection.model_inference.RiskScorer` directly
- **Bulk re-scoring without validation**: Always start with `--dry-run`

---

## How to Choose the Replay Window

### Considerations

1. **Time Range**
   - How far back do you need to go?
   - How much data is there? (estimate: 1M events ≈ 5-10 min replay time at 10x speed)
   - Do you have Kafka retention?

2. **Operational Impact**
   - Replay consumes Kafka brokers and DB connections
   - Run during low-traffic windows if possible
   - Limit to non-peak hours (e.g., 2 AM UTC)

3. **Validation**
   - Always do `--dry-run` first to estimate runtime
   - Compare sample scores before and after
   - Check for anomalies (all scores 0? data issue?)

### Recommended Approach

1. **Daily incremental replays**
   - Replay last 24 hours each day
   - Lower impact than batch replays
   - Keep scores fresher

2. **Weekly full replay**
   - One larger batch replay (e.g., 7 days)
   - Catch edge cases and long-tail effects
   - Off-peak hours only

3. **Event-driven replays**
   - After model deployment: replay last N days
   - After bug fix: replay affected time window only
   - After data correction: replay specific range

---

## Usage Examples

### Example 1: Dry-Run on Last 24 Hours

Test replay without writing to DB (recommended first step):

```bash
python -m scripts.replay_stream \
    --from-timestamp -86400 \
    --dry-run \
    --confirm
```

**Output:**
- Logs show predicted scores for each wallet
- No DB writes
- Runtime should be < 10 minutes for ~100k events

**What to check:**
- Are scores reasonable? (not all 0s or 100s)
- Are error rates acceptable?
- Does throughput look good?

### Example 2: Actual Replay with Confirmation Prompt

Replay last 7 days with interactive confirmation:

```bash
python -m scripts.replay_stream \
    --from-timestamp -604800 \
    --to-timestamp 0
```

**Flow:**
1. Script displays configuration
2. Prompts: "Confirm stream replay? (type 'yes' to proceed):"
3. User types `yes` to proceed
4. Replay begins, logs progress every 100 messages
5. Final summary shows total events, wallets scored, throughput

### Example 3: Specific Time Range with Explicit Confirm

Replay January 1-7, 2024 without prompt:

```bash
python -m scripts.replay_stream \
    --from-timestamp 1704067200 \
    --to-timestamp 1704672000 \
    --confirm
```

**Time range:**
- `1704067200`: Jan 1, 2024 00:00:00 UTC
- `1704672000`: Jan 8, 2024 00:00:00 UTC

### Example 4: Resume Interrupted Replay

If replay was interrupted (e.g., network issue), resume from last offset:

```bash
python -m scripts.replay_stream \
    --resume \
    --confirm
```

**Behavior:**
- Reads last committed offset from replay consumer group
- Resumes from that point
- Avoids re-processing already-handled events

### Example 5: Custom Kafka Broker

Replay against specific Kafka cluster:

```bash
python -m scripts.replay_stream \
    --bootstrap-servers kafka1.internal:9092,kafka2.internal:9092 \
    --from-timestamp -86400 \
    --confirm
```

---

## Monitoring Replay Progress

### Log Output

Replay logs to stdout with progress every 100 messages:

```
[INFO] Replay progress: 500 events, 250 scored, 1200.5 events/sec
[INFO] Replay progress: 1000 events, 500 scored, 1150.0 events/sec
[INFO] Replay completed: 2000 events processed, 1000 wallets scored, 
       1100.0 events/sec (1.8 min total)
```

### Interpretation

- **events**: Total Kafka messages consumed
- **scored**: Wallets that met min_trades threshold and were scored
- **events/sec**: Throughput (target: >1000 for 10x real-time)
- **total time**: Wall-clock duration

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Very low throughput (<100 events/sec) | Kafka lag or slow model inference | Check Kafka cluster health, reduce batch size |
| All scores 0 | Features not computed | Check FeatureBuffer for errors, ensure model loaded |
| Out of memory | Too many events in FeatureBuffer | Reduce buffer max_trades or replay smaller window |
| DB locked errors | SQLite concurrency issue | Use PostgreSQL for production, or replay during off-peak |

---

## Comparing Replay Scores to Live Scores

After replay completes, validate changes:

### Manual Spot Check

```python
from detection.risk_score_store import RiskScoreStore

store = RiskScoreStore()

# Check a specific wallet
wallet = "GABC123..."
pair = "USDC:GA.../XLM:native"

record = store.get(wallet, pair)
print(f"Score: {record.score}")
print(f"Benford flag: {record.benford_flag}")
print(f"ML flag: {record.ml_flag}")
print(f"Updated at: {record.updated_at}")
```

### Bulk Comparison

Export before/after scores and compare:

```bash
# Before replay (backup current scores)
sqlite3 ledgerlens.db "SELECT wallet, asset_pair, score FROM risk_scores LIMIT 100" > before.csv

# Run replay

# After replay (export new scores)
sqlite3 ledgerlens.db "SELECT wallet, asset_pair, score FROM risk_scores LIMIT 100" > after.csv

# Compare
diff before.csv after.csv
```

### Statistical Validation

```python
import pandas as pd

before = pd.read_csv('before.csv')
after = pd.read_csv('after.csv')

# Merge on (wallet, pair)
comparison = before.merge(after, on=['wallet', 'asset_pair'], suffixes=('_before', '_after'))

# Score difference distribution
comparison['score_delta'] = comparison['score_after'] - comparison['score_before']
print(comparison['score_delta'].describe())

# How many wallets had significant changes?
print((comparison['score_delta'].abs() > 10).sum(), "wallets changed >10 points")

# New flags triggered?
print((comparison['benford_flag_after'] & ~comparison['benford_flag_before']).sum(), "new Benford flags")
```

---

## Architecture: How Replay Works

```
Historical Kafka Topic
    ↓
Seek to timestamp/offset (--from-timestamp)
    ↓
StreamReplayer Consumer (group: ledgerlens-replay)
    ├─ Poll batches of messages
    ├─ Reconstruct Trade objects
    ├─ Update FeatureBuffer (per-wallet history)
    ├─ Score wallets via StreamingScorer
    └─ Store scores with replay_model_version tag
    ↓
Risk Score DB (persisted)
    ├─ (wallet, asset_pair) → score, benford_flag, ml_flag, confidence
    └─ + replay_model_version tag for audit
```

### Key Design Choices

1. **No-Op Alert Dispatcher**: Replay never sends live alerts
   - Prevents alert fatigue from historical scores
   - Distinguishes replay from live scoring

2. **Separate Consumer Group**: `ledgerlens-replay`
   - Tracks replay offsets independently
   - Allows resume without losing progress
   - Live scoring unaffected

3. **Replay Tag in DB**: `replay_model_version = "replay"`
   - Audit trail: know scores came from replay
   - Comparison: query before/after separately
   - Future: support multiple model versions

---

## Performance Tuning

### Throughput Target

**Goal**: 10x real-time = 10 events/sec × 10 = 100 events/sec actual

For typical streaming:
- 1 hour of events ≈ 36k trade messages
- Target replay time: < 6 minutes
- Target throughput: > 100 events/sec

### If Replay is Too Slow

1. **Check Kafka**: brokers healthy? Lag acceptable?
   ```bash
   kafka-consumer-groups.sh --group ledgerlens-replay --describe
   ```

2. **Profile model inference**: Which scorer is slow?
   - Benford computation
   - ML model inference (RandomForest, XGBoost, LightGBM)
   - Feature engineering

3. **Reduce batch size**: If memory-bound
   - Current: 100 events/batch
   - Tune `max_records` in KafkaConsumer.poll()

4. **Parallelize**: Run multiple replays on different partitions
   - Manual: `--group replay-partition-0`, `--group replay-partition-1`
   - Assign each to subset of partitions

---

## Safety Checklist

Before running replay in production:

- [ ] **Backup DB**: `sqlite3 ledgerlens.db ".backup backup.db"`
- [ ] **Start with `--dry-run`**: Validate logic without side effects
- [ ] **Review replay window**: Does time range make sense?
- [ ] **Check Kafka retention**: Is data still available?
- [ ] **Schedule off-peak**: Replay during low-traffic windows
- [ ] **Monitor progress**: Check logs for errors
- [ ] **Validate results**: Spot-check scores, compare before/after
- [ ] **Document changes**: Note replay date/time and model version
- [ ] **Notify stakeholders**: API team, alerting team, operations

---

## Example Runbook: Deploy New Model

### Step 1: Train and Validate Model
```bash
python -m detection.model_training --data-path data/labeled_2024.parquet
# Models saved to ./models/
```

### Step 2: Dry-Run Replay (Last 7 Days)
```bash
python -m scripts.replay_stream \
    --from-timestamp -604800 \
    --dry-run \
    --confirm
# Verify scores, throughput, no errors
# Expected: 1-2 hours, no DB writes
```

### Step 3: Sample Validation
```python
# Spot-check 5-10 wallets
from detection.risk_score_store import RiskScoreStore
store = RiskScoreStore()

sample_wallets = ["GABC123...", "GXYZ789...", ...]
for wallet in sample_wallets:
    record = store.get(wallet, "USDC:GA.../XLM:native")
    print(f"{wallet}: score={record.score}")
```

### Step 4: Full Replay (Last 30 Days)
```bash
python -m scripts.replay_stream \
    --from-timestamp -2592000 \
    --confirm
# Runs during off-peak (2 AM UTC)
# Estimated runtime: 4-6 hours
```

### Step 5: Post-Replay Validation
```bash
# Check final scores
sqlite3 ledgerlens.db "SELECT COUNT(*) FROM risk_scores WHERE score >= 70"

# Monitor for anomalies
sqlite3 ledgerlens.db "SELECT COUNT(*) FROM risk_scores WHERE score = 0"
```

### Step 6: Deploy to Production
```bash
# If validation passed:
# 1. Deploy updated model code
# 2. Restart live streaming pipeline
# 3. Monitor alerts for expected patterns
```

---

## Troubleshooting Guide

### Replay Never Completes

**Symptom**: Script runs but never reaches "Replay completed"

**Diagnosis**:
```bash
# Check Kafka lag
kafka-consumer-groups.sh --group ledgerlens-replay --describe

# Check broker connectivity
kafka-topics.sh --list --bootstrap-server localhost:9092
```

**Solutions**:
- Reduce `--from-timestamp` (smaller time window)
- Increase Kafka consumer timeouts in code
- Check DB connection pooling limits

### Low Throughput (< 50 events/sec)

**Likely causes**:
1. Model inference too slow: Check which model/feature is bottleneck
2. DB lock contention: Use WAL mode or PostgreSQL
3. Kafka broker lag: Check broker CPU/memory

**Fix**:
```bash
# Profile model inference
time python -c "
from detection.model_inference import RiskScorer
from streaming.feature_buffer import FeatureBuffer
# Time score_wallet()
"
```

### DB Locked Error (SQLite Only)

**Error**: `database is locked`

**Root cause**: SQLite single-writer limitation

**Solutions** (in order):
1. Enable WAL mode (already done in code)
2. Increase timeout: Edit `detection/persistence.py` timeout
3. Replay during completely off-peak hours
4. Switch to PostgreSQL for production

### Out of Memory

**Symptom**: Replay crashes with OOM error

**Cause**: FeatureBuffer too large (default 1000 trades/wallet)

**Fix**: Reduce buffer size or replay smaller window
```python
# In replay_stream.py
buffer = FeatureBuffer(max_trades=100)  # Reduce from 1000
```

---

## FAQ

**Q: Can I replay while live pipeline is running?**
A: Yes, but with caveats. Live and replay use different consumer groups, so no conflicts. However, competing for DB writes might cause slowdowns. Recommended: replay during off-peak.

**Q: What if Kafka is deleted the events?**
A: Replay fails silently (no events to replay). Use `--dry-run` first to validate data exists.

**Q: Can I replay multiple models at once?**
A: No, only current model version is active. For A/B testing, save replay tag and compare separately.

**Q: How long should I keep replays in DB?**
A: Use `replay_model_version` tag to audit. Consider archiving old replays after validation.

**Q: Does replay affect live alerts?**
A: No, replay uses no-op dispatcher. Live alerts continue unaffected.

---

## Related Documentation

- [Stream Architecture](streaming_architecture.md) - Kafka partitioning and workers
- [Model Training](../detection/model_training.py) - How to train new model versions
- [Risk Score Store](../detection/risk_score_store.py) - DB schema and queries
- [Feature Engineering](../detection/feature_engineering.py) - Feature computation

