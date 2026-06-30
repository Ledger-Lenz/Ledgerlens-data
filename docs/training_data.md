# Training Data Management

## Reservoir Sampler

LedgerLens uses a `DriftAwareReservoirSampler` to maintain a fixed-size training
buffer that adapts to evolving Stellar DEX trading patterns without unbounded
storage growth.

### Architecture

```
Streaming trade data
        │
        ▼
DriftAwareReservoirSampler
        │
        ├─ Stable mode: Random replacement (Algorithm R)
        │
        └─ Drift mode: Recency-biased replacement
        │
        ▼
data/reservoir.parquet (persisted buffer)
        │
        ▼
Mini-batch training (sample(n))
```

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `RESERVOIR_SIZE` | 10000 | Maximum number of examples in the buffer |
| `RESERVOIR_FLUSH_INTERVAL` | 1000 | Write to disk every N updates |

### Drift-Biased Replacement Logic

The sampler operates in two modes based on the CUSUM detector alarm state:

1. **Stable Mode (alarm = False)**: Standard reservoir sampling (Algorithm R)
   - Each incoming example has equal probability of replacing any existing entry
   - Maintains uniform random sample from the stream history

2. **Drift Mode (alarm = True)**: Recency-biased replacement
   - When drift is detected (via CUSUM in `monitoring/cusum_detector.py`), incoming
     examples preferentially replace older entries
   - Replacement probability is proportional to entry age
   - Older entries have probability `age/max_age`, newer entries have probability
     `(age + max_age)/max_age`
   - This rapidly incorporates new patterns while gracefully aging out old data

### Usage

```python
from data.reservoir_sampler import DriftAwareReservoirSampler

# Create sampler with defaults
sampler = DriftAwareReservoirSampler()

# Add examples
sampler.update(
    example={"feat_a": 0.5, "feat_b": 1.2, "label": 1},
    timestamp=1234567890.0
)

# Sample mini-batch for training
batch = sampler.sample(32)  # Returns DataFrame with 32 examples

# Manual reset (clears buffer and removes persisted file)
sampler.reset()
```

### Triggering Manual Reservoir Reset

To reset the reservoir buffer manually:

```python
sampler = DriftAwareReservoirSampler()
sampler.reset()  # Clears in-memory buffer and removes data/reservoir.parquet
```

This is useful when:
- Deploying a major model update and wanting fresh training data collection
- Recovering from corrupted persisted state
- Starting a new training cycle with a clean slate

### Persistence Format

The reservoir is stored as `data/reservoir.parquet` with columns:
- All feature columns from the examples
- `timestamp`: Unix timestamp for recency weighting

Atomic writes ensure the file is never left in a partially-written state.
The temp file (`*.parquet.tmp`) is written first, then renamed to the target
path only after successful completion.