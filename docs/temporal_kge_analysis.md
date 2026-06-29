# Temporal Knowledge Graph Embedding for Evolving Wash Trade Ring Detection

## Overview

Static knowledge graph embeddings (TransE, RotatE) capture entity relationships at a single point in time but miss the temporal evolution of wash trade rings: new participants join, funding sources change, and rings evolve their trading patterns in response to detection pressure.

**Temporal Knowledge Graph Embedding (TComplEx)** extends standard KGE with time-aware relation embeddings, enabling the model to reason about *when* relationships hold, not just *whether* they hold. This enables prediction of future trading relationships and identification of emerging wash-trading coalitions.

## The Problem: Static vs Temporal Graphs

### Static KGE (TransE/RotatE)
- Captures: Wallet A traded with Wallet B
- Misses: When did this trading pattern emerge? Are new participants joining?
- Risk: False negatives when rings evolve faster than detection models retrain

### Temporal KGE (TComplEx)
- Captures: Wallet A traded with Wallet B in hour T, T+1, T+2 (temporal bins)
- Enables: Prediction of future collaboration (e.g., if C traded with D in hour T, predict C-D in T+1)
- Advantage: Detects emerging coalitions and predicts ring expansion before it manifests

## Architecture

### Temporal Knowledge Graph Construction

**Triples** are (wallet_A, traded_with, wallet_B, timestamp_bin):

```
(wallet_A, traded_with, wallet_B, hour_2024_01_01_12)
(wallet_A, traded_with, wallet_B, hour_2024_01_01_13)
(wallet_B, traded_with, wallet_C, hour_2024_01_01_13)
```

**Binning**: Timestamps are quantized to 1-hour intervals (configurable via `TEMPORAL_BINNING_HOURS`):
- 2024-01-01 12:00-12:59 → bin 2024-01-01 12:00
- 2024-01-01 13:00-13:59 → bin 2024-01-01 13:00

Advantages:
- Reduces cardinality (360+ hourly bins vs millions of microsecond timestamps)
- Aligns with market-making behavior patterns (hourly activity profiles)
- Matches retraining cadence (models trained daily)

### TComplEx Model

**Why TComplEx over TNTComplEx?**

| Model | Scoring | Parameters | Use Case |
|---|---|---|---|
| **TComplEx** | S(h,r,t,τ) = Re(⟨h, r, t̄⟩ + ⟨r, τ, h̄⟩) | Simple; O(d·n·T) | Hourly financial bins |
| **TNTComplEx** | Learned temporal decay + complex rotations | Heavy; O(d·n·T²) | Microsecond precision |

**Choice: TComplEx** for hourly-binned financial data
- Sufficient temporal granularity for wash-trading (ops happen within hours)
- Linear scaling vs quadratic in transaction count
- Faster inference (< 5ms per link prediction)
- Simpler hyperparameter tuning

### TComplEx Scoring Function

```
S(h, r, t, τ) = Re(⟨h ⊙ r, t̄⟩ + ⟨r ⊙ τ, h̄⟩)
```

Where:
- h, r, t are complex-valued embeddings
- τ is the time-dependent relation embedding
- ⊙ is Hadamard product
- ̄ denotes complex conjugate
- Re() extracts real part

**Interpretation**:
- First term: compatibility of (head, relation, tail) in feature space
- Second term: temporal modulat ion based on relation + time
- High score → likely link exists at that time

## Feature: `temporal_kge_collab_score`

### Definition

**Feature Name**: `temporal_kge_collab_score`

**Type**: Float in range [0.0, 1.0]

**Description**: Maximum predicted collaboration likelihood for a wallet across its historical counterparties, as computed by TComplEx link prediction.

### Computation

For a wallet W with counterparties [C1, C2, ..., Cn]:

```python
collab_scores = [
    predict_collaboration_score(W, C1, target_time=T+1),
    predict_collaboration_score(W, C2, target_time=T+1),
    ...
]
temporal_kge_collab_score = max(collab_scores)  # Highest risk
```

Where `predict_collaboration_score(W, C, T)` returns the TComplEx link prediction score for triple (W, traded_with, C, T).

### Wash-Trading Signal

High values indicate predicted imminent collaboration, suggesting:

| Value | Interpretation |
|---|---|
| **0.0–0.2** | No predicted collaboration; isolated or dormant wallets |
| **0.2–0.5** | Weak collaboration signal; normal market-making |
| **0.5–0.8** | Moderate signal; wallet coordinating with known partners |
| **0.8–1.0** | Strong signal; predicted imminent collaboration, likely coordinated wash trading |

## Integration with GNN

The temporal KGE captures different structural patterns than the GNN:

| Model | Captures | Used For |
|---|---|---|
| **GraphSAGE (GNN)** | Multi-hop funding relationships, ring topology | Static community detection, ring membership |
| **TComplEx (Temporal KGE)** | Temporal co-trading patterns, emerging coalitions | Predicting future relationships, early warning |

**Combined scoring**: When both are available, ensemble prediction uses:

```python
risk_score = w_gnn * gnn_embedding_score + w_kge * temporal_kge_collab_score
```

Where weights are learned via NSGA-II Pareto front search (see `ensemble_calibrator.py`).

## Incremental Training

The temporal KGE is rebuilt daily (not from scratch) using:

1. **Daily batch**: Trades from past 24 hours
2. **Full graph**: Complete wallet entity set + relation embeddings (frozen)
3. **Update strategy**: Warm-start with previous day's model, train on new triples only
4. **Fallback**: If warm-start diverges (loss increases), full retrain from scratch

```python
# Pseudocode
if day_num > 1:
    load(yesterday_kge_model)  # Warm-start
    train_new_triples(new_trades_24h)  # Fine-tune
else:
    train_from_scratch()
```

## Requirements & Constraints

### Functional

- ✓ Temporal KG rebuilt incrementally daily
- ✓ Embedding dimension configurable (KGE_EMBEDDING_DIM, default 64)
- ✓ Inference (collab score for single wallet pair) completes in < 5ms

### Performance

- **Training time**: ~10 min/day for 10K wallets, 100K triples (batch SGD)
- **Inference time**: ~1-3ms per wallet pair (cached embeddings)
- **Memory**: ~50MB for 10K wallets, dim=64 (embeddings + relation matrix)

### Security

- ✓ Model artifacts versioned and SHA-256 signed
- ✓ Metadata includes: trained_at, embedding_dim, n_entities, entity_id_map
- ✓ Load-time verification: SHA-256 mismatch raises ModelIntegrityError

### Data Integrity

- Timestamps binned must not extend beyond current time
- Unknown wallets default to 0.0 score
- Graceful fallback when model absent: feature = 0.0

## Testing

### Unit Test: Ring Link Prediction

```python
def test_ring_members_score_higher_than_random():
    """After training on 3-wallet ring, ring members score higher than random."""
    trades_df = sample_3_wallet_ring_trades()  # A↔B, B↔C, C↔A for 24 hours
    
    encoder = TemporalKGEncoder(embedding_dim=32)
    encoder.train(trades_df, num_epochs=50)
    
    # Ring member scores
    ring_scores = [
        encoder.predict_collaboration_score(A, B),  # Should be high
        encoder.predict_collaboration_score(B, C),  # Should be high
        encoder.predict_collaboration_score(C, A),  # Should be high
    ]
    ring_mean = np.mean(ring_scores)
    
    # Random wallet scores
    random_wallet = generate_unused_wallet()
    random_scores = [
        encoder.predict_collaboration_score(random_wallet, A),  # Should be low
        encoder.predict_collaboration_score(random_wallet, B),  # Should be low
    ]
    
    assert ring_mean > 0.3
    assert np.mean(random_scores) < ring_mean
```

### Test: Temporal Binning Correctness

```python
def test_temporal_binning_no_future_bins():
    """Timestamp bins must not exceed current time."""
    trades = trades_within_past_24h()  # All in past
    kg_info = build_temporal_kg_from_trades(trades)
    
    min_time, max_time = kg_info["timestamp_range"]
    now_bin = int(datetime.now(tz=UTC).timestamp() // 3600)
    
    assert max_time <= now_bin, "Max bin must not be in future"
```

### Performance Test: Inference < 5ms

```python
def test_inference_within_time_budget():
    """Single link prediction must complete within 5ms."""
    encoder = load_trained_model()
    wallet_a, wallet_b = get_random_wallet_pair()
    
    start = time.perf_counter()
    score = encoder.predict_collaboration_score(wallet_a, wallet_b)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    assert elapsed_ms < 5.0, f"Inference took {elapsed_ms:.2f}ms"
    assert 0.0 <= score <= 1.0
```

## PyKEEN Integration Details

```python
from pykeen.models import TComplEx
from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop

# Create TComplEx model
model = TComplEx(
    embedding_dim=64,
    entity_initializer="uniform",
    relation_initializer="uniform",
    time_initializer="uniform"
)

# Train loop
training_loop = SLCWATrainingLoop(
    model=model,
    optimizer_cls="Adam",
    optimizer_kwargs={"lr": 0.1}
)

for epoch in range(num_epochs):
    loss = training_loop.train(
        triples_tensor,  # (head, rel, tail, time)
        batch_size=256,
        use_tqdm=False
    )
```

## Future Enhancements

### 1. Dynamic Temporal Granularity
Adapt binning based on trading frequency:
- High-frequency wallets → 15-min bins
- Low-frequency wallets → 6-hour bins

### 2. Relation-Specific Temporal Decay
Model weakening of relationships over time:
- "traded_with": decay = 0.95 per day
- "funded_by": decay = 1.0 (permanent)

### 3. Cross-Venue Temporal Alignment
Correlate SDEX + AMM temporal patterns:
- If wallet X trades USDC/XLM on SDEX at hour T, predict AMM collab at T+1

### 4. Anomaly Detection on Temporal Graph
Detect sudden changes in temporal patterns:
- New participants joining ring at time T
- Trading time shifts (e.g., moving from UTC 12-14 to 00-02)

## References

- **TComplEx Model**: Lacroix et al. (2021) — "Temporal Knowledge Graph Completion using a Linear Temporal Regularizer"
- **PyKEEN**: Ali et al. (2021) — PyKEEN: Knowledge Graph Embedding Made Easy
- **Knowledge Graph Embeddings**: Nickel et al. (2016) — A Review of Relational Machine Learning for Knowledge Graphs

## Contributor Guidance

**Area of Specialty**: Knowledge graph embeddings, graph ML, Python (PyKEEN), temporal reasoning.

**How to Contribute**:

1. Comment describing your experience with:
   - Knowledge graph embeddings (PyKEEN, DGL-KE, Torch-Geometric-KG)
   - TComplEx vs TNTComplEx tradeoffs
   - Temporal reasoning in financial/blockchain networks

2. Propose enhancements:
   - Dynamic temporal binning based on transaction frequency
   - Relation-specific decay functions
   - Cross-venue temporal alignment (SDEX + AMM)

3. PR Requirements:
   - [ ] Ring-link-prediction unit test passes
   - [ ] < 5ms inference performance test passes
   - [ ] Temporal binning never produces future bins
   - [ ] Unknown wallets return 0.0 score
   - [ ] Model SHA-256 signed and verified on load
   - [ ] Documentation updated with examples

## Configuration

Add to `config.py`:

```python
# Temporal KGE settings
KGE_EMBEDDING_DIM: int = int(os.getenv("KGE_EMBEDDING_DIM", "64"))
TEMPORAL_BINNING_HOURS: int = int(os.getenv("TEMPORAL_BINNING_HOURS", "1"))
KGE_TRAINING_EPOCHS: int = int(os.getenv("KGE_TRAINING_EPOCHS", "50"))
KGE_BATCH_SIZE: int = int(os.getenv("KGE_BATCH_SIZE", "256"))
KGE_LEARNING_RATE: float = float(os.getenv("KGE_LEARNING_RATE", "0.1"))
KGE_INFERENCE_TIME_BUDGET_MS: float = float(os.getenv("KGE_INFERENCE_TIME_BUDGET_MS", "5.0"))
```

