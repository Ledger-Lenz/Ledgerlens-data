# Sequence Model Architecture — #182

## Overview

The LedgerLens transformer sequence model captures **temporal wash-trading patterns** that are invisible to the aggregate 37-feature vector consumed by the Random Forest / XGBoost / LightGBM ensemble.

Wash-trading manifests as sequential structure:

- **Ping-pong cycles**: alternating buy → sell → buy at nearly identical amounts.
- **Volume escalation**: steadily increasing trade size before a pump event.
- **Bot regularity**: mechanical inter-trade intervals (constant Δt) that stand out against the irregular cadence of organic market-making.

Aggregate features (mean, std, concentration ratios) collapse this temporal signal. The transformer reads the ordered sequence directly and distils it into a sequence-level risk embedding that is concatenated to the feature vector before scoring.

---

## Data flow

```
Stellar Horizon SSE
        │
        ▼
FeatureBuffer (streaming/feature_buffer.py)
  rolling deque, per wallet, up to 1 000 trades
        │
        ├── build_feature_vector() ──────────────────────────────────────────┐
        │   (detection/feature_engineering.py)                               │
        │   • Benford features (15)                                          │
        │   • Trade pattern features                                         │
        │   • Volume/timing features                                         │
        │   • Wallet graph features                                          │
        │   • Cross-asset coordination features (6)                         │
        │   • GNN embedding (32 dims)                                       │
        │                                                                     │
        └── build_sequence_embedding()  ─────────────────────────────────── ─┤
            (detection/trade_sequence_transformer.py)                        │
            TradeSequenceTransformer forward pass                            │
            ↓                                                                 │
            seq_0 … seq_63  (64-dim sequence embedding)                     │
                                                                              ▼
                                               Concatenated feature vector (37 + 64 = 101 dims)
                                                              │
                                                              ▼
                                              Ensemble (RF / XGBoost / LightGBM)
                                                              │
                                                              ▼
                                                LedgerLens Risk Score (0–100)
```

---

## Model Architecture

### Input representation

Each trade event is encoded into a fixed-width vector:

| Field | Dim | Notes |
|---|---|---|
| `log_amount` | 1 | `log(base_amount)` — captures amount magnitude |
| `log1p(time_delta_s)` | 1 | log(1 + seconds since previous trade) — compresses long tails |
| `direction` | 1 | `+1.0` = buyer, `-1.0` = seller, `0.0` = unknown |
| `pair_one_hot` | N | One-hot encoding of the asset pair index (N = `SEQ_MODEL_NUM_PAIRS`, default 32) |

**Total input dim** = 3 + N (default: 35)

### Transformer encoder

```
Input sequence  (batch, seq_len, input_dim)
      │
      ▼
Linear projection  →  (batch, seq_len, embed_dim)
      │
      ▼
Sinusoidal positional encoding  (fixed, not learned)
      │
      ▼
TransformerEncoder × num_layers
  • Multi-head self-attention  (num_heads, embed_dim / num_heads per head)
  • Pre-LayerNorm (norm_first=True) for stable gradient flow
  • Feed-forward expansion:  embed_dim → ffn_dim → embed_dim
  • Padding mask: padded positions are excluded from attention
      │
      ▼
Mean pooling over non-padding positions
      │
      ▼
LayerNorm
      │
      ▼
Output embedding  (batch, embed_dim)
```

### Default hyperparameters

| Config variable | Default | Description |
|---|---|---|
| `SEQ_MODEL_NUM_PAIRS` | `32` | Asset-pair one-hot vocabulary size |
| `SEQ_MODEL_EMBED_DIM` | `64` | Token / output embedding dimension |
| `SEQ_MODEL_NUM_HEADS` | `4` | Self-attention heads |
| `SEQ_MODEL_NUM_LAYERS` | `2` | Transformer encoder layers |
| `SEQ_MODEL_FFN_DIM` | `128` | Feed-forward expansion dimension |
| `SEQ_MODEL_DROPOUT` | `0.1` | Dropout (training only) |
| `SEQ_MODEL_MAX_LENGTH` | `512` | Maximum sequence length (security limit) |
| `SEQ_MODEL_ENABLED` | `true` | Load model at inference time |

### Why sinusoidal, not learned, positional encoding?

Wash-trading bots operate on **absolute timing patterns** (e.g., every 5 seconds). Sinusoidal positional encoding generalises well to sequence lengths not seen at training time and avoids the lookup-table position aliasing problem where identical token patterns at different positions in a learned PE table produce divergent embeddings.

---

## Variable-length input handling

Wallets have between 1 and 512 buffered trades. The model handles this via **dynamic padding and masking**:

1. Within a batch, sequences are padded to the length of the longest sequence with zero vectors.
2. A boolean `padding_mask` of shape `(batch, seq_len)` marks padded positions as `True` (ignored).
3. The `TransformerEncoder` passes `src_key_padding_mask=padding_mask` to suppress attention to and from padded positions.
4. Mean pooling sums only non-padded positions: `sum(encoded * non_pad) / count(non_pad)`.

**Security**: inputs longer than `SEQ_MODEL_MAX_LENGTH` are rejected with a `ValueError` *before* reaching the model (enforced in `validate_sequence`). This prevents memory exhaustion attacks via the API.

---

## Training

```bash
# Train on the synthetic dataset (2 epochs default)
python -m scripts.train_sequence_transformer

# With benchmark
python -m scripts.train_sequence_transformer --benchmark

# Custom settings
python -m scripts.train_sequence_transformer \
    --data-path data/synthetic_dataset.parquet \
    --epochs 5 \
    --batch-size 64 \
    --lr 1e-3
```

### Optimiser

- **AdamW** with cosine annealing LR schedule (`CosineAnnealingLR`, η_min = 1% of peak LR).
- **Gradient clipping**: `max_norm=1.0` prevents gradient explosions on long sequences.
- **BCEWithLogitsLoss**: binary cross-entropy with logits for wash vs. legitimate classification.
- **80/20 stratified split** of wallets for training and validation.

---

## CPU latency budget

The inference latency requirement is **< 50 ms per batch of 32 wallets** on CPU (no GPU required in production).

| Configuration | Measured p95 (CPU) |
|---|---|
| 2 layers, embed=64, heads=4, seq=64 | ~8 ms |
| 4 layers, embed=128, heads=8, seq=64 | ~25 ms |
| 4 layers, embed=128, heads=8, seq=512 | ~180 ms |

Use `benchmark_latency()` to verify on your hardware:

```python
model = TradeSequenceTransformer.load()
result = model.benchmark_latency(batch_size=32, seq_len=64, n_runs=50)
print(f"p95={result['p95_ms']:.1f} ms")
```

---

## Artifact persistence

Weights are saved to `{MODEL_DIR}/sequence_transformer.pt`.  An integrity entry is written to `metrics.json`:

```json
{
  "sequence_transformer": {
    "artifact_sha256": "<sha256>",
    "embed_dim": 64,
    "num_layers": 2,
    "num_pairs": 32
  }
}
```

On load, the SHA-256 of the file is recomputed and compared against the manifest.  Any mismatch raises `RuntimeError` — same pattern as the GNN encoder and ensemble models.

---

## Feature schema impact

Adding the sequence embedding appends `seq_0` … `seq_{embed_dim-1}` columns to the feature vector. This changes the `feature_schema_hash` stored in `model_metadata.json`.

**After training the sequence model, retrain the ensemble** with the updated feature matrix so the schema hash stays consistent:

```bash
python -m scripts.train_sequence_transformer
python -m detection.model_training --data-path data/synthetic_dataset.parquet
```

Before the first sequence model training run, `build_sequence_embedding()` returns an all-zeros vector so the feature schema stays stable and the ensemble can still score without the sequence model.

---

## Integration with the feature pipeline

```
detection/feature_engineering.py::build_feature_vector(
    wallet, wallet_trades, ...,
    seq_model=loaded_transformer,   # ← new parameter
    pair_vocab={...},
)
```

The `seq_model` parameter is optional. When `None` (or `SEQ_MODEL_ENABLED=false`), zeros are appended:

```python
features.update({f"seq_{i}": 0.0 for i in range(config.SEQ_MODEL_EMBED_DIM)})
```

`RiskScorer` loads the sequence model in `_load_seq_model()` at initialisation time and makes it available as `self.seq_model`. Callers that build feature rows outside of `RiskScorer` (e.g., `run_pipeline.py`) should pass `seq_model=risk_scorer.seq_model` through to `build_feature_vector`.

---

## Security considerations

1. **Input length validation** (`validate_sequence`): sequences longer than `SEQ_MODEL_MAX_LENGTH` are rejected before reaching the model. This prevents API callers from triggering O(seq²) attention computation with adversarially long inputs.
2. **No model code in artifacts**: weights are saved with `torch.save(state_dict)` — not `torch.save(model)`. The model class must exist locally; unpickling a remote model artifact cannot execute arbitrary code.
3. **SHA-256 integrity check**: identical to the GNN encoder and ensemble artifacts — tampering is detected on load.
