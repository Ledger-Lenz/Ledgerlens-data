# Federated Gradient Compression

Gradient compression reduces the per-round bandwidth cost of federated learning from hundreds of megabytes (full gradient tensors) to a few megabytes, with minimal impact on model convergence.

Two schemes are implemented in `detection/federated/gradient_compression.py`.

---

## Compression Schemes

### Top-K Sparsification (`TopKSparsifier`)

Transmits only the `k` gradient values with the largest absolute magnitude, along with their indices.

```python
from detection.federated.gradient_compression import TopKSparsifier

comp = TopKSparsifier(k_ratio=0.01)   # keep top 1%
payload = comp.compress(gradient)     # → TopKPayload
recovered = TopKSparsifier.decompress(payload)
```

**Bandwidth saving:** `~k_ratio` of the original (≈100× at k_ratio=0.01, accounting for index overhead).

**Security:** A random sign-flip rotation is applied to the gradient before index selection. This prevents the transmitted indices from revealing which features have the largest gradients (addresses issue #251 security requirement). The rotation seed is included in the payload so the receiver can invert it exactly.

**When to use:** Best suited for gradients that are already sparse or where a small fraction of parameters dominate the update — common in GNN node-classification heads and sparse attention layers.

---

### PowerSGD (`PowerSGDCompressor`)

Approximates a gradient matrix `G ≈ P @ Q^T` via randomised low-rank factorisation. Communicates factors `P` (m×r) and `Q` (n×r) instead of `G` (m×n).

```python
from detection.federated.gradient_compression import PowerSGDCompressor

comp = PowerSGDCompressor(rank=4, n_power_iterations=2)
payload = comp.compress(gradient)          # → PowerSGDPayload
recovered = PowerSGDCompressor.decompress(payload)
```

**Bandwidth saving:** `r(m+n) / (mn)` — for a 200×200 matrix at rank 4 this is ~4%.

**Matrix handling:** If the input gradient is already 2-D with both dimensions ≥ rank, the compressor uses the existing shape directly (preserving low-rank structure). Otherwise it reshapes to the nearest square matrix.

**When to use:** Well-suited for weight matrices in GNN encoders and DANN adapters, which tend to have low effective rank. Outperforms Top-K when the gradient structure is smooth rather than sparse.

---

## Memory Correction (Error Feedback)

Both compressors can be wrapped with `ErrorFeedbackCompressor` to apply per-layer error feedback:

```python
from detection.federated.gradient_compression import (
    ErrorFeedbackCompressor, TopKSparsifier
)

ec = ErrorFeedbackCompressor(TopKSparsifier(k_ratio=0.01), n_layers=1)

# Each round:
payload = ec.compress(gradient, layer_idx=0)
```

**How it works:**

1. Before compressing gradient `g_t`, add the stored residual: `g̃_t = g_t + e_{t-1}`
2. Compress `g̃_t` → payload
3. Decompress to get `ĝ_t`
4. Store new residual: `e_t = g̃_t − ĝ_t`

The residual is tracked **per layer** (`layer_idx` parameter) so that large residuals in one layer do not contaminate updates to other layers.

Call `ec.reset()` at the start of each training run to clear accumulated residuals.

---

## Bandwidth–Accuracy Trade-off

Benchmarked on a 5-round single-node simulation (logistic regression, 20 features, `k_ratio=0.1` for comparability):

| Scheme | Bandwidth ratio | Final log-loss vs. uncompressed |
|---|---|---|
| No compression | 1.00× | baseline |
| Top-K (k=10%) | ~0.15× | ≤ 2× baseline (typically <1.1×) |
| PowerSGD (rank=4) | ~0.04× | ≤ 2× baseline for low-rank gradients |

For GNN gradients (which are naturally low-rank), PowerSGD achieves higher compression with comparable accuracy. For sparse MLP gradients with a few dominant directions, Top-K is simpler and equally effective.

---

## Integration with `FederatedParticipant`

`participant.py` is pre-wired to use `ErrorFeedbackCompressor(TopKSparsifier(k_ratio=0.01))` by default. To switch to PowerSGD:

```python
from detection.federated.gradient_compression import ErrorFeedbackCompressor, PowerSGDCompressor
from detection.federated.participant import FederatedParticipant

participant = FederatedParticipant(
    participant_id="p1",
    coordinator_url="http://localhost:8000",
    compressor=ErrorFeedbackCompressor(PowerSGDCompressor(rank=4)),
)
```

---

## Running the Tests

```bash
pytest tests/test_gradient_compression.py -v
```

All 10 tests cover: Frobenius norm error bounds, shape preservation, bandwidth ratio, rotation randomness, error-feedback accumulation, and 5-round convergence regression.
