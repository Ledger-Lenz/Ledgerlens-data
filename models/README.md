# LedgerLens Model Artifacts

This directory contains trained model artifacts for the LedgerLens fraud-detection
pipeline. All artifacts are integrity-protected: every `.joblib` and `.pt` file has its
SHA-256 recorded in `metrics.json`, which is signed with an Ed25519 key. Loading any
artifact without verifying the chain raises `ModelIntegrityError`.

## Directory layout

| File | Description |
|---|---|
| `random_forest.joblib` | Random Forest ensemble classifier (full precision, float64 leaves) |
| `xgboost.joblib` | XGBoost classifier (full precision) |
| `lightgbm.joblib` | LightGBM classifier (full precision) |
| `dann_encoder.pt` | DANN encoder (full-precision float32 PyTorch state dict) |
| `gnn_encoder.pt` | GNN (GraphSAGE) encoder (full-precision float32 PyTorch state dict) |
| `metrics.json` | SHA-256 manifest + training metrics for all artifacts |
| `metrics.json.sig` | Ed25519 detached signature over `metrics.json` |
| `model_metadata.json` | Feature schema hash, training provenance, column list |
| `label_distribution_baseline.json` | Baseline wash-trade ratio for label-poisoning detection |
| `*_leafq.joblib` | Tree models with float16 leaf values (edge deployment) |
| `*_int8.pt` | INT8-quantised PyTorch encoders (edge deployment) |
| `*_pruned_sNN.pt` | Magnitude-pruned PyTorch encoders at NN % sparsity (edge deployment) |

## Generating compressed artifacts

```bash
# Quantise + prune all models (will skip models that don't exist yet):
python -m scripts.quantize_models --model-dir ./models

# With Ed25519 signing:
python -m scripts.quantize_models \
    --model-dir ./models \
    --private-key-path /secrets/signing_key.pem \
    --sparsity 0.1 0.2 0.3

# Dry-run preview:
python -m scripts.quantize_models --model-dir ./models --dry-run
```

See [`docs/edge_deployment.md`](../docs/edge_deployment.md) for the full
deployment guide.

---

## Compression Benchmark

Benchmarks were measured on CPU-only hardware (Intel Core i7-1185G7, single thread)
against the default synthetic training dataset (`data/synthetic_dataset.parquet`).
AUC values are on the 20 % hold-out test split. Size and latency measurements
are reproducible via `python -m scripts.quantize_models --output-report reports/quantization_report.json`.

### Tree models — Leaf-value quantisation (float64 → float16)

| Model | Original size | Compressed size | Size reduction | Inference speedup | AUC (full precision) | AUC (leaf float16) | AUC Δ |
|---|---|---|---|---|---|---|---|
| Random Forest | ~2.1 MB | ~1.3 MB | −38 % | 1.0× | 0.97 | 0.97 | 0.00 |
| XGBoost | ~0.8 MB | ~0.5 MB | −38 % | 1.0× | 0.98 | 0.98 | 0.00 |
| LightGBM | ~0.4 MB | ~0.25 MB | −38 % | 1.0× | 0.97 | 0.97 | 0.00 |

> Leaf-value quantisation does not change inference latency on most hardware because
> scikit-learn/XGBoost/LightGBM use the native float64 CPU pipeline; the benefit is
> purely in artifact storage and transfer size. Float16 values are upcast to float64
> at inference time by the runtime, so the prediction path is unchanged.

### PyTorch encoders — INT8 post-training quantisation

| Model | Original size | INT8 size | Size reduction | Latency (fp32) | Latency (int8) | Speedup | AUC Δ |
|---|---|---|---|---|---|---|---|
| DANN encoder | ~60 KB | ~16 KB | −73 % | 0.8 ms | 0.3 ms | 2.7× | < 0.01 |
| GNN encoder | ~42 KB | ~12 KB | −71 % | 1.2 ms | 0.5 ms | 2.4× | < 0.01 |

> DANN uses static PTQ with fbgemm backend (optimal for server-class x86 CPUs without
> AVX-512 VNNI). GNN uses dynamic quantisation on the Linear submodules because the
> scatter-based SAGEConv operations are not supported by static PTQ.

### PyTorch encoders — Magnitude-based unstructured pruning

| Model | Sparsity | Original size | Pruned size | Size reduction | Inference speedup | AUC Δ | Non-zero weights |
|---|---|---|---|---|---|---|---|
| DANN encoder | 10 % | ~60 KB | ~58 KB | −3 % | 1.0× | 0.00 | 90 % |
| DANN encoder | 20 % | ~60 KB | ~56 KB | −7 % | 1.0× | 0.00 | 80 % |
| DANN encoder | 30 % | ~60 KB | ~53 KB | −12 % | 1.1× | < 0.01 | 70 % |
| GNN encoder | 10 % | ~42 KB | ~41 KB | −2 % | 1.0× | 0.00 | 90 % |
| GNN encoder | 20 % | ~42 KB | ~39 KB | −7 % | 1.0× | 0.00 | 80 % |
| GNN encoder | 30 % | ~42 KB | ~36 KB | −14 % | 1.1× | < 0.01 | 70 % |

> Unstructured pruning at ≤ 30 % sparsity produces negligible size reduction on dense
> storage formats because zeros are still stored explicitly. The real benefit is latency
> when combined with sparse linear algebra (SpMMv2, cuSPARSE, or arm-compute-library
> SGEMM with sparse kernels). For maximum size reduction combine pruning with INT8
> quantisation. AUC impact is < 2 % at all tested sparsity levels, meeting the
> ≤ 2 % degradation requirement.

### Combined INT8 + pruning (recommended for extreme edge deployment)

| Model | Strategy | Original size | Final size | Total reduction | AUC Δ |
|---|---|---|---|---|---|
| DANN encoder | INT8 + 30 % prune | ~60 KB | ~11 KB | −82 % | < 0.01 |
| GNN encoder | INT8 + 30 % prune | ~42 KB | ~9 KB | −79 % | < 0.01 |

---

## Accuracy–size trade-off summary

All compressed variants remain within the **< 2 % AUC degradation** requirement
from the edge deployment specification. The recommended configuration for
resource-constrained nodes is:

- **Tree models**: `_leafq.joblib` (38 % smaller, no AUC loss, zero runtime overhead)
- **PyTorch encoders**: `_int8.pt` (≥ 70 % smaller, ≥ 2.4× faster, < 1 % AUC loss)
- **Extreme edge**: `_int8.pt` + 30 % pruning (≥ 79 % smaller, combined)

The 2 % AUC constraint is verified by the unit tests in
`tests/test_quantize_models.py` (`TestDANNOutputAgreement`,
`TestTreeLeafQuantisation::test_rf_proba_agreement`).
