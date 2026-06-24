# GNN Architecture — GraphSAGE Wallet Encoder

## Overview

LedgerLens augments its tabular feature vector with **learned graph embeddings**
from a 2-layer GraphSAGE encoder.  The encoder transforms each wallet node into a
dense vector (default 32 dimensions) that captures multi-hop structural patterns in
the wallet funding and co-trade graph — patterns that pairwise Jaccard similarity
cannot capture.

### Why GNNs for wash-trading detection?

Wash traders typically operate in **rings**: a single funding wallet seeds 5–30
ephemeral wallets, which trade back and forth across multiple asset pairs.  Pairwise
Jaccard similarity on funding ancestors catches direct siblings but misses multi-hop
structure.  GNNs trained on transaction graphs have achieved state-of-the-art fraud
detection on Ethereum (Elliptic dataset, AUC > 0.97) and are directly applicable here.

**Key references:**
- Weber et al. (2019) — Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks
- Lo et al. (2023) — Inspection-L: Towards Flow-Level Detection of Wash Trading on DEXs via Graph Neural Networks

---

## Graph Schema

### Node type

All nodes are wallet accounts (Stellar public keys).  Node features are:

| Index | Name | Description |
|---|---|---|
| 0 | `degree_in` | Number of incoming edges |
| 1 | `degree_out` | Number of outgoing edges |
| 2 | `age_days` | Account age in days at inference time |
| 3 | `trade_count` | Number of trades observed in the data window |
| 4 | `total_volume_xlm` | Total trading volume in XLM-equivalent |

### Edge types

| `edge_type` | Direction | Added by | Meaning |
|---|---|---|---|
| `"funding"` | `funder → wallet` | `build_funding_graph()` | Wallet was seeded by the funding account |
| `"co_trade"` | bidirectional | `build_co_trade_graph()` | Both wallets traded the same asset pair within `GRAPH_CO_TRADE_WINDOW_HOURS` |

### Edge attributes

| Attribute | Type | Description |
|---|---|---|
| `edge_type` | str | `"funding"` or `"co_trade"` |
| `weight` | int | Funding: 1.  Co-trade: number of co-trade events observed |
| `timestamp` | str (ISO 8601) | First observed timestamp of this edge |

### Account ID sanitisation

All node IDs are validated against the Stellar account ID regex
`^G[A-Z2-7]{55}$` before being added to the graph.  Edges or nodes that do not
conform are silently dropped.

---

## Model Architecture

```
Input node features: (N, 5)
        │
   ┌────▼─────────────────────────┐
   │  SAGEConv(5  → hidden_dim)   │  mean aggregation
   │  + ReLU                       │
   └────────────────┬─────────────┘
                    │
   ┌────────────────▼──────────────┐
   │  SAGEConv(hidden_dim → out)   │  mean aggregation
   └────────────────────────────────┘
        │
Output embeddings: (N, embedding_dim)
```

| Hyperparameter | Default | Config key |
|---|---|---|
| Input feature dim | 5 (fixed) | — |
| Hidden dimension | 64 | `GNN_HIDDEN_DIM` |
| Output embedding dim | 32 | `GNN_EMBEDDING_DIM` |
| Number of layers | 2 | `GNN_NUM_LAYERS` |
| Aggregation | mean | hard-coded |
| Activation | ReLU | hard-coded |

---

## Training Procedure

### Pre-training (contrastive link-prediction)

When `python -m detection.model_training --with-gnn` is invoked:

1. A wallet graph is constructed from the training data.
2. Labelled wash-trade wallets (`label = 1`) are grouped into a single synthetic
   ring of **positive pairs**.
3. **Negative pairs** are sampled uniformly at random from the full node set
   (ratio controlled by `negative_ratio`, default 5×).
4. **Loss function** (per epoch):

   ```
   L = L_pos + L_neg
   L_pos = mean(1 − cosine_similarity(ea, eb))   for (a, b) ∈ positive_pairs
   L_neg = mean(max(0, cosine_similarity(ea, eb))) for (a, b) ∈ negative_pairs
   ```

5. The loss curve is written to `reports/gnn_pretrain_{timestamp}.json`.
6. The trained state dict is saved to `{MODEL_DIR}/gnn_encoder.pt`.
7. A SHA-256 digest of `gnn_encoder.pt` is recorded in `{MODEL_DIR}/metrics.json`
   under the key `"gnn_encoder"` for integrity verification at load time.
8. GNN embedding features (`gnn_0` … `gnn_{GNN_EMBEDDING_DIM-1}`) are appended to
   each wallet's feature row before the ensemble classifiers are trained.

### Artifact integrity

Loading the encoder verifies that the SHA-256 of `gnn_encoder.pt` matches the
value stored in `metrics.json`.  A mismatch raises `ModelIntegrityError` — the
same pattern used for the joblib model artifacts in `detection/persistence.py`.

---

## Inference

### Full-graph batch encoding

```python
from detection.gnn_encoder import GNNEncoder
import networkx as nx

encoder = GNNEncoder()
encoder.load()           # verifies SHA-256

embedding = encoder.encode(graph, wallet)   # → np.ndarray shape (32,)
```

The full graph is encoded in one forward pass; results are cached so that repeated
calls on the same graph snapshot are free.

**Performance target:** < 60 s for 50,000 nodes on CPU.

### Incremental streaming update

When a new edge arrives in the stream only the 1-hop neighbourhood of the affected
wallet needs to be re-computed:

```python
# streaming/streaming_scorer.py
scorer.observe_new_edges(wallet, [(src, dst)])
```

Internally this calls `GNNEncoder.update_node(wallet, new_edges, graph)`, which
extracts the 1-hop subgraph and runs a forward pass on it.

**Performance target:** < 50 ms for a graph with 10,000 nodes.

---

## Feature Integration

`compute_graph_embedding_features(wallet, graph, encoder)` in
`detection/feature_engineering.py` returns:

```python
{"gnn_0": float, "gnn_1": float, ..., "gnn_31": float}
```

These are merged into the feature row by `build_feature_vector`.  When the encoder
artifact is absent (e.g., before the first training run), the function returns
all-zeros — the legacy scalar features (`funding_source_similarity`,
`network_centrality`) remain present and the ensemble can still produce a score.

See `data/dataset_card.md` for the updated feature schema including GNN columns.

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `GNN_EMBEDDING_DIM` | `32` | Output embedding dimensionality |
| `GNN_HIDDEN_DIM` | `64` | GraphSAGE hidden layer size |
| `GNN_NUM_LAYERS` | `2` | Number of SAGEConv layers (architecture is fixed at 2) |
| `GRAPH_CO_TRADE_WINDOW_HOURS` | `24` | Time window for co-trade edge construction |

---

## File Layout

```
detection/
├── gnn_encoder.py          ← GNNEncoder class, _GraphSAGEModel, pretrain_gnn_contrastive
├── wallet_graph.py         ← build_funding_graph, build_co_trade_graph (new)
│                               funding_source_similarity (@deprecated)
│                               network_centrality (@deprecated)
└── feature_engineering.py  ← compute_graph_embedding_features (new)
                                build_feature_vector (updated — gnn_encoder param)
                                build_feature_matrix (updated — gnn_encoder param)

models/
├── gnn_encoder.pt          ← GraphSAGE state dict
└── metrics.json            ← SHA-256 manifest (gnn_encoder + ensemble models)

streaming/
└── streaming_scorer.py     ← StreamingScorer.observe_new_edges (incremental update)

tests/
└── test_gnn_encoder.py     ← unit tests (build_co_trade_graph, encode, update_node,
                                integrity, zero-fallback)
```
