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
- Xu et al. (2020) — Inductive Representation Learning on Temporal Graphs (TGAT), ICLR 2020

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
├── gnn_encoder.py          ← GNNEncoder, _GraphSAGEModel, GraphLevelPooling,
│                               pretrain_gnn_contrastive
├── wallet_graph.py         ← build_funding_graph, build_co_trade_graph
└── feature_engineering.py  ← compute_graph_embedding_features, build_feature_vector

models/
├── gnn_encoder.pt          ← GraphSAGE state dict
└── metrics.json            ← SHA-256 manifest

streaming/
└── streaming_scorer.py     ← StreamingScorer.observe_new_edges (incremental update)

tests/
├── test_gnn_encoder.py     ← encoder unit tests
└── test_cluster_scoring.py ← DiffPool / cluster scoring unit tests
```

---

## DiffPool Graph-Level Pooling & Cluster Scoring

### Motivation

The GraphSAGE encoder produces **per-wallet node embeddings** used in individual
wallet scoring.  Wash-trade rings are **cluster-level phenomena** — the entire ring
is suspicious, not just individual wallets.  A cluster-level risk score captures
ring-level patterns (circular trade flows, shared funding ancestry) that per-node
scores cannot aggregate in a permutation-invariant way.

### Architecture

`GraphLevelPooling` (in `detection/gnn_encoder.py`) implements hierarchical DiffPool:

```
Per-node embeddings X  (N × D)  ←  GNNEncoder
         │
         ▼
_DiffPoolAssignNet   →  S  (N × K)       soft cluster assignment
         │
         ▼
Pooled features:  X_out = S^T · X       (K × D)
         │
         ▼
Global mean-readout: mean(X_out, dim=0)  (D,)
         │
         ▼
Linear head:  logit = W · emb + b        (scalar)
         │
         ▼
Cluster score = sigmoid(logit) × 100     ∈ [0, 100]
```

| Hyperparameter | Default | Config key |
|---|---|---|
| Target cluster count K | 10 | `GNN_DIFFPOOL_CLUSTERS` |
| Max input nodes | 50 | hard-coded |
| Assignment net hidden dim | `GNN_HIDDEN_DIM` | `GNN_HIDDEN_DIM` |

The assignment network uses the **same SAGEConv + Softmax** architecture.
It is initialised with random weights; a trained linear head gives interpretable
cluster-level logits.

### Cluster Scoring API

```python
from detection.model_inference import score_cluster
from detection.gnn_encoder import GNNEncoder, GraphLevelPooling

encoder = GNNEncoder()
encoder.load()

pooler = GraphLevelPooling()   # uses GNN_DIFFPOOL_CLUSTERS, GNN_HIDDEN_DIM

result = score_cluster(
    wallet_ids=["GABC...", "GXYZ...", "GDEF...", "GHIJ...", "GKLM..."],
    graph=wallet_graph,
    scorer=risk_scorer,
    feature_matrix=feature_matrix,
    pooler=pooler,
    encoder=encoder,
)

print(result["cluster_score"])     # int 0–100
print(result["cluster_id"])        # SHA-256 of sorted wallet addresses
print(result["individual_scores"]) # per-wallet scores
```

### Permutation Invariance

The cluster score is **order-independent**:

- Node ordering is canonically sorted (`sorted(wallet_ids)`) before graph encoding.
- DiffPool's assignment matrix `S` is a function of sorted node features.
- Global mean-readout is permutation-invariant by construction.

### Cluster ID

Each result carries a `cluster_id` = SHA-256 of `"|".join(sorted(wallet_ids))`.
This prevents duplicate scoring of the same ring under different orderings and
allows cluster scores to be deduplicated in the database.

### How Cluster Score Relates to Individual Wallet Scores

| Mode | Description |
|---|---|
| Pooler + features available | 50% DiffPool graph embedding score + 50% mean individual score |
| Features only (no pooler) | 90th-percentile of individual scores |
| Pooler only (no features) | DiffPool graph embedding score |
| Neither | 0 (no information available) |

### Prometheus Counter

```
ledgerlens_cluster_scored_total  ← incremented on each score_cluster() call
```

### Testing

```bash
pytest tests/test_cluster_scoring.py -v
```

Tests cover permutation invariance, cluster ID stability, correct return shape,
high-risk ring produces score > 80, and DiffPool pooling to `n_clusters`.


---

## Contrastive Pre-training Strategy

`detection/contrastive/` implements **SimCLR-style contrastive pre-training**
for the `TransactionEncoder` (a 2-layer MLP encoder used before the GNN
pooling stage).  The default random-augmentation approach is augmented with
**domain-aware negative mining** that exploits labelled wash-trade structure
to produce more discriminative embeddings.

### Why domain-aware negatives?

Standard SimCLR draws negatives uniformly at random from the mini-batch.  For
wash-trade detection this is suboptimal: the encoder quickly learns to separate
clearly different wallets but struggles to distinguish borderline wash-traders
from high-frequency legitimate market makers — both produce high-volume,
clustered trade patterns.  Domain-aware negatives address this by forcing the
encoder to separate the most confusable pairs:

- **Semi-hard negatives**: confirmed-clean wallets that are *currently closest*
  to a wash-trade anchor in embedding space (hard in the model's representation,
  correct in label).
- **Ring positives**: wallet pairs from the same detected wash-trading ring are
  pulled together as additional positives beyond the standard augmentation views.

### Negative mining algorithm (`detection/contrastive/negative_miner.py`)

```
Input:  anchor embeddings  A  (batch × dim)
        clean-wallet pool  C  (n_clean × dim)
        epoch index        e

1. L2-normalise A and C.
2. Query FAISS HNSW index (built over C) for the k nearest neighbours
   of each anchor.                              ← O(k log n_clean)
3. Curriculum mix:
     hard_fraction = min(e / curriculum_epochs, 1.0)
     n_hard = round(hard_fraction × n_negatives)
     n_easy = n_negatives − n_hard
4. Return top-n_hard ANN results + n_easy random indices.
```

**Complexity**: HNSW query is O(k log n) vs O(n·k) brute-force, enabling
large clean-wallet pools without training throughput degradation.  A brute-
force cosine fallback is used automatically when `faiss` is not installed.

**Privacy**: All wallet G-addresses are replaced with HMAC-SHA256 digests
(keyed by `EVENT_HMAC_SECRET`) before entering the miner's data structures.
Raw addresses never appear in training state, logs, or the ring registry.

### Ring-positive construction

`RingRegistry` stores hashed wallet IDs grouped by wash-trading ring
(from Louvain community detection in `detection/wallet_graph.py`).  During
each batch, `get_ring_positives(hashed_batch_ids)` returns `(i, j)` index
pairs where wallets `i` and `j` are confirmed colluders.  A cosine-similarity
auxiliary loss pulls these pairs together in embedding space.

### Curriculum schedule

```
hard_fraction(epoch) = min(epoch / CONTRASTIVE_CURRICULUM_EPOCHS, 1.0)
```

| Epoch | hard_fraction | Negative source |
|---|---|---|
| 0 | 0.00 | 100 % random |
| curriculum_epochs / 2 | 0.50 | 50 % hard, 50 % random |
| curriculum_epochs | 1.00 | 100 % hard (ANN) |
| > curriculum_epochs | 1.00 | 100 % hard (ANN) |

The warm-up prevents training instability when the encoder is still random:
hard negatives are only informative once the embedding space has some
structure.

**Configuration**:

| Variable | Default | Description |
|---|---|---|
| `CONTRASTIVE_CURRICULUM_EPOCHS` | `5` | Warm-up length (set to 0 for instant hard negatives) |
| `EVENT_HMAC_SECRET` | (required) | HMAC key for wallet ID hashing |

### Training the domain-aware encoder

```bash
# Domain-aware pre-training on the synthetic dataset:
python -m detection.contrastive.pretrain \
    --data data/synthetic_dataset.parquet \
    --epochs 20 \
    --domain-sampling \
    --curriculum-epochs 5 \
    --output models/pretrained_encoder.pt

# AUC benchmark (random vs domain-aware):
python -m detection.contrastive.pretrain \
    --data data/synthetic_dataset.parquet \
    --epochs 10 \
    --benchmark
```

### Benchmark: random vs domain-aware AUC

A linear probe (logistic regression on frozen encoder embeddings) is used
to measure downstream classification quality on the synthetic dataset.

| Pre-training strategy | Downstream AUC | Notes |
|---|---|---|
| Random augmentation (SimCLR) | baseline | No label information used |
| Domain-aware (hard negatives + ring positives) | ≥ random AUC | Tested by `tests/test_negative_miner.py::TestAUCRegression` |

The AUC regression test (`TestAUCRegression::test_domain_aware_auc_not_worse`)
enforces that domain-aware pre-training does not degrade below the random
baseline by more than 2 % on the synthetic dataset, with a tolerance for
the variance introduced by short training runs.

### Integration with downstream ML pipeline

After pre-training, the encoder's hidden representation `h` (before the
SimCLR projector head `z`) is used as an additional feature source:

```python
from detection.contrastive.encoder import TransactionEncoder
import torch

encoder = TransactionEncoder(input_dim=feature_dim)
encoder.load_state_dict(torch.load("models/pretrained_encoder.pt", map_location="cpu"))
encoder.eval()

with torch.no_grad():
    h, _ = encoder(feature_tensor)   # h: (batch, 256)
```

The 256-dim `h` vectors can be appended to the tabular feature matrix
before training the RF / XGBoost / LightGBM ensemble, similar to how
`gnn_0 … gnn_31` GNN embeddings are appended (see
`detection/gnn_encoder.py::compute_graph_embedding_features`).
