# Multi-relational vs. homogeneous community detection (Issue #885)

## Setup
- **Method:** `detect_multirelational_communities` (Louvain on a graph collapsed with
  per-edge-type weights: same-chain transfer 1.0, cross-chain bridge 1.5,
  behavioural similarity 0.5) vs. `detect_communities` on the same edges with types
  discarded (every edge weight 1.0).
- **Data:** labelled wash-ring wallets from `data/synthetic_dataset.parquet`, with
  bridge edges drawn from `data/bridge_anchors.json` and behavioural-similarity edges
  added between wallets whose feature vectors have cosine similarity > 0.95.
- **Metric:** adjusted Rand index (ARI) and ring-member recall (a labelled ring counts
  as recovered when ≥80% of its wallets share one community).
- **Reproduce:** use seed `config.WASH_RING_LOUVAIN_SEED`, resolution 1.0, min size 3.

## Findings
| Scenario | Homogeneous | Multi-relational |
|---|---|---|
| Rings on a single chain | baseline | matches baseline (same-chain weight 1.0 → same partition) |
| Rings split across chains through bridges | the ring fragments per chain when bridge edges are sparse | bridge up-weighting merges the fragments |
| Dense behavioural-similarity noise | similarity edges pull unrelated bot wallets into rings | down-weighting similarity edges keeps them out |

**Justification:** on homogeneous inputs the method gives exactly the baseline partition
(there is a backward-compatible fallback), so it cannot regress. Where the edge types
differ, weighting the edge types keeps the signal that collapsing them throws away:
bridge hops that mark deliberate cross-chain obfuscation, and weak correlational
similarity edges. The weights can be set per call through `edge_type_weights`; tune
them again whenever the labelled set is refreshed.
