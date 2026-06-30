# Graph Features: Motif Census

## Overview

Once Louvain community detection (`detection/community_detector.py`) partitions the wallet graph into suspected wash-trading rings, the motif census (`detection/motif_census.py`) characterises the *internal structure* of each community by counting k-node subgraph patterns (motifs). The resulting structural fingerprints distinguish coordinated wash rings (which favour dense triangles and reciprocal cycles) from organic market-maker networks (which tend toward hub-and-spoke star topologies with low reciprocity).

Entry point:

```python
from detection.community_detector import enrich_communities_with_motifs

motif_features = enrich_communities_with_motifs(graph, community_map)
# Returns: {community_id: {"triangle_density": ..., "star_ratio": ..., ...}}
```

---

## 3-Node Motif Taxonomy

| Motif | Pattern | Edges | Detection method |
|-------|---------|-------|-----------------|
| **Triangle** (K₃) | All three nodes mutually connected | 3 | A³ matrix trace |
| **Open wedge / Star** (P₃) | One centre node connected to two leaves; leaves not connected | 2 | Degree-sum formula |

### Triangle Density

```
triangle_density = triangle_count / C(n, 3)
```

where `n` is the community size and `C(n, 3) = n(n−1)(n−2)/6` is the maximum possible number of triangles.  A value of 1.0 means every triple of nodes forms a triangle (i.e., the community is a clique).

**Efficient computation — A³ trace method**

Rather than enumerating every node triple (O(n³)), triangle count is computed via:

```
triangles = trace(A³) / 6
```

where A is the adjacency matrix of the undirected community subgraph. This reduces to two matrix multiplications (O(n^{2.37…}) with optimised BLAS, O(n³) worst-case) followed by a diagonal sum, which is faster in practice because NumPy delegates to LAPACK/OpenBLAS.

### Star Ratio

```
total_wedges    = Σ_v C(deg_v, 2)
open_wedges     = total_wedges  −  3 × triangle_count
star_ratio      = open_wedges / (open_wedges + triangle_count)
```

Open wedges (P₃ patterns) are wedges whose two endpoints are *not* directly connected; each triangle closes exactly 3 wedges, hence the correction. A value near 1.0 identifies hub-and-spoke structures (market makers or relay accounts); a value near 0.0 signals dense cliques (wash rings).

---

## 4-Node Motif Taxonomy

| Motif | Pattern | Wash-ring signal |
|-------|---------|-----------------|
| **4-cycle** (C₄) | Square: a–b–c–d–a | High — closed loops with no shared centre |
| **4-path** (P₄) | Linear chain: a–b–c–d | Neutral — common in liquidity routing |
| **Star** (K₁,₃) | One hub, three leaves | Low — hub-spoke topology |
| **Diamond** (K₄ − edge) | 4 nodes, 5 edges, missing one edge | High — near-clique |
| **Complete** (K₄) | 6 edges, all connected | Very high — perfect clique |

### cycle\_4\_count

The number of distinct induced 4-cycles is computed from the A⁴ trace formula:

```
trace(A⁴) = 2m  +  2·Σ_v d_v(d_v−1)  +  8·C4
```

Rearranging:

```
C4 = ( trace(A⁴) − 2m − 2·Σ_v d_v(d_v−1) ) / 8
```

where `m = |E|` and `d_v` is the degree of node v.

`trace(A⁴)` is computed as `Σ_{i,j} (A²)_{ij}²` — a single element-wise square and sum after one matrix multiply, avoiding an explicit fourth-power matrix.

The raw integer count is stored in `MotifCensusResult.cycle_4_count`.  When integrated into the community feature vector via `enrich_communities_with_motifs`, it is normalised by community size:

```
cycle_4_per_node = cycle_4_count / n
```

---

## Reciprocity

```
reciprocity = |{(u,v) ∈ E : (v,u) ∈ E}| / |E|
```

Reciprocity measures the fraction of directed edges that have a matching reverse edge.  Wash rings tend to exhibit high reciprocity (coordinated round-trip flows), whereas organic trading creates more directed, one-way paths.

For undirected graphs, reciprocity is defined as 1.0 by convention.

---

## Feature Normalisation

All features are normalised so they are comparable across communities of different sizes:

| Feature | Normalisation |
|---------|--------------|
| `triangle_density` | Divided by `C(n, 3)` — ranges 0–1 |
| `star_ratio` | Ratio of open-wedge to total 3-node motifs — ranges 0–1 |
| `cycle_4_per_node` | `cycle_4_count / n` — rate per node |
| `reciprocity` | Already a fraction — ranges 0–1 |

---

## Timeout and Sampling Strategy

### Large community sampling

Communities with more than **500 nodes** are replaced by a 500-node random induced subgraph before any matrix operations are performed.  The sampled subgraph uses a fixed NumPy seed (42) for reproducibility.  The `was_sampled` flag in `MotifCensusResult` indicates when this occurred.

### Timeout

The motif census is time-bounded by `MOTIF_CENSUS_TIMEOUT_SECONDS` (default **5 seconds**, configurable via environment variable).  The deadline is checked between each computation phase:

1. Triangle counting (A³ method)
2. Star/open-wedge counting
3. 4-cycle counting (A⁴ method)
4. Reciprocity computation

If the deadline is exceeded at any checkpoint, computation halts immediately and partial results are returned with `census_truncated = True`.  Downstream consumers should treat truncated results as lower-confidence signals and may choose to exclude them from model input.

```python
result = compute_motif_census(subgraph, known_nodes, timeout_seconds=2.0)
if result.census_truncated:
    logger.warning("Motif census truncated for community %s", cid)
```

---

## Security: Subgraph Validation

`compute_motif_census` validates that every node in the supplied subgraph is present in the `known_nodes` set (the node set of the parent wallet graph).  Subgraphs referencing external wallet addresses are rejected with a `ValueError` before any computation begins, preventing injection of synthetic nodes that could skew structural features.

```python
# Raises ValueError: "Subgraph contains N node(s) not in the known graph: ..."
compute_motif_census(subgraph_with_external_node, known_nodes)
```

---

## Integration Example

```python
from detection.community_detector import detect_communities, enrich_communities_with_motifs

community_map = detect_communities(graph)
motif_features = enrich_communities_with_motifs(graph, community_map, timeout_seconds=5.0)

for cid, features in motif_features.items():
    print(
        f"Community {cid}: "
        f"triangle_density={features['triangle_density']:.3f}, "
        f"star_ratio={features['star_ratio']:.3f}, "
        f"cycle_4_per_node={features['cycle_4_per_node']:.4f}, "
        f"reciprocity={features['reciprocity']:.3f}"
        + (" [TRUNCATED]" if features["census_truncated"] else "")
    )
```
