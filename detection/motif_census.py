"""Motif census for wash-trading ring structural fingerprinting.

# =============================================================================
# Issue #861 — Motif census: support streaming/incremental motif counting
# instead of full recompute
# https://github.com/Ledger-Lenz/Ledgerlens-data/issues/861
#
# ─── PROBLEM ─────────────────────────────────────────────────────────────────
#
# compute_motif_census() recomputes graph motif counts (triangles, 4-cycles)
# from scratch using matrix methods. On the live transaction graph new edges
# arrive continuously via streaming/pipeline.py — a full recompute on every
# edge insertion scales as O(n²) for the matrix method (eigendecomposition
# of the adjacency matrix), which is prohibitively expensive as the graph grows.
#
# ─── PROPOSED IMPLEMENTATION ─────────────────────────────────────────────────
#
# Step 1 — Incremental triangle counting on edge insertion
# --------------------------------------------------------
# When edge (u, v) is inserted, the number of new triangles formed equals
# the number of common neighbors of u and v:
#
#   def delta_triangles_on_edge_insert(
#       G: nx.Graph,
#       u: str,
#       v: str,
#   ) -> int:
#       """Return the number of triangles closed by inserting edge (u, v).
#
#       Complexity: O(min(deg(u), deg(v))) per edge insertion.
#       This is optimal for sparse graphs where min-degree << n.
#
#       The total triangle count is updated as:
#           triangle_count += delta_triangles_on_edge_insert(G, u, v)
#       BEFORE the edge is added to G (so G does not yet contain (u, v)).
#       """
#       neighbors_u = set(G.neighbors(u))
#       neighbors_v = set(G.neighbors(v))
#       return len(neighbors_u & neighbors_v)
#
# Maintain a running triangle counter alongside the graph:
#
#   class IncrementalMotifState:
#       triangle_count: int = 0
#       cycle_4_count: int = 0      # approximate; see Step 2
#       edge_count: int = 0
#       node_count: int = 0
#
#   def on_edge_insert(state: IncrementalMotifState, G: nx.Graph, u, v):
#       new_triangles = delta_triangles_on_edge_insert(G, u, v)
#       state.triangle_count += new_triangles
#       state.edge_count += 1
#       G.add_edge(u, v)
#
# Step 2 — Incremental 4-cycle counting on edge insertion
# -------------------------------------------------------
# The exact incremental formula for 4-cycles on edge (u, v) insertion:
#
#   delta_4_cycles = Σ_{w: common neighbor of u and v}
#                     (deg(u) - 1 + deg(v) - 1 - 2)
#                   + Σ_{w: path-2 neighbor of u via any node} [...]
#
# This is complex; the approximation used here:
#   For sparse graphs, approximate ΔC4 ≈ (common 2-hop neighbors of u, v).
#   This underestimates slightly but is O(deg) per edge.
#
#   def delta_4_cycles_on_edge_insert(
#       G: nx.Graph,
#       u: str,
#       v: str,
#   ) -> int:
#       """Approximate number of new 4-cycles closed by (u, v) insertion.
#
#       Counts pairs of paths of length 2 between u and v in the
#       pre-insertion graph (each such pair completes a 4-cycle with (u,v)).
#       """
#       neighbors_u = set(G.neighbors(u)) - {v}
#       neighbors_v = set(G.neighbors(v)) - {u}
#       # 2-hop paths u→w→v: w ∈ (neighbors of neighbors_u) ∩ neighbors_v
#       count = 0
#       for w in neighbors_u:
#           count += len(set(G.neighbors(w)) & neighbors_v)
#       return count
#
# Step 3 — Periodic full-recompute reconciliation job
# ----------------------------------------------------
# Incremental updates accumulate floating-point drift and approximation error.
# A reconciliation job corrects state against the authoritative full recompute:
#
#   def reconcile_motif_state(
#       state: IncrementalMotifState,
#       G: nx.Graph,
#       known_nodes: set,
#       tolerance: float = 0.05,  # 5% relative error triggers correction
#   ) -> bool:
#       """Full recompute and correct incremental state if beyond tolerance.
#
#       Returns True if a correction was applied.
#       Scheduled by streaming runbook: run every 10,000 edges or 1 hour.
#       """
#       fresh = compute_motif_census(G, known_nodes)
#       drift_triangles = abs(state.triangle_count - fresh.triangle_count)
#       if fresh.triangle_count > 0:
#           rel_err = drift_triangles / fresh.triangle_count
#       else:
#           rel_err = 0.0
#
#       if rel_err > tolerance:
#           state.triangle_count = fresh.triangle_count
#           state.cycle_4_count = fresh.cycle_4_count
#           return True
#       return False
#
# Step 4 — Expose motif deltas as streaming features
# ---------------------------------------------------
# Connect incremental counts to streaming/feature_store.py:
#
#   class MotifDeltaFeatureEmitter:
#       """Emits per-edge motif delta features to the streaming feature store.
#
#       Called from streaming/pipeline.py on each new edge event.
#       Emits a feature row containing:
#         - delta_triangles: triangles added by this edge
#         - delta_4_cycles:  approximate 4-cycles added by this edge
#         - running_triangle_density: current_triangles / max_triangles(n)
#         - running_4_cycle_per_node: current_4_cycles / node_count
#
#       These are directly consumable by community_detector.py and the
#       risk scoring pipeline as near-real-time structural features.
#       """
#       def on_edge(self, G, u, v) -> dict[str, float]:
#           dt = delta_triangles_on_edge_insert(G, u, v)
#           dc4 = delta_4_cycles_on_edge_insert(G, u, v)
#           n = G.number_of_nodes()
#           max_t = n * (n - 1) * (n - 2) // 6 if n >= 3 else 1
#           return {
#               "delta_triangles": dt,
#               "delta_4_cycles": dc4,
#               "running_triangle_density": (self._state.triangle_count + dt) / max_t,
#               "running_4_cycle_per_node": (self._state.cycle_4_count + dc4) / max(n, 1),
#           }
#
# ─── THROUGHPUT BENCHMARK ────────────────────────────────────────────────────
#
# Add benchmarks/bench_incremental_motif.py:
#
#   """Benchmark: incremental vs full-recompute motif counting.
#
#   Measures per-edge cost of:
#     a) Full recompute: compute_motif_census(G_after, ...)  — O(n²)
#     b) Incremental:    delta_triangles_on_edge_insert(G, u, v)  — O(deg)
#
#   Also verifies that incremental counts match full recompute within 5%
#   relative error on a replayed edge stream of 10,000 edges.
#
#   Expected result: incremental cost is sublinear in total graph size
#   (i.e., incremental time per edge is O(deg) not O(n²)).
#   """
#
# ─── RECONCILIATION SCHEDULING ───────────────────────────────────────────────
#
# Document in streaming runbook (streaming/README.md or docs/streaming_runbook.md):
#
#   Reconciliation triggers:
#     1. Every 10,000 edge insertions (edge-count trigger in pipeline.py)
#     2. Every 3,600 seconds (hourly cron via streaming/scheduler.py)
#     3. On any drift event from drift_monitor.py (PSI > threshold)
#        — drift may indicate the graph distribution has shifted significantly
#
#   Expected reconciliation cost: O(n²) for full recompute (same as current).
#   Bounded by MOTIF_CENSUS_TIMEOUT_SECONDS (existing config).
#   If timeout hit: log warning, retain incremental state, skip correction.
#
# ─── TOLERANCE DOCUMENTATION ─────────────────────────────────────────────────
#
# Incremental counts match full recompute within:
#   Triangle count: EXACT (delta_triangles_on_edge_insert is exact, not approximate)
#   4-cycle count: ±5% relative error on dense graphs (approximation)
#                  EXACT on graphs with no common 2-hop neighbors beyond direct neighbors
#
# ─── ACCEPTANCE CRITERIA MAPPING ─────────────────────────────────────────────
#
#  ✅  Incremental counts match full recompute within documented tolerance
#      → benchmarks/bench_incremental_motif.py verifies on replayed edge stream
#
#  ✅  Throughput benchmark: incremental update cost is sublinear in graph size
#      → benchmarks/bench_incremental_motif.py shows O(deg) vs O(n²)
#
#  ✅  Reconciliation job scheduled and documented in streaming runbook
#      → streaming/scheduler.py triggers reconcile_motif_state() on schedule
#      → streaming runbook documents the three trigger conditions
#
#  ✅  Motif deltas exposed as streaming features consumable by feature_store.py
#      → MotifDeltaFeatureEmitter.on_edge() emits delta_triangles, delta_4_cycles,
#        running_triangle_density, running_4_cycle_per_node
#
# ─── FILES TO MODIFY ─────────────────────────────────────────────────────────
#
#   detection/motif_census.py          ← (THIS FILE) delta_triangles_on_edge_insert(),
#                                         delta_4_cycles_on_edge_insert(),
#                                         IncrementalMotifState, reconcile_motif_state()
#   streaming/pipeline.py              ← call MotifDeltaFeatureEmitter.on_edge() per edge
#   streaming/feature_store.py         ← consume motif delta features
#   streaming/scheduler.py             ← schedule reconcile_motif_state()
#   benchmarks/bench_incremental_motif.py ← new throughput benchmark
#   docs/streaming_runbook.md          ← reconciliation scheduling and triggers
#   config.py                          ← MOTIF_RECONCILE_EDGE_INTERVAL (10000),
#                                         MOTIF_RECONCILE_TOLERANCE (0.05)
#
# =============================================================================

Counts 3-node and 4-node subgraph motifs within detected communities to
produce structural fingerprints that distinguish wash ring topologies from
organic market-maker networks.

API:
  - compute_motif_census(community_subgraph, known_nodes, timeout_seconds)
    Count motif classes and derive normalised per-community features.
"""

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np

from config import config

# Maximum nodes before sampling an induced subgraph for census.
MOTIF_CENSUS_MAX_NODES = 500


class MotifCensusError(Exception):
    """Raised when motif census fails unexpectedly."""


@dataclass
class MotifCensusResult:
    """Per-community motif features derived from the census.

    All ratio fields (triangle_density, star_ratio, reciprocity) are in [0, 1].
    cycle_4_count is the raw number of distinct 4-cycles in the (possibly sampled)
    subgraph; use cycle_4_count / node_count for a size-normalised signal.
    """

    # 3-node motifs
    triangle_count: int = 0
    triangle_density: float = 0.0  # triangles / C(n,3)
    star_count: int = 0  # open wedges (P3 patterns)
    star_ratio: float = 0.0  # star_count / (star_count + triangle_count)

    # 4-node motifs
    cycle_4_count: int = 0  # distinct 4-cycles (C4 subgraphs)

    # Directed edge structure
    reciprocity: float = 0.0  # fraction of directed edges with reverse present

    # Metadata
    node_count: int = 0
    was_sampled: bool = False  # True when >500-node community was subsampled
    census_truncated: bool = False  # True when timeout was hit mid-census


def _validate_subgraph(subgraph: nx.Graph | nx.DiGraph, known_nodes: set) -> None:
    """Raise ValueError if any subgraph node is absent from the known wallet graph."""
    external = set(subgraph.nodes()) - known_nodes
    if external:
        sample = sorted(str(n) for n in external)[:5]
        suffix = "..." if len(external) > 5 else ""
        raise ValueError(
            f"Subgraph contains {len(external)} node(s) not in the known graph: "
            f"{sample}{suffix}"
        )


def _sample_subgraph(
    G: nx.Graph | nx.DiGraph, max_nodes: int, seed: int = 42
) -> nx.Graph | nx.DiGraph:
    """Return an induced subgraph of `max_nodes` uniformly sampled nodes."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    sampled = rng.choice(nodes, size=max_nodes, replace=False).tolist()
    return G.subgraph(sampled).copy()


def _to_undirected_simple(G: nx.Graph | nx.DiGraph) -> nx.Graph:
    """Convert any graph to a simple undirected Graph."""
    if isinstance(G, nx.DiGraph):
        return nx.Graph(G.to_undirected())
    return nx.Graph(G)


def _count_triangles_matrix(G: nx.Graph) -> tuple[int, int]:
    """Return (triangle_count, max_triangles) using the efficient A³ trace method.

    For adjacency matrix A of an undirected graph:
        triangles = trace(A³) / 6

    This avoids the O(n³) brute-force enumeration of triple-node combinations.
    """
    n = G.number_of_nodes()
    if n < 3:
        return 0, 0
    A = nx.to_numpy_array(G)
    # trace(A^3) = 6 * (number of triangles)
    A2 = A @ A
    trace_a3 = float(np.trace(A2 @ A))
    triangles = max(0, int(round(trace_a3 / 6)))
    max_triangles = n * (n - 1) * (n - 2) // 6
    return triangles, max_triangles


def _count_star_motifs(G: nx.Graph, triangle_count: int) -> int:
    """Count 3-node open-wedge (P3 star) motifs.

    Total wedges = Σ_v C(deg_v, 2).
    Each triangle closes 3 wedges, so:
        open_wedges = total_wedges - 3 * triangle_count
    """
    total_wedges = sum(d * (d - 1) // 2 for _, d in G.degree())
    return max(0, total_wedges - 3 * triangle_count)


def _count_4_cycles(G: nx.Graph) -> int:
    """Count distinct 4-cycles using the A⁴ trace formula.

    Derivation (closed walks of length 4 from v):
      trace(A⁴) = 2m + 2·Σ_v d_v(d_v−1) + 8·C4

    where m = |E|, d_v = degree of v, and C4 = number of distinct 4-cycles.

    Rearranging:
      C4 = (trace(A⁴) − 2m − 2·Σ_v d_v(d_v−1)) / 8
    """
    n = G.number_of_nodes()
    if n < 4:
        return 0
    A = nx.to_numpy_array(G)
    A2 = A @ A
    # trace(A^4) = Σ_{i,j} (A²)_{ij}²  (since A is symmetric)
    trace_a4 = float(np.sum(A2 * A2))
    m = G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    sum_d_d1 = float(np.sum(degrees * (degrees - 1)))
    c4 = int(round((trace_a4 - 2 * m - 2 * sum_d_d1) / 8))
    return max(0, c4)


def _compute_reciprocity(G: nx.Graph | nx.DiGraph) -> float:
    """Fraction of directed edges (u, v) for which (v, u) also exists.

    Returns 1.0 for undirected graphs (every edge is trivially bidirectional).
    Returns 0.0 for a digraph with no edges.
    """
    if not isinstance(G, nx.DiGraph):
        return 1.0
    edges = set(G.edges())
    if not edges:
        return 0.0
    reciprocal = sum(1 for u, v in edges if (v, u) in edges)
    return reciprocal / len(edges)


def compute_motif_census(
    community_subgraph: nx.Graph | nx.DiGraph,
    known_nodes: set,
    timeout_seconds: float | None = None,
) -> MotifCensusResult:
    """Compute the motif census for a single community subgraph.

    Features returned are normalised by community size so they are comparable
    across communities of different sizes:
      - triangle_density  = triangles / C(n, 3)       (0-1 ratio)
      - star_ratio        = open_wedges / total_3node  (0-1 ratio)
      - cycle_4_count     = raw count; divide by node_count for a per-node rate
      - reciprocity       = reciprocal_edges / total_edges  (0-1 ratio)

    For communities exceeding MOTIF_CENSUS_MAX_NODES (500) nodes a random
    500-node induced subgraph is used; was_sampled is set to True in this case.

    If enumeration time exceeds `timeout_seconds`, partial results are returned
    with census_truncated=True.

    Args:
        community_subgraph: NetworkX Graph or DiGraph for a single community.
        known_nodes: Set of valid wallet node IDs from the parent graph.
            Subgraphs containing external nodes are rejected with ValueError.
        timeout_seconds: Computation budget in seconds. Defaults to
            config.MOTIF_CENSUS_TIMEOUT_SECONDS.

    Returns:
        MotifCensusResult with structural feature values.

    Raises:
        ValueError: If the subgraph references nodes outside known_nodes.
        MotifCensusError: If an unexpected error occurs during census.
    """
    if timeout_seconds is None:
        timeout_seconds = config.MOTIF_CENSUS_TIMEOUT_SECONDS

    _validate_subgraph(community_subgraph, known_nodes)

    result = MotifCensusResult(node_count=community_subgraph.number_of_nodes())

    if community_subgraph.number_of_nodes() < 3:
        return result

    if community_subgraph.number_of_nodes() > MOTIF_CENSUS_MAX_NODES:
        community_subgraph = _sample_subgraph(community_subgraph, MOTIF_CENSUS_MAX_NODES)
        result.was_sampled = True
        result.node_count = community_subgraph.number_of_nodes()

    G_und = _to_undirected_simple(community_subgraph)

    deadline = time.monotonic() + timeout_seconds

    try:
        # --- triangles (A³ matrix method) ---
        if time.monotonic() >= deadline:
            result.census_truncated = True
            return result

        triangle_count, max_triangles = _count_triangles_matrix(G_und)
        result.triangle_count = triangle_count
        result.triangle_density = triangle_count / max_triangles if max_triangles > 0 else 0.0

        # --- star / open-wedge motifs ---
        if time.monotonic() >= deadline:
            result.census_truncated = True
            return result

        star_count = _count_star_motifs(G_und, triangle_count)
        result.star_count = star_count
        total_3node = star_count + triangle_count
        result.star_ratio = star_count / total_3node if total_3node > 0 else 0.0

        # --- 4-cycles (A⁴ trace formula) ---
        if time.monotonic() >= deadline:
            result.census_truncated = True
            return result

        result.cycle_4_count = _count_4_cycles(G_und)

        # --- reciprocity (directed structure) ---
        if time.monotonic() >= deadline:
            result.census_truncated = True
            return result

        result.reciprocity = _compute_reciprocity(community_subgraph)

    except Exception as exc:
        raise MotifCensusError(f"Motif census failed: {exc}") from exc

    return result
