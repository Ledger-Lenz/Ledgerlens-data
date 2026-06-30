"""Motif census for wash-trading ring structural fingerprinting.

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
    star_count: int = 0            # open wedges (P3 patterns)
    star_ratio: float = 0.0        # star_count / (star_count + triangle_count)

    # 4-node motifs
    cycle_4_count: int = 0         # distinct 4-cycles (C4 subgraphs)

    # Directed edge structure
    reciprocity: float = 0.0       # fraction of directed edges with reverse present

    # Metadata
    node_count: int = 0
    was_sampled: bool = False       # True when >500-node community was subsampled
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
        result.triangle_density = (
            triangle_count / max_triangles if max_triangles > 0 else 0.0
        )

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
