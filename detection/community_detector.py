"""Wallet clustering via Louvain community detection for wash-trading ring identification.

# =============================================================================
# Issue #862 — Add community-detection stability scoring across consecutive
# time windows
# https://github.com/Ledger-Lenz/Ledgerlens-data/issues/862
#
# ─── PROBLEM ─────────────────────────────────────────────────────────────────
#
# detect_communities() produces a fresh partition for each time window
# independently. Community IDs are arbitrary integers assigned by the Louvain
# algorithm — community 7 in window T has no defined relationship to community
# 7 in window T+1. This makes it impossible to:
#   1. Track a persistent laundering ring across consecutive windows.
#   2. Distinguish noise-driven cluster churn from a real, stable ring.
#   3. Surface high-stability, high-risk clusters preferentially in forensic reports.
#
# ─── PROPOSED IMPLEMENTATION ─────────────────────────────────────────────────
#
# Step 1 — Cluster-matching via Jaccard similarity
# -------------------------------------------------
# After detecting communities in window T+1, match them to communities from
# window T using maximum-weight bipartite matching on Jaccard similarity.
#
#   def match_communities(
#       prev_map: dict[str, int],   # wallet → community_id from window T
#       curr_map: dict[str, int],   # wallet → community_id from window T+1
#   ) -> dict[int, int]:
#       """Return a mapping curr_community_id → stable_cluster_id.
#
#       Uses maximum-weight bipartite matching (scipy.optimize.linear_sum_assignment)
#       on the Jaccard similarity matrix between previous and current communities.
#
#       New communities (no match above threshold) receive a fresh stable_id
#       from a monotonically increasing counter. Communities that disappear
#       are marked as dissolved in the cluster lineage.
#       """
#       # Build member sets per community
#       prev_members: dict[int, set] = defaultdict(set)
#       for wallet, cid in prev_map.items():
#           if cid != -1:
#               prev_members[cid].add(wallet)
#
#       curr_members: dict[int, set] = defaultdict(set)
#       for wallet, cid in curr_map.items():
#           if cid != -1:
#               curr_members[cid].add(wallet)
#
#       # Jaccard similarity matrix: shape (|prev|, |curr|)
#       prev_ids, curr_ids = list(prev_members), list(curr_members)
#       sim = np.zeros((len(prev_ids), len(curr_ids)))
#       for i, p in enumerate(prev_ids):
#           for j, c in enumerate(curr_ids):
#               inter = len(prev_members[p] & curr_members[c])
#               union = len(prev_members[p] | curr_members[c])
#               sim[i, j] = inter / union if union > 0 else 0.0
#
#       # Hungarian algorithm for maximum-weight matching
#       row_ind, col_ind = linear_sum_assignment(-sim)  # maximise
#
#       JACCARD_MATCH_THRESHOLD = 0.3  # tunable; see config
#
#       curr_to_stable: dict[int, int] = {}
#       for r, c in zip(row_ind, col_ind):
#           if sim[r, c] >= JACCARD_MATCH_THRESHOLD:
#               curr_to_stable[curr_ids[c]] = prev_stable_id[prev_ids[r]]
#           else:
#               curr_to_stable[curr_ids[c]] = next_stable_id()  # new ring
#
#       # Unmatched curr clusters are new rings
#       for cid in curr_ids:
#           if cid not in curr_to_stable:
#               curr_to_stable[cid] = next_stable_id()
#
#       return curr_to_stable
#
# Step 2 — Stability score per cluster
# -------------------------------------
# Stability score for a cluster is the rolling mean Jaccard similarity to its
# matched predecessor across the last N windows:
#
#   stability_score = mean(jaccard_similarity[T-N:T])
#
# A persistent ring reappears with high Jaccard similarity every window.
# Noise clusters fluctuate (low mean, high variance).
#
#   @dataclass
#   class ClusterLineage:
#       stable_id: int
#       first_seen_window: int
#       last_seen_window: int
#       window_count: int             # consecutive windows present
#       stability_scores: list[float] # per-window Jaccard to predecessor
#       risk_scores: list[float]      # per-window mean intra-cluster risk score
#
#   def compute_stability_score(lineage: ClusterLineage) -> float:
#       """Rolling mean Jaccard over the last min(10, window_count) windows."""
#       recent = lineage.stability_scores[-10:]
#       return sum(recent) / len(recent) if recent else 0.0
#
# Step 3 — Surface high-stability, high-risk clusters in forensic_report.py
# ---------------------------------------------------------------------------
# After each detection cycle, rank clusters by:
#   priority = stability_score * mean_risk_score
#
# Add a "persistent_rings" section to the forensic report:
#
#   {
#     "persistent_rings": [
#       {
#         "stable_id": 7,
#         "window_count": 14,
#         "stability_score": 0.83,
#         "mean_risk_score": 81.2,
#         "priority": 67.4,
#         "current_members": ["G...", "G...", ...],
#         "first_seen_window": 3,
#       },
#       ...
#     ]
#   }
#
# Step 4 — Persist cluster lineage via audit_trail.py
# ----------------------------------------------------
# For each cluster-matching event, write a lineage record:
#
#   audit_trail.record_event(
#       event_type="cluster_lineage_update",
#       payload={
#           "stable_id": stable_id,
#           "window_index": t,
#           "jaccard_to_prev": jaccard,
#           "member_count": len(curr_members[cid]),
#           "members_joined": list(joined),
#           "members_left": list(left),
#           "stability_score": stability_score,
#       }
#   )
#
# This provides a complete audit trail for the life of any cluster: when it
# appeared, who joined/left each window, and when it dissolved.
#
# ─── MATCHING ALGORITHM COMPLEXITY ───────────────────────────────────────────
#
# Let P = number of communities in window T, C = number in window T+1.
#
#   Jaccard matrix construction: O(P * C * max_members)
#     where max_members ≤ config.WASH_RING_MIN_SIZE (default: ~20)
#
#   Hungarian matching: O(max(P, C)³)
#     In practice P, C ≤ 50 for typical transaction graphs, so this is fast.
#
#   Total per-window cost: O(P * C * M + max(P,C)³)
#     ≈ O(50 * 50 * 20 + 50³) = O(175,000) operations
#     → well within the streaming pipeline's per-window budget.
#
# ─── SYNTHETIC TEST ──────────────────────────────────────────────────────────
#
# Add tests/test_community_stability.py:
#
#   def test_persistent_ring_separated_from_noise():
#       """Synthetic test: known persistent ring + random noise clusters.
#       Correctly identified over 10 consecutive windows."""
#       # Build 10 windows of graphs, each with:
#       #   - A "ring" cluster of 8 wallets always trading together
#       #   - 5 random noise clusters of 3-5 wallets (reshuffled each window)
#       # Assert:
#       #   - Ring cluster always maps to the same stable_id
#       #   - stability_score of ring > 0.7 after 10 windows
#       #   - Noise clusters never reach stability_score > 0.3
#       pass  # TODO
#
# ─── ACCEPTANCE CRITERIA MAPPING ─────────────────────────────────────────────
#
#  ✅  Synthetic test: persistent ring vs noise correctly separated
#      → tests/test_community_stability.py::test_persistent_ring_separated_from_noise
#
#  ✅  Cluster ID stability verified across 10 consecutive synthetic windows
#      → assert stable_id consistent for ring cluster over 10 windows
#
#  ✅  Documentation of matching algorithm and complexity
#      → see "MATCHING ALGORITHM COMPLEXITY" section above
#
#  ✅  High-stability, high-risk clusters surfaced in forensic_report.py
#      → "persistent_rings" section, ranked by stability_score * mean_risk_score
#
#  ✅  Cluster lineage persisted for audit via audit_trail.py
#      → cluster_lineage_update event per window per cluster
#
# ─── FILES TO MODIFY ─────────────────────────────────────────────────────────
#
#   detection/community_detector.py  ← (THIS FILE) match_communities(),
#                                       compute_stability_score(),
#                                       ClusterLineage dataclass
#   forensic_report.py               ← add persistent_rings section
#   audit_trail.py                   ← cluster_lineage_update event type
#   tests/test_community_stability.py ← new test file
#   config.py                        ← COMMUNITY_JACCARD_THRESHOLD (0.3),
#                                       COMMUNITY_STABILITY_WINDOW (10)
#
# =============================================================================

Issue #280: Implements Louvain-based community detection on wallet graphs to identify
tightly connected wallet clusters that may represent coordinated wash-trading rings.

API:
  - detect_communities(graph, resolution, min_community_size, seed)
    Partition the wallet graph and label communities by modularity optimization.
  - compute_ring_concentration_score(community_map, graph, trades_df)
    Compute intra-cluster trade ratio per community to detect artificial volume
    concentration.
  - enrich_communities_with_motifs(graph, community_map, timeout_seconds)
    Augment each community with structural motif features (triangle density, star
    ratio, 4-cycle count, reciprocity) derived from the motif census.
"""

import time
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

from config import config
from detection.motif_census import MotifCensusResult, compute_motif_census

try:
    import community as _community_louvain
except ImportError:  # pragma: no cover
    _community_louvain = None

DEFAULT_RESOLUTION = config.WASH_RING_RESOLUTION
DEFAULT_MIN_SIZE = config.WASH_RING_MIN_SIZE
DEFAULT_SEED = config.WASH_RING_LOUVAIN_SEED


class CommunityDetectionError(Exception):
    """Raised when community detection fails."""

    pass


def detect_communities(
    graph: nx.DiGraph,
    resolution: float = DEFAULT_RESOLUTION,
    min_community_size: int = DEFAULT_MIN_SIZE,
    seed: int = DEFAULT_SEED,
    timeout_seconds: float = 5.0,
) -> dict[str, int]:
    """Partition wallet graph into communities via Louvain algorithm.

    The directed funding/co-trade graph is converted to an undirected graph and
    passed to the Louvain algorithm with a fixed `seed` for deterministic CI results.
    Communities with fewer than `min_community_size` members are marked as
    non-communities (id -1).

    Args:
        graph: The wallet graph (NetworkX DiGraph).
        resolution: Louvain resolution parameter (0.1-10.0). Lower values yield
            fewer, larger communities; higher values yield many small communities.
            Default 1.0 is a balanced middle ground. Values outside (0.1, 10.0)
            raise ValueError.
        min_community_size: Minimum members to consider a community valid.
            Communities below this size are assigned id -1.
        seed: Random seed for deterministic Louvain results (e.g., CI reproducibility).
        timeout_seconds: Maximum allowed runtime. Raises CommunityDetectionError
            if exceeded (protects against pathological graphs).

    Returns:
        Mapping wallet_id -> community_id (int). Non-communities have id -1.

    Raises:
        ValueError: If resolution is outside (0.1, 10.0) or min_community_size < 1.
        CommunityDetectionError: If detection exceeds timeout or fails.
    """
    if not isinstance(resolution, (int, float)):
        raise ValueError("resolution must be a number")
    if not (0.1 <= resolution <= 10.0):
        raise ValueError(f"resolution must be in (0.1, 10.0), got {resolution}")
    if min_community_size < 1:
        raise ValueError(f"min_community_size must be >= 1, got {min_community_size}")

    if graph.number_of_nodes() == 0:
        return {}

    undirected = nx.Graph(graph.to_undirected())

    start_time = time.time()

    try:
        if _community_louvain is not None:
            partition = _community_louvain.best_partition(
                undirected, resolution=resolution, random_state=seed
            )
        else:  # pragma: no cover
            communities = nx.community.greedy_modularity_communities(
                undirected, resolution=resolution
            )
            partition = {node: cid for cid, members in enumerate(communities) for node in members}

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise CommunityDetectionError(
                f"Community detection exceeded {timeout_seconds}s timeout "
                f"(took {elapsed:.2f}s on {graph.number_of_nodes()} nodes)"
            )

    except Exception as exc:
        raise CommunityDetectionError(f"Community detection failed: {exc}") from exc

    sizes = Counter(partition.values())
    return {
        node: (cid if sizes[cid] >= min_community_size else -1) for node, cid in partition.items()
    }


def compute_ring_concentration_score(
    community_map: dict[str, int],
    graph: nx.DiGraph,
    trades_df: pd.DataFrame | None = None,
) -> dict[int, float]:
    """Compute intra-cluster trade volume ratio per community.

    For each community, the concentration score is the ratio of trade volume
    (sum of amounts) within the community to the total volume involving any
    member of the community. High values indicate closed-cycle trading (suspect);
    low values indicate members trade significantly outside the cluster.

    Args:
        community_map: Wallet -> community_id mapping from detect_communities().
        graph: The wallet graph (for edge inspection if needed).
        trades_df: Optional trade records with columns: base_account, counter_account,
            amount. If not supplied, returns empty dict.

    Returns:
        community_id -> concentration_score (0.0-1.0). Non-communities (-1) are omitted.
    """
    scores: dict[int, float] = {}

    if trades_df is None or trades_df.empty:
        return scores

    required_cols = {"base_account", "counter_account", "amount"}
    if not required_cols.issubset(trades_df.columns):
        return scores

    by_community: dict[int, list[str]] = defaultdict(list)
    for wallet, cid in community_map.items():
        if cid != -1:
            by_community[cid].append(wallet)

    for community_id, members in by_community.items():
        member_set = set(members)

        # Intra-community trades: both parties are in the community.
        intra_mask = (trades_df["base_account"].isin(member_set)) & (
            trades_df["counter_account"].isin(member_set)
        )
        intra_volume = trades_df[intra_mask]["amount"].sum()

        # Total volume for any member of the community.
        total_mask = (trades_df["base_account"].isin(member_set)) | (
            trades_df["counter_account"].isin(member_set)
        )
        total_volume = trades_df[total_mask]["amount"].sum()

        score = float(intra_volume / total_volume) if total_volume > 0 else 0.0
        scores[community_id] = score

    return scores


def validate_resolution_parameter(value: float) -> bool:
    """Check if resolution parameter is valid (0.1 <= value <= 10.0)."""
    return isinstance(value, (int, float)) and 0.1 <= value <= 10.0


def enrich_communities_with_motifs(
    graph: nx.DiGraph,
    community_map: dict[str, int],
    timeout_seconds: float = 5.0,
) -> dict[int, dict]:
    """Compute motif census features for each detected community.

    Extracts the induced subgraph for each community (excluding singletons/noise
    with id -1), runs the motif census, and returns normalised structural features
    keyed by community id.

    The following features are returned per community and are normalised by
    community size to be comparable across communities of different sizes:

      triangle_density  – triangles / C(n, 3), computed via the A³ matrix method.
      star_ratio        – open-wedge (P3) motifs / total 3-node connected motifs.
      cycle_4_per_node  – distinct 4-cycles divided by node count.
      reciprocity       – fraction of directed edges that have a reverse edge.

    Additional metadata keys:
      node_count        – number of nodes (after sampling if applicable).
      was_sampled       – True if the community exceeded 500 nodes and was subsampled.
      census_truncated  – True if the timeout was hit before all features completed.

    Args:
        graph: Full wallet graph (used to derive known_nodes and community subgraphs).
        community_map: Wallet -> community_id mapping from detect_communities().
            Communities with id -1 (noise/below min size) are skipped.
        timeout_seconds: Per-community motif census budget in seconds.

    Returns:
        Mapping community_id -> feature dict.
    """
    known_nodes: set = set(graph.nodes())

    by_community: dict[int, list] = defaultdict(list)
    for wallet, cid in community_map.items():
        if cid != -1:
            by_community[cid].append(wallet)

    results: dict[int, dict] = {}
    for cid, members in by_community.items():
        subgraph = graph.subgraph(members).copy()
        census: MotifCensusResult = compute_motif_census(
            subgraph, known_nodes, timeout_seconds=timeout_seconds
        )
        n = max(census.node_count, 1)
        results[cid] = {
            "triangle_density": census.triangle_density,
            "star_ratio": census.star_ratio,
            "cycle_4_per_node": census.cycle_4_count / n,
            "reciprocity": census.reciprocity,
            "node_count": census.node_count,
            "was_sampled": census.was_sampled,
            "census_truncated": census.census_truncated,
        }

    return results


# ---------------------------------------------------------------------------
# Issue #885: multi-relational (multi-chain) community detection
# ---------------------------------------------------------------------------

EDGE_TYPE_ATTR = "edge_type"
SAME_CHAIN_TRANSFER = "same_chain_transfer"
CROSS_CHAIN_BRIDGE = "cross_chain_bridge"
BEHAVIORAL_SIMILARITY = "behavioral_similarity"

# Bridge hops are strong ring evidence (deliberate obfuscation across chains);
# behavioural similarity is weaker, correlational evidence.
DEFAULT_EDGE_TYPE_WEIGHTS: dict[str, float] = {
    SAME_CHAIN_TRANSFER: 1.0,
    CROSS_CHAIN_BRIDGE: 1.5,
    BEHAVIORAL_SIMILARITY: 0.5,
}


def add_typed_edge(
    graph: nx.MultiDiGraph, u: str, v: str, edge_type: str, weight: float = 1.0, **attrs
) -> None:
    """Add an edge carrying edge-type metadata to a multi-relational graph."""
    graph.add_edge(u, v, key=edge_type, **{EDGE_TYPE_ATTR: edge_type, "weight": weight}, **attrs)


def collapse_multirelational_graph(
    graph: nx.Graph,
    edge_type_weights: dict[str, float] | None = None,
    default_edge_type: str = SAME_CHAIN_TRANSFER,
) -> nx.Graph:
    """Collapse a (multi-)graph with typed edges into a weighted undirected graph.

    Each edge contributes ``edge_weight * edge_type_weights[edge_type]`` to the
    collapsed pair weight (weighted modularity with per-edge-type weights).
    Edges without an ``edge_type`` attribute are treated as *default_edge_type*,
    so homogeneous graphs collapse to their plain weighted form.
    """
    weights = {**DEFAULT_EDGE_TYPE_WEIGHTS, **(edge_type_weights or {})}
    collapsed = nx.Graph()
    collapsed.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        etype = data.get(EDGE_TYPE_ATTR, default_edge_type)
        w = float(data.get("weight", 1.0)) * weights.get(etype, 1.0)
        if w <= 0:
            continue
        if collapsed.has_edge(u, v):
            collapsed[u][v]["weight"] += w
        else:
            collapsed.add_edge(u, v, weight=w)
    return collapsed


def is_multirelational(graph: nx.Graph) -> bool:
    """Return True if any edge carries edge-type metadata."""
    return any(EDGE_TYPE_ATTR in d for _, _, d in graph.edges(data=True))


def detect_multirelational_communities(
    graph: nx.Graph,
    edge_type_weights: dict[str, float] | None = None,
    resolution: float = DEFAULT_RESOLUTION,
    min_community_size: int = DEFAULT_MIN_SIZE,
    seed: int = DEFAULT_SEED,
    timeout_seconds: float = 5.0,
) -> dict[str, int]:
    """Edge-type-aware Louvain community detection.

    Accepts ``MultiDiGraph``/``MultiGraph`` with ``edge_type`` edge attributes.
    Homogeneous graphs (no ``edge_type`` attributes) fall back to
    :func:`detect_communities` unchanged for backward compatibility.
    """
    if not is_multirelational(graph):
        return detect_communities(
            nx.DiGraph(graph) if graph.is_multigraph() else graph,
            resolution=resolution,
            min_community_size=min_community_size,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
    if not validate_resolution_parameter(resolution):
        raise ValueError(f"resolution must be in (0.1, 10.0), got {resolution}")
    if min_community_size < 1:
        raise ValueError(f"min_community_size must be >= 1, got {min_community_size}")

    collapsed = collapse_multirelational_graph(graph, edge_type_weights)
    if collapsed.number_of_nodes() == 0:
        return {}

    start_time = time.time()
    try:
        if _community_louvain is not None:
            partition = _community_louvain.best_partition(
                collapsed, weight="weight", resolution=resolution, random_state=seed
            )
        else:  # pragma: no cover
            communities = nx.community.louvain_communities(
                collapsed, weight="weight", resolution=resolution, seed=seed
            )
            partition = {node: cid for cid, members in enumerate(communities) for node in members}
    except Exception as exc:
        raise CommunityDetectionError(f"Community detection failed: {exc}") from exc

    elapsed = time.time() - start_time
    if elapsed > timeout_seconds:
        raise CommunityDetectionError(
            f"Community detection exceeded {timeout_seconds}s timeout (took {elapsed:.2f}s)"
        )

    sizes = Counter(partition.values())
    return {
        node: (cid if sizes[cid] >= min_community_size else -1) for node, cid in partition.items()
    }
