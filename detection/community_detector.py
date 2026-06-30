"""Wallet clustering via Louvain community detection for wash-trading ring identification.

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
