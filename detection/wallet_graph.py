"""Wallet funding-graph features: `funding_source_similarity` and
`network_centrality`.

Builds a directed graph of "funded by" relationships from
`AccountActivity.funding_account` and derives two signals used by
`feature_engineering.compute_wallet_graph_features`:

- `funding_source_similarity`: the highest Jaccard similarity between a
  wallet's set of funding ancestors and any other wallet's funding-ancestor
  set. A high value means two wallets trace back to the same funding
  source(s) — a common pattern for sock-puppet / wash-trading rings.
- `network_centrality`: degree centrality of the wallet within the funding
  graph, a proxy for how connected/influential an account is within the
  observed funding network.

Also provides:

- `build_co_trade_graph`: builds edges between wallets that co-traded the
  same asset pair within a configurable time window.
"""

import hashlib
import re
import threading
import warnings
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from itertools import combinations
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

from config import config
from ingestion.data_models import AccountActivity

try:
    from prometheus_client import Gauge

    _graph_nodes_total = Gauge(
        "wallet_graph_nodes_total",
        "Current number of nodes in the incremental wallet graph",
    )
    _graph_edges_total = Gauge(
        "wallet_graph_edges_total",
        "Current number of edges in the incremental wallet graph",
    )
    _graph_stale_edges_total = Gauge(
        "wallet_graph_stale_edges_total",
        "Number of stale edges removed in the last removal pass",
    )
except Exception:  # pragma: no cover
    _graph_nodes_total = None  # type: ignore[assignment]
    _graph_edges_total = None  # type: ignore[assignment]
    _graph_stale_edges_total = None  # type: ignore[assignment]


def _parse_edge_timestamp(ts) -> datetime:
    """Parse an edge timestamp to a UTC-aware datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime().replace(tzinfo=UTC) if ts.tzinfo is None else ts.to_pydatetime()
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=UTC)

try:  # python-louvain — preferred community detector
    import community as _community_louvain
except ImportError:  # pragma: no cover - fallback path exercised only without the dep
    _community_louvain = None

# Stellar account ID format: G followed by 55 uppercase base-32 chars
_STELLAR_ACCOUNT_RE = re.compile(r"^G[A-Z2-7]{55}$")

# Wash-trading ring detection defaults (Issue #11).
DEFAULT_MAX_DEPTH = config.WALLET_GRAPH_MAX_DEPTH
DEFAULT_MIN_RING_SIZE = config.WASH_RING_MIN_SIZE
DEFAULT_RESOLUTION = config.WASH_RING_RESOLUTION
DEFAULT_SEED = config.WASH_RING_LOUVAIN_SEED

NO_RING = -1


def _validate_account_id(account_id: str) -> bool:
    """Return True if *account_id* matches the Stellar account ID format."""
    return bool(_STELLAR_ACCOUNT_RE.match(account_id))


def build_funding_graph(
    activities: Iterable[AccountActivity],
    trades: pd.DataFrame | None = None,
    validate_account_ids: bool = False,
    *,
    co_trade_window: str | pd.Timedelta = "5min",
    output_format: Literal["networkx", "pyg"] = "networkx",
    node_features: pd.DataFrame | Mapping[str, Sequence[float]] | None = None,
) -> "nx.DiGraph":
    """Build the wallet graph, preserving the historical NetworkX default.

    Funding edges point from funder to funded account. When ``trades`` is
    supplied, wallets active in the same asset pair within ``co_trade_window``
    are connected in both directions. Set ``output_format="pyg"`` to obtain a
    :class:`torch_geometric.data.Data` object suitable for GraphSAGE.
    """
    if output_format not in {"networkx", "pyg"}:
        raise ValueError("output_format must be 'networkx' or 'pyg'")

    graph: nx.DiGraph = nx.DiGraph()
    for activity in activities:
        if validate_account_ids and not _validate_account_id(activity.account_id):
            continue
        graph.add_node(activity.account_id)
        if activity.funding_account:
            if validate_account_ids and not _validate_account_id(activity.funding_account):
                continue
            graph.add_edge(activity.funding_account, activity.account_id, edge_type="funding")

    if trades is not None and not trades.empty:
        _add_co_trade_edges(graph, trades, pd.Timedelta(co_trade_window))

    if output_format == "pyg":
        return to_pyg_data(graph, node_features=node_features)
    return graph


def _add_co_trade_edges(graph: nx.DiGraph, trades: pd.DataFrame, window: pd.Timedelta) -> None:
    """Add bidirectional edges between wallets co-active on an asset pair."""
    required = {"base_account", "counter_account", "ledger_close_time"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing required columns: {sorted(missing)}")

    frame = trades.copy()
    if "pair_id" not in frame:
        if not {"base_asset", "counter_asset"}.issubset(frame.columns):
            raise ValueError("trades require pair_id or base_asset/counter_asset columns")
        frame["pair_id"] = (
            frame["base_asset"].astype(str) + "/" + frame["counter_asset"].astype(str)
        )
    frame["ledger_close_time"] = pd.to_datetime(frame["ledger_close_time"], utc=True)

    for _, pair_trades in frame.groupby("pair_id", sort=False):
        records = pair_trades.sort_values("ledger_close_time").to_dict("records")
        for left_index, left in enumerate(records):
            left_time = left["ledger_close_time"]
            active_wallets = {left["base_account"], left["counter_account"]}
            for right in records[left_index + 1 :]:
                if right["ledger_close_time"] - left_time > window:
                    break
                active_wallets.update((right["base_account"], right["counter_account"]))
            for source, target in combinations(sorted(active_wallets), 2):
                graph.add_edge(source, target, edge_type="co_trade")
                graph.add_edge(target, source, edge_type="co_trade")


def to_pyg_data(
    graph: nx.DiGraph,
    node_features: pd.DataFrame | Mapping[str, Sequence[float]] | None = None,
):
    """Convert a wallet graph to PyG ``Data`` with a stable wallet mapping.

    ``node_features`` may be a feature matrix containing a ``wallet`` column,
    or a wallet-to-vector mapping. Missing wallets receive zero vectors. With
    no feature input, a constant feature is used so topology remains usable.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - exercised without GNN extra
        raise ImportError("PyG output requires torch and torch-geometric") from exc

    wallet_ids = list(graph.nodes)
    node_index = {wallet: index for index, wallet in enumerate(wallet_ids)}
    feature_map: dict[str, np.ndarray] = {}
    width = 1
    if isinstance(node_features, pd.DataFrame):
        if "wallet" not in node_features:
            raise ValueError("node feature DataFrame must contain a wallet column")
        feature_columns = [
            column
            for column in node_features.columns
            if column not in {"wallet", "label"}
            and pd.api.types.is_numeric_dtype(node_features[column])
        ]
        width = len(feature_columns) or 1
        feature_map = {
            str(row["wallet"]): row[feature_columns].to_numpy(dtype=np.float32)
            for _, row in node_features.iterrows()
        }
    elif node_features:
        feature_map = {
            str(wallet): np.asarray(values, dtype=np.float32)
            for wallet, values in node_features.items()
        }
        widths = {len(values) for values in feature_map.values()}
        if len(widths) != 1:
            raise ValueError("all node feature vectors must have equal length")
        width = widths.pop()

    default = (
        np.ones(width, dtype=np.float32) if not feature_map else np.zeros(width, dtype=np.float32)
    )
    x = (
        np.stack([feature_map.get(str(wallet), default) for wallet in wallet_ids])
        if wallet_ids
        else np.empty((0, width), dtype=np.float32)
    )
    edges = [(node_index[source], node_index[target]) for source, target in graph.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index)
    data.wallet_ids = wallet_ids
    data.node_index = node_index
    return data


def build_co_trade_graph(
    trades_df: pd.DataFrame,
    window_hours: int,
) -> nx.DiGraph:
    """Build a directed co-trade graph from a trades DataFrame."""
    graph: nx.DiGraph = nx.DiGraph()
    if trades_df.empty:
        return graph

    required_cols = {
        "base_account",
        "counter_account",
        "base_asset",
        "counter_asset",
        "ledger_close_time",
        "amount",
    }
    if not required_cols.issubset(trades_df.columns):
        return graph

    df = trades_df.copy()
    df["ledger_close_time"] = pd.to_datetime(df["ledger_close_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["ledger_close_time"])

    if "pair_id" not in df.columns:
        df["pair_id"] = df.apply(
            lambda row: "/".join(sorted([str(row["base_asset"]), str(row["counter_asset"])])),
            axis=1,
        )

    window_td = pd.Timedelta(hours=window_hours)
    for _, pair_df in df.groupby("pair_id"):
        pair_df = pair_df.sort_values("ledger_close_time")
        events: list[tuple[str, pd.Timestamp]] = []
        for _, row in pair_df.iterrows():
            for account in (row["base_account"], row["counter_account"]):
                if _validate_account_id(str(account)):
                    events.append((str(account), row["ledger_close_time"]))

        if len(events) < 2:
            continue

        events.sort(key=lambda item: item[1])
        for index, (wallet_a, time_a) in enumerate(events):
            for wallet_b, time_b in events[index + 1 :]:
                if time_b - time_a > window_td:
                    break
                if wallet_a == wallet_b:
                    continue
                for source, target in ((wallet_a, wallet_b), (wallet_b, wallet_a)):
                    if graph.has_edge(source, target):
                        graph[source][target]["weight"] += 1
                    else:
                        graph.add_edge(
                            source,
                            target,
                            edge_type="co_trade",
                            weight=1,
                            timestamp=time_a.isoformat(),
                        )

    return graph


def multi_hop_ancestors(graph: nx.DiGraph, wallet: str, max_depth: int) -> set[str]:
    """Return funding ancestors of `wallet` reachable within `max_depth` hops.

    Performs a bounded breadth-first traversal over predecessor edges. A
    `visited` set guarantees each node is expanded once, so the traversal is
    linear in the size of the reachable subgraph and safe on cyclic graphs.
    """
    if wallet not in graph or max_depth < 1:
        return set()

    visited: set[str] = set()
    frontier = {wallet}
    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for node in frontier:
            for pred in graph.predecessors(node):
                if pred != wallet and pred not in visited:
                    visited.add(pred)
                    next_frontier.add(pred)
        if not next_frontier:
            break
        frontier = next_frontier
    return visited


def funding_source_similarity(wallet: str, graph: nx.DiGraph) -> float:
    r"""Highest Jaccard similarity between ``wallet`` funding ancestors and
    any other node's funding ancestors in ``graph``.

    Returns ``0.0`` if ``wallet`` isn't in the graph or has no funding
    ancestors.

    .. deprecated::
        Use :class:`detection.gnn_encoder.GNNEncoder` embeddings instead.
        This scalar feature is preserved for backwards compatibility with the
        existing model artifact.
    """
    warnings.warn(
        "funding_source_similarity is deprecated; use GNNEncoder embeddings instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _funding_source_similarity(wallet, graph)


def _funding_source_similarity(
    wallet: str, graph: nx.DiGraph, max_depth: int = DEFAULT_MAX_DEPTH
) -> float:
    """Internal (non-deprecated) implementation used by compute_wallet_graph_metrics.

    Uses bounded multi-hop ancestor traversal (Issue #11) so wallets that share
    a funding source several hops back are still detected.
    """
    if wallet not in graph:
        return 0.0

    wallet_ancestors = multi_hop_ancestors(graph, wallet, max_depth)
    if not wallet_ancestors:
        return 0.0

    best = 0.0
    for other in graph.nodes:
        if other == wallet:
            continue
        other_ancestors = multi_hop_ancestors(graph, other, max_depth)
        if not other_ancestors:
            continue
        union = wallet_ancestors | other_ancestors
        if not union:
            continue
        jaccard = len(wallet_ancestors & other_ancestors) / len(union)
        best = max(best, jaccard)

    return float(best)


def network_centrality(wallet: str, graph: nx.DiGraph) -> float:
    """Degree centrality of ``wallet`` within the funding graph.

    .. deprecated::
        Use :class:`detection.gnn_encoder.GNNEncoder` embeddings instead.
        This scalar feature is preserved for backwards compatibility with the
        existing model artifact.
    """
    warnings.warn(
        "network_centrality is deprecated; use GNNEncoder embeddings instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _network_centrality(wallet, graph)


def _network_centrality(wallet: str, graph: nx.DiGraph) -> float:
    """Internal (non-deprecated) implementation used by compute_wallet_graph_metrics."""
    if wallet not in graph or graph.number_of_nodes() < 2:
        return 0.0
    return float(nx.degree_centrality(graph)[wallet])


def compute_wallet_graph_metrics(wallet: str, graph: nx.DiGraph) -> dict:
    """Return ``{funding_source_similarity, network_centrality}`` for *wallet*.

    Calls the internal implementations directly to avoid emitting deprecation
    warnings from internal code paths.
    """
    return {
        "funding_source_similarity": _funding_source_similarity(wallet, graph),
        "network_centrality": _network_centrality(wallet, graph),
    }


# ---------------------------------------------------------------------------
# Wash-trading ring detection (community detection) — Issue #11
# ---------------------------------------------------------------------------


def detect_wash_trading_rings(
    graph: nx.DiGraph,
    resolution: float = DEFAULT_RESOLUTION,
    min_ring_size: int = DEFAULT_MIN_RING_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Partition `graph` into communities and label likely wash-trading rings.

    The directed funding graph is converted to an undirected graph and passed
    to the Louvain algorithm (python-louvain) with a fixed `seed` so results
    are deterministic for CI. Communities with fewer than `min_ring_size`
    members are not considered rings; their wallets are assigned `NO_RING`
    (`-1`). Returns a mapping `wallet_id -> community_id`.
    """
    if graph.number_of_nodes() == 0:
        return {}

    undirected = nx.Graph(graph.to_undirected())

    if _community_louvain is not None:
        partition = _community_louvain.best_partition(
            undirected, resolution=resolution, random_state=seed
        )
    else:  # pragma: no cover - fallback only when python-louvain is absent
        communities = nx.community.greedy_modularity_communities(undirected, resolution=resolution)
        partition = {node: cid for cid, members in enumerate(communities) for node in members}

    sizes = Counter(partition.values())
    return {
        node: (cid if sizes[cid] >= min_ring_size else NO_RING) for node, cid in partition.items()
    }


def ring_id_for_members(members: Iterable[str]) -> str:
    """Stable string ring id derived from the sorted member set."""
    digest = hashlib.sha1(",".join(sorted(members)).encode("utf-8")).hexdigest()
    return f"ring_{digest[:12]}"


def ring_statistics(community_id: int, community_map: dict[str, int], graph: nx.DiGraph) -> dict:
    """Summarise a detected community.

    Returns `{ring_id, ring_size, internal_edge_density, avg_funding_depth}`.
    `internal_edge_density` is the fraction of possible undirected edges within
    the community that actually exist (1.0 = fully connected clique).
    """
    members = sorted(node for node, cid in community_map.items() if cid == community_id)
    ring_size = len(members)

    undirected = graph.to_undirected()
    internal_edges = undirected.subgraph(members).number_of_edges()
    possible_edges = ring_size * (ring_size - 1) // 2
    density = float(internal_edges / possible_edges) if possible_edges > 0 else 0.0

    return {
        "ring_id": ring_id_for_members(members),
        "ring_size": ring_size,
        "internal_edge_density": density,
        "avg_funding_depth": _avg_funding_depth(graph, members),
    }


def build_ring_statistics(community_map: dict[str, int], graph: nx.DiGraph) -> dict[int, dict]:
    """Compute `ring_statistics` for every non-`NO_RING` community in the map."""
    stats: dict[int, dict] = {}
    for community_id in set(community_map.values()):
        if community_id == NO_RING:
            continue
        stats[community_id] = ring_statistics(community_id, community_map, graph)
    return stats


def ring_id_map(community_map: dict[str, int], graph: nx.DiGraph) -> dict[str, str | None]:
    """Map each wallet to its stable `ring_id` string (`None` if not in a ring)."""
    by_community: dict[int, list[str]] = defaultdict(list)
    for node, community_id in community_map.items():
        by_community[community_id].append(node)

    result: dict[str, str | None] = {}
    for community_id, members in by_community.items():
        rid = None if community_id == NO_RING else ring_id_for_members(members)
        for node in members:
            result[node] = rid
    return result


def _avg_funding_depth(graph: nx.DiGraph, members: list[str]) -> float:
    """Average funding depth of `members` within their induced directed subgraph.

    Depth is the number of funding hops from the nearest in-ring funding source
    (a member with no in-ring funder). Returns 0.0 when the subgraph has no such
    source (e.g. a pure cycle).
    """
    if not members:
        return 0.0

    sub = graph.subgraph(members)
    sources = [node for node in sub.nodes if sub.in_degree(node) == 0]
    if not sources:
        return 0.0

    distance = {source: 0 for source in sources}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for successor in sub.successors(node):
            if successor not in distance:
                distance[successor] = distance[node] + 1
                queue.append(successor)

    depths = [distance.get(node, 0) for node in sub.nodes]
    return float(sum(depths) / len(depths)) if depths else 0.0


# ---------------------------------------------------------------------------
# Incremental wallet graph (Issue #203)
# ---------------------------------------------------------------------------


class IncrementalWalletGraph:
    """Thread-safe in-memory wallet graph with incremental edge updates.

    Wallet IDs are normalised to lowercase on ingestion to prevent duplicate
    nodes from capitalisation variants.  The graph stores a ``last_seen``
    timestamp on every edge so that :meth:`remove_stale_edges` can identify
    and evict edges that have not been refreshed within the configured window.

    Prometheus gauges ``wallet_graph_nodes_total`` and
    ``wallet_graph_edges_total`` are updated on every mutating operation.
    ``wallet_graph_stale_edges_total`` is updated after each stale-removal pass.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add_trade_edge(self, src_wallet: str, dst_wallet: str, trade: dict) -> None:
        """Add or refresh a directed trade edge. O(1) amortised per edge.

        ``trade`` may contain ``ledger_close_time`` (str, pd.Timestamp, or
        datetime) which is stored as ``last_seen`` for staleness tracking.
        """
        src = src_wallet.lower()
        dst = dst_wallet.lower()
        ts = _parse_edge_timestamp(trade.get("ledger_close_time", datetime.now(UTC)))
        ts_str = ts.isoformat()

        with self._lock:
            if self._graph.has_edge(src, dst):
                attrs = self._graph[src][dst]
                attrs["weight"] = attrs.get("weight", 1) + 1
                attrs["last_seen"] = ts_str
            else:
                self._graph.add_edge(
                    src,
                    dst,
                    edge_type="co_trade",
                    weight=1,
                    timestamp=ts_str,
                    last_seen=ts_str,
                )
            self._update_gauges()

    def remove_stale_edges(self, max_age_hours: float | None = None) -> int:
        """Remove edges whose ``last_seen`` is older than *max_age_hours*.

        Defaults to ``config.GRAPH_STALE_EDGE_MAX_AGE_HOURS`` (168 h / 7 days).
        Isolated nodes that result from edge removal are also pruned.
        Returns the number of edges removed.
        """
        hours = max_age_hours if max_age_hours is not None else config.GRAPH_STALE_EDGE_MAX_AGE_HOURS
        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        with self._lock:
            stale = [
                (u, v)
                for u, v, d in self._graph.edges(data=True)
                if _parse_edge_timestamp(d.get("last_seen", d.get("timestamp"))) < cutoff
            ]
            self._graph.remove_edges_from(stale)
            isolated = list(nx.isolates(self._graph))
            self._graph.remove_nodes_from(isolated)
            count = len(stale)
            self._update_gauges(stale_count=count)

        return count

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def get_ego_subgraph(self, wallet_id: str, hops: int = 2) -> nx.DiGraph:
        """Return the *hops*-hop ego subgraph for *wallet_id*.

        Uses BFS over the underlying undirected view so both incoming and
        outgoing edges contribute neighbourhood.  Designed to run in < 10 ms
        for wallets with up to 500 direct neighbours at ``hops=2``.

        Returns an empty DiGraph when the wallet is not in the graph.
        """
        node = wallet_id.lower()
        with self._lock:
            if node not in self._graph:
                return nx.DiGraph()

            visited: set[str] = {node}
            frontier: set[str] = {node}
            undirected = self._graph.to_undirected(as_view=True)
            for _ in range(hops):
                next_frontier: set[str] = set()
                for n in frontier:
                    for nb in undirected.neighbors(n):
                        if nb not in visited:
                            visited.add(nb)
                            next_frontier.add(nb)
                if not next_frontier:
                    break
                frontier = next_frontier

            return self._graph.subgraph(visited).copy()

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        with self._lock:
            return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        with self._lock:
            return self._graph.number_of_edges()

    def stale_edge_count(self, max_age_hours: float | None = None) -> int:
        """Count edges that would be removed by :meth:`remove_stale_edges`."""
        hours = max_age_hours if max_age_hours is not None else config.GRAPH_STALE_EDGE_MAX_AGE_HOURS
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        with self._lock:
            return sum(
                1
                for _, _, d in self._graph.edges(data=True)
                if _parse_edge_timestamp(d.get("last_seen", d.get("timestamp"))) < cutoff
            )

    def graph_metrics(self, max_age_hours: float | None = None) -> dict:
        """Return ``{nodes, edges, stale_edges}`` in a single lock acquisition."""
        hours = max_age_hours if max_age_hours is not None else config.GRAPH_STALE_EDGE_MAX_AGE_HOURS
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        with self._lock:
            stale = sum(
                1
                for _, _, d in self._graph.edges(data=True)
                if _parse_edge_timestamp(d.get("last_seen", d.get("timestamp"))) < cutoff
            )
            return {
                "nodes": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
                "stale_edges": stale,
            }

    # ------------------------------------------------------------------
    # Read-only graph access
    # ------------------------------------------------------------------

    def snapshot(self) -> nx.DiGraph:
        """Return a copy of the current graph (safe for read-only callers)."""
        with self._lock:
            return self._graph.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_gauges(self, stale_count: int | None = None) -> None:
        if _graph_nodes_total is not None:
            _graph_nodes_total.set(self._graph.number_of_nodes())
        if _graph_edges_total is not None:
            _graph_edges_total.set(self._graph.number_of_edges())
        if stale_count is not None and _graph_stale_edges_total is not None:
            _graph_stale_edges_total.set(stale_count)


def build_co_trade_graph(trades: pd.DataFrame, window_hours: float = 24.0) -> nx.DiGraph:
    """Build a directed graph of wallets co-active on an asset pair within window_hours.

    Edges must be bidirectional and carry edge_type="co_trade", weight, and timestamp.
    Invalid Stellar accounts are dropped using _validate_account_id.
    """
    g = nx.DiGraph()
    if trades.empty:
        return g

    required = {"base_account", "counter_account", "ledger_close_time"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing required columns: {sorted(missing)}")

    frame = trades.copy()
    if "pair_id" not in frame:
        if not {"base_asset", "counter_asset"}.issubset(frame.columns):
            raise ValueError("trades require pair_id or base_asset/counter_asset columns")
        frame["pair_id"] = (
            frame["base_asset"].astype(str) + "/" + frame["counter_asset"].astype(str)
        )
    frame["ledger_close_time"] = pd.to_datetime(frame["ledger_close_time"], utc=True)
    window = pd.Timedelta(hours=window_hours)

    for _, pair_trades in frame.groupby("pair_id", sort=False):
        records = pair_trades.sort_values("ledger_close_time").to_dict("records")
        for left_index, left in enumerate(records):
            left_time = left["ledger_close_time"]
            active_wallets = {left["base_account"], left["counter_account"]}
            for right in records[left_index + 1 :]:
                if right["ledger_close_time"] - left_time > window:
                    break
                active_wallets.update((right["base_account"], right["counter_account"]))

            # Filter to valid accounts only
            active_wallets = {w for w in active_wallets if _validate_account_id(w)}

            # Add bidirectional edges
            for source, target in combinations(sorted(active_wallets), 2):
                # Update or insert source -> target edge
                if g.has_edge(source, target):
                    g[source][target]["weight"] += 1
                    g[source][target]["timestamp"] = max(g[source][target]["timestamp"], left_time)
                else:
                    g.add_edge(source, target, edge_type="co_trade", weight=1, timestamp=left_time)

                # Update or insert target -> source edge
                if g.has_edge(target, source):
                    g[target][source]["weight"] += 1
                    g[target][source]["timestamp"] = max(g[target][source]["timestamp"], left_time)
                else:
                    g.add_edge(target, source, edge_type="co_trade", weight=1, timestamp=left_time)
    return g
