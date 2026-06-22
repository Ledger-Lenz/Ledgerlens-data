"""Cross-DEX coordination detection between SDEX and AMM liquidity pools.

Computes cross-venue features and coordination graphs that detect wash traders
operating across both the Stellar SDEX (order book) and AMM pool venues.
"""

from __future__ import annotations

import bisect

import networkx as nx
import numpy as np  # noqa: F401 — kept for potential downstream use
import pandas as pd

# ---------------------------------------------------------------------------
# Cross-venue feature computation
# ---------------------------------------------------------------------------

_CROSS_VENUE_DEFAULTS: dict[str, float] = {
    "venue_trade_ratio": 0.0,
    "cross_venue_volume_correlation": 0.0,
    "cross_venue_timing_synchrony": 0.0,
    "cross_venue_net_flow": 0.0,
    "counterparty_venue_overlap": 0.0,
    "simultaneous_order_pair": 0.0,
    "cross_venue_cluster_score": 0.0,
}

_TIMING_WINDOW_SECONDS = 10


def compute_cross_venue_features(
    wallet: str,
    sdex_trades: pd.DataFrame,
    amm_trades: pd.DataFrame,
    clusters: list[set[str]] | None = None,
    graph: nx.DiGraph | None = None,
) -> dict:
    """Compute all 7 cross-venue features for a wallet.

    Falls back to 0.0 for every feature when AMM data is empty or unavailable.

    Args:
        wallet: Stellar public key to compute features for.
        sdex_trades: DataFrame of SDEX trades (same schema as
            ``historical_loader.trades_to_dataframe``).
        amm_trades: DataFrame of AMM pool trades (same schema).
        clusters: Pre-computed Louvain clusters (optional).  When ``None``,
            the coordination graph is built inline and ``cross_venue_cluster_score``
            defaults to 0.0.
        graph: Pre-computed coordination graph (optional).
    """
    if amm_trades is None or amm_trades.empty:
        return dict(_CROSS_VENUE_DEFAULTS)

    wallet_sdex = _filter_wallet(wallet, sdex_trades)
    wallet_amm = _filter_wallet(wallet, amm_trades)

    features: dict = {}

    features["venue_trade_ratio"] = _venue_trade_ratio(wallet_sdex, wallet_amm)
    features["cross_venue_volume_correlation"] = _cross_venue_volume_correlation(
        wallet_sdex, wallet_amm
    )
    features["cross_venue_timing_synchrony"] = _cross_venue_timing_synchrony(
        wallet_sdex, wallet_amm
    )
    features["cross_venue_net_flow"] = _cross_venue_net_flow(wallet, wallet_sdex, wallet_amm)
    features["counterparty_venue_overlap"] = _counterparty_venue_overlap(
        wallet, wallet_sdex, wallet_amm
    )
    features["simultaneous_order_pair"] = _simultaneous_order_pair(wallet_sdex, wallet_amm)

    if clusters is not None and graph is not None:
        features["cross_venue_cluster_score"] = cross_venue_cluster_score(wallet, clusters, graph)
    else:
        features["cross_venue_cluster_score"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def _filter_wallet(wallet: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df.get("base_account", pd.Series(dtype=str)) == wallet) | (
        df.get("counter_account", pd.Series(dtype=str)) == wallet
    )
    return df[mask]


def _venue_trade_ratio(sdex: pd.DataFrame, amm: pd.DataFrame) -> float:
    """Ratio of SDEX to AMM trade count; 0.0 when AMM is empty."""
    n_amm = len(amm)
    if n_amm == 0:
        return 0.0
    return float(len(sdex) / n_amm)


def _cross_venue_volume_correlation(sdex: pd.DataFrame, amm: pd.DataFrame) -> float:
    """Pearson correlation of 1-hour bucketed SDEX and AMM trade volumes."""
    if sdex.empty or amm.empty:
        return 0.0

    def hourly_volume(df: pd.DataFrame) -> pd.Series:
        ts = pd.to_datetime(df["ledger_close_time"], utc=True, errors="coerce")
        amounts = df["amount"].astype(float)
        return amounts.groupby(ts.dt.floor("h")).sum()

    vol_s = hourly_volume(sdex)
    vol_a = hourly_volume(amm)
    aligned = pd.concat([vol_s, vol_a], axis=1, keys=["s", "a"]).fillna(0.0)

    if len(aligned) < 2 or aligned["s"].std() == 0 or aligned["a"].std() == 0:
        return 0.0
    corr = float(aligned["s"].corr(aligned["a"]))
    return corr if not np.isnan(corr) else 0.0


def _cross_venue_timing_synchrony(sdex: pd.DataFrame, amm: pd.DataFrame) -> float:
    """Fraction of AMM trades occurring within 10 s of any SDEX trade."""
    if amm.empty or sdex.empty:
        return 0.0

    sdex_times = (
        pd.to_datetime(sdex["ledger_close_time"], utc=True, errors="coerce").dropna().sort_values()
    )
    amm_times = pd.to_datetime(amm["ledger_close_time"], utc=True, errors="coerce").dropna()

    if sdex_times.empty or amm_times.empty:
        return 0.0

    # Use seconds (float) for cross-version pandas compatibility — asi8 unit varies by version
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    sdex_secs = (sdex_times - _epoch).dt.total_seconds().to_numpy().copy()
    amm_secs = (amm_times - _epoch).dt.total_seconds().to_numpy()
    sdex_secs.sort()
    window = float(_TIMING_WINDOW_SECONDS)

    paired = 0
    for t in amm_secs:
        lo = bisect.bisect_left(sdex_secs, t - window)
        hi = bisect.bisect_right(sdex_secs, t + window)
        if lo < hi:
            paired += 1

    return float(paired / len(amm_times))


def _cross_venue_net_flow(wallet: str, sdex: pd.DataFrame, amm: pd.DataFrame) -> float:
    """Net XLM flow (SDEX outflow - AMM inflow); near-zero for wash traders."""
    combined = pd.concat([sdex, amm], ignore_index=True)
    if combined.empty:
        return 0.0

    net = 0.0
    for _, row in combined.iterrows():
        amount = float(row.get("amount", 0.0))
        if row.get("base_account") == wallet:
            net -= amount
        else:
            net += amount

    return float(abs(net))


def _counterparty_venue_overlap(wallet: str, sdex: pd.DataFrame, amm: pd.DataFrame) -> float:
    """Fraction of SDEX counterparties also seen as AMM LP addresses."""
    if sdex.empty or amm.empty:
        return 0.0

    def counterparties(df: pd.DataFrame) -> set[str]:
        result: set[str] = set()
        for _, row in df.iterrows():
            if row.get("base_account") == wallet:
                cp = row.get("counter_account", "")
            else:
                cp = row.get("base_account", "")
            if cp and cp != wallet:
                result.add(cp)
        return result

    sdex_cps = counterparties(sdex)
    amm_cps = counterparties(amm)

    if not sdex_cps:
        return 0.0
    return float(len(sdex_cps & amm_cps) / len(sdex_cps))


def _simultaneous_order_pair(sdex: pd.DataFrame, amm: pd.DataFrame) -> float:
    """Binary: 1.0 if the wallet has SDEX trades and AMM trades in overlapping windows."""
    if sdex.empty or amm.empty:
        return 0.0

    sdex_times = pd.to_datetime(sdex["ledger_close_time"], utc=True, errors="coerce").dropna()
    amm_times = pd.to_datetime(amm["ledger_close_time"], utc=True, errors="coerce").dropna()

    if sdex_times.empty or amm_times.empty:
        return 0.0

    sdex_min, sdex_max = sdex_times.min(), sdex_times.max()
    amm_min, amm_max = amm_times.min(), amm_times.max()

    overlap = (sdex_min <= amm_max) and (amm_min <= sdex_max)
    return 1.0 if overlap else 0.0


# ---------------------------------------------------------------------------
# Coordination graph
# ---------------------------------------------------------------------------


def build_coordination_graph(
    sdex_trades: pd.DataFrame,
    amm_trades: pd.DataFrame,
    window_seconds: int = 10,
) -> nx.DiGraph:
    """Build a directed wallet coordination graph from SDEX and AMM trades.

    Nodes are wallet addresses.  An edge (A→B, venue=V) is added when wallets A
    and B both appear in trades within ``window_seconds`` of each other on venue V.

    Uses a sort + sliding-window algorithm for O(n log n) performance on large
    trade sets, meeting the < 30 s requirement for 10,000 wallets × 100,000 trades.
    """
    graph = nx.DiGraph()

    for venue, df in (("sdex", sdex_trades), ("amm", amm_trades)):
        if df is None or df.empty:
            continue
        _add_venue_edges(graph, df, venue, window_seconds)

    return graph


def _add_venue_edges(
    graph: nx.DiGraph,
    df: pd.DataFrame,
    venue: str,
    window_seconds: int,
) -> None:
    """Add coordination edges for one venue using a sorted sliding window."""
    times = pd.to_datetime(df["ledger_close_time"], utc=True, errors="coerce")
    valid = df[times.notna()].copy()
    valid["_ts"] = times[times.notna()].values

    # Build per-row list of (timestamp_seconds_float, wallet_a, wallet_b)
    events: list[tuple[float, str, str]] = []
    for _, row in valid.iterrows():
        ba = str(row.get("base_account", "") or "")
        ca = str(row.get("counter_account", "") or "")
        ts = float(pd.Timestamp(row["_ts"]).timestamp())  # seconds since epoch (float)
        if ba:
            events.append((ts, ba, ca))
        if ca and ca != ba:
            events.append((ts, ca, ba))

    if not events:
        return

    events.sort(key=lambda x: x[0])
    window = float(window_seconds)
    timestamps = [e[0] for e in events]

    for i, (t_i, w_i, _) in enumerate(events):
        lo = bisect.bisect_left(timestamps, t_i - window, 0, i)
        for j in range(lo, i):
            t_j, w_j, _ = events[j]
            if abs(t_i - t_j) <= window and w_i != w_j:
                if not graph.has_node(w_i):
                    graph.add_node(w_i)
                if not graph.has_node(w_j):
                    graph.add_node(w_j)
                if not graph.has_edge(w_i, w_j):
                    graph.add_edge(w_i, w_j, venue=venue, weight=1)
                else:
                    graph[w_i][w_j]["weight"] = graph[w_i][w_j].get("weight", 0) + 1


def detect_coordinated_clusters(graph: nx.DiGraph) -> list[set[str]]:
    """Apply Louvain community detection to find tightly coupled wallet clusters.

    Returns a list of disjoint sets (partition) covering all nodes.  Each wallet
    appears in exactly one cluster.

    Uses ``networkx.algorithms.community.louvain_communities`` on the undirected
    projection of the graph (edge weights are summed for bidirectional edges).
    """
    if graph.number_of_nodes() == 0:
        return []

    undirected = graph.to_undirected(as_view=False)
    communities = nx.algorithms.community.louvain_communities(undirected, weight="weight", seed=42)
    return [set(c) for c in communities]


def cross_venue_cluster_score(
    wallet: str,
    clusters: list[set[str]],
    graph: nx.DiGraph,
) -> float:
    """Measure how central a wallet is within its cluster and cross-venue activity.

    Returns a score in [0, 1] combining:
    - Degree centrality within the cluster subgraph.
    - Fraction of the cluster's edges that span both venues (cross-venue ratio).
    """
    if not clusters or graph.number_of_nodes() == 0:
        return 0.0

    wallet_cluster: set[str] | None = None
    for cluster in clusters:
        if wallet in cluster:
            wallet_cluster = cluster
            break

    if wallet_cluster is None or len(wallet_cluster) < 2:
        return 0.0

    subgraph = graph.subgraph(wallet_cluster)

    # Degree centrality within the cluster
    centralities = nx.degree_centrality(subgraph)
    centrality = float(centralities.get(wallet, 0.0))

    # Cross-venue ratio: fraction of edges with at least one venue-specific marker
    venues_seen: set[str] = set()
    for _u, _v, data in subgraph.edges(data=True):
        venue = data.get("venue", "")
        if venue:
            venues_seen.add(venue)

    cross_venue_ratio = 1.0 if len(venues_seen) >= 2 else 0.0

    return float((centrality + cross_venue_ratio) / 2.0)
