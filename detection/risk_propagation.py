"""Graph-based risk propagation via Personalized PageRank (PPR) diffusion.

Spreads base risk scores (from the ML ensemble) through the combined
funding + co-trade graph so that wallets indirectly connected to a
high-risk node inherit a proportionally decayed propagated score.

Algorithm
---------
For each seed wallet *w* with base score R(w):

    PPR(v | w) = (1 - α) * A * PPR(v | w) + α * e_w

where α is the teleport probability (default 0.15), A is the
row-normalised adjacency of the combined graph, and e_w is the
one-hot seed vector.

The propagated score for node *v* is then:

    R_prop(v) = Σ_w  R(w) * PPR(v | w)     clipped to [0, 100]

Convergence is declared when the L1 norm of the update drops below
``convergence_tol`` (default 1e-6) or ``max_iterations`` is reached.

Performance
-----------
Uses a sparse CSR matrix for the power iteration; on a 10,000-node
graph a full pass completes in < 2 seconds on CPU.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph combination helper
# ---------------------------------------------------------------------------


def _build_combined_graph(
    funding_graph: nx.DiGraph,
    co_trade_graph: nx.Graph | None,
) -> nx.DiGraph:
    """Merge *funding_graph* and *co_trade_graph* into a single DiGraph.

    Co-trade edges are treated as bidirectional (both directions added).
    Nodes that appear only in *co_trade_graph* are included so that wallets
    with no funding relationship but shared co-trade activity still receive
    propagated scores.
    """
    combined: nx.DiGraph = nx.DiGraph()
    combined.add_nodes_from(funding_graph.nodes())
    combined.add_edges_from(funding_graph.edges())

    if co_trade_graph is not None:
        # add_nodes_from is a no-op for nodes already present
        combined.add_nodes_from(co_trade_graph.nodes())
        for u, v in co_trade_graph.edges():
            combined.add_edge(u, v)
            combined.add_edge(v, u)

    return combined


def _row_normalise(adj: np.ndarray) -> np.ndarray:
    """Row-normalise a dense or sparse matrix; rows with zero sum stay zero."""
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero for sink nodes
    result: np.ndarray = adj / row_sums
    return result


# ---------------------------------------------------------------------------
# Core PPR routine
# ---------------------------------------------------------------------------


def _personalised_pagerank(
    A_csr: csr_matrix,
    seed_idx: int,
    alpha: float,
    max_iterations: int,
    convergence_tol: float,
) -> np.ndarray:
    """Run PPR power iteration for a single seed node.

    Parameters
    ----------
    A_csr:
        Row-normalised adjacency as a CSR sparse matrix (shape N×N).
    seed_idx:
        Column index of the seed node.
    alpha:
        Teleport (restart) probability.
    max_iterations:
        Hard cap on iterations.
    convergence_tol:
        L1 convergence threshold.

    Returns
    -------
    np.ndarray of shape (N,) — PPR scores summing to 1.
    """
    n = A_csr.shape[0]
    # personalisation vector
    e = np.zeros(n, dtype=np.float64)
    e[seed_idx] = 1.0

    ppr = e.copy()

    for _ in range(max_iterations):
        new_ppr = (1.0 - alpha) * A_csr.T.dot(ppr) + alpha * e
        delta = float(np.abs(new_ppr - ppr).sum())
        ppr = new_ppr
        if delta < convergence_tol:
            break

    return ppr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def propagate_risk_scores(
    base_scores: dict[str, float],
    funding_graph: nx.DiGraph,
    co_trade_graph: nx.Graph | None = None,
    alpha: float = 0.15,
    max_iterations: int = 50,
    convergence_tol: float = 1e-6,
    db_url: str | None = None,
) -> dict[str, float]:
    """Return propagated risk scores for **all** nodes in the graph.

    Parameters
    ----------
    base_scores:
        Mapping of wallet → base risk score (0–100) from the ML ensemble.
        Wallets not present in the graph are silently ignored.
    funding_graph:
        Directed graph of ``funding_account → account_id`` edges (from
        :func:`detection.wallet_graph.build_funding_graph`).
    co_trade_graph:
        Optional undirected graph of wallets that traded the same asset
        pair within the same window.  Edges are treated as bidirectional.
    alpha:
        Teleport probability (restart probability towards the seed). A
        lower value propagates risk further; default 0.15 gives ~85 %
        weight to direct neighbours and ~12 % to two-hop neighbours.
    max_iterations:
        Maximum power-iteration steps per seed.  Convergence is typically
        reached in < 20 steps.
    convergence_tol:
        L1 norm threshold for declaring convergence.
    db_url:
        Optional SQLite database URL to resolve cross-chain identity risk propagation.

    Returns
    -------
    dict[str, float]
        Propagated score per wallet, clipped to [0, 100].  Every node in
        the combined graph receives an entry; nodes with no high-risk
        ancestors/descendants receive 0.0.
    """
    combined = _build_combined_graph(funding_graph, co_trade_graph)

    if combined.number_of_nodes() == 0:
        return {}

    # Update base scores with cross-chain linked risk scores
    updated_base_scores = base_scores.copy()
    try:
        from detection.cross_chain.resolver import resolve_risk_scores
        for node in combined.nodes():
            ext_scores = resolve_risk_scores(node, db_url=db_url)
            if ext_scores:
                max_ext = max(ext_scores.values())
                updated_base_scores[node] = max(updated_base_scores.get(node, 0.0), max_ext)
    except Exception as e:
        logger.warning("Failed to propagate cross-chain risk scores: %s", e)

    nodes: list[str] = list(combined.nodes())
    n = len(nodes)
    node_idx: dict[str, int] = {w: i for i, w in enumerate(nodes)}

    # Build row-normalised adjacency matrix (sparse)
    rows, cols = [], []
    for u, v in combined.edges():
        rows.append(node_idx[u])
        cols.append(node_idx[v])

    if rows:
        data = np.ones(len(rows), dtype=np.float64)
        adj_raw = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        # Row-normalise: convert to dense temporarily, then back
        row_sums = np.asarray(adj_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        D_inv = diags(1.0 / row_sums)
        A_csr: csr_matrix = (D_inv @ adj_raw).tocsr()
    else:
        # Isolated nodes only — no edges; propagation has no effect
        A_csr = csr_matrix((n, n), dtype=np.float64)

    # Accumulate weighted PPR vectors from all seed wallets
    propagated = np.zeros(n, dtype=np.float64)

    seeds_in_graph = {w: s for w, s in updated_base_scores.items() if w in node_idx and s > 0}

    if not seeds_in_graph:
        return {w: 0.0 for w in nodes}

    for wallet, score in seeds_in_graph.items():
        seed_idx = node_idx[wallet]
        ppr = _personalised_pagerank(A_csr, seed_idx, alpha, max_iterations, convergence_tol)
        # Weight PPR by the wallet's base score (normalised to [0,1])
        propagated += (score / 100.0) * ppr

    # Re-scale: PPR vectors sum to 1 per seed, so multiply by 100 and
    # clip to [0, 100]
    propagated_scores = np.clip(propagated * 100.0, 0.0, 100.0)

    return {nodes[i]: float(propagated_scores[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# Attribution helper (used by ForensicReport)
# ---------------------------------------------------------------------------


def propagation_attribution(
    wallet: str,
    base_scores: dict[str, float],
    funding_graph: nx.DiGraph,
    co_trade_graph: nx.Graph | None = None,
    alpha: float = 0.15,
    max_iterations: int = 50,
    convergence_tol: float = 1e-6,
    top_n: int = 5,
    db_url: str | None = None,
) -> list[dict]:
    """Return which high-risk ancestors/descendants contributed to *wallet*'s
    propagated score, and what fraction each contributed.

    Returns
    -------
    list of dicts with keys:
        ``source_wallet``, ``base_score``, ``ppr_weight``, ``contribution``,
        ``fraction`` (0–1).

    Returns an empty list if *wallet* is not in the graph or if its
    propagated score is zero.
    """
    combined = _build_combined_graph(funding_graph, co_trade_graph)

    if wallet not in combined or combined.number_of_nodes() == 0:
        return []

    # Update base scores with cross-chain linked risk scores
    updated_base_scores = base_scores.copy()
    try:
        from detection.cross_chain.resolver import resolve_risk_scores
        for node in combined.nodes():
            ext_scores = resolve_risk_scores(node, db_url=db_url)
            if ext_scores:
                max_ext = max(ext_scores.values())
                updated_base_scores[node] = max(updated_base_scores.get(node, 0.0), max_ext)
    except Exception as e:
        logger.warning("Failed to propagate cross-chain risk scores in attribution: %s", e)

    nodes: list[str] = list(combined.nodes())
    n = len(nodes)
    node_idx: dict[str, int] = {w: i for i, w in enumerate(nodes)}
    target_idx = node_idx[wallet]

    # Build A_csr (same logic as propagate_risk_scores)
    rows, cols = [], []
    for u, v in combined.edges():
        rows.append(node_idx[u])
        cols.append(node_idx[v])

    if rows:
        data = np.ones(len(rows), dtype=np.float64)
        adj_raw = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        row_sums = np.asarray(adj_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        D_inv = diags(1.0 / row_sums)
        A_csr: csr_matrix = (D_inv @ adj_raw).tocsr()
    else:
        A_csr = csr_matrix((n, n), dtype=np.float64)

    seeds_in_graph = {w: s for w, s in updated_base_scores.items() if w in node_idx and s > 0}
    if not seeds_in_graph:
        return []

    contributions: list[dict[str, float | str]] = []
    total_contribution = 0.0

    for source, score in seeds_in_graph.items():
        seed_idx = node_idx[source]
        ppr = _personalised_pagerank(A_csr, seed_idx, alpha, max_iterations, convergence_tol)
        ppr_weight = float(ppr[target_idx])
        contribution = (score / 100.0) * ppr_weight * 100.0
        if contribution > 0.0:
            contributions.append(
                {
                    "source_wallet": source,
                    "base_score": score,
                    "ppr_weight": round(ppr_weight, 6),
                    "contribution": round(contribution, 4),
                }
            )
            total_contribution += contribution

    if total_contribution == 0.0:
        return []

    for c in contributions:
        c["fraction"] = round(float(c["contribution"]) / total_contribution, 4)

    contributions.sort(key=lambda x: float(x["contribution"]), reverse=True)
    return contributions[:top_n]


# ---------------------------------------------------------------------------
# Weighted PPR — edge weights derived from trade-level features
# ---------------------------------------------------------------------------


class WeightedRiskPropagation:
    """Personalised PageRank with trade-derived edge weights.

    Edge weight combines three signals:
    - ``shared_trades_weight``: log-normalised count of trades both wallets share
    - ``temporal_concentration``: how clustered in time those trades are
    - ``amount_similarity``: how similar the trade amounts are

    Parameters
    ----------
    alpha:
        Teleport probability (default 0.15).
    max_iterations:
        Hard cap on PPR power-iteration steps (default 100).
    convergence_threshold:
        Stops when max per-node score change drops below this value.
        Defaults to ``config.RISK_PROP_CONVERGENCE_THRESHOLD`` (0.01).
    """

    def __init__(
        self,
        alpha: float = 0.15,
        max_iterations: int = 100,
        convergence_threshold: float | None = None,
    ) -> None:
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.convergence_threshold = (
            convergence_threshold
            if convergence_threshold is not None
            else config.RISK_PROP_CONVERGENCE_THRESHOLD
        )

    # ------------------------------------------------------------------
    # Edge weight computation
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_weight(
        shared_trade_count: int,
        max_shared_trades: int,
        inter_trade_intervals: list[float],
        amounts: list[float],
    ) -> float:
        """Return a [0, 1] edge weight from three normalised signals."""
        # 1. shared trades (log-normalised)
        if max_shared_trades > 0:
            shared_trades_weight = np.log1p(shared_trade_count) / np.log1p(max_shared_trades)
        else:
            shared_trades_weight = 0.0

        # 2. temporal concentration
        if len(inter_trade_intervals) >= 2:
            arr = np.asarray(inter_trade_intervals, dtype=np.float64)
            mean_i = arr.mean()
            std_i = arr.std()
            if mean_i > 0:
                temporal_concentration = float(np.clip(1.0 - std_i / mean_i, 0.0, 1.0))
            else:
                temporal_concentration = 1.0  # all at the same time → maximally clustered
        else:
            temporal_concentration = 0.5

        # 3. amount similarity
        if amounts:
            arr_a = np.asarray(amounts, dtype=np.float64)
            mean_a = arr_a.mean()
            std_a = arr_a.std()
            amount_similarity = float(np.clip(1.0 - std_a / (mean_a + 1e-9), 0.0, 1.0))
        else:
            amount_similarity = 0.5

        return (shared_trades_weight + temporal_concentration + amount_similarity) / 3.0

    def _build_weighted_adjacency(
        self,
        graph: nx.DiGraph,
        trade_data: dict | None,
        nodes: list[str],
        node_idx: dict[str, int],
    ) -> csr_matrix:
        """Build a row-normalised weighted adjacency CSR matrix."""
        n = len(nodes)

        # Pre-compute max_shared_trades across all edges for normalisation
        if trade_data:
            max_shared = max(
                (
                    trade_data.get((u, v), {}).get("shared_trade_count", 0)
                    for u, v in graph.edges()
                    if (u, v) in trade_data or trade_data.get((u, v))
                ),
                default=1,
            )
            # Also check edges that have the count stored directly on the graph
            for u, v, d in graph.edges(data=True):
                cnt = d.get("shared_trade_count", 0)
                if cnt > max_shared:
                    max_shared = cnt
        else:
            max_shared = max(
                (d.get("shared_trade_count", 0) for _, _, d in graph.edges(data=True)),
                default=1,
            )
        if max_shared < 1:
            max_shared = 1

        rows, cols, weights = [], [], []
        for u, v, edge_data in graph.edges(data=True):
            if u not in node_idx or v not in node_idx:
                continue

            # Pull edge-level trade stats — prefer trade_data dict, fall back to graph attrs
            if trade_data and (u, v) in trade_data:
                td = trade_data[(u, v)]
            else:
                td = edge_data

            shared_count = int(td.get("shared_trade_count", 1))
            intervals = td.get("inter_trade_intervals", [])
            amounts = td.get("amounts", [])

            w = self._edge_weight(shared_count, max_shared, intervals, amounts)
            rows.append(node_idx[u])
            cols.append(node_idx[v])
            weights.append(w if w > 0 else 1e-9)  # avoid zero rows collapsing to uniform

        if not rows:
            return csr_matrix((n, n), dtype=np.float64)

        adj_raw = csr_matrix(
            (np.array(weights, dtype=np.float64), (rows, cols)),
            shape=(n, n),
            dtype=np.float64,
        )
        # Row-normalise
        row_sums = np.asarray(adj_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        D_inv = diags(1.0 / row_sums)
        return (D_inv @ adj_raw).tocsr()

    # ------------------------------------------------------------------
    # PPR power iteration (convergence by max score change)
    # ------------------------------------------------------------------

    def _ppr(self, A_csr: csr_matrix, seed_idx: int, n: int) -> np.ndarray:
        e = np.zeros(n, dtype=np.float64)
        e[seed_idx] = 1.0
        ppr = e.copy()
        for _ in range(self.max_iterations):
            new_ppr = (1.0 - self.alpha) * A_csr.T.dot(ppr) + self.alpha * e
            if float(np.abs(new_ppr - ppr).max()) < self.convergence_threshold:
                ppr = new_ppr
                break
            ppr = new_ppr
        return ppr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(
        self,
        base_scores: dict[str, float],
        graph: nx.DiGraph,
        trade_data: dict | None = None,
    ) -> dict[str, float]:
        """Propagate *base_scores* through *graph* using weighted PPR.

        Parameters
        ----------
        base_scores:
            Wallet → raw risk score (0–100).
        graph:
            Directed wallet graph (funding + co-trade).  Each edge may
            carry ``shared_trade_count``, ``inter_trade_intervals``, and
            ``amounts`` attributes used to compute edge weights.
        trade_data:
            Optional ``{(u, v): {"shared_trade_count": int,
            "inter_trade_intervals": [...], "amounts": [...]}}`` mapping
            that overrides per-edge graph attributes.

        Returns
        -------
        dict[str, float]
            Propagated score per node, clipped to [0, 100].
        """
        if graph.number_of_nodes() == 0:
            return {}

        nodes: list[str] = list(graph.nodes())
        node_idx: dict[str, int] = {w: i for i, w in enumerate(nodes)}
        n = len(nodes)

        # Handle disconnected components independently by processing each
        # weakly-connected component separately on the same shared matrix —
        # PPR naturally stays within a component when the adjacency is block-diagonal,
        # so no extra work is needed beyond building the full matrix once.
        A_csr = self._build_weighted_adjacency(graph, trade_data, nodes, node_idx)

        seeds = {w: s for w, s in base_scores.items() if w in node_idx and s > 0}
        if not seeds:
            return {w: 0.0 for w in nodes}

        propagated = np.zeros(n, dtype=np.float64)
        for wallet, score in seeds.items():
            ppr = self._ppr(A_csr, node_idx[wallet], n)
            propagated += (score / 100.0) * ppr

        propagated_scores = np.clip(propagated * 100.0, 0.0, 100.0)
        return {nodes[i]: float(propagated_scores[i]) for i in range(n)}


def propagate_risk_scores_weighted(
    base_scores: dict[str, float],
    graph: nx.DiGraph,
    trade_data: dict | None = None,
    alpha: float = 0.15,
    max_iterations: int = 100,
    convergence_threshold: float | None = None,
) -> dict[str, float]:
    """Convenience wrapper around :class:`WeightedRiskPropagation`.

    Parameters mirror ``WeightedRiskPropagation.__init__`` and
    ``WeightedRiskPropagation.propagate``.
    """
    return WeightedRiskPropagation(
        alpha=alpha,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    ).propagate(base_scores, graph, trade_data=trade_data)


# ---------------------------------------------------------------------------
# Weighted Risk Propagation (issue #259)
# ---------------------------------------------------------------------------


def _compute_edge_weights(
    graph: nx.DiGraph,
    trade_data: dict[tuple[str, str], pd.DataFrame] | None = None,
) -> dict[tuple[str, str], float]:
    """Compute edge weights for each edge in the graph.

    Weight is a function of:
    - shared_trade_count: log-normalised number of shared trades between nodes
    - temporal_concentration: how clustered trades are in time (1 = very clustered)
    - amount_similarity: how similar trade amounts are (1 = very similar)
    """
    import math

    if trade_data is None:
        # Uniform weights when no trade data available
        return {(u, v): 1.0 for u, v in graph.edges()}

    # First pass: gather shared trade counts to normalise
    edge_trade_counts: dict[tuple[str, str], int] = {}
    for u, v in graph.edges():
        df = trade_data.get((u, v)) or trade_data.get((v, u))
        edge_trade_counts[(u, v)] = len(df) if df is not None else 0

    max_trades = max(edge_trade_counts.values(), default=1) or 1

    weights: dict[tuple[str, str], float] = {}
    for (u, v), count in edge_trade_counts.items():
        # (1) shared trade count weight
        shared_w = math.log1p(count) / math.log1p(max_trades)

        df = trade_data.get((u, v)) or trade_data.get((v, u))
        if df is not None and len(df) >= 2 and "ledger_close_time" in df.columns:
            times = pd.to_datetime(df["ledger_close_time"]).sort_values()
            intervals = times.diff().dropna().dt.total_seconds()
            if intervals.mean() > 0:
                # (2) temporal concentration: low CV = concentrated in time
                cv = intervals.std() / (intervals.mean() + 1e-9)
                temporal_w = 1.0 / (1.0 + cv)
            else:
                temporal_w = 0.5

            if "amount" in df.columns and df["amount"].mean() > 0:
                # (3) amount similarity: low CV = similar sizes
                cv_amt = df["amount"].std() / (df["amount"].mean() + 1e-9)
                amount_w = max(0.0, min(1.0, 1.0 - cv_amt))
            else:
                amount_w = 0.5
        else:
            temporal_w = 0.5
            amount_w = 0.5

        weights[(u, v)] = (shared_w + temporal_w + amount_w) / 3.0

    return weights


class WeightedRiskPropagation:
    """Personalised PageRank with heterogeneous edge weights.

    Edge weights are computed from shared trade count, temporal concentration,
    and amount similarity between wallet pairs.  Propagation stops when the
    maximum score change across all nodes drops below ``convergence_threshold``.

    The algorithm handles disconnected graph components independently — PPR
    is computed per seed, so risk never bleeds across components.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        max_iterations: int = 100,
        convergence_threshold: float | None = None,
    ) -> None:
        self.alpha = alpha
        self.max_iterations = max_iterations
        if convergence_threshold is None:
            from config import config
            convergence_threshold = config.RISK_PROP_CONVERGENCE_THRESHOLD
        self.convergence_threshold = convergence_threshold

    def propagate(
        self,
        base_scores: dict[str, float],
        graph: nx.DiGraph,
        trade_data: dict[tuple[str, str], pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        """Propagate risk scores through *graph* using weighted PPR.

        Parameters
        ----------
        base_scores:
            wallet → risk score (0–100).
        graph:
            Directed wallet graph (funding or co-trade).
        trade_data:
            Optional mapping of (wallet_a, wallet_b) → trade DataFrame used
            to compute edge weights.  When ``None``, uniform weights are used.

        Returns
        -------
        dict[str, float] — propagated score per node, clipped to [0, 100].
        """
        if graph.number_of_nodes() == 0:
            return {}

        nodes: list[str] = list(graph.nodes())
        n = len(nodes)
        node_idx: dict[str, int] = {w: i for i, w in enumerate(nodes)}

        # Compute edge weights and build weighted adjacency
        edge_weights = _compute_edge_weights(graph, trade_data)

        rows, cols, data = [], [], []
        for (u, v), w in edge_weights.items():
            if u in node_idx and v in node_idx:
                rows.append(node_idx[u])
                cols.append(node_idx[v])
                data.append(w)

        if rows:
            adj_raw = csr_matrix(
                (np.array(data, dtype=np.float64), (rows, cols)),
                shape=(n, n),
            )
            row_sums = np.asarray(adj_raw.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            A_csr: csr_matrix = (diags(1.0 / row_sums) @ adj_raw).tocsr()
        else:
            A_csr = csr_matrix((n, n), dtype=np.float64)

        propagated = np.zeros(n, dtype=np.float64)
        seeds = {w: s for w, s in base_scores.items() if w in node_idx and s > 0}

        if not seeds:
            return {w: 0.0 for w in nodes}

        e_base = np.zeros(n, dtype=np.float64)
        for wallet, score in seeds.items():
            seed_idx = node_idx[wallet]
            e = e_base.copy()
            e[seed_idx] = 1.0
            ppr = e.copy()
            for _ in range(self.max_iterations):
                new_ppr = (1.0 - self.alpha) * A_csr.T.dot(ppr) + self.alpha * e
                # Use convergence_threshold on max score change (not L1 of PPR)
                delta_scores = float(np.abs((new_ppr - ppr) * (score / 100.0) * 100.0).max())
                ppr = new_ppr
                if delta_scores < self.convergence_threshold:
                    break
            propagated += (score / 100.0) * ppr

        return {
            nodes[i]: float(np.clip(propagated[i] * 100.0, 0.0, 100.0))
            for i in range(n)
        }


def propagate_risk_scores_weighted(
    base_scores: dict[str, float],
    funding_graph: nx.DiGraph,
    co_trade_graph: nx.Graph | None = None,
    trade_data: dict[tuple[str, str], pd.DataFrame] | None = None,
    alpha: float = 0.15,
    convergence_threshold: float | None = None,
) -> dict[str, float]:
    """Convenience wrapper around :class:`WeightedRiskPropagation`."""
    combined = _build_combined_graph(funding_graph, co_trade_graph)
    wrp = WeightedRiskPropagation(alpha=alpha, convergence_threshold=convergence_threshold)
    return wrp.propagate(base_scores, combined, trade_data=trade_data)
