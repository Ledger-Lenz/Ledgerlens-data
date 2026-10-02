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
        from detection.cross_chain.resolver import resolve_weighted_risk_scores_bulk

        # Linked risk is scaled by link confidence, so risk imported over a
        # proof-verified bridge link outweighs a heuristic one (#884).
        linked = resolve_weighted_risk_scores_bulk(combined.nodes(), db_url=db_url)
        for node, ext_scores in linked.items():
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
        from detection.cross_chain.resolver import resolve_weighted_risk_scores_bulk

        # Linked risk is scaled by link confidence, so risk imported over a
        # proof-verified bridge link outweighs a heuristic one (#884).
        linked = resolve_weighted_risk_scores_bulk(combined.nodes(), db_url=db_url)
        for node, ext_scores in linked.items():
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

        return {nodes[i]: float(np.clip(propagated[i] * 100.0, 0.0, 100.0)) for i in range(n)}


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
