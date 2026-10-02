"""Cross-chain identity resolver API.

Provides methods to resolve a Stellar address to its linked Ethereum/Solana counterparts
and retrieve their corresponding risk scores.

Risk-score resolution only follows links whose combined confidence reaches
``config.CROSS_CHAIN_MIN_CONFIDENCE`` (Issue #879); the evidence behind any
link can be pulled with ``get_link_evidence`` for forensic reporting.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from detection.cross_chain.confidence import combine_confidence
from detection.cross_chain.identity_graph import IdentityGraph, normalize_address
from detection.persistence import get_engine, get_session_factory


def _default_min_confidence() -> float:
    from config import config

    return config.CROSS_CHAIN_MIN_CONFIDENCE


def resolve(
    stellar_address: str, db_url: str | None = None, min_confidence: float = 0.0
) -> dict[str, list[str]]:
    """Resolve a Stellar address to counterpart Ethereum and Solana addresses.

    ``min_confidence`` drops links whose combined confidence is below it.

    Returns:
        dict: {"eth": [...], "sol": [...]}
    """
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    graph = IdentityGraph(session_factory)

    component = graph.get_connected_component(stellar_address, min_confidence=min_confidence)

    return {
        "eth": [node["address"] for node in component.get("eth", [])],
        "sol": [node["address"] for node in component.get("sol", [])],
    }


def resolve_risk_scores(
    stellar_address: str, db_url: str | None = None, min_confidence: float | None = None
) -> dict[str, float]:
    """Retrieve risk scores for all EVM/Solana addresses linked to a Stellar wallet.

    Used by risk propagation, so links below ``min_confidence`` (default
    ``config.CROSS_CHAIN_MIN_CONFIDENCE``) are excluded.

    Returns:
        dict: {linked_address: risk_score}
    """
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    graph = IdentityGraph(session_factory)

    if min_confidence is None:
        min_confidence = _default_min_confidence()
    component = graph.get_connected_component(stellar_address, min_confidence=min_confidence)

    risk_scores = {}
    for node in component.get("eth", []):
        risk_scores[node["address"]] = node["risk_score"]
    for node in component.get("sol", []):
        risk_scores[node["address"]] = node["risk_score"]

    return risk_scores


def get_link_evidence(address_a: str, address_b: str, db_url: str | None = None) -> dict[str, Any]:
    """Return the combined confidence and every piece of evidence for a link.

    Returns:
        dict: {"confidence": float, "evidence": [Evidence.to_dict(), ...]}
    """
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    graph = IdentityGraph(session_factory)

    evidence = graph.get_edge_evidence(address_a, address_b)
    return {
        "confidence": combine_confidence(evidence),
        "evidence": [item.to_dict() for item in evidence],
    }


def resolve_weighted_risk_scores(
    stellar_address: str, db_url: str | None = None
) -> dict[str, float]:
    """Like :func:`resolve_risk_scores`, but each score is scaled by the
    strongest-path link confidence, so risk imported over a verified bridge
    link counts for more than risk imported over a heuristic one (#884).
    """
    engine = get_engine(db_url)
    graph = IdentityGraph(get_session_factory(engine))
    component = graph.get_connected_component(stellar_address)
    return {
        node["address"]: node["risk_score"] * node["link_confidence"]
        for key in ("eth", "sol")
        for node in component.get(key, [])
    }


def resolve_weighted_risk_scores_bulk(
    addresses: Iterable[str], db_url: str | None = None
) -> dict[str, dict[str, float]]:
    """Batch form of :func:`resolve_weighted_risk_scores`.

    Loads the identity graph with two queries and resolves every requested
    address in memory, instead of one DB round trip per visited node per
    address. Addresses absent from the identity graph are skipped, so a
    batch of wallets with no cross-chain links costs two queries total.

    Returns ``{address: {linked_address: risk_score * link_confidence}}`` for
    addresses that have at least one linked EVM/Solana counterpart, keyed by
    the address exactly as passed in.
    """
    graph = IdentityGraph(get_session_factory(get_engine(db_url)))
    nodes = graph.load_nodes()
    wanted = {a: normalize_address(a) for a in addresses}
    wanted = {orig: norm for orig, norm in wanted.items() if norm in nodes}
    if not wanted:
        return {}
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for u, v, conf in graph.load_edges():
        conf = max(0.0, min(1.0, conf))
        adjacency[u].append((v, conf))
        adjacency[v].append((u, conf))

    result: dict[str, dict[str, float]] = {}
    for orig, start in wanted.items():
        best = {start: 1.0}
        heap = [(-1.0, start)]
        done: set[str] = set()
        scores: dict[str, float] = {}
        while heap:
            neg, cur = heapq.heappop(heap)
            if cur in done:
                continue
            done.add(cur)
            info = nodes.get(cur)
            if (
                cur != start
                and info
                and info["chain"] in ("ethereum", "eth", "evm", "solana", "sol")
            ):
                scores[cur] = info["risk_score"] * -neg
            for nxt, conf in adjacency.get(cur, ()):
                c = -neg * conf
                if nxt not in done and c > best.get(nxt, -1.0):
                    best[nxt] = c
                    heapq.heappush(heap, (-c, nxt))
        if scores:
            result[orig] = scores
    return result
