"""Cross-chain identity resolver API.

Provides methods to resolve a Stellar address to its linked Ethereum/Solana counterparts
and retrieve their corresponding risk scores.
"""

from __future__ import annotations

from detection.cross_chain.identity_graph import IdentityGraph
from detection.persistence import get_engine, get_session_factory


def resolve(stellar_address: str, db_url: str | None = None) -> dict[str, list[str]]:
    """Resolve a Stellar address to counterpart Ethereum and Solana addresses.

    Returns:
        dict: {"eth": [...], "sol": [...]}
    """
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    graph = IdentityGraph(session_factory)

    component = graph.get_connected_component(stellar_address)

    return {
        "eth": [node["address"] for node in component.get("eth", [])],
        "sol": [node["address"] for node in component.get("sol", [])],
    }


def resolve_risk_scores(stellar_address: str, db_url: str | None = None) -> dict[str, float]:
    """Retrieve risk scores for all EVM/Solana addresses linked to a Stellar wallet.

    Returns:
        dict: {linked_address: risk_score}
    """
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    graph = IdentityGraph(session_factory)

    component = graph.get_connected_component(stellar_address)

    risk_scores = {}
    for node in component.get("eth", []):
        risk_scores[node["address"]] = node["risk_score"]
    for node in component.get("sol", []):
        risk_scores[node["address"]] = node["risk_score"]

    return risk_scores
