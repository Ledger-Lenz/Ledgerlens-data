"""Cross-chain identity resolver API.

Provides methods to resolve a Stellar address to its linked Ethereum/Solana counterparts
and retrieve their corresponding risk scores.

Risk-score resolution only follows links whose combined confidence reaches
``config.CROSS_CHAIN_MIN_CONFIDENCE`` (Issue #879); the evidence behind any
link can be pulled with ``get_link_evidence`` for forensic reporting.
"""

from __future__ import annotations

from typing import Any

from detection.cross_chain.confidence import combine_confidence
from detection.cross_chain.identity_graph import IdentityGraph
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
