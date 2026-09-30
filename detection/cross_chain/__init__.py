"""Cross-chain identity resolution module."""

from __future__ import annotations

from detection.cross_chain.behavioral_matcher import BehavioralMatcher
from detection.cross_chain.bridge_detector import BridgeDetector
from detection.cross_chain.bridge_mechanisms import (
    BridgeMechanismHandler,
    analyze_bridge_transaction,
    classify_bridge_mechanism,
    register_bridge_mechanism,
)
from detection.cross_chain.confidence import Evidence, combine_confidence
from detection.cross_chain.identity_graph import (
    CrossChainEdge,
    CrossChainNode,
    IdentityGraph,
)
from detection.cross_chain.resolver import get_link_evidence, resolve, resolve_risk_scores

__all__ = [
    "BridgeDetector",
    "BehavioralMatcher",
    "BridgeMechanismHandler",
    "register_bridge_mechanism",
    "classify_bridge_mechanism",
    "analyze_bridge_transaction",
    "Evidence",
    "combine_confidence",
    "IdentityGraph",
    "CrossChainNode",
    "CrossChainEdge",
    "resolve",
    "resolve_risk_scores",
    "get_link_evidence",
]
