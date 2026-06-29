"""Cross-chain identity resolution module."""

from __future__ import annotations

from detection.cross_chain.behavioral_matcher import BehavioralMatcher
from detection.cross_chain.bridge_detector import BridgeDetector
from detection.cross_chain.identity_graph import (
    CrossChainEdge,
    CrossChainNode,
    IdentityGraph,
)
from detection.cross_chain.resolver import resolve, resolve_risk_scores

__all__ = [
    "BridgeDetector",
    "BehavioralMatcher",
    "IdentityGraph",
    "CrossChainNode",
    "CrossChainEdge",
    "resolve",
    "resolve_risk_scores",
]
