"""SQLite-backed cross-chain identity graph.

Stores wallets on different chains (Stellar, Ethereum, Solana) and the links
between them (from bridges, behavioral matching, shared deposits, etc.).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import Float, Integer, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from detection.persistence import Base, get_session_factory

logger = logging.getLogger(__name__)


class CrossChainNode(Base):
    """Represents a wallet on a specific blockchain."""

    __tablename__ = "cross_chain_nodes"

    address: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    chain: Mapped[str] = mapped_column(String, nullable=False)  # "stellar", "ethereum", "solana"
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class CrossChainEdge(Base):
    """Represents a link/connection between two wallet addresses."""

    __tablename__ = "cross_chain_edges"
    __table_args__ = (
        UniqueConstraint("source_address", "target_address", "link_type", name="uq_source_target_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_address: Mapped[str] = mapped_column(String, nullable=False, index=True)
    target_address: Mapped[str] = mapped_column(String, nullable=False, index=True)
    link_type: Mapped[str] = mapped_column(String, nullable=False)  # "bridge", "amount_fingerprint", "timing_correlation", "shared_deposit"
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    metadata_json: Mapped[str | None] = mapped_column(String, nullable=True)


def normalize_address(address: str) -> str:
    """Normalize address to lowercase if it is an EVM address."""
    addr = address.strip()
    if addr.startswith("0x") or addr.lower().startswith("0x"):
        return addr.lower()
    return addr


class IdentityGraph:
    """Graph manager for cross-chain identity links in SQLite."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or get_session_factory()

    def add_node(self, address: str, chain: str, risk_score: float = 0.0) -> CrossChainNode:
        """Insert or update a cross-chain wallet node."""
        address = normalize_address(address)
        with self._session_factory() as session:
            node = session.get(CrossChainNode, address)
            if node is None:
                node = CrossChainNode(address=address, chain=chain.lower())
                session.add(node)
            node.risk_score = float(risk_score)
            node.chain = chain.lower()
            session.commit()
            session.refresh(node)
            return node

    def add_edge(
        self,
        source: str,
        target: str,
        link_type: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> CrossChainEdge:
        """Insert or update a link between two wallet addresses."""
        source = normalize_address(source)
        target = normalize_address(target)
        meta_str = json.dumps(metadata) if metadata else None
        # Ensure we always add/fetch nodes first so foreign keys or existence constraints are satisfied
        # Note: we don't have strict foreign key constraints in SQLite schema but good practice to register them
        with self._session_factory() as session:
            # Check source and target nodes exist, create if they don't
            for addr in (source, target):
                node = session.get(CrossChainNode, addr)
                if node is None:
                    # Guess chain based on address format
                    chain = "stellar"
                    if addr.startswith("0x") and len(addr) == 42:
                        chain = "ethereum"
                    elif not addr.startswith("G") and len(addr) >= 32 and len(addr) <= 44:
                        chain = "solana"
                    
                    new_node = CrossChainNode(address=addr, chain=chain, risk_score=0.0)
                    session.add(new_node)
            
            existing = session.scalar(
                select(CrossChainEdge).where(
                    CrossChainEdge.source_address == source,
                    CrossChainEdge.target_address == target,
                    CrossChainEdge.link_type == link_type,
                )
            )
            if existing is None:
                existing = CrossChainEdge(
                    source_address=source,
                    target_address=target,
                    link_type=link_type,
                )
                session.add(existing)

            existing.confidence = float(confidence)
            existing.metadata_json = meta_str
            session.commit()
            session.refresh(existing)
            return existing

    def get_connected_component(self, start_address: str) -> dict[str, list[dict[str, Any]]]:
        """BFS traversal to find all transitively linked addresses, grouped by chain."""
        start_address = normalize_address(start_address)
        visited = set()
        queue = [start_address]
        nodes_info = {}

        # Get initial node info
        with self._session_factory() as session:
            start_node = session.get(CrossChainNode, start_address)
            if not start_node:
                return {"eth": [], "sol": [], "stellar": []}
            nodes_info[start_address] = {
                "address": start_node.address,
                "chain": start_node.chain,
                "risk_score": start_node.risk_score,
            }

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            with self._session_factory() as session:
                edges = session.scalars(
                    select(CrossChainEdge).where(
                        (CrossChainEdge.source_address == current) |
                        (CrossChainEdge.target_address == current)
                    )
                ).all()

                for edge in edges:
                    neighbor = edge.target_address if edge.source_address == current else edge.source_address
                    if neighbor not in visited and neighbor not in queue:
                        neighbor_node = session.get(CrossChainNode, neighbor)
                        if neighbor_node:
                            nodes_info[neighbor] = {
                                "address": neighbor_node.address,
                                "chain": neighbor_node.chain,
                                "risk_score": neighbor_node.risk_score,
                            }
                            queue.append(neighbor)

        # Group by chain (standardizing keys to 'eth', 'sol', 'stellar')
        result: dict[str, list[dict[str, Any]]] = {"eth": [], "sol": [], "stellar": []}
        for addr, info in nodes_info.items():
            if addr == start_address:
                continue
            chain_key = info["chain"].lower()
            if chain_key in ("ethereum", "eth", "evm"):
                result["eth"].append(info)
            elif chain_key in ("solana", "sol"):
                result["sol"].append(info)
            elif chain_key in ("stellar",):
                result["stellar"].append(info)
            else:
                result[chain_key] = result.get(chain_key, [])
                result[chain_key].append(info)

        return result
