"""SQLite-backed cross-chain identity graph.

Stores wallets on different chains (Stellar, Ethereum, Solana) and the links
between them (from bridges, behavioral matching, shared deposits, etc.).

Each edge row is one piece of evidence (one ``link_type``) for an address
pair; the detector's payload is kept in ``metadata_json`` so it can be
retrieved for forensic reports.  The confidence of a pair is the noisy-OR
combination of all its evidence rows (see ``confidence.py``, Issue #879).
"""

from __future__ import annotations

import heapq
import json
import logging
from typing import Any

from sqlalchemy import Float, Integer, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from detection.cross_chain.confidence import Evidence, combine_confidence
from detection.cross_chain.graph_partition import PartitionedResolver
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
        UniqueConstraint(
            "source_address", "target_address", "link_type", name="uq_source_target_type"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_address: Mapped[str] = mapped_column(String, nullable=False, index=True)
    target_address: Mapped[str] = mapped_column(String, nullable=False, index=True)
    link_type: Mapped[str] = mapped_column(
        String, nullable=False
    )  # "bridge", "amount_fingerprint", "timing_correlation", "shared_deposit"
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

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        include_solana_pdas: bool | None = None,
    ):
        self._session_factory = session_factory or get_session_factory()
        # None defers to config.SOLANA_INCLUDE_PDA_EDGES at edge-insert time (#881).
        self._include_solana_pdas = include_solana_pdas

    def _is_excluded_solana_address(self, address: str, metadata: dict[str, Any] | None) -> bool:
        """True if ``address`` is a Solana PDA/program account that must not become a
        user-identity edge (see ``solana_resolver.classify_solana_address``)."""
        from detection.cross_chain.solana_resolver import (
            NON_USER_ADDRESS_KINDS,
            classify_solana_address,
        )

        include = self._include_solana_pdas
        if include is None:
            from config import config

            include = bool(getattr(config, "SOLANA_INCLUDE_PDA_EDGES", False))
        # Stellar G-addresses are 56 chars, so the 32-44 char Solana range
        # never collides with them even though Solana keys may start with "G".
        if include or address.startswith("0x") or not 32 <= len(address) <= 44:
            return False
        owner = (metadata or {}).get("owner_program_ids", {}).get(address)
        return classify_solana_address(address, owner_program_id=owner).kind in (
            NON_USER_ADDRESS_KINDS
        )

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
    ) -> CrossChainEdge | None:
        """Insert or update a link between two wallet addresses.

        Returns None (and writes nothing) when either endpoint is a Solana
        program-derived / program-owned address and PDA edges are disabled.
        ``metadata["owner_program_ids"]`` may map an address to its on-chain
        owner program to catch on-curve program-owned accounts.
        """
        source = normalize_address(source)
        target = normalize_address(target)
        for addr in (source, target):
            if self._is_excluded_solana_address(addr, metadata):
                logger.info("Skipping %s edge: %s is a Solana program address", link_type, addr)
                return None
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
                    elif 32 <= len(addr) <= 44:
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

    def add_links(self, links: list[dict[str, Any]], link_type: str) -> int:
        """Store detector output (BridgeDetector / BehavioralMatcher links) as evidence.

        Each link needs ``stellar_address``, ``linked_address`` and
        ``confidence``; everything else (tx ids, memo, mechanism, correlation
        stats, ...) is kept as the edge's forensic evidence payload.
        """
        stored = 0
        for link in links:
            metadata = dict(link.get("metadata") or {})
            for key, value in link.items():
                if key not in ("stellar_address", "linked_address", "confidence", "metadata"):
                    metadata.setdefault(key, value)
            self.add_edge(
                link["stellar_address"],
                link["linked_address"],
                link_type,
                confidence=float(link.get("confidence", 1.0)),
                metadata=metadata,
            )
            stored += 1
        return stored

    def get_edge_evidence(self, address_a: str, address_b: str) -> list[Evidence]:
        """Return every piece of evidence linking two addresses (either direction)."""
        a = normalize_address(address_a)
        b = normalize_address(address_b)
        with self._session_factory() as session:
            edges = session.scalars(
                select(CrossChainEdge).where(
                    ((CrossChainEdge.source_address == a) & (CrossChainEdge.target_address == b))
                    | ((CrossChainEdge.source_address == b) & (CrossChainEdge.target_address == a))
                )
            ).all()
            return [_edge_to_evidence(edge) for edge in edges]

    def get_link_confidence(self, address_a: str, address_b: str) -> float:
        """Combined confidence that two addresses belong to the same entity."""
        return combine_confidence(self.get_edge_evidence(address_a, address_b))

    def get_connected_component(
        self, start_address: str, min_confidence: float = 0.0
    ) -> dict[str, list[dict[str, Any]]]:
        """Find all transitively linked addresses, grouped by chain.

        Edges between an address pair are combined into one confidence from all
        of their evidence (``combine_confidence``). Pairs below
        ``min_confidence`` are not traversed, so low-confidence links neither
        appear in the result nor bridge to further addresses.

        Each returned node carries ``link_confidence`` (also exposed as
        ``confidence``): the strongest path from ``start_address`` to it, i.e.
        the maximum over paths of the product of link confidences (best-first
        search). A zk/guardian-attested bridge edge therefore propagates more
        weight than a heuristic one (#884).
        """
        start_address = normalize_address(start_address)
        best: dict[str, float] = {start_address: 1.0}
        heap: list[tuple[float, str]] = [(-1.0, start_address)]
        done: set[str] = set()
        nodes_info: dict[str, dict[str, Any]] = {}

        with self._session_factory() as session:
            if session.get(CrossChainNode, start_address) is None:
                return {"eth": [], "sol": [], "stellar": []}

            while heap:
                neg_conf, current = heapq.heappop(heap)
                if current in done:
                    continue
                done.add(current)
                node = session.get(CrossChainNode, current)
                if node is None:
                    continue
                nodes_info[current] = {
                    "address": node.address,
                    "chain": node.chain,
                    "risk_score": node.risk_score,
                    "link_confidence": -neg_conf,
                    "confidence": -neg_conf,
                }
                edges = session.scalars(
                    select(CrossChainEdge).where(
                        (CrossChainEdge.source_address == current)
                        | (CrossChainEdge.target_address == current)
                    )
                ).all()

                evidence_by_neighbor: dict[str, list[Evidence]] = {}
                for edge in edges:
                    neighbor = (
                        edge.target_address
                        if edge.source_address == current
                        else edge.source_address
                    )
                    evidence_by_neighbor.setdefault(neighbor, []).append(_edge_to_evidence(edge))

                for neighbor, evidence in evidence_by_neighbor.items():
                    edge_conf = max(0.0, min(1.0, combine_confidence(evidence)))
                    if edge_conf < min_confidence:
                        continue
                    conf = -neg_conf * edge_conf
                    if neighbor not in done and conf > best.get(neighbor, -1.0):
                        best[neighbor] = conf
                        heapq.heappush(heap, (-conf, neighbor))

        return _group_by_chain(nodes_info, exclude=start_address)

    def load_edges(self) -> list[tuple[str, str, float]]:
        """Return every edge as ``(source, target, confidence)`` in one query."""
        with self._session_factory() as session:
            rows = session.execute(
                select(
                    CrossChainEdge.source_address,
                    CrossChainEdge.target_address,
                    CrossChainEdge.confidence,
                )
            ).all()
        return [(r[0], r[1], float(r[2])) for r in rows]

    def load_nodes(self) -> dict[str, dict[str, Any]]:
        """Return every node as ``{address: {"address", "chain", "risk_score"}}``."""
        with self._session_factory() as session:
            rows = session.execute(
                select(CrossChainNode.address, CrossChainNode.chain, CrossChainNode.risk_score)
            ).all()
        return {r[0]: {"address": r[0], "chain": r[1], "risk_score": float(r[2])} for r in rows}

    def build_partitioned_resolver(
        self, num_shards: int = 16, strategy: str = "anchor"
    ) -> PartitionedResolver:
        """Load all edges once into a :class:`PartitionedResolver` (#883)."""
        resolver = PartitionedResolver(num_shards=num_shards, strategy=strategy)
        resolver.add_edges((u, v) for u, v, _ in self.load_edges())
        return resolver


def _group_by_chain(
    nodes_info: dict[str, dict[str, Any]], exclude: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Group node dicts under standardized 'eth' / 'sol' / 'stellar' keys."""
    result: dict[str, list[dict[str, Any]]] = {"eth": [], "sol": [], "stellar": []}
    for addr, info in nodes_info.items():
        if addr == exclude:
            continue
        chain_key = info["chain"].lower()
        if chain_key in ("ethereum", "eth", "evm"):
            result["eth"].append(info)
        elif chain_key in ("solana", "sol"):
            result["sol"].append(info)
        elif chain_key in ("stellar",):
            result["stellar"].append(info)
        else:
            result.setdefault(chain_key, []).append(info)
    return result


def _edge_to_evidence(edge: CrossChainEdge) -> Evidence:
    metadata = json.loads(edge.metadata_json) if edge.metadata_json else {}
    metadata.setdefault("source_address", edge.source_address)
    metadata.setdefault("target_address", edge.target_address)
    return Evidence(evidence_type=edge.link_type, strength=edge.confidence, metadata=metadata)
