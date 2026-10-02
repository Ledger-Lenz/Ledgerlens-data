"""Partitioned identity resolution for the cross-chain graph (#883).

Whole-graph resolution (``IdentityGraph.get_connected_component``) walks the
graph with one DB round trip per visited node and has to be rerun from scratch
whenever links change. This module resolves identities over *shards* instead:

1. **Partition**: every edge is assigned to exactly one shard (see
   :func:`shard_key`). Nodes are not partitioned; a node touched by edges in
   several shards is a *boundary node* and appears in each of them.
2. **Local resolution**: each shard computes its own connected components
   (union-find) independently. Shards are self-contained, so they can be
   resolved in parallel / on separate workers, and after a batch of edge
   changes only the *dirty* shards are recomputed.
3. **Boundary reconciliation**: each shard-local component gets a label
   ``(shard, local_root)``. For every boundary node, all its labels are
   unioned in a small coordinator union-find. A node's global identity is the
   coordinator root of any of its labels.

Correctness: two nodes are connected in the whole graph iff there is a path
between them. Every edge on that path lies in some shard, so consecutive edges
in the same shard share a local label, and consecutive edges in different
shards meet at a node present in both, i.e. a boundary node whose labels are
reconciled. Hence partitioned components are *identical* to whole-graph
components, for every partitioning (the scheme only affects cost).

See ``docs/cross_chain_graph_partitioning.md`` for the design and benchmark.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable

Edge = tuple[str, str]


class UnionFind:
    """Union-find with path halving and union by size."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1

    def find(self, x: str) -> str:
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]

    def __iter__(self):
        return iter(self._parent)

    def __contains__(self, x: object) -> bool:
        return x in self._parent


def resolve_whole_graph(edges: Iterable[Edge]) -> dict[str, str]:
    """Baseline: map every node to a component representative in one pass."""
    uf = UnionFind()
    for u, v in edges:
        uf.union(u, v)
    return {node: uf.find(node) for node in uf}


def canonical_components(assignment: dict[str, str]) -> set[frozenset[str]]:
    """Representative-independent view of a node -> component mapping."""
    groups: dict[str, set[str]] = defaultdict(set)
    for node, root in assignment.items():
        groups[root].add(node)
    return {frozenset(g) for g in groups.values()}


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")


def guess_chain(address: str) -> str:
    """Same address-format heuristic ``IdentityGraph.add_edge`` uses."""
    if address.startswith("0x") and len(address) == 42:
        return "ethereum"
    if len(address) == 56 and address.startswith("G"):
        return "stellar"
    if 32 <= len(address) <= 44:
        return "solana"
    return "stellar"


def shard_key(edge: Edge, strategy: str, chain_of: Callable[[str], str] = guess_chain) -> str:
    """Return the partition key for an edge.

    ``anchor`` (default): the edge's Stellar endpoint (else the smaller
        endpoint). Identity clusters are mostly stars around a user's Stellar
        wallet, so a cluster lands in one shard and only external addresses
        shared by users in different shards (exchange deposit addresses,
        bridge custody) become boundary nodes.
    ``chain_pair``: the unordered chain pair (``ethereum|stellar`` ...). Few,
        coarse shards; useful when per-chain workers own the data, but every
        address bridged to two other chains is a boundary node.
    ``hash``: the smaller endpoint. Balanced but locality-blind; the
        worst case for boundary size, kept as a reference point.
    """
    u, v = edge
    if strategy == "anchor":
        cu, cv = chain_of(u), chain_of(v)
        if cu == "stellar" and cv != "stellar":
            return u
        if cv == "stellar" and cu != "stellar":
            return v
        return min(u, v)
    if strategy == "chain_pair":
        return "|".join(sorted((chain_of(u), chain_of(v))))
    if strategy == "hash":
        return min(u, v)
    raise ValueError(f"Unknown partition strategy {strategy!r}")


class PartitionedResolver:
    """Incremental, shard-parallelisable identity resolution.

    Usage::

        resolver = PartitionedResolver(num_shards=64)
        resolver.add_edges(pairs)
        resolver.component_of("G...")          # set of linked addresses
        resolver.remove_edges([...])           # e.g. retracted / PDA links
        resolver.component_of("G...")          # recomputes dirty shards only

    After a batch of changes, :meth:`resolve` costs
    O(edges in dirty shards + boundary nodes), not O(whole graph): clean
    shards keep their local components and the coordinator only ever holds
    labels of boundary nodes.
    """

    def __init__(
        self,
        num_shards: int = 16,
        strategy: str = "anchor",
        chain_of: Callable[[str], str] = guess_chain,
    ) -> None:
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        self.num_shards = num_shards
        self.strategy = strategy
        self._chain_of = chain_of
        self._shard_edges: list[set[Edge]] = [set() for _ in range(num_shards)]
        # Per-shard node -> local root and local root -> members; rebuilt only
        # for dirty shards.
        self._local: list[dict[str, str]] = [{} for _ in range(num_shards)]
        self._members: list[dict[str, list[str]]] = [{} for _ in range(num_shards)]
        # node -> {shard: incident edge count}; nodes in >1 shard are boundary.
        self._node_shards: dict[str, dict[int, int]] = {}
        self._boundary: set[str] = set()
        self._dirty: set[int] = set()
        self._coordinator: UnionFind | None = None
        self._groups: dict[str, list[str]] = {}
        self.last_recomputed_shards: int = 0

    # -- partitioning ------------------------------------------------------

    def shard_of(self, edge: Edge) -> int:
        return _stable_hash(shard_key(edge, self.strategy, self._chain_of)) % self.num_shards

    @staticmethod
    def _norm(edge: Edge) -> Edge:
        u, v = edge
        return (u, v) if u <= v else (v, u)

    def _touch(self, node: str, idx: int, delta: int) -> None:
        shards = self._node_shards.setdefault(node, {})
        count = shards.get(idx, 0) + delta
        if count > 0:
            shards[idx] = count
        else:
            shards.pop(idx, None)
        if not shards:
            del self._node_shards[node]
        if len(shards) > 1:
            self._boundary.add(node)
        else:
            self._boundary.discard(node)

    def add_edges(self, edges: Iterable[Edge]) -> None:
        for edge in edges:
            edge = self._norm(edge)
            if edge[0] == edge[1]:
                continue
            idx = self.shard_of(edge)
            if edge not in self._shard_edges[idx]:
                self._shard_edges[idx].add(edge)
                self._touch(edge[0], idx, 1)
                self._touch(edge[1], idx, 1)
                self._dirty.add(idx)
                self._coordinator = None

    def remove_edges(self, edges: Iterable[Edge]) -> None:
        for edge in edges:
            edge = self._norm(edge)
            idx = self.shard_of(edge)
            if edge in self._shard_edges[idx]:
                self._shard_edges[idx].discard(edge)
                self._touch(edge[0], idx, -1)
                self._touch(edge[1], idx, -1)
                self._dirty.add(idx)
                self._coordinator = None

    def shard_sizes(self) -> list[int]:
        return [len(e) for e in self._shard_edges]

    @property
    def boundary_nodes(self) -> int:
        return len(self._boundary)

    # -- resolution --------------------------------------------------------

    @staticmethod
    def resolve_shard(edges: Iterable[Edge]) -> dict[str, str]:
        """Local components of one shard. Pure function: safe to run on a worker."""
        return resolve_whole_graph(edges)

    def resolve(self) -> None:
        """Recompute dirty shards, then reconcile boundary nodes."""
        if self._coordinator is not None:
            return
        dirty = sorted(self._dirty)
        for idx in dirty:
            local = self.resolve_shard(self._shard_edges[idx])
            members: dict[str, list[str]] = defaultdict(list)
            for node, root in local.items():
                members[root].append(node)
            self._local[idx] = local
            self._members[idx] = dict(members)
        self._dirty.clear()
        self.last_recomputed_shards = len(dirty)

        coordinator = UnionFind()
        for node in self._boundary:
            labels = [f"{idx}:{self._local[idx][node]}" for idx in self._node_shards[node]]
            coordinator.add(labels[0])
            for label in labels[1:]:
                coordinator.union(labels[0], label)
        groups: dict[str, list[str]] = defaultdict(list)
        for label in coordinator:
            groups[coordinator.find(label)].append(label)
        self._coordinator = coordinator
        self._groups = dict(groups)

    def _label(self, node: str) -> str | None:
        shards = self._node_shards.get(node)
        if not shards:
            return None
        idx = next(iter(shards))
        return f"{idx}:{self._local[idx][node]}"

    def component_id(self, address: str) -> str | None:
        """Stable-per-resolution identifier of ``address``'s identity cluster."""
        self.resolve()
        label = self._label(address)
        if label is None:
            return None
        assert self._coordinator is not None
        return self._coordinator.find(label) if label in self._coordinator else label

    def component_of(self, address: str) -> set[str]:
        """All addresses linked to ``address`` (including itself)."""
        root = self.component_id(address)
        if root is None:
            return set()
        members: set[str] = set()
        for label in self._groups.get(root, [root]):
            idx, local_root = label.split(":", 1)
            members.update(self._members[int(idx)][local_root])
        return members

    def assignment(self) -> dict[str, str]:
        """Full node -> component id map (O(N); for verification and export)."""
        self.resolve()
        return {node: self.component_id(node) for node in self._node_shards}  # type: ignore[misc]
