"""Tests for partitioned cross-chain identity resolution (Issue #883)."""

from __future__ import annotations

import random

import pytest

from detection.cross_chain.graph_partition import (
    PartitionedResolver,
    UnionFind,
    canonical_components,
    resolve_whole_graph,
    shard_key,
)
from detection.cross_chain.identity_graph import IdentityGraph
from detection.cross_chain.resolver import (
    resolve_weighted_risk_scores,
    resolve_weighted_risk_scores_bulk,
)
from detection.persistence import Base, get_engine, get_session_factory
from scripts.benchmark_identity_partitioning import run, stellar_address, synthetic_edges


def _random_edges(rng: random.Random, n_nodes: int, n_edges: int) -> list[tuple[str, str]]:
    nodes = [f"n{i}" for i in range(n_nodes)]
    return [tuple(rng.sample(nodes, 2)) for _ in range(n_edges)]


def test_union_find_basics():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("c", "d")
    assert uf.find("a") == uf.find("b") != uf.find("c")
    uf.union("b", "d")
    assert uf.find("a") == uf.find("c")
    assert "a" in uf and "z" not in uf


@pytest.mark.parametrize("strategy", ["anchor", "chain_pair", "hash"])
@pytest.mark.parametrize("num_shards", [1, 3, 17, 64])
def test_partitioned_equals_whole_graph_random(strategy, num_shards):
    rng = random.Random(num_shards)
    for _ in range(5):
        edges = _random_edges(rng, 300, 280)
        resolver = PartitionedResolver(num_shards, strategy)
        resolver.add_edges(edges)
        assert canonical_components(resolver.assignment()) == canonical_components(
            resolve_whole_graph(edges)
        )


@pytest.mark.parametrize("strategy", ["anchor", "chain_pair", "hash"])
def test_partitioned_equals_whole_graph_after_incremental_updates(strategy):
    rng = random.Random(42)
    resolver = PartitionedResolver(16, strategy)
    live: set[tuple[str, str]] = set()
    for _ in range(10):
        added = _random_edges(rng, 400, 60)
        removed = rng.sample(sorted(live), min(15, len(live)))
        resolver.add_edges(added)
        resolver.remove_edges(removed)
        live |= {tuple(sorted(e)) for e in added}
        live -= {tuple(sorted(e)) for e in removed}
        assert canonical_components(resolver.assignment()) == canonical_components(
            resolve_whole_graph(live)
        )


def test_synthetic_identity_graph_equivalence():
    edges = synthetic_edges(3_000, seed=5)
    resolver = PartitionedResolver(12, "anchor")
    resolver.add_edges(edges)
    assert canonical_components(resolver.assignment()) == canonical_components(
        resolve_whole_graph(edges)
    )
    g = stellar_address(0)
    whole = resolve_whole_graph(edges)
    assert resolver.component_of(g) == {n for n, r in whole.items() if r == whole[g]}


def test_cross_shard_link_is_reconciled():
    # Two users in different shards share one exchange deposit address.
    resolver = PartitionedResolver(64, "anchor")
    hub, xa, xb = "0x" + "f" * 40, "0x" + "a" * 40, "0x" + "b" * 40
    a = stellar_address(1)
    b = next(
        stellar_address(i)
        for i in range(2, 200)
        if resolver.shard_of((stellar_address(i), hub)) != resolver.shard_of((a, hub))
    )
    resolver.add_edges([(a, xa), (a, hub), (b, xb), (b, hub)])
    assert resolver.component_of(xa) == {a, b, xa, xb, hub}
    assert resolver.boundary_nodes == 1
    resolver.remove_edges([(b, hub)])
    assert resolver.component_of(xa) == {a, xa, hub}
    assert resolver.component_of(xb) == {b, xb}
    assert resolver.boundary_nodes == 0


def test_only_dirty_shards_are_recomputed():
    resolver = PartitionedResolver(32, "anchor")
    resolver.add_edges(synthetic_edges(2_000, seed=1))
    resolver.resolve()
    assert resolver.last_recomputed_shards == 32
    resolver.add_edges([(stellar_address(99_999), "0xnew")])
    resolver.resolve()
    assert resolver.last_recomputed_shards == 1


def test_anchor_strategy_keeps_user_cluster_in_one_shard():
    g = stellar_address(3)
    keys = {shard_key((g, x), "anchor") for x in ("0x" + "1" * 40, "0x" + "2" * 40)}
    assert keys == {g}


def test_unknown_strategy_and_bad_shard_count():
    with pytest.raises(ValueError):
        shard_key(("a", "b"), "nope")
    with pytest.raises(ValueError):
        PartitionedResolver(0)


def test_component_of_unknown_address():
    assert PartitionedResolver(4).component_of("nobody") == set()


def test_benchmark_smoke():
    [row] = run(sizes=(2_000,), edges_per_shard=500)
    assert row["shards"] >= 2
    assert row["largest_shard_peak_mib"] < row["whole_peak_mib"]


# -- IdentityGraph integration -------------------------------------------------


@pytest.fixture
def graph_and_url(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'part.db'}"
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return IdentityGraph(get_session_factory(engine)), db_url


def test_identity_graph_partitioned_matches_bfs(graph_and_url):
    graph, _ = graph_and_url
    g1, g2 = stellar_address(10), stellar_address(11)
    graph.add_edge(g1, "0x" + "a" * 40, "bridge")
    graph.add_edge("0x" + "a" * 40, "0x" + "b" * 40, "amount_fingerprint", confidence=0.7)
    graph.add_edge(g2, "0x" + "c" * 40, "bridge")
    resolver = graph.build_partitioned_resolver(num_shards=8)
    comp = graph.get_connected_component(g1)
    bfs = {n["address"] for nodes in comp.values() for n in nodes} | {g1}
    assert resolver.component_of(g1) == bfs


def test_bulk_weighted_scores_match_single_lookup(graph_and_url):
    graph, db_url = graph_and_url
    g = stellar_address(20)
    graph.add_node("0x" + "d" * 40, "ethereum", risk_score=90.0)
    graph.add_node("0x" + "e" * 40, "ethereum", risk_score=50.0)
    graph.add_edge(g, "0x" + "d" * 40, "bridge", confidence=0.9)
    graph.add_edge("0x" + "d" * 40, "0x" + "e" * 40, "amount_fingerprint", confidence=0.5)
    graph.add_edge(g, "0x" + "e" * 40, "timing_correlation", confidence=0.8)

    single = resolve_weighted_risk_scores(g, db_url)
    bulk = resolve_weighted_risk_scores_bulk([g, "GUNKNOWN"], db_url)
    assert bulk == {g: single}
    # Strongest path to 0xeee… is the direct 0.8 edge, not 0.9 * 0.5.
    assert single["0x" + "e" * 40] == pytest.approx(50.0 * 0.8)
