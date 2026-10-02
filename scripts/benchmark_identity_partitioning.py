"""Benchmark partitioned vs whole-graph cross-chain identity resolution (#883).

Builds a synthetic identity graph (one Stellar wallet per user linked to 1-3
EVM and 0-2 Solana addresses, plus shared exchange-deposit hub addresses that
merge unrelated users), then measures for growing graph sizes:

* ``full``: resolving the whole graph from scratch (union-find baseline) vs
  the initial partitioned resolve (all shards dirty).
* ``update``: resolution latency after a batch of edge additions/removals.
  Whole-graph resolution must recompute everything; partitioned resolution
  recomputes only dirty shards plus boundary reconciliation.
* ``peak memory``: tracemalloc peak of resolving the whole graph vs resolving
  the largest single shard (the working set one worker needs).

Shard count scales with graph size (fixed target edges per shard) so each
worker's load stays constant as the graph grows.

Usage::

    python scripts/benchmark_identity_partitioning.py
    python scripts/benchmark_identity_partitioning.py --sizes 5000 20000 --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import tracemalloc
from typing import Any

from detection.cross_chain.graph_partition import (
    PartitionedResolver,
    canonical_components,
    resolve_whole_graph,
)

_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _digest(tag: str) -> bytes:
    return hashlib.sha256(tag.encode()).digest()


def stellar_address(i: int) -> str:
    d = _digest(f"xlm{i}") * 2
    return "G" + "".join(_B32[b % 32] for b in d[:55])


def evm_address(tag: str) -> str:
    return "0x" + _digest(tag).hex()[:40]


def solana_address(tag: str) -> str:
    return "".join(_B58[b % 58] for b in _digest(tag)[:32]) + "1" * 12


def synthetic_edges(
    n_users: int, hub_fraction: float = 0.002, seed: int = 0, offset: int = 0
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    hubs = [evm_address(f"hub{h}") for h in range(max(1, int(n_users * hub_fraction)))]
    edges = []
    for u in range(offset, offset + n_users):
        g = stellar_address(u)
        for k in range(rng.randint(1, 3)):
            edges.append((g, evm_address(f"u{u}e{k}")))
        for k in range(rng.randint(0, 2)):
            edges.append((g, solana_address(f"u{u}s{k}")))
        if rng.random() < 0.05:  # deposits to a shared exchange address
            edges.append((g, rng.choice(hubs)))
    return edges


def _timed(fn, *args) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def _peak_mib(fn, *args) -> float:
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def run(
    sizes: tuple[int, ...] = (5_000, 20_000, 80_000),
    edges_per_shard: int = 2_000,
    batch: int = 100,
    verify_max_users: int = 20_000,
    seed: int = 0,
) -> list[dict[str, float]]:
    rows = []
    rng = random.Random(seed)
    for n_users in sizes:
        edges = synthetic_edges(n_users, seed=seed)
        shards = max(1, len(edges) // edges_per_shard)
        resolver = PartitionedResolver(num_shards=shards, strategy="anchor")
        resolver.add_edges(edges)
        whole, whole_full_s = _timed(resolve_whole_graph, edges)
        _, part_full_s = _timed(resolver.resolve)
        if n_users <= verify_max_users:
            assert canonical_components(resolver.assignment()) == canonical_components(whole)

        # Incremental batch: new users' links plus a few retracted links.
        new_edges = synthetic_edges(batch, seed=seed + 1, offset=n_users)
        removed = rng.sample(edges, batch // 10)
        removed_set = set(removed)
        current = [e for e in edges if e not in removed_set] + new_edges

        _, whole_update_s = _timed(resolve_whole_graph, current)
        resolver.add_edges(new_edges)
        resolver.remove_edges(removed)
        _, part_update_s = _timed(resolver.resolve)
        if n_users <= verify_max_users:
            assert canonical_components(resolver.assignment()) == canonical_components(
                resolve_whole_graph(current)
            )

        largest = max(range(shards), key=lambda i: resolver.shard_sizes()[i])
        whole_mem = _peak_mib(resolve_whole_graph, current)
        shard_mem = _peak_mib(PartitionedResolver.resolve_shard, resolver._shard_edges[largest])

        rows.append(
            {
                "users": n_users,
                "edges": len(edges),
                "shards": shards,
                "boundary_nodes": resolver.boundary_nodes,
                "dirty_shards_after_batch": resolver.last_recomputed_shards,
                "whole_full_s": whole_full_s,
                "partitioned_full_s": part_full_s,
                "whole_update_s": whole_update_s,
                "partitioned_update_s": part_update_s,
                "whole_peak_mib": whole_mem,
                "largest_shard_peak_mib": shard_mem,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[5_000, 20_000, 80_000])
    parser.add_argument("--edges-per-shard", type=int, default=2_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = run(tuple(args.sizes), edges_per_shard=args.edges_per_shard)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(
        "| users | edges | shards | boundary | dirty | whole full (s) | part. full (s)"
        " | whole update (s) | part. update (s) | whole peak MiB | shard peak MiB |"
    )
    print("|" + "---:|" * 11)
    for r in rows:
        print(
            f"| {r['users']} | {r['edges']} | {r['shards']} | {r['boundary_nodes']}"
            f" | {r['dirty_shards_after_batch']} | {r['whole_full_s']:.3f}"
            f" | {r['partitioned_full_s']:.3f} | {r['whole_update_s']:.3f}"
            f" | {r['partitioned_update_s']:.4f} | {r['whole_peak_mib']:.1f}"
            f" | {r['largest_shard_peak_mib']:.2f} |"
        )


if __name__ == "__main__":
    main()
