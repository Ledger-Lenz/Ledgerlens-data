# Cross-chain identity graph partitioning (#883)

## Problem

Identity resolution in `detection/cross_chain/identity_graph.py` walks the graph with
one database round trip per visited node, and it must rerun from scratch after any
link change: new bridge links, links retracted because a proof failed, or PDA edges
excluded (#881). `propagate_risk_scores` also resolved each wallet separately. On a
10k-wallet graph that took **68 s** against a 2 s budget
(`test_10k_node_graph_completes_under_2_seconds` was failing on `main`).

## Design

Implemented in `detection/cross_chain/graph_partition.py`.

### 1. Partitioning (edges, not nodes)

Each edge goes to exactly one shard, `hash(shard_key(edge)) mod num_shards`. Nodes are
not partitioned. A node with edges in several shards is a **boundary node**.

| strategy | shard key | when to use |
|---|---|---|
| `anchor` (default) | the edge's Stellar endpoint, else the smaller endpoint | Identity clusters are stars around a user's Stellar wallet, so a cluster stays in one shard. Only external addresses shared across users (exchange deposit addresses, bridge custody) become boundary nodes. |
| `chain_pair` | unordered chain pair, e.g. `ethereum\|stellar` | Per-chain workers own the data. Coarse: every address bridged to two other chains is a boundary node. |
| `hash` | the smaller endpoint | Locality-blind reference point with the most boundary nodes. |

This is the "connected component with overlap" option from the issue in practice. The
overlap is exactly the boundary nodes, which are replicated into every shard that
touches them.

### 2. Local resolution

Each shard computes its connected components (union-find) from its own edges only.
`PartitionedResolver.resolve_shard(edges)` is a pure function, so shards can run on
separate workers. Each worker's working set is one shard. After `add_edges` or
`remove_edges`, only the shards containing changed edges are marked dirty and
recomputed.

### 3. Boundary-edge reconciliation

Every shard-local component gets a label `shard:local_root`. A coordinator union-find
holds **only boundary nodes' labels**: for each boundary node, it unions the labels of
all its shards. A node's identity is `coordinator.find(label)` if its label is in the
coordinator, or its local label otherwise. Reconciliation costs O(boundary nodes), not
O(graph). Boundary membership is maintained incrementally from per-(node, shard) edge
counts, so removals shrink it correctly.

### Correctness

Two nodes are connected in the whole graph iff a path joins them. Every edge on the
path lies in some shard. Consecutive edges in one shard share a local label, and
consecutive edges in different shards meet at a node present in both shards, which is
a boundary node whose labels the coordinator unions. The converse holds because every
union corresponds to a real edge or a shared node. So partitioned components are
**identical** to whole-graph components for every strategy and shard count; the
strategy only changes cost.

`tests/test_graph_partition.py` checks exact equality (as sets of components) against
whole-graph union-find:

- random graphs × 3 strategies × {1, 3, 17, 64} shards
- 10 rounds of interleaved additions and removals
- a 3,000-user synthetic identity graph
- a hub address linking users from different shards, then un-linking them

It also checks that the partitioned resolver agrees with `IdentityGraph`'s BFS on a
real SQLite graph.

## Integration

- `IdentityGraph.build_partitioned_resolver(num_shards, strategy)` loads all edges with
  one query (`load_edges`).
- `resolve_weighted_risk_scores_bulk(addresses)` loads nodes and edges once and
  resolves a batch of wallets in memory, including the strongest-path link confidence
  from #884. `propagate_risk_scores` and `propagation_attribution` use it: the 10k-node
  test now runs in 0.4 s instead of 68 s.
- `get_connected_component` (single-address BFS) is unchanged in semantics and now also
  reports `link_confidence`.

## Benchmark

`python -m scripts.benchmark_identity_partitioning --sizes 5000 20000 80000 320000`.
Each user has one Stellar wallet with 1–3 EVM and 0–2 Solana links, and 5% of users
deposit to shared exchange hubs. Shard count is scaled at 2,000 edges per shard. The
update batch is 100 new users plus 10 retracted links. Components were asserted
identical to whole-graph at the two smaller sizes, before and after the update.

| users | edges | shards | boundary nodes | dirty shards | whole full (s) | partitioned full (s) | whole update (s) | partitioned update (s) | whole peak MiB | largest-shard peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 15,279 | 7 | 10 | 7 | 0.016 | 0.018 | 0.011 | 0.018 | 1.4 | 0.35 |
| 20,000 | 61,044 | 30 | 40 | 30 | 0.091 | 0.097 | 0.065 | 0.081 | 6.4 | 0.35 |
| 80,000 | 243,694 | 121 | 160 | 71 | 0.478 | 0.418 | 0.444 | 0.222 | 25.7 | 0.35 |
| 320,000 | 975,128 | 487 | 640 | 97 | 2.095 | 1.604 | 2.045 | 0.312 | 102.7 | 0.35 |

Over a 64× larger graph:

- **Update latency:** whole-graph re-resolution grows 186× (linear); partitioned
  grows 18× (sublinear). It is 6.5× faster at 320k users, and the gap widens with size
  because the cost follows the dirty shards and the boundary, not the graph.
- **Per-worker memory** is constant at 0.35 MiB per shard, against 103 MiB to hold
  the whole graph.
- **Full resolution** is about the same either way (it is the same union-find work).
  Partitioning pays off in incremental updates and distribution, not in cold starts.

At small sizes a 100-user batch dirties every shard, so partitioning has nothing to
save and adds a little overhead. Use it once the graph spans many shards.

## Limitations and follow-ups

- Resolution computes connectivity (who is linked). Path-confidence scoring still runs
  per queried address in `resolve_weighted_risk_scores_bulk`. That is cheap because
  identity clusters are small, but a very large component (a hub address that should
  be excluded, like a PDA) makes it expensive. Exclude hubs at insertion time.
- Shards are in-process today. Distributing them means shipping `resolve_shard` inputs
  to workers and gathering boundary labels, and the interfaces are shaped for that.
- The `anchor` strategy assumes Stellar-centred clusters. For EVM↔Solana-heavy graphs,
  pick `chain_pair` or add a strategy.
