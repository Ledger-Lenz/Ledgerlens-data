# Parallel executor: per-category isolation

`ingestion.parallel_executor.PartitionedExecutor` runs each task category in
its own bounded worker pool so one slow source cannot starve others.

```python
with PartitionedExecutor({"horizon": 8, "amm": 4, "file": 2}) as ex:
    ex.submit("horizon", load_pair, pair)
    print(ex.metrics())  # per-category queue_depth, in_flight, durations
```

Categories not listed use `default_limit` (4).

## Recommended defaults

| Deployment | horizon | amm | orderbook | file | default |
|------------|---------|-----|-----------|------|---------|
| Small (≤4 cores)   | 4  | 2 | 2 | 1 | 2 |
| Medium (8–16 cores)| 8  | 4 | 4 | 2 | 4 |
| Large (32+ cores)  | 16 | 8 | 8 | 4 | 8 |

Network-bound sources (Horizon) tolerate more workers than CPU/disk-bound
ones (file parsing). A category whose `queue_depth` stays high or whose
`max_duration_s` spikes is a noisy-neighbor candidate.
