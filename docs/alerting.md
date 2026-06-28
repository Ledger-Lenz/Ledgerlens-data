# Alerting

## Correlated alert deduplication

Independent detectors (Benford engine, GNN embedder, Isolation Forest, ...)
can each flag the same underlying wallet activity within seconds of one
another. Without correlation, an analyst receives one notification per
detector for what is really a single event. `alerts/deduplicator.py` groups
these correlated signals into a single enriched alert.

### Architecture

```
 Benford engine ──┐
 GNN embedder ────┼──► alert_stream ──► deduplicate() ──► grouped alert ──► AlertDispatcher
 Isolation Forest ┘                         │
                                             ▼
                                   in-memory sliding window,
                                   keyed by (wallet_address, asset_pair)
```

* Alerts are buffered per `(wallet_address, asset_pair)` key.
* A group is flushed once `ALERT_DEDUP_WINDOW_SECONDS` (default 60s, clamped
  to 5-300s) of silence has elapsed for that key, measured against each
  alert's `detected_at` event time rather than wall-clock time. This keeps
  the function deterministic for both replay and live use.
* The flushed alert contains the union of contributing detector names, the
  maximum risk score across signals, the union of evidence fields, and the
  earliest `detected_at` timestamp in the group.
* Buffering is purely in-memory and is not persisted across process
  restarts -- a restart simply starts new groups.
* The `ledgerlens_alerts_deduplicated_total` Prometheus counter tracks how
  many raw alerts were folded into an existing group rather than emitted
  standalone.

### Continuous alerting with no silence gap

If a wallet keeps generating alerts back-to-back with no gap longer than
`ALERT_DEDUP_WINDOW_SECONDS`, the group never flushes on its own and keeps
absorbing new signals indefinitely (bounded only by stream end). For a truly
unbounded live stream, pair `deduplicate()` with an external max-group-age
or periodic forced-flush check upstream if you need a hard upper bound on
notification latency; the current implementation intentionally favours
correctness of grouping over a hard latency ceiling.
