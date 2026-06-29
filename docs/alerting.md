# Alerting Architecture

LedgerLens uses a consensus-driven alert escalation model to reduce false positives while maintaining sensitivity.

## Consensus Escalation

The alerting system requires agreement from multiple detectors before escalating to high-severity alerts:

- **Single-detector alerts**: Emitted on the `low_confidence_alerts` channel when one detector fires
- **Consensus alerts**: Emitted when `MIN_DETECTOR_CONSENSUS` (default 2) distinct detectors fire within `CONSENSUS_WINDOW_SECONDS` (default 120)

## Sliding Window

The consensus window is a sliding window, not a tumbling window. Signals arriving 119 seconds apart still trigger consensus if they meet the threshold.

## Redis Buffer Architecture

Detector signals are buffered in Redis with the following structure:

- Key: `consensus:{wallet}:{pair}`
- Fields: `{detector_name: timestamp}`
- TTL: `CONSENSUS_WINDOW_SECONDS + 10` seconds

This ensures state is preserved across restarts and automatically expires old signals.

## Configuration

Consensus thresholds can be configured per alert severity level in `config.py`.

## Detector Allowlist

Detector names are validated against a known allowlist to prevent injection attacks. Valid detectors include: benford, ml, graph, liquidity, cross_pair.
