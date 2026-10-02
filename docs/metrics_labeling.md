# Metrics Labeling Guidelines

This document describes safe labeling practices for contributors adding new
metrics to the monitoring stack. Following these guidelines prevents unbounded
label cardinality, which is a common cause of metrics-backend overload and cost
blowup in production observability systems.

## Why cardinality matters

Every unique combination of metric name and label values creates a distinct
time series. A single label whose value is drawn from an unbounded set (wallet
addresses, transaction ids, block hashes, user ids, request ids, free-form
error strings, etc.) can multiply the number of series without limit. This
overloads the metrics backend, inflates storage cost, and degrades query
performance for everyone.

## Rules

1. **Never use unbounded identifiers as label values.** Do not label metrics
   with raw wallet addresses, transaction ids, block hashes, user ids, session
   ids, request ids, or any other value that is unique per event.
2. **Prefer bounded, enumerable label values.** Good labels are things like
   `status` (`ok`/`error`), `chain` (`ethereum`/`polygon`), `method`
   (`GET`/`POST`), or `severity` (`info`/`warn`/`error`).
3. **Cap the number of distinct values.** If a label can legitimately take many
   values, bucket them (e.g. `status_class="4xx"`) or drop the label and record
   the detail in logs/traces instead.
4. **Keep the label set small.** Aim for a handful of labels per metric. Each
   additional label multiplies the series count.
5. **Put high-cardinality detail in logs or traces, not metrics.** Metrics are
   for aggregate, low-cardinality signals; logs and traces carry per-event
   detail.

## Runtime guardrail

The metrics collector enforces a cardinality guardrail at emission time. When a
metric is emitted with a label value that looks like a high-cardinality
identifier (for example a long hex string or a UUID), the guardrail flags or
rejects the emission so the problem is caught before it reaches the backend.

If you hit the guardrail while adding a metric, it means the label value is not
safe. Replace it with a bounded value or move the detail to logs/traces.

## CI check

A CI check scans new metric-emission code for known high-cardinality-risk
patterns, such as label values sourced directly from user or transaction
identifiers. The check fails the build when it detects a risky label so the
issue is caught in review rather than in production.

## Checklist for new metrics

- [ ] Metric name is stable and namespaced.
- [ ] Every label has a bounded, enumerable set of values.
- [ ] No label value comes from a wallet address, transaction id, block hash,
      user id, or other per-event identifier.
- [ ] High-cardinality detail is recorded in logs or traces instead.
- [ ] The metric passes the runtime guardrail and the CI cardinality check.
