# Data Quality Validation Framework

## Why

Validation logic in this repo is currently scattered and ad hoc: range
checks live next to `data/feature_ranges.json`, config presence checks are
bespoke in `tests/test_config_validation.py`, and trade-record shape checks
are implicit in `ingestion/data_models.py`. Each new importer or feature
pipeline that needs to validate incoming data re-derives its own checks,
and failures are usually reported as an opaque exception rather than "row
17, field `score`, value -5 is below minimum 0."

## What this adds

`utils/data_quality.py` provides a small rule-composition contract:

- `ValidationRule` — the `Protocol` every rule satisfies: a `name` and a
  `check(record) -> str | None` method.
- Built-in rules: `RequiredFieldRule`, `TypeRule`, `RangeRule`, `RegexRule`.
- `RangeRule.from_feature_ranges(path)` — optionally builds `RangeRule`
  instances directly from the existing `data/feature_ranges.json`, so
  bounds don't have to be duplicated by hand; returns `[]` (not an
  exception) if the file is absent, so it's safe to use in environments
  without that fixture.
- `DataQualityValidator` — composes rules, runs them against a single
  record (`validate`) or a batch (`validate_batch`), and returns a
  `ValidationReport` with **every** issue found, tagged with the failing
  rule name, field, message, and (for batches) record index.

```python
from utils.data_quality import DataQualityValidator, RequiredFieldRule, RangeRule

validator = DataQualityValidator([
    RequiredFieldRule("wallet"),
    RequiredFieldRule("score"),
    RangeRule("score", minimum=0, maximum=100),
])
report = validator.validate_batch(incoming_records)
if not report.passed:
    for issue in report.issues:
        log.warning("record %s failed %s: %s", issue.record_index, issue.rule_name, issue.message)
```

## Developer commands

```
pytest tests/test_data_quality.py -v

python -m scripts.validate_dataset --input data/some_export.jsonl \
    --required wallet --required score --range score:0:100
```

`scripts/validate_dataset.py` runs the framework over a JSON-lines file and
exits non-zero on any validation failure, so it can be dropped into a CI
step or a pre-ingest gate ahead of a bulk import (pairs naturally with
`ingestion/batch_processor.py` from this same change set — validate a
sample before running `BatchProcessor.run` over the full dataset).

Tests cover: passing/failing on required/type/range/regex rules, bool
values correctly rejected by numeric type/range checks, absent fields
being silent for type/range rules (so `RequiredFieldRule` is the single
source of truth for "missing" diagnostics), batch aggregation with
per-record indices, `fail_fast` early exit, loading rules from
`feature_ranges.json`, and the missing-file fallback.

## Design tradeoffs

- **Composable rule objects, not a schema DSL.** Matches the repo's
  existing preference for small explicit Python objects (e.g. the
  dataclass-based models in `ingestion/data_models.py`) over introducing a
  schema language or new dependency (e.g. `jsonschema`, `pydantic`
  validators) purely for this.
- **Non-raising `check()` contract.** Rules return a message string instead
  of raising, so a `DataQualityValidator` can always finish a full batch
  and report every issue at once, rather than stopping at the first
  exception — important for triaging a bad ingest run in one pass instead
  of a fix-one-fail-again loop.
- **Follow-up work:** wiring `RangeRule.from_feature_ranges()` into the
  feature pipeline (`features/feature_pipeline.py`) as an optional runtime
  guard, and adding a `SchemaRule` that validates against
  `data/trade_avro_schema.json` directly for parity with the Avro codec
  path (`ingestion/avro_codec.py`).

## Stream-level anomaly detection (issue #913)

Per-record rules cannot see a systemic upstream fault whose records are each
valid but collectively wrong: a feed that silently drops most trades, or a
field that starts arriving null for every record. `StreamQualityMonitor` in
`ingestion/data_quality.py` adds rolling statistical checks on top of the
per-record rules.

For every batch passed to `observe_batch(source, batch)` it computes:

| Metric | Meaning |
|---|---|
| `volume` | number of records in the batch |
| `null_rate:<field>` | null ratio of each configured `key_fields` column |
| `mean:<field>` | mean of each configured numeric `distribution_fields` column |

Each value is compared with a rolling window (`window`, default 24 batches) of
that source's previous values. Once `min_history` batches exist, a value whose
z-score exceeds `z_threshold` (default 4) is a `StreamAnomaly` (`spike` or
`drop`). The spread is floored at 5% of the baseline mean (1 percentage point
for null rates), so a near-constant baseline does not turn normal jitter into
alerts. Anomalous values are kept out of the baseline; after an intended level
change, call `reset_baseline(source)`.

Unsuppressed anomalies go through `alerts.router.AlertRouter` with
`detectors: ["stream_quality_monitor"]` plus `source`, `metric`, `direction`,
`observed`, `baseline` and `magnitude` (the z-score). The shipped
`alerts/routing_config.yaml` sends these to `#data-quality`. `AlertRouter` only
picks destinations; the monitor calls the `dispatch(destination, alert)`
function you pass in to deliver each one.

```python
from alerts.router import AlertRouter
from ingestion.data_quality import StreamQualityMonitor

monitor = StreamQualityMonitor(
    router=AlertRouter.from_yaml("alerts/routing_config.yaml"),
    dispatch=send_alert,
    key_fields=["amount", "price"],
    distribution_fields=["amount"],
)
monitor.observe_batch("horizon_trades", trades_df)
```

### Acknowledging expected changes

Planned maintenance or a known backfill will change volume on purpose.
Acknowledge the window so it does not flood on-call:

```python
monitor.acknowledge(
    "horizon_trades", start, end,
    reason="Horizon maintenance CHG-1234",
    metrics=["volume"],  # omit to cover every metric; source="*" covers every source
)
```

Anomalies inside the window are still returned and counted
(`suppressed="true"`), but they are not routed. Expired windows are dropped
automatically.

Metrics `ledgerlens_ingestion_stream_metric_value`,
`ledgerlens_ingestion_stream_metric_zscore` and
`ledgerlens_ingestion_stream_anomalies_total` are shown on the
`monitoring/grafana/dashboards/ingestion_stream_quality.json` dashboard.

```
pytest tests/test_stream_quality_monitor.py -v
```
