# Project health summary (Issue #605)

Dashboard-ready **project health** data for LedgerLens-data: a stable JSON contract, collectors, CLI, API endpoint, and Grafana board that operators can use without scraping ad-hoc paths.

## Why

Ops and dashboard consumers previously pieced together health from scattered files (`models/metrics.json`, Grafana, adversarial reports). Issue #605 asks for a **durable engineering capability**: one validated document that summarises project foundation health with actionable diagnostics.

## Contract (schema `1.0`)

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Always `"1.0"` for this contract |
| `generated_at` | string | UTC ISO-8601 timestamp |
| `overall_status` | string | `healthy` \| `degraded` \| `critical` \| `unknown` (worst-of rollup) |
| `components` | array | Per-subsystem health records |
| `signals` | object | Machine-readable rollups for dashboards |
| `diagnostics` | string[] | Actionable messages (highest severity first) |
| `thresholds` | object | Thresholds used for evaluation |

### Component record

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable component id |
| `name` | string | Human label |
| `status` | string | Same enum as overall |
| `score` | number | 0–100 |
| `metrics` | object | Component-specific numbers / paths |
| `diagnostics` | string[] | Actionable failures for this component |
| `checked_at` | string | UTC timestamp |

### Default components

| id | What it checks |
|---|---|
| `model_artifacts` | `models/` presence, baseline files, weight files |
| `training_metrics` | `models/metrics.json` AUC degradation / MIA rates |
| `data_contracts` | feature ranges, dictionary, Avro schema, dataset card |
| `adversarial_posture` | `reports/adversarial_benchmark.json` evasion rates |
| `ops_surface` | Prometheus/alert config + Grafana dashboard JSON validity |
| `pipeline_entrypoints` | `run_pipeline.py`, API, stream script, requirements |

## Module layout

| Path | Role |
|---|---|
| `monitoring/project_health.py` | Contract, collectors, builder, validation |
| `scripts/project_health_summary.py` | CLI / CI entrypoint |
| `api/app.py` → `GET /v1/ops/project-health` | Authenticated API surface |
| `monitoring/grafana/dashboards/project_health.json` | Grafana board |
| `tests/test_project_health.py` | Unit + contract tests |

## CLI usage

```bash
# Pretty-print summary
python -m scripts.project_health_summary

# Write dashboard artifact
python -m scripts.project_health_summary --output reports/project_health.json

# CI gate: fail when overall is critical (or worse than threshold)
python -m scripts.project_health_summary --fail-on critical --quiet

# Makefile helper
make project-health
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Summary built; gate passed (or `--fail-on never`) |
| `1` | Overall status reached/exceeded `--fail-on` severity |

## API usage

```http
GET /v1/ops/project-health
X-API-Key: <key>
```

Requires a valid API key (same as score endpoints). Response body matches schema `1.0`.

Liveness remains on `GET /v1/health` (DB/model probe only) — project health is intentionally deeper and auth-gated.

## Thresholds

Defaults in `monitoring.project_health.DEFAULT_THRESHOLDS`:

| Key | Default | Meaning |
|---|---|---|
| `max_auc_degradation` | `0.05` | Max allowed AUC drop vs baseline in DP metrics |
| `max_mia_success_rate` | `0.65` | Max membership-inference success rate |
| `max_adversarial_evasion_rate` | `0.25` | Max evasion rate from adversarial benchmark |
| `min_component_score` | `50.0` | Reserved for future score-based gates |

Override by passing `thresholds=` to `build_project_health_summary`.

## Extending

Add a collector:

```python
from monitoring.project_health import ComponentHealth, build_project_health_summary

def collect_my_thing(repo_root, thresholds) -> ComponentHealth:
    ...

summary = build_project_health_summary(collectors=[..., collect_my_thing])
```

Validate any external document with `validate_project_health_summary` / `assert_valid_project_health_summary`.

## Design choices

1. **Filesystem-first collectors** — unit tests and offline CI work without Redis/Kafka/DB.
2. **Worst-of status rollup** — dashboards get a single unambiguous overall status.
3. **Degrade, don't crash** — missing optional artifacts emit diagnostics; invalid JSON on critical paths is `critical`.
4. **Contract validation always** — `build_project_health_summary` asserts schema validity before return.

## Follow-up risks

- Prometheus gauges for the Grafana board are **not** auto-exported yet; operators can scrape JSON via API/CLI or wire exporters later.
- Training metrics depend on whatever structure training scripts write into `models/metrics.json`; new model families should extend `collect_training_metrics` carefully.
- Live scoring latency / queue depth are out of scope here (covered by capacity + Kafka dashboards).
