# Importer conformance requirements

Importers register via `@register_importer(...)` in
`ingestion/registered_importers.py`, declaring `ImporterCapability` flags.
With `strict=True` (used by all built-in importers), each declared capability
is checked against a minimal contract at registration; a failure raises
`ImporterConformanceError` and the importer is not registered.

| Capability           | Contract |
|----------------------|----------|
| `STREAMING`          | a public method starting with `stream` |
| `BULK`               | a public method starting with `load`/`get`/`fetch`/`list`/`compute`/`reconstruct` |
| `DATAFRAME_OUTPUT`   | a method named `*dataframe*` or annotated to return `DataFrame` |
| `POOL_DISCOVERY`     | `list_active_pools` or a `discover*` method |
| `MULTI_HOP_ANALYSIS` | a `reconstruct*` method |

Behavioural capabilities (`RETRY`, `VALIDATION`, `FAILOVER`, ...) have no
static contract. Extend `CAPABILITY_CONTRACTS` in
`ingestion/importer_registry.py` to add one.

CI: `.github/workflows/importer-conformance.yml` runs
`scripts/check_importer_conformance.py` (`ImporterRegistry.verify_conformance()`)
on any PR touching importer files.
