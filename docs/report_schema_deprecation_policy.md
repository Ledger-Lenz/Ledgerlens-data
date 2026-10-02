# Report Schema Deprecation Policy

*Applies to: `reporting/schemas/*.json` — the JSON Schema files that define the
structure of LedgerLens exported reports (IVMS101 forensic reports,
model-card metadata, and any future report formats).*

*Related: `docs/deprecation_policy.md` (code deprecation), `data/schema_evolution.md`
(Avro/Kafka schema evolution), `scripts/check_report_schema_compatibility.py`
(automated enforcement).*

---

## Why report schemas need versioning

Downstream consumers of LedgerLens exported reports — regulatory bodies,
exchange partners, compliance platforms — integrate against the **field names,
types, and required-field sets** of each report format. An unannounced change
that removes a field, narrows a type, or adds a required field will silently
break their parsers and may invalidate legally submitted reports.

The versioning scheme and deprecation policy in this document ensure that
consumers always have a machine-readable signal that a schema changed, a
documented runway before old formats are retired, and a compatibility contract
enforced in CI.

---

## Schema version field

Every JSON Schema file under `reporting/schemas/` must carry an explicit
version identifier. Two patterns are accepted:

**Pattern A — top-level `schema_version` property (model_metadata.json)**

```json
{
  "schema_version": "1.0.0",
  "model_name": "xgboost",
  ...
}
```

**Pattern B — `payloadMetadata.schemaVersion` property (ivms101.json)**

```json
{
  "payloadMetadata": {
    "schemaVersion": "1.0.0",
    ...
  }
}
```

The version string must follow **semver** (`MAJOR.MINOR.PATCH`):

| Component | When to increment |
|---|---|
| **MAJOR** | Breaking change — removed required field, narrowed type, removed enum value |
| **MINOR** | Backward-compatible addition — new optional field, new enum value, relaxed type |
| **PATCH** | Non-functional change — description update, comment, formatting |

---

## Compatibility rules (enforced by CI)

`scripts/check_report_schema_compatibility.py` compares every schema in
`reporting/schemas/` against the same files **at the PR target branch** and
fails CI on:

| Change | Classification |
|---|---|
| Required field removed | ❌ Breaking (MAJOR bump required) |
| New required field added | ❌ Breaking — old producers will omit the field |
| Field type narrowed | ❌ Breaking — previously valid values now rejected |
| Enum value removed | ❌ Breaking — existing exports may carry the removed value |
| Schema `title` changed | ❌ Breaking — consumers that identify the schema by title |
| New *optional* field added | ✅ Compatible (MINOR bump) |
| New enum value added | ✅ Compatible (MINOR bump) |
| Type relaxed (e.g. `string` → `["string","null"]`) | ✅ Compatible (MINOR bump) |
| Description / comment updated | ✅ Compatible (PATCH bump) |

Run the check locally before opening a PR:

```bash
python scripts/check_report_schema_compatibility.py
# or
make check-report-schemas
```

---

## Deprecation runway

### Minor / compatible changes

No deprecation runway required. Bump MINOR, add the change, update the version
field in the schema file, and merge.

### Breaking (MAJOR) changes

1. **Announce** the upcoming breaking change in the CHANGELOG under
   `## [Unreleased]`, referencing the affected schema file and planned MAJOR
   version. Post a notice in the `#consumers` channel (or equivalent).
2. **Dual-field migration window** (minimum one full calendar month):
   - Add the new field/structure alongside the old one, marking the old field
     with `"deprecated": true` in its JSON Schema `description`.
   - Bump the MINOR version to signal the addition.
   - Consumers must migrate to the new field during this window.
3. **Remove** the old field in a subsequent release. Bump MAJOR.
4. **Communicate** the removal in the release notes with a migration guide.

### Fast-track (exceptional circumstances)

If a breaking change is required urgently (e.g. a security-sensitive field
must be removed immediately), skip the runway but:

- Coordinate directly with all known consumers before merging.
- Open a companion issue in each affected downstream repo.
- Document the fast-track rationale in the PR description.

---

## Procedure for adding a new report format

1. Create `reporting/schemas/<format_name>.json`.
2. Include the `schema_version` field (starting at `"1.0.0"`).
3. Add the new schema to `reporting/schemas/` (CI will detect it as a new
   schema and skip the compatibility check on its first appearance).
4. Reference the schema in the exporter that produces reports of this type.
5. Document the schema in `docs/reporting.md`.

---

## Cross-reference

- `scripts/check_report_schema_compatibility.py` — automated CI enforcement.
- `scripts/check_deprecation_policy.py` — enforcement for *code* deprecations.
- `data/schema_evolution.md` — Avro/Kafka schema evolution (different rules).
- `docs/deprecation_policy.md` — code module deprecation policy.
