# Avro Schema Evolution Guide

This document describes the compatibility rules, versioning procedure, and
migration path for `data/trade_avro_schema.json` — the Avro schema used by the
LedgerLens Kafka ingestion layer.

---

## Background

The trade event schema is shared by:

- `ingestion/kafka_producer.py` (writer) — serialises `Trade` objects to Avro.
- `streaming/kafka_worker.py` (reader) — deserialises Avro messages back to `Trade`.

When the Stellar Horizon API evolves (new fields in responses) or when LedgerLens
adds new feature fields, the schema must evolve while remaining compatible with
messages already in the Kafka topic that were written under earlier schema versions.

---

## Compatibility Rules

### Backward compatibility

**Messages written with the OLD schema can be read with the NEW schema.**

| Change | Backward compatible? |
|---|---|
| Add optional field (has `"default"` or nullable type) | ✅ Yes |
| Add required field (no default) | ❌ No — old messages are missing the value |
| Remove optional field | ✅ Yes — new reader ignores or supplies default |
| Remove required field | ❌ No — new reader cannot reconstruct the value |
| Rename a field | ❌ No (use `"aliases"` instead) |
| Change field type (e.g. `string` → `bytes`) | ❌ Always breaking |

### Forward compatibility

**Messages written with the NEW schema can be read with the OLD schema.**

| Change | Forward compatible? |
|---|---|
| Add optional field (has `"default"`) | ✅ Yes — old reader supplies the default |
| Add field without default | ❌ No — old reader cannot supply a value |
| Remove field that old reader expects (no default) | ❌ No |
| Remove optional field | ✅ Yes |

### Full (bidirectional) compatibility

A change is **fully compatible** only if it passes both checks simultaneously.
The safe subset is: *add optional fields with defaults*.

---

## Versioning Procedure

### Step 1: Author the change

Edit `data/trade_avro_schema.json`.  For every added field, include a `"default"`:

```json
{
  "name": "new_optional_field",
  "type": ["null", "string"],
  "default": null
}
```

### Step 2: Run the compatibility check (CI)

The CI step `make check-schema-compatibility` runs:

```python
from ingestion.avro_codec import (
    SchemaRegistry, check_backward_compatibility, check_forward_compatibility
)
import json

old_schema = json.load(open("data/trade_avro_schema.json.prev"))
new_schema = json.load(open("data/trade_avro_schema.json"))

back_ok, back_errs = check_backward_compatibility(old_schema, new_schema)
fwd_ok,  fwd_errs  = check_forward_compatibility(old_schema, new_schema)

if not back_ok or not fwd_ok:
    raise SystemExit(f"Schema incompatible:\n{back_errs + fwd_errs}")
```

Any change to `data/trade_avro_schema.json` must pass both checks before merge.

### Step 3: Update the schema fingerprint header

Each Kafka message carries an `avro-schema-version` header whose value is the
hex-encoded CRC-32 fingerprint of the encoding schema.  The `HorizonKafkaProducer`
computes this automatically from the loaded schema — no manual change required.

### Step 4: Update consuming services

Services that consume from the Kafka topic should:

1. Read the `avro-schema-version` header from each message.
2. Look up the schema for that fingerprint in their local `SchemaRegistry`.
3. Deserialise using the **writer schema** (from the header) and project to their
   own reader schema using Avro schema resolution.

---

## SchemaRegistry API

```python
from ingestion.avro_codec import SchemaRegistry, load_schema

registry = SchemaRegistry()
fp_v1 = registry.register(load_schema("data/trade_avro_schema_v1.json"))
fp_v2 = registry.register(load_schema("data/trade_avro_schema.json"))

# Check compatibility
back_ok, errors = registry.check_backward_compatibility(fp_v1, fp_v2)
fwd_ok, errors  = registry.check_forward_compatibility(fp_v1, fp_v2)

# Fingerprint lookup
schema = registry.get_schema(fp_v2)
version = registry.get_version(fp_v2)
```

Fingerprints use Avro's CRC-64-AVRO canonical fingerprinting algorithm (with
CRC-32 fallback in environments without the optional dependency).

---

## Handling Breaking Changes

Some changes are unavoidable (e.g. a field's type must change from `string` to
`bytes`).  These are always breaking.  The migration procedure is:

1. **Do not mutate** the existing field.  Instead, add a new field with a `"default"`.
2. **Dual-write**: producers write both the old and the new field for a migration
   window (typically one full Kafka retention period).
3. **Consumers migrate**: update all consumer code to read the new field (falling
   back to the old field when the new field is absent/null).
4. **Remove** the old field in a subsequent release once all messages with the
   old field have expired from the topic.

Never remove a field without checking that it has been absent from all messages
in the topic for at least one full retention window.

---

## Security

Schema files must only be loaded from the bundled `data/` directory at
startup.  Runtime schema negotiation from untrusted external sources (e.g.
operator-supplied URLs, user-submitted JSON) is not supported and must not be
added without a security review.
