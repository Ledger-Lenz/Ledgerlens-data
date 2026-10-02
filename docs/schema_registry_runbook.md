# Schema Registry Runbook: Evolving the Trade Schema Safely

This runbook covers how the trade Avro schema (`data/trade_avro_schema.json`)
is registered, how the compatibility mode is enforced, and what to do when a
change is rejected. The compatibility rules are explained in
[`data/schema_evolution.md`](../data/schema_evolution.md).

---

## How registration works

`HorizonKafkaProducer` registers its schema **when it is constructed**, before
it produces anything. The subject is `{KAFKA_TOPIC_PREFIX}-value`, which is
`ledgerlens.trades-value` by default.

| Setting | Default | Effect |
|---|---|---|
| `SCHEMA_REGISTRY_URL` | unset | When set, schemas are registered with that Confluent-compatible Schema Registry (`ConfluentSchemaRegistry`). When unset, they go to an in-process `SchemaRegistry` seeded with the bundled schema. |
| `SCHEMA_COMPATIBILITY_MODE` | `BACKWARD` | `NONE`, `BACKWARD`, `FORWARD` or `FULL`. Enforced against the subject's latest registered version. |

On registration against a remote registry, `ConfluentSchemaRegistry`:

1. Sets the subject's compatibility level to `SCHEMA_COMPATIBILITY_MODE`
   (`PUT /config/{subject}`), so the registry enforces the same mode for every
   other client too.
2. Tests the schema against the latest version
   (`POST /compatibility/subjects/{subject}/versions/latest`).
3. Registers it (`POST /subjects/{subject}/versions`) only if step 2 passed.

A rejected schema raises `SchemaCompatibilityError` (error code
`ingestion_schema_incompatible`). Its `details` include the subject, the mode
and the list of violations. Because this happens in the producer's
constructor, a breaking change stops the producer from starting. It is never
published as a message that consumers cannot decode.

Registry network failures raise `IngestionTransportError`. HTTP 5xx responses
are marked retryable.

---

## Making a schema change

1. **Only make additive changes.** Add new fields with a `"default"`, usually
   `["null", <type>]` with `"default": null`. Do not rename a field (use
   `"aliases"`), change its type, or remove a required field.
2. **Check it locally.**

   ```bash
   make check-schema-compatibility
   pytest tests/test_schema_registry.py tests/test_avro_codec.py -v
   ```

   `make check-schema-compatibility` checks the change against the target
   branch. The `schema-compatibility` CI job runs the same check on every PR.
3. **Fill in the review gate.** Changes to `data/trade_avro_schema.json`
   require a "Kafka wire schema" section in the PR body (see
   [`.github/review-checklists.md`](../.github/review-checklists.md)).
4. **Deploy consumers first when the mode is `BACKWARD`.** Upgraded consumers
   can read messages written with the old schema, so upgrade
   `streaming/kafka_worker.py` deployments before producers. Under `FORWARD`
   the order is reversed: producers first. Under `FULL` either order is safe.
5. **Deploy producers.** On startup each producer registers the new version.
   Check the producer logs for a clean start and confirm the new version on
   the registry:

   ```bash
   curl -s "$SCHEMA_REGISTRY_URL/subjects/ledgerlens.trades-value/versions"
   ```

---

## When a producer fails with `SchemaCompatibilityError`

1. Read the violations in the error message or in `exc.details["violations"]`.
   Each one names the field that caused it.
2. Fix the schema instead of loosening the mode. The usual fix is to add a
   `"default"` to a new field, or to keep a field you were removing.
3. If the change really is breaking (for example a type change), follow the
   dual-write procedure in
   [`data/schema_evolution.md`](../data/schema_evolution.md#handling-breaking-changes):
   add a new field alongside the old one, migrate consumers, and remove the old
   field only after one full topic retention period.
4. Set `SCHEMA_COMPATIBILITY_MODE=NONE` only for a coordinated, reviewed
   migration in which every consumer is redeployed together, and only for the
   producers doing that migration. Change it back immediately afterwards.
   `NONE` also sets the subject's level on the shared registry, so other
   producers lose the protection while it is in effect.

---

## Rolling back

Registered versions are never deleted by LedgerLens. To roll back, redeploy
the producer with the previous schema file. After an additive change the
previous schema only lacks optional fields, so it is still compatible with the
latest version. Registration succeeds, and the registry returns the existing
schema id.
