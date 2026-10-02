"""Schema registry integration and compatibility-mode enforcement (Issue #914)."""

import copy
import json
from unittest.mock import MagicMock

import pytest
import requests

from ingestion import avro_codec
from ingestion.avro_codec import (
    CompatibilityMode,
    ConfluentSchemaRegistry,
    SchemaRegistry,
    check_compatibility,
    deserialize,
    load_schema,
    read_schema,
)
from ingestion.exceptions import (
    IngestionTransportError,
    InvalidInputError,
    SchemaCompatibilityError,
)
from ingestion.kafka_producer import HorizonKafkaProducer
from tests.test_kafka_producer import make_trade

SUBJECT = "ledgerlens.trades-value"


@pytest.fixture
def base_schema() -> dict:
    return read_schema()


@pytest.fixture
def additive_schema(base_schema) -> dict:
    schema = copy.deepcopy(base_schema)
    schema["fields"].append({"name": "venue", "type": ["null", "string"], "default": None})
    return schema


@pytest.fixture
def breaking_schema(base_schema) -> dict:
    schema = copy.deepcopy(base_schema)
    schema["fields"].append({"name": "venue", "type": "string"})
    return schema


# ---------------------------------------------------------------------------
# In-process registry
# ---------------------------------------------------------------------------


def test_breaking_change_is_rejected_at_registration(base_schema, breaking_schema):
    registry = SchemaRegistry(CompatibilityMode.BACKWARD)
    registry.register(base_schema, SUBJECT)

    with pytest.raises(SchemaCompatibilityError) as excinfo:
        registry.register(breaking_schema, SUBJECT)

    assert "venue" in str(excinfo.value)
    assert excinfo.value.details["compatibility"] == "BACKWARD"
    assert registry.latest_schema(SUBJECT) == base_schema


def test_removing_a_required_field_is_rejected(base_schema):
    registry = SchemaRegistry("FULL")
    registry.register(base_schema, SUBJECT)
    removed = copy.deepcopy(base_schema)
    removed["fields"] = [f for f in removed["fields"] if f["name"] != "price"]

    with pytest.raises(SchemaCompatibilityError):
        registry.register(removed, SUBJECT)


def test_additive_change_is_accepted(base_schema, additive_schema):
    registry = SchemaRegistry(CompatibilityMode.FULL)
    v1 = registry.register(base_schema, SUBJECT)
    v2 = registry.register(additive_schema, SUBJECT)

    assert v1 != v2
    assert registry.get_version(v2) == 2
    assert registry.latest_schema(SUBJECT) == additive_schema
    # Re-registering the latest schema is a no-op.
    assert registry.register(additive_schema, SUBJECT) == v2
    assert registry.get_version(v2) == 2


def test_subjects_are_checked_independently(base_schema, breaking_schema):
    registry = SchemaRegistry(CompatibilityMode.BACKWARD)
    registry.register(base_schema, SUBJECT)

    registry.register(breaking_schema, "other-topic-value")

    assert registry.latest_schema("other-topic-value") == breaking_schema


def test_none_mode_accepts_breaking_changes(base_schema, breaking_schema):
    registry = SchemaRegistry(CompatibilityMode.NONE)
    registry.register(base_schema, SUBJECT)
    registry.register(breaking_schema, SUBJECT)

    assert registry.latest_schema(SUBJECT) == breaking_schema


def test_registry_defaults_to_configured_mode(monkeypatch):
    monkeypatch.setattr(avro_codec.config, "SCHEMA_COMPATIBILITY_MODE", "forward")
    assert SchemaRegistry().compatibility is CompatibilityMode.FORWARD


def test_unknown_mode_is_rejected():
    with pytest.raises(InvalidInputError):
        SchemaRegistry("SIDEWAYS")


@pytest.mark.parametrize("mode", list(CompatibilityMode))
def test_check_compatibility_modes(base_schema, additive_schema, breaking_schema, mode):
    assert check_compatibility(base_schema, additive_schema, mode) == (True, [])

    ok, violations = check_compatibility(base_schema, breaking_schema, mode)
    assert ok is (mode is CompatibilityMode.NONE)
    assert bool(violations) is (mode is not CompatibilityMode.NONE)


# ---------------------------------------------------------------------------
# Producer: breaking schemas are rejected before anything is published
# ---------------------------------------------------------------------------


def _write(tmp_path, schema: dict) -> str:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema))
    return str(path)


def test_producer_rejects_breaking_schema_before_publish(tmp_path, base_schema, breaking_schema):
    registry = SchemaRegistry(CompatibilityMode.BACKWARD)
    registry.register(base_schema, SUBJECT)
    kafka = MagicMock()

    with pytest.raises(SchemaCompatibilityError):
        HorizonKafkaProducer(
            producer=kafka,
            topic_prefix="ledgerlens.trades",
            schema_path=_write(tmp_path, breaking_schema),
            schema_registry=registry,
        )

    kafka.produce.assert_not_called()
    assert registry.latest_schema(SUBJECT) == base_schema


def test_producer_accepts_additive_schema_and_publishes(tmp_path, base_schema, additive_schema):
    registry = SchemaRegistry(CompatibilityMode.FULL)
    registry.register(base_schema, SUBJECT)
    kafka = MagicMock()
    path = _write(tmp_path, additive_schema)

    producer = HorizonKafkaProducer(
        producer=kafka, topic_prefix="ledgerlens.trades", schema_path=path, schema_registry=registry
    )
    producer.produce_trade(make_trade())

    assert registry.latest_schema(SUBJECT) == additive_schema
    value = kafka.produce.call_args.kwargs["value"]
    # Consumers still on the previous schema can read the new messages...
    assert deserialize(value, load_schema())["trade_id"] == "trade-001"
    # ...and the new field falls back to its default.
    assert deserialize(value, load_schema(path))["venue"] is None


# ---------------------------------------------------------------------------
# Confluent-compatible registry client (HTTP mocked)
# ---------------------------------------------------------------------------


def _response(status: int, body: dict | None = None) -> MagicMock:
    resp = MagicMock(status_code=status, text=json.dumps(body or {}))
    resp.json.return_value = body or {}
    return resp


def _client(*responses: MagicMock) -> tuple[ConfluentSchemaRegistry, MagicMock]:
    session = MagicMock()
    session.request.side_effect = list(responses)
    return ConfluentSchemaRegistry("http://registry:8081/", "BACKWARD", session=session), session


def test_confluent_register_sets_mode_checks_then_registers(base_schema):
    client, session = _client(
        _response(200, {"compatibility": "BACKWARD"}),
        _response(200, {"is_compatible": True}),
        _response(200, {"id": 7}),
    )

    assert client.register(base_schema, SUBJECT) == 7

    calls = [(c.args[0], c.args[1]) for c in session.request.call_args_list]
    assert calls == [
        ("PUT", f"http://registry:8081/config/{SUBJECT}"),
        (
            "POST",
            f"http://registry:8081/compatibility/subjects/{SUBJECT}/versions/latest?verbose=true",
        ),
        ("POST", f"http://registry:8081/subjects/{SUBJECT}/versions"),
    ]
    assert session.request.call_args_list[0].kwargs["json"] == {"compatibility": "BACKWARD"}
    assert json.loads(session.request.call_args_list[2].kwargs["json"]["schema"]) == base_schema
    assert client.get_schema(7) == base_schema


def test_confluent_rejects_incompatible_schema_without_registering(breaking_schema):
    client, session = _client(
        _response(200),
        _response(
            200, {"is_compatible": False, "messages": ["READER_FIELD_MISSING_DEFAULT_VALUE"]}
        ),
    )

    with pytest.raises(SchemaCompatibilityError, match="READER_FIELD_MISSING_DEFAULT_VALUE"):
        client.register(breaking_schema, SUBJECT)

    assert session.request.call_count == 2


def test_confluent_first_version_of_new_subject_registers(base_schema):
    client, _ = _client(_response(200), _response(404), _response(200, {"id": 1}))

    assert client.register(base_schema, SUBJECT) == 1


def test_confluent_registry_side_conflict_is_a_compatibility_error(base_schema):
    client, _ = _client(_response(200), _response(200, {"is_compatible": True}), _response(409))

    with pytest.raises(SchemaCompatibilityError):
        client.register(base_schema, SUBJECT)


def test_confluent_lookup_by_id(base_schema):
    client, session = _client(
        _response(200, {"schema": json.dumps(base_schema)}), _response(404, {"error_code": 40403})
    )

    assert client.get_schema(3) == base_schema
    assert client.get_schema(3) == base_schema  # cached
    assert client.get_schema(99) is None
    assert session.request.call_count == 2


def test_confluent_network_failure_is_a_transport_error(base_schema):
    session = MagicMock()
    session.request.side_effect = requests.ConnectionError("refused")
    client = ConfluentSchemaRegistry("http://registry:8081", session=session)

    with pytest.raises(IngestionTransportError):
        client.register(base_schema, SUBJECT)


def test_default_registry_uses_confluent_when_url_configured(monkeypatch):
    monkeypatch.setattr(avro_codec, "_default_registry", None)
    monkeypatch.setattr(avro_codec.config, "SCHEMA_REGISTRY_URL", "http://registry:8081")

    registry = avro_codec.get_default_registry()

    assert isinstance(registry, ConfluentSchemaRegistry)
    assert registry.url == "http://registry:8081"


def test_default_registry_is_in_process_and_seeded_without_url(monkeypatch, base_schema):
    monkeypatch.setattr(avro_codec, "_default_registry", None)
    monkeypatch.setattr(avro_codec.config, "SCHEMA_REGISTRY_URL", None)

    registry = avro_codec.get_default_registry()

    assert isinstance(registry, SchemaRegistry)
    assert registry.latest_schema() == base_schema
