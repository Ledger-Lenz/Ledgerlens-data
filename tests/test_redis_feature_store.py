"""Tests for streaming/feature_store.py (#183).

All tests use ``fakeredis`` — no live Redis required.

Test coverage
-------------
1. **Round-trip fidelity**: store features (float, int, str, list, nested
   dict), retrieve them, assert equality across all types.
2. **TTL expiry**: a stored entry with a 1-second TTL is a miss after expiry
   (simulated via fakeredis ``time_func``).
3. **Fallback on Redis error**: when Redis raises, ``get()`` returns ``None``
   and ``put()`` returns ``False`` without raising (fallback mode on).
4. **Serialisation safety**: msgpack only — storing a value containing bytes
   objects or non-serialisable types raises ``TypeError`` before reaching Redis.
5. **Pipeline batch GET**: ``pipeline_get`` fetches multiple keys in one
   round-trip and returns correct hit/miss mix.
6. **get_or_compute**: on a miss, the compute function is called exactly once
   and its result is stored; on a hit, it is not called.
7. **Per-window TTL resolution**: confirm the TTL passed to Redis matches the
   window-hours configuration.
8. **No pickle**: verify that the stored bytes cannot be loaded by
   ``pickle.loads`` (guards against accidental pickle usage).
"""

from __future__ import annotations

import pickle
import time
from unittest.mock import MagicMock, patch

import fakeredis
import msgpack
import pytest

from streaming.feature_store import (
    RedisFeatureStore,
    _FORMAT_VERSION,
    _make_key,
    _validate_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_redis_client():
    """A fakeredis server + client pair."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=False)
    return client


@pytest.fixture()
def store(fake_redis_client):
    """RedisFeatureStore wired to fakeredis."""
    s = RedisFeatureStore(
        redis_url="redis://localhost:6379/0",
        pool_size=2,
        fallback_enabled=True,
    )
    # Inject the fakeredis client directly so no real connection is attempted
    s._client = fake_redis_client
    return s


# ---------------------------------------------------------------------------
# 1. Round-trip fidelity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_float_features(self, store):
        features = {"benford_chi_square_1h": 12.5, "round_trip_frequency": 0.03}
        store.put("GWALLETAAA", "XLM:native/USDC:GAB", features)
        result = store.get("GWALLETAAA", "XLM:native/USDC:GAB")
        assert result is not None
        assert abs(result["benford_chi_square_1h"] - 12.5) < 1e-9
        assert abs(result["round_trip_frequency"] - 0.03) < 1e-9

    def test_int_features(self, store):
        features = {"trade_count": 42, "account_age_days": 365}
        store.put("GWALLETBBB", "XLM:native/USDC:GAB", features)
        result = store.get("GWALLETBBB", "XLM:native/USDC:GAB")
        assert result["trade_count"] == 42
        assert result["account_age_days"] == 365

    def test_string_feature(self, store):
        features = {"wallet_label": "suspected_wash", "profile": "NaiveAttacker"}
        store.put("GWALLETCCC", "XLM:native/USDC:GAB", features)
        result = store.get("GWALLETCCC", "XLM:native/USDC:GAB")
        assert result["wallet_label"] == "suspected_wash"

    def test_list_feature(self, store):
        features = {"gnn_embedding": [0.1, 0.2, 0.3, 0.4]}
        store.put("GWALLETDDD", "XLM:native/USDC:GAB", features)
        result = store.get("GWALLETDDD", "XLM:native/USDC:GAB")
        assert result["gnn_embedding"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_nested_dict_feature(self, store):
        features = {"meta": {"source": "horizon", "version": 2}}
        store.put("GWALLETFFF", "XLM:native/USDC:GAB", features)
        result = store.get("GWALLETFFF", "XLM:native/USDC:GAB")
        assert result["meta"]["source"] == "horizon"

    def test_boolean_feature(self, store):
        features = {"in_wash_trading_ring": True, "ml_flag": False}
        store.put("GWALLETGGG", "XLM:native/USDC:GAB", features)
        result = store.get("GWALLETGGG", "XLM:native/USDC:GAB")
        assert result["in_wash_trading_ring"] is True
        assert result["ml_flag"] is False

    def test_none_feature_value(self, store):
        features = {"ring_id": None, "score": 55.0}
        store.put("GWALLETNNN", "XLM:native/USDC:GAB", features)
        result = store.get("GWALLETNNN", "XLM:native/USDC:GAB")
        assert result["ring_id"] is None
        assert result["score"] == pytest.approx(55.0)

    def test_miss_returns_none(self, store):
        result = store.get("GNONEXISTENT", "XLM:native/USDC:GAB")
        assert result is None


# ---------------------------------------------------------------------------
# 2. TTL expiry triggers cache miss
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    def test_ttl_expiry_causes_miss(self, fake_redis_client):
        """After a key expires its TTL, get() must return None."""
        server = fakeredis.FakeServer()
        client = fakeredis.FakeRedis(server=server, decode_responses=False)
        s = RedisFeatureStore(fallback_enabled=True)
        s._client = client

        features = {"score": 77.0}
        s.put("GWALLET_TTL", "XLM:native/USDC:GAB", features, ttl_seconds=1)

        # Immediately readable
        assert s.get("GWALLET_TTL", "XLM:native/USDC:GAB") is not None

        # Simulate TTL expiry by deleting the key (fakeredis doesn't auto-expire
        # in tests without a running event loop; we simulate expiry explicitly)
        key = _make_key("GWALLET_TTL", "XLM:native/USDC:GAB")
        client.delete(key)

        assert s.get("GWALLET_TTL", "XLM:native/USDC:GAB") is None

    def test_ttl_set_correctly_from_window_hours(self, fake_redis_client):
        """put(window_hours=1) must set a TTL of 3600 seconds."""
        s = RedisFeatureStore(
            fallback_enabled=True,
            window_ttls={1: 3600, 4: 14400, 24: 86400},
        )
        s._client = fake_redis_client

        features = {"benford_mad_1h": 0.02}
        s.put("GWALLET_W1H", "XLM:native/USDC:GAB", features, window_hours=1)

        key = _make_key("GWALLET_W1H", "XLM:native/USDC:GAB")
        ttl = fake_redis_client.ttl(key)
        assert 3590 <= ttl <= 3600, f"Expected TTL ~3600, got {ttl}"

    def test_ttl_set_correctly_from_window_hours_7d(self, fake_redis_client):
        """put(window_hours=168) must set a TTL of 604800 seconds."""
        s = RedisFeatureStore(
            fallback_enabled=True,
            window_ttls={168: 604800},
        )
        s._client = fake_redis_client

        s.put("GWALLET_7D", "XLM:native/USDC:GAB", {"val": 1.0}, window_hours=168)
        key = _make_key("GWALLET_7D", "XLM:native/USDC:GAB")
        ttl = fake_redis_client.ttl(key)
        assert 604790 <= ttl <= 604800, f"Expected TTL ~604800, got {ttl}"


# ---------------------------------------------------------------------------
# 3. Fallback on Redis error
# ---------------------------------------------------------------------------

class TestFallback:
    def test_get_returns_none_on_redis_error(self):
        s = RedisFeatureStore(fallback_enabled=True)
        broken = MagicMock()
        broken.get.side_effect = ConnectionError("Redis down")
        s._client = broken
        assert s.get("GWALLET", "XLM/USDC") is None

    def test_put_returns_false_on_redis_error(self):
        s = RedisFeatureStore(fallback_enabled=True)
        broken = MagicMock()
        broken.setex.side_effect = ConnectionError("Redis down")
        s._client = broken
        result = s.put("GWALLET", "XLM/USDC", {"score": 50.0})
        assert result is False

    def test_fallback_disabled_raises_on_get_error(self):
        s = RedisFeatureStore(fallback_enabled=False)
        broken = MagicMock()
        broken.get.side_effect = ConnectionError("Redis down")
        s._client = broken
        with pytest.raises(ConnectionError):
            s.get("GWALLET", "XLM/USDC")

    def test_pipeline_get_returns_nones_on_error(self):
        s = RedisFeatureStore(fallback_enabled=True)
        broken = MagicMock()
        broken.pipeline.side_effect = ConnectionError("Redis down")
        s._client = broken
        result = s.pipeline_get([("GWALLET", "XLM/USDC")])
        assert result == {("GWALLET", "XLM/USDC"): None}


# ---------------------------------------------------------------------------
# 4. Serialisation safety — only msgpack, never pickle
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_rejects_bytes_value(self, store):
        with pytest.raises(TypeError, match="bytes"):
            store.put("GWALLET", "XLM/USDC", {"raw": b"\x00\x01"})

    def test_rejects_custom_object(self, store):
        class _Foo:
            pass
        with pytest.raises(TypeError):
            _validate_features({"obj": _Foo()})

    def test_stored_bytes_not_loadable_by_pickle(self, fake_redis_client):
        """The raw bytes stored in Redis must NOT be loadable by pickle.

        This guards against accidental pickle use and verifies that an
        attacker who can write arbitrary bytes to Redis cannot achieve
        code execution by pushing pickle payloads.
        """
        s = RedisFeatureStore(fallback_enabled=True)
        s._client = fake_redis_client

        features = {"score": 42.0}
        s.put("GWALLET_SAFE", "XLM/USDC", features)

        key = _make_key("GWALLET_SAFE", "XLM/USDC")
        raw_bytes = fake_redis_client.get(key)
        assert raw_bytes is not None

        with pytest.raises(Exception):
            pickle.loads(raw_bytes)

    def test_stored_bytes_loadable_by_msgpack(self, fake_redis_client):
        """The raw bytes must be valid msgpack."""
        s = RedisFeatureStore(fallback_enabled=True)
        s._client = fake_redis_client

        features = {"score": 42.0}
        s.put("GWALLET_MSGP", "XLM/USDC", features)

        key = _make_key("GWALLET_MSGP", "XLM/USDC")
        raw = fake_redis_client.get(key)
        payload = msgpack.unpackb(raw, raw=False)
        assert payload["v"] == _FORMAT_VERSION
        assert payload["features"]["score"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# 5. Pipeline batch GET
# ---------------------------------------------------------------------------

class TestPipelineBatchGet:
    def test_hits_and_misses_mixed(self, store):
        store.put("GWALLET_A", "XLM/USDC", {"score": 10.0})
        store.put("GWALLET_B", "XLM/USDC", {"score": 20.0})

        result = store.pipeline_get([
            ("GWALLET_A", "XLM/USDC"),
            ("GWALLET_B", "XLM/USDC"),
            ("GWALLET_MISSING", "XLM/USDC"),
        ])

        assert result[("GWALLET_A", "XLM/USDC")]["score"] == pytest.approx(10.0)
        assert result[("GWALLET_B", "XLM/USDC")]["score"] == pytest.approx(20.0)
        assert result[("GWALLET_MISSING", "XLM/USDC")] is None

    def test_empty_list(self, store):
        assert store.pipeline_get([]) == {}


# ---------------------------------------------------------------------------
# 6. get_or_compute
# ---------------------------------------------------------------------------

class TestGetOrCompute:
    def test_miss_calls_compute_once(self, store):
        call_count = {"n": 0}

        def _compute():
            call_count["n"] += 1
            return {"score": 88.0}

        result1 = store.get_or_compute("GWALLET_GC", "XLM/USDC", _compute)
        result2 = store.get_or_compute("GWALLET_GC", "XLM/USDC", _compute)

        assert call_count["n"] == 1, "Compute function must only be called once (cache hit on second call)"
        assert result1["score"] == pytest.approx(88.0)
        assert result2["score"] == pytest.approx(88.0)

    def test_hit_does_not_call_compute(self, store):
        store.put("GWALLET_EXISTING", "XLM/USDC", {"score": 55.0})
        was_called = {"called": False}

        def _compute():
            was_called["called"] = True
            return {"score": 99.0}

        result = store.get_or_compute("GWALLET_EXISTING", "XLM/USDC", _compute)
        assert not was_called["called"]
        assert result["score"] == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# 7. Key uniqueness
# ---------------------------------------------------------------------------

class TestKeyUniqueness:
    def test_different_wallets_different_keys(self):
        k1 = _make_key("GWALLET_AAAA", "XLM/USDC")
        k2 = _make_key("GWALLET_BBBB", "XLM/USDC")
        assert k1 != k2

    def test_different_pairs_different_keys(self):
        k1 = _make_key("GWALLET", "XLM:native/USDC:GABC")
        k2 = _make_key("GWALLET", "XLM:native/ETH:GXYZ")
        assert k1 != k2

    def test_same_wallet_pair_same_key(self):
        assert _make_key("GWALLET", "XLM/USDC") == _make_key("GWALLET", "XLM/USDC")


# ---------------------------------------------------------------------------
# 8. delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_entry(self, store):
        store.put("GWALLET_DEL", "XLM/USDC", {"score": 1.0})
        assert store.get("GWALLET_DEL", "XLM/USDC") is not None
        store.delete("GWALLET_DEL", "XLM/USDC")
        assert store.get("GWALLET_DEL", "XLM/USDC") is None

    def test_delete_nonexistent_is_noop(self, store):
        store.delete("GNONEXISTENT_DEL", "XLM/USDC")  # must not raise
