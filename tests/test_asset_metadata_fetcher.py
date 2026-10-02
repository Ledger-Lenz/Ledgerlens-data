"""Staleness-aware caching and trust-tier fallback for asset metadata (Issue #917).

Horizon is never contacted: ``urllib.request.urlopen`` is replaced with a fake
that serves or fails per base URL.
"""

import io
import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from detection.forensic_report import ForensicReportGenerator
from features import feature_pipeline
from ingestion import asset_metadata_fetcher as amf
from ingestion.asset_metadata_fetcher import TrustTier, get_asset_metadata, get_asset_supply

PRIMARY = "https://horizon.primary"
ALTERNATE = "https://horizon.alternate"
CODE, ISSUER = "USDC", "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
KEY = f"ledgerlens:asset_supply:{CODE}:{ISSUER}"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
WALLET = "GBRPYHIL2CI3FNQ4BXLFMNDLFJUNPU2HY3ZMFXYSFZW2BV3FL224GKO7"


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttls[key] = ttl


class FakeHorizon:
    """Serves ``supply`` per base URL; URLs in ``down`` raise like an outage."""

    def __init__(self, monkeypatch, supply: dict[str, float]):
        self.supply = supply
        self.down: set[str] = set()
        self.calls: list[str] = []
        monkeypatch.setattr(amf.urllib.request, "urlopen", self._urlopen)

    def _urlopen(self, url, timeout):  # noqa: ARG002
        base = url.split("/assets?")[0]
        self.calls.append(base)
        if base in self.down:
            raise OSError("connection refused")
        records = [{"amount": str(self.supply[base])}] if base in self.supply else []
        return io.BytesIO(json.dumps({"_embedded": {"records": records}}).encode())


@pytest.fixture(autouse=True)
def _clear_local_cache(monkeypatch):
    monkeypatch.setattr(amf, "_local_cache", {})


@pytest.fixture
def horizon(monkeypatch):
    return FakeHorizon(monkeypatch, {PRIMARY: 1_000_000.0, ALTERNATE: 990_000.0})


def _fetch(redis=None, alternate=ALTERNATE, now=T0):
    return get_asset_metadata(
        CODE, ISSUER, PRIMARY, redis, alternate_horizon_url=alternate, now=now
    )


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


def test_primary_available_is_primary_tier(horizon):
    record = _fetch()

    assert record.circulating_supply == 1_000_000.0
    assert record.trust_tier is TrustTier.PRIMARY
    assert record.source == PRIMARY
    assert record.fetched_at == T0
    assert record.age_seconds == 0
    assert record.is_stale is False
    assert record.is_degraded is False


def test_fresh_cache_hit_is_primary_tier_without_refetch(horizon):
    _fetch()

    record = _fetch(now=T0 + timedelta(minutes=30))

    assert horizon.calls == [PRIMARY]
    assert record.trust_tier is TrustTier.PRIMARY
    assert record.age_seconds == 1800
    assert record.is_stale is False


def test_expired_cache_refreshes_from_primary(horizon):
    _fetch()
    horizon.supply[PRIMARY] = 1_200_000.0

    record = _fetch(now=T0 + timedelta(hours=2))

    assert record.circulating_supply == 1_200_000.0
    assert record.trust_tier is TrustTier.PRIMARY
    assert record.fetched_at == T0 + timedelta(hours=2)


def test_outage_falls_back_to_stale_cache(horizon):
    _fetch()
    horizon.down.add(PRIMARY)

    record = _fetch(now=T0 + timedelta(hours=5))

    assert record.circulating_supply == 1_000_000.0
    assert record.trust_tier is TrustTier.CACHE
    assert record.source == PRIMARY
    assert record.fetched_at == T0
    assert record.age_seconds == 5 * 3600
    assert record.is_stale is True
    assert record.is_degraded is True
    assert ALTERNATE not in horizon.calls


def test_outage_without_cache_falls_back_to_alternate(horizon):
    horizon.down.add(PRIMARY)

    record = _fetch()

    assert record.circulating_supply == 990_000.0
    assert record.trust_tier is TrustTier.ALTERNATE
    assert record.source == ALTERNATE
    assert record.is_stale is False
    assert record.is_degraded is True
    # Alternate data is never cached as if it came from the primary.
    assert _fetch(alternate=None).trust_tier is TrustTier.UNAVAILABLE


def test_outage_with_cache_past_stale_window_uses_alternate(horizon):
    _fetch()
    horizon.down.add(PRIMARY)

    record = _fetch(now=T0 + timedelta(days=8))

    assert record.trust_tier is TrustTier.ALTERNATE


def test_everything_down_is_unavailable(horizon):
    horizon.down.update({PRIMARY, ALTERNATE})

    record = _fetch()

    assert record.circulating_supply is None
    assert record.trust_tier is TrustTier.UNAVAILABLE
    assert record.fetched_at is None
    assert record.is_degraded is True


def test_primary_answering_without_supply_does_not_fall_back(monkeypatch):
    horizon = FakeHorizon(monkeypatch, {ALTERNATE: 990_000.0})

    record = _fetch()

    assert record.circulating_supply is None
    assert record.trust_tier is TrustTier.PRIMARY
    assert horizon.calls == [PRIMARY]


# ---------------------------------------------------------------------------
# Cache storage
# ---------------------------------------------------------------------------


def test_redis_entry_carries_provenance_and_outlives_ttl(horizon):
    redis = FakeRedis()
    _fetch(redis)

    assert json.loads(redis.store[KEY]) == {
        "supply": 1_000_000.0,
        "fetched_at": T0.isoformat(),
        "source": PRIMARY,
    }
    assert redis.ttls[KEY] == amf._MAX_STALE_SECONDS

    # A different process (empty local cache) serves the stale Redis entry.
    amf._local_cache.clear()
    horizon.down.add(PRIMARY)
    record = _fetch(redis, now=T0 + timedelta(hours=3))
    assert record.trust_tier is TrustTier.CACHE
    assert record.age_seconds == 3 * 3600


def test_legacy_bare_float_redis_entry_is_ignored(horizon):
    redis = FakeRedis()
    redis.store[KEY] = "123.0"

    record = _fetch(redis)

    assert record.trust_tier is TrustTier.PRIMARY
    assert record.circulating_supply == 1_000_000.0


def test_get_asset_supply_returns_just_the_number(horizon):
    assert get_asset_supply(CODE, ISSUER, PRIMARY) == 1_000_000.0


def test_record_to_dict_exposes_provenance(horizon):
    horizon.down.add(PRIMARY)

    assert _fetch().to_dict() == {
        "asset_code": CODE,
        "asset_issuer": ISSUER,
        "circulating_supply": 990_000.0,
        "trust_tier": "alternate",
        "source": ALTERNATE,
        "fetched_at": T0.isoformat(),
        "age_seconds": 0.0,
        "is_stale": False,
    }


# ---------------------------------------------------------------------------
# Downstream consumers surface trust tier / staleness
# ---------------------------------------------------------------------------


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": ["t1", "t2"],
            "id": ["t1", "t2"],
            "ledger": [1, 2],
            "base_account": [WALLET] * 2,
            "counter_account": ["GOTHER"] * 2,
            "base_amount": [10.0, 20.0],
            "counter_amount": [10.0, 20.0],
            "amount": [100.0, 400.0],
            "price": [1.0, 1.0],
            "ledger_close_time": [T0 - timedelta(minutes=10), T0 - timedelta(hours=3)],
        }
    )


def test_forensic_report_surfaces_degraded_asset_metadata(horizon):
    _fetch()
    horizon.down.add(PRIMARY)
    metadata = _fetch(now=T0 + timedelta(hours=5))

    report = ForensicReportGenerator().generate(
        WALLET, _trades(), risk_score_dict={"score": 55}, asset_metadata=metadata
    )

    assert report.to_dict()["asset_metadata"]["trust_tier"] == "cache"
    assert report.to_dict()["asset_metadata"]["is_stale"] is True
    assert report.asset_metadata_degraded is True
    assert report.verify_integrity()
    markdown = report.to_markdown()
    assert "## Asset Metadata Provenance" in markdown
    assert "not fresh primary-source data" in markdown
    assert "| Trust Tier | `cache` |" in markdown


def test_forensic_report_with_primary_metadata_has_no_warning(horizon):
    report = ForensicReportGenerator().generate(
        WALLET, _trades(), risk_score_dict={"score": 55}, asset_metadata=_fetch()
    )

    markdown = report.to_markdown()
    assert "| Trust Tier | `primary` |" in markdown
    assert "not fresh primary-source data" not in markdown


def test_forensic_report_without_asset_metadata_is_unchanged():
    report = ForensicReportGenerator().generate(WALLET, _trades(), risk_score_dict={"score": 5})

    assert "asset_metadata" not in report.to_dict()
    assert "Asset Metadata Provenance" not in report.to_markdown()


def test_feature_pipeline_propagates_trust_tier(horizon, monkeypatch):
    monkeypatch.setattr(feature_pipeline, "build_feature_vector", lambda *a, **k: {})
    horizon.down.add(PRIMARY)
    metadata = _fetch()

    features = feature_pipeline.build_extended_feature_vector(
        WALLET, _trades(), now=T0, asset_metadata=metadata
    )

    assert features["asset_metadata_trust_tier"] == "alternate"
    assert features["asset_metadata_is_stale"] is False
    assert features["asset_metadata_age_seconds"] == 0.0
    assert features["token_velocity_1h"] == pytest.approx(100.0 / 990_000.0)
