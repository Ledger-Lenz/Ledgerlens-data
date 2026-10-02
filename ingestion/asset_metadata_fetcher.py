"""Asset metadata fetcher — circulating supply from Stellar Horizon (issues #292, #917).

Fetches asset circulating supply from the Horizon /assets endpoint and returns
it as an :class:`AssetMetadataRecord` that always states where the value came
from (``trust_tier``) and how old it is (``fetched_at`` / ``age_seconds`` /
``is_stale``), so consumers never mistake stale or lower-trust data for current.

Fallback chain (see ``docs/asset_metadata_trust_tiers.md``)
------------------------------------------------------------
1. Fresh cache hit (primary data younger than the 1-hour TTL) -> ``primary``.
2. Primary Horizon source                                    -> ``primary``.
3. Primary unavailable: cached primary data up to 7 days old -> ``cache``.
4. Primary unavailable, no usable cache: alternate source   -> ``alternate``.
5. Nothing available                                         -> ``unavailable``.

Cache entries live in Redis when available (plus an in-process fallback) and
are kept for the full stale window; freshness is judged from ``fetched_at``.

Public API
----------
get_asset_metadata(asset_code, asset_issuer, horizon_url, redis_client,
                   alternate_horizon_url) -> AssetMetadataRecord
get_asset_supply(asset_code, asset_issuer, horizon_url, redis_client) -> float | None
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ingestion.exceptions import SourceUnavailableError

logger = logging.getLogger(__name__)

_SUPPLY_CACHE_TTL_SECONDS = 3_600
# How long cached primary data may still be served while the primary is down.
_MAX_STALE_SECONDS = 7 * 24 * 3_600
# In-process fallback when Redis is unavailable
_local_cache: dict[str, dict[str, Any]] = {}


class TrustTier(StrEnum):
    """How much a metadata value can be trusted, highest first."""

    PRIMARY = "primary"
    CACHE = "cache"
    ALTERNATE = "alternate"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AssetMetadataRecord:
    """Asset metadata plus the provenance consumers must surface."""

    asset_code: str
    asset_issuer: str
    circulating_supply: float | None
    trust_tier: TrustTier
    source: str | None
    fetched_at: datetime | None
    age_seconds: float | None
    is_stale: bool

    @property
    def is_degraded(self) -> bool:
        """True when the value is not fresh data from the primary source."""
        return self.trust_tier is not TrustTier.PRIMARY or self.is_stale

    def provenance(self) -> dict[str, Any]:
        """Trust tier and staleness, for embedding in downstream features/reports."""
        return {
            "trust_tier": self.trust_tier.value,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "age_seconds": self.age_seconds,
            "is_stale": self.is_stale,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_code": self.asset_code,
            "asset_issuer": self.asset_issuer,
            "circulating_supply": self.circulating_supply,
            **self.provenance(),
        }


def get_asset_metadata(
    asset_code: str,
    asset_issuer: str,
    horizon_url: str,
    redis_client=None,
    alternate_horizon_url: str | None = None,
    now: datetime | None = None,
) -> AssetMetadataRecord:
    """Return asset metadata labelled with its trust tier and staleness.

    Args:
        asset_code: Stellar asset code (e.g. "USDC").
        asset_issuer: Stellar account ID of the asset issuer.
        horizon_url: Primary Horizon base URL.
        redis_client: Optional ``redis.Redis`` instance for distributed cache.
        alternate_horizon_url: Optional lower-trust Horizon-compatible source,
            consulted only when the primary is unavailable and no cached value
            is usable.
        now: Reference time (for tests and backtesting). Defaults to UTC now.
    """
    cache_key = f"ledgerlens:asset_supply:{asset_code}:{asset_issuer}"
    now = now or datetime.now(UTC)

    def record(tier: TrustTier, entry: dict[str, Any] | None) -> AssetMetadataRecord:
        if entry is None:
            return AssetMetadataRecord(
                asset_code, asset_issuer, None, tier, None, None, None, is_stale=False
            )
        age = _age(entry, now)
        return AssetMetadataRecord(
            asset_code,
            asset_issuer,
            entry["supply"],
            tier,
            entry["source"],
            datetime.fromisoformat(entry["fetched_at"]),
            age,
            is_stale=age >= _SUPPLY_CACHE_TTL_SECONDS,
        )

    cached = _read_cache(cache_key, redis_client)
    if cached is not None and _age(cached, now) < _SUPPLY_CACHE_TTL_SECONDS:
        return record(TrustTier.PRIMARY, cached)

    try:
        supply = _fetch_from_horizon(asset_code, asset_issuer, horizon_url)
    except SourceUnavailableError as exc:
        logger.warning("Primary asset metadata source unavailable: %s", exc)
    else:
        entry = {"supply": supply, "fetched_at": now.isoformat(), "source": horizon_url}
        _write_cache(cache_key, entry, redis_client)
        return record(TrustTier.PRIMARY, entry)

    if cached is not None and _age(cached, now) < _MAX_STALE_SECONDS:
        return record(TrustTier.CACHE, cached)

    if alternate_horizon_url:
        try:
            supply = _fetch_from_horizon(asset_code, asset_issuer, alternate_horizon_url)
        except SourceUnavailableError as exc:
            logger.warning("Alternate asset metadata source unavailable: %s", exc)
        else:
            # Not cached: the cache only ever holds primary-source data.
            entry = {
                "supply": supply,
                "fetched_at": now.isoformat(),
                "source": alternate_horizon_url,
            }
            return record(TrustTier.ALTERNATE, entry)

    return record(TrustTier.UNAVAILABLE, None)


def get_asset_supply(
    asset_code: str,
    asset_issuer: str,
    horizon_url: str,
    redis_client=None,
) -> float | None:
    """Return circulating supply for an asset, or ``None`` if unavailable.

    Thin wrapper over :func:`get_asset_metadata` for callers that only need
    the number; use :func:`get_asset_metadata` wherever the trust tier and
    staleness should be surfaced.
    """
    metadata = get_asset_metadata(asset_code, asset_issuer, horizon_url, redis_client)
    return metadata.circulating_supply


def _age(entry: dict[str, Any], now: datetime) -> float:
    return (now - datetime.fromisoformat(entry["fetched_at"])).total_seconds()


def _read_cache(cache_key: str, redis_client) -> dict[str, Any] | None:
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached is not None:
                entry = json.loads(cached)
                if isinstance(entry, dict) and {"supply", "fetched_at", "source"} <= entry.keys():
                    return entry
        except Exception as exc:  # noqa: BLE001
            # Broad catch justified: Redis read can fail on network/auth/timeout issues,
            # and entries written before issue #917 are bare floats rather than JSON
            # objects. Fall through to the local cache.
            logger.warning("Redis supply cache read failed: %s", exc)
    return _local_cache.get(cache_key)


def _write_cache(cache_key: str, entry: dict[str, Any], redis_client) -> None:
    _local_cache[cache_key] = entry
    if redis_client is not None:
        try:
            # Keep entries for the whole stale window; freshness comes from fetched_at.
            redis_client.setex(cache_key, _MAX_STALE_SECONDS, json.dumps(entry))
        except Exception as exc:  # noqa: BLE001
            # Broad catch justified: Redis write can fail on network/auth/memory/timeout.
            # The local cache still holds the entry.
            logger.warning("Redis supply cache write failed: %s", exc)


def _fetch_from_horizon(
    asset_code: str,
    asset_issuer: str,
    horizon_url: str,
) -> float | None:
    """Fetch circulating supply from a Horizon-compatible source.

    Returns ``None`` when the source answers but has no positive supply for
    the asset.

    Raises:
        SourceUnavailableError: The source could not be reached or returned an
            unusable response.
    """
    params = f"asset_code={asset_code}&asset_issuer={asset_issuer}&limit=1"
    url = f"{horizon_url.rstrip('/')}/assets?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        records = data.get("_embedded", {}).get("records", [])
        if not records:
            return None
        supply = float(records[0].get("amount", 0))
    except Exception as exc:  # noqa: BLE001
        # Broad catch justified: network errors, malformed JSON, an unexpected
        # response shape or a bad amount all mean this source is unusable right now.
        raise SourceUnavailableError(
            f"Failed to fetch supply for {asset_code}:{asset_issuer} from {horizon_url}",
            source="asset_metadata_fetcher._fetch_from_horizon",
            reason=str(exc),
        ) from exc
    return supply if supply > 0 else None
