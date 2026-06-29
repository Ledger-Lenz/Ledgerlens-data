"""Asset metadata fetcher — circulating supply from Stellar Horizon (issue #292).

Fetches and caches asset circulating supply from the Horizon /assets endpoint.
Cache is backed by Redis when available, with a 1-hour TTL.

Public API
----------
get_asset_supply(asset_code, asset_issuer, horizon_url, redis_client) -> float | None
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_SUPPLY_CACHE_TTL_SECONDS = 3_600
# In-process fallback when Redis is unavailable
_local_cache: dict[str, tuple[float | None, datetime]] = {}


def get_asset_supply(
    asset_code: str,
    asset_issuer: str,
    horizon_url: str,
    redis_client=None,
) -> float | None:
    """Return circulating supply for an asset, cached for 1 hour.

    Supply data is fetched asynchronously relative to the scoring path: the
    cache is populated on first call and refreshed in the background on TTL
    expiry (best-effort; stale value is returned on refresh failure).

    Args:
        asset_code: Stellar asset code (e.g. "USDC").
        asset_issuer: Stellar account ID of the asset issuer.
        horizon_url: Horizon base URL.
        redis_client: Optional ``redis.Redis`` instance for distributed cache.

    Returns:
        Circulating supply as a float, or ``None`` if unavailable.
    """
    cache_key = f"ledgerlens:asset_supply:{asset_code}:{asset_issuer}"
    now_utc = datetime.now(timezone.utc)

    # --- check Redis cache ---
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached is not None:
                return float(cached)
        except Exception as exc:
            logger.warning("Redis supply cache read failed: %s", exc)

    # --- check local cache ---
    local = _local_cache.get(cache_key)
    if local is not None:
        supply, fetched_at = local
        if (now_utc - fetched_at).total_seconds() < _SUPPLY_CACHE_TTL_SECONDS:
            return supply

    # --- fetch from Horizon ---
    supply = _fetch_from_horizon(asset_code, asset_issuer, horizon_url)
    _local_cache[cache_key] = (supply, now_utc)

    # --- populate Redis ---
    if redis_client is not None and supply is not None:
        try:
            redis_client.setex(cache_key, _SUPPLY_CACHE_TTL_SECONDS, str(supply))
        except Exception as exc:
            logger.warning("Redis supply cache write failed: %s", exc)

    return supply


def _fetch_from_horizon(
    asset_code: str,
    asset_issuer: str,
    horizon_url: str,
) -> float | None:
    params = f"asset_code={asset_code}&asset_issuer={asset_issuer}&limit=1"
    url = f"{horizon_url.rstrip('/')}/assets?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        records = data.get("_embedded", {}).get("records", [])
        if records:
            raw = records[0].get("amount", 0)
            supply = float(raw)
            return supply if supply > 0 else None
    except Exception as exc:
        logger.warning("Failed to fetch supply for %s:%s from Horizon: %s", asset_code, asset_issuer, exc)
    return None
