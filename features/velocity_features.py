"""Token velocity ratio features (issue #292).

Computes volume-to-circulating-supply ratios over 1h / 24h / 7d windows.
Asset supply is fetched from Stellar Horizon /assets and cached in Redis
(or an in-process dict when Redis is unavailable) with a 1-hour TTL.

Public API
----------
compute_token_velocity(trades_df, asset_supply, now)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

_VELOCITY_WINDOWS: dict[str, int] = {
    "token_velocity_1h": 1,
    "token_velocity_24h": 24,
    "token_velocity_7d": 168,
}

# In-process fallback cache: asset_code -> (supply, fetched_at)
_supply_cache: dict[str, tuple[float, datetime]] = {}
_SUPPLY_CACHE_TTL_SECONDS = 3_600  # 1 hour


def _validate_supply(supply: float | None) -> bool:
    """Return True iff supply is a positive finite float suitable for division."""
    if supply is None:
        return False
    try:
        f = float(supply)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0.0


def fetch_asset_supply(asset_code: str, asset_issuer: str, horizon_url: str) -> float | None:
    """Fetch circulating supply from Horizon /assets endpoint (cached 1h).

    This is intentionally synchronous; callers that need non-blocking
    behaviour should run it in a thread pool.
    """
    import json
    import urllib.request

    cache_key = f"{asset_code}:{asset_issuer}"
    now_utc = datetime.now(timezone.utc)
    cached = _supply_cache.get(cache_key)
    if cached is not None:
        supply, fetched_at = cached
        if (now_utc - fetched_at).total_seconds() < _SUPPLY_CACHE_TTL_SECONDS:
            return supply

    params = f"asset_code={asset_code}&asset_issuer={asset_issuer}&limit=1"
    url = f"{horizon_url.rstrip('/')}/assets?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        records = data.get("_embedded", {}).get("records", [])
        if records:
            raw = records[0].get("amount") or records[0].get("accounts", {}).get("authorized", 0)
            supply = float(raw)
            _supply_cache[cache_key] = (supply, now_utc)
            return supply
    except Exception as exc:
        logger.warning("Failed to fetch supply for %s: %s", cache_key, exc)

    _supply_cache[cache_key] = (0.0, now_utc)
    return None


def compute_token_velocity(
    trades_df: pd.DataFrame,
    asset_supply: float | None,
    now: datetime | None = None,
) -> dict[str, float]:
    """Compute token velocity ratio features.

    Args:
        trades_df: DataFrame with ``amount`` and ``ledger_close_time`` columns.
        asset_supply: Circulating supply of the asset. If ``None`` or ``<= 0``,
            all velocity features are set to ``NaN`` and a warning is logged.
        now: Reference timestamp for backtesting reproducibility. Defaults to
            ``datetime.now(UTC)``.

    Returns:
        Dict with keys ``token_velocity_1h``, ``token_velocity_24h``,
        ``token_velocity_7d``. Values are ``float('nan')`` when supply is
        unavailable or zero.
    """
    nan = float("nan")

    if not _validate_supply(asset_supply):
        logger.warning(
            "Asset supply unavailable or zero (%r); velocity features set to NaN.", asset_supply
        )
        return {k: nan for k in _VELOCITY_WINDOWS}

    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    supply = float(asset_supply)  # validated above
    result: dict[str, float] = {}

    if trades_df.empty:
        return {k: nan for k in _VELOCITY_WINDOWS}

    trade_times = pd.to_datetime(trades_df["ledger_close_time"], utc=True)
    amounts = trades_df["amount"].astype(float)

    for feature_name, hours in _VELOCITY_WINDOWS.items():
        cutoff = now - pd.Timedelta(hours=hours)
        mask = trade_times >= cutoff
        window_volume = float(amounts[mask].sum())
        result[feature_name] = window_volume / supply

    return result
