"""Wallet lifecycle feature engineering (issue #293).

Derives temporal behavioural signals from wallet age and trading activity
history. All timestamps are relative to a caller-supplied ``now`` parameter
so computations are reproducible during backtesting.

Public API
----------
compute_lifecycle_features(wallet_address, trades_df, account_created_at, now)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

_STELLAR_ACCOUNT_RE = re.compile(r"^G[A-Z2-7]{55}$")

# Cache: wallet_address -> (account_created_at, fetched_at)
_account_cache: dict[str, tuple[datetime | None, datetime]] = {}
_CACHE_TTL_SECONDS = 86_400  # 24 hours


def _validate_wallet(wallet_address: str) -> None:
    if not _STELLAR_ACCOUNT_RE.match(wallet_address):
        raise ValueError(
            f"Invalid Stellar account ID: {wallet_address!r}. "
            "Must start with 'G' and be 56 characters."
        )


def _fetch_account_created_at(wallet_address: str, horizon_url: str) -> datetime | None:
    """Fetch account creation timestamp from Horizon, with 24h TTL cache."""
    import urllib.request
    import json

    now_utc = datetime.now(timezone.utc)
    cached = _account_cache.get(wallet_address)
    if cached is not None:
        created_at, fetched_at = cached
        if (now_utc - fetched_at).total_seconds() < _CACHE_TTL_SECONDS:
            return created_at

    url = f"{horizon_url.rstrip('/')}/accounts/{wallet_address}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        raw = data.get("last_modified_time") or data.get("created_at")
        if raw:
            created_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            _account_cache[wallet_address] = (created_at, now_utc)
            return created_at
    except Exception as exc:
        logger.warning("Failed to fetch account creation time for %s: %s", wallet_address, exc)

    _account_cache[wallet_address] = (None, now_utc)
    return None


def compute_lifecycle_features(
    wallet_address: str,
    trades_df: pd.DataFrame,
    account_created_at: datetime | None,
    now: datetime | None = None,
) -> dict[str, float]:
    """Compute wallet lifecycle features.

    Args:
        wallet_address: Stellar account ID (validated before use).
        trades_df: DataFrame with a ``ledger_close_time`` column (UTC-aware or
            naive ISO-8601 strings). May be empty.
        account_created_at: UTC datetime of account creation from Horizon, or
            ``None`` if unavailable.
        now: Reference timestamp for reproducible backtesting. Defaults to
            ``datetime.now(UTC)`` when ``None``.

    Returns:
        Dict with keys:
            - ``wallet_age_days``
            - ``days_since_first_trade``
            - ``days_since_last_trade``
            - ``active_days_ratio``
            - ``burst_score``

        Age features are ``float('nan')`` when ``account_created_at`` is
        ``None``. ``burst_score`` is ``float('nan')`` when the 30-day average
        trade count is zero.
    """
    _validate_wallet(wallet_address)

    nan = float("nan")
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # --- age features (require account_created_at) ---
    if account_created_at is None:
        wallet_age_days = nan
        active_days_ratio = nan
    else:
        if account_created_at.tzinfo is None:
            account_created_at = account_created_at.replace(tzinfo=timezone.utc)
        wallet_age_days = max((now - account_created_at).total_seconds() / 86_400.0, 0.0)

        if not trades_df.empty:
            trade_times = pd.to_datetime(trades_df["ledger_close_time"], utc=True)
            active_days = trade_times.dt.normalize().nunique()
            age_days_floor = max(wallet_age_days, 1.0)
            active_days_ratio = float(active_days) / age_days_floor
        else:
            active_days_ratio = 0.0

    # --- trade recency features (do not require account_created_at) ---
    if trades_df.empty:
        days_since_first_trade = nan
        days_since_last_trade = nan
    else:
        trade_times = pd.to_datetime(trades_df["ledger_close_time"], utc=True)
        first_trade = trade_times.min()
        last_trade = trade_times.max()
        days_since_first_trade = max((now - first_trade).total_seconds() / 86_400.0, 0.0)
        days_since_last_trade = max((now - last_trade).total_seconds() / 86_400.0, 0.0)

    # --- burst score ---
    burst_score: float
    if trades_df.empty:
        burst_score = nan
    else:
        trade_times = pd.to_datetime(trades_df["ledger_close_time"], utc=True)
        window_24h = now - pd.Timedelta(hours=24)
        window_30d = now - pd.Timedelta(days=30)

        count_24h = int((trade_times >= window_24h).sum())
        trades_30d = trade_times[(trade_times >= window_30d) & (trade_times < window_24h)]
        # Average trades per day over the preceding 29 days (30d window minus last 24h)
        avg_per_day_30d = len(trades_30d) / 29.0
        if avg_per_day_30d == 0.0:
            burst_score = nan
        else:
            burst_score = count_24h / avg_per_day_30d

    return {
        "wallet_age_days": wallet_age_days,
        "days_since_first_trade": days_since_first_trade,
        "days_since_last_trade": days_since_last_trade,
        "active_days_ratio": active_days_ratio,
        "burst_score": burst_score,
    }
