"""Feature pipeline integrating lifecycle and velocity features (issues #293, #292).

Wraps the existing ``detection.feature_engineering`` builders and appends the
new lifecycle and velocity feature groups so downstream ML code only needs to
call ``build_extended_feature_vector``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

from detection.feature_engineering import build_feature_vector
from features.ohlcv_features import compute_ohlcv_features
from features.velocity_features import compute_token_velocity
from features.wallet_lifecycle_features import compute_lifecycle_features

if TYPE_CHECKING:
    from ingestion.asset_metadata_fetcher import AssetMetadataRecord


def build_extended_feature_vector(
    wallet: str,
    wallet_trades: pd.DataFrame,
    account_created_at: datetime | None = None,
    asset_supply: float | None = None,
    now: datetime | None = None,
    asset_metadata: AssetMetadataRecord | None = None,
    **kwargs,
) -> dict:
    """Build a full feature row including lifecycle and velocity features.

    Args:
        wallet: Stellar account ID.
        wallet_trades: Trade DataFrame filtered to this wallet.
        account_created_at: UTC datetime of account creation (or ``None``).
        asset_supply: Circulating supply for velocity computation (or ``None``).
        now: Reference timestamp for reproducibility. Defaults to UTC now.
        asset_metadata: Asset metadata from
            ``ingestion.asset_metadata_fetcher.get_asset_metadata``. Supplies
            ``asset_supply`` when that is not given, and adds its trust tier
            and staleness as ``asset_metadata_*`` fields next to the velocity
            features derived from it.
        **kwargs: Forwarded verbatim to ``build_feature_vector`` (e.g.
            ``orderbook_events``, ``funding_graph``, ``all_pairs_df``).

    Returns:
        Feature dict containing all existing features plus lifecycle and
        velocity feature groups.
    """
    if now is None:
        now = datetime.now(UTC)

    features = build_feature_vector(wallet, wallet_trades, **kwargs)
    features.update(compute_lifecycle_features(wallet, wallet_trades, account_created_at, now=now))
    if asset_metadata is not None:
        if asset_supply is None:
            asset_supply = asset_metadata.circulating_supply
        features.update(
            asset_metadata_trust_tier=asset_metadata.trust_tier.value,
            asset_metadata_age_seconds=asset_metadata.age_seconds,
            asset_metadata_is_stale=asset_metadata.is_stale,
        )
    features.update(compute_token_velocity(wallet_trades, asset_supply, now=now))

    # OHLCV-derived candle features (pair-level microstructure signals)
    features.update(compute_ohlcv_features(wallet_trades))

    return features
