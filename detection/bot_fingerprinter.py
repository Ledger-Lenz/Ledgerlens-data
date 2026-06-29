"""Bot detection fingerprinting from Stellar Horizon event patterns.

Wash trade bots have distinctive Horizon event fingerprints:
- Trust line creation latency: created for new asset pairs within seconds
- Trading cadence: mechanically regular inter-trade intervals (low CV)
- Account management: operations cluster at predictable times (low entropy)

This module extracts these patterns from raw Horizon effects and trade history.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TypedDict

import numpy as np
import pandas as pd

from ingestion.data_models import BotFingerprint


class Effect(TypedDict, total=False):
    """Horizon effect record (partial typing for our use case)."""

    id: str
    type: str
    created_at: str
    account: str
    timestamp: str


MIN_TRADES_FOR_INTERVAL_CV = 5
STELLAR_GENESIS_TIMESTAMP = datetime(2015, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def extract_bot_fingerprint(
    account_id: str,
    effects: list[Effect] | None = None,
    trades_df: pd.DataFrame | None = None,
    account_created_at: datetime | None = None,
) -> BotFingerprint:
    """Extract bot detection fingerprint from Horizon events and trades.

    Args:
        account_id: Stellar account ID.
        effects: List of Horizon effect dicts with 'type', 'created_at', 'account'.
            Should include trust_line_created and manage_offer effects.
        trades_df: DataFrame of trades with 'ledger_close_time' and 'base_account'/'counter_account'.
        account_created_at: Account creation timestamp (required for latency calculation).

    Returns:
        BotFingerprint with three scores:
        - trust_line_creation_latency_seconds: Time from account creation to first trust line
        - inter_trade_interval_cv: Coefficient of variation of inter-trade intervals
        - account_management_cluster_score: Entropy of operation type distribution
    """
    effects = effects or []
    trades_df = trades_df or pd.DataFrame()

    fingerprint = BotFingerprint(account_id=account_id)

    # 1. Trust line creation latency
    if account_created_at:
        latency = _compute_trust_line_latency(account_created_at, effects)
        fingerprint.trust_line_creation_latency_seconds = latency

    # 2. Inter-trade interval coefficient of variation
    if not trades_df.empty:
        interval_cv = _compute_inter_trade_interval_cv(account_id, trades_df)
        fingerprint.inter_trade_interval_cv = interval_cv
        fingerprint.is_valid = interval_cv is not None

    # 3. Account management cluster score (entropy)
    if effects:
        cluster_score = _compute_account_management_entropy(effects)
        fingerprint.account_management_cluster_score = cluster_score

    # Set confidence based on data availability
    fingerprint.confidence = _compute_confidence(
        fingerprint.inter_trade_interval_cv,
        fingerprint.trust_line_creation_latency_seconds,
    )

    return fingerprint


def _compute_trust_line_latency(account_created_at: datetime, effects: list[Effect]) -> float | None:
    """Compute latency from account creation to first trust line creation.

    Returns:
        Seconds as float, or None if no trust line found or invalid timestamp.
    """
    if not effects:
        return None

    try:
        created_at = account_created_at
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        # Validate timestamp is plausible
        if not _is_plausible_timestamp(created_at):
            return None

        # Find first trust_line_created effect
        for effect in effects:
            if effect.get("type") == "trust_line_created":
                effect_time_str = effect.get("created_at")
                if effect_time_str:
                    try:
                        effect_time = datetime.fromisoformat(
                            effect_time_str.replace("Z", "+00:00")
                        )
                        if _is_plausible_timestamp(effect_time):
                            latency = (effect_time - created_at).total_seconds()
                            # Only return if plausible (non-negative and < 1 year)
                            if 0 <= latency < (365 * 24 * 3600):
                                return float(latency)
                    except (ValueError, TypeError):
                        continue

    except Exception:
        pass

    return None


def _compute_inter_trade_interval_cv(account_id: str, trades_df: pd.DataFrame) -> float | None:
    """Compute coefficient of variation of inter-trade intervals.

    Uses population standard deviation (not sample) to avoid division instability
    on small samples. Returns None if fewer than MIN_TRADES_FOR_INTERVAL_CV trades.

    Args:
        account_id: Account to analyze.
        trades_df: DataFrame with 'base_account', 'counter_account', 'ledger_close_time'.

    Returns:
        CV as float in [0, inf), or None if insufficient trades or invalid data.
    """
    if trades_df.empty:
        return None

    # Filter trades involving the account
    mask = (trades_df["base_account"] == account_id) | (trades_df["counter_account"] == account_id)
    wallet_trades = trades_df[mask].copy()

    if len(wallet_trades) < MIN_TRADES_FOR_INTERVAL_CV:
        return None

    try:
        # Convert timestamps and sort
        timestamps = pd.to_datetime(wallet_trades["ledger_close_time"], utc=True)
        timestamps = timestamps.sort_values().reset_index(drop=True)

        if len(timestamps) < 2:
            return None

        # Compute inter-trade intervals in seconds
        intervals = timestamps.diff().dt.total_seconds().dropna()

        if len(intervals) < 1 or (intervals == 0).all():
            return None

        # Use population standard deviation (ddof=0)
        mean_interval = float(intervals.mean())
        std_interval = float(intervals.std(ddof=0))  # Population std

        # Avoid division by zero
        if mean_interval == 0:
            return None

        cv = std_interval / mean_interval
        return float(cv)

    except Exception:
        return None


def _compute_account_management_entropy(effects: list[Effect]) -> float:
    """Compute entropy of operation type distribution from effects.

    Lower entropy indicates clustering (bot-like), higher entropy indicates
    diverse operations (human-like).

    Args:
        effects: List of Horizon effects.

    Returns:
        Shannon entropy in [0, log(n_types)].
    """
    if not effects:
        return 0.0

    # Count effect types (proxy for operation type distribution)
    type_counts: dict[str, int] = {}
    for effect in effects:
        effect_type = effect.get("type", "unknown")
        type_counts[effect_type] = type_counts.get(effect_type, 0) + 1

    total = sum(type_counts.values())
    if total == 0:
        return 0.0

    # Compute Shannon entropy: H = -sum(p_i * log(p_i))
    entropy = 0.0
    for count in type_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return float(entropy)


def _is_plausible_timestamp(ts: datetime) -> bool:
    """Validate that timestamp is plausible.

    Must be after Stellar genesis and before current time + 60 seconds
    (allowing some clock skew).
    """
    if not isinstance(ts, datetime):
        return False

    now = datetime.now(timezone.utc)
    future_bound = now.replace(microsecond=0) + pd.Timedelta(seconds=60)

    return STELLAR_GENESIS_TIMESTAMP <= ts <= future_bound


def _compute_confidence(interval_cv: float | None, latency: float | None) -> float:
    """Compute confidence score for fingerprint validity.

    Lower confidence when data is sparse or missing.

    Args:
        interval_cv: Inter-trade interval CV (None if insufficient trades).
        latency: Trust line creation latency (None if no trust line).

    Returns:
        Confidence in [0, 1].
    """
    if interval_cv is None and latency is None:
        return 0.2  # Very low confidence with no data

    if interval_cv is None or latency is None:
        return 0.6  # Moderate confidence with partial data

    return 1.0  # High confidence with both features


def extract_bot_fingerprints_batch(
    accounts_data: list[dict],
) -> list[BotFingerprint]:
    """Extract bot fingerprints for a batch of accounts.

    Args:
        accounts_data: List of dicts with keys:
            - account_id
            - account_created_at
            - effects (optional)
            - trades_df (optional)

    Returns:
        List of BotFingerprint objects.
    """
    results = []
    for account_data in accounts_data:
        try:
            fingerprint = extract_bot_fingerprint(
                account_id=account_data["account_id"],
                account_created_at=account_data.get("account_created_at"),
                effects=account_data.get("effects"),
                trades_df=account_data.get("trades_df"),
            )
            results.append(fingerprint)
        except Exception:
            # Return minimal fingerprint on error
            results.append(BotFingerprint(account_id=account_data.get("account_id", "unknown")))

    return results


def is_likely_bot(fingerprint: BotFingerprint, threshold: float = 0.7) -> bool:
    """Simple heuristic to classify account as likely bot.

    Combines the three fingerprint features:
    - Fast trust line creation (<60s) suggests pre-planned trading
    - Low interval CV (<0.1) suggests mechanical regularity
    - Low entropy (<1.5 bits) suggests clustered operations

    Args:
        fingerprint: BotFingerprint to classify.
        threshold: Classification threshold in [0, 1].

    Returns:
        True if account meets multiple bot criteria.
    """
    bot_scores = []

    # Fast trust line creation (< 60 seconds)
    if fingerprint.trust_line_creation_latency_seconds is not None:
        if fingerprint.trust_line_creation_latency_seconds < 60:
            bot_scores.append(0.9)
        elif fingerprint.trust_line_creation_latency_seconds < 600:
            bot_scores.append(0.6)

    # Low inter-trade interval CV (< 0.1 = very regular)
    if fingerprint.inter_trade_interval_cv is not None:
        if fingerprint.inter_trade_interval_cv < 0.1:
            bot_scores.append(0.95)
        elif fingerprint.inter_trade_interval_cv < 0.3:
            bot_scores.append(0.7)

    # Low operation entropy (< 1.5 bits = clustered)
    if fingerprint.account_management_cluster_score < 1.5:
        bot_scores.append(0.8)

    if not bot_scores:
        return False

    # Average bot score
    return float(np.mean(bot_scores)) > threshold
