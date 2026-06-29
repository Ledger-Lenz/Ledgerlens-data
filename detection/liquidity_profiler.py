"""Per-asset liquidity regime profiler for calibrated Benford scoring.


Clusters assets into liquidity regimes using k-means on four observable
features, then computes an empirical Benford baseline for each regime from
known-clean trade data.  All downstream Benford metrics are expressed as
deviations from the regime baseline rather than from the theoretical
Benford distribution, reducing false positives on legitimate market-makers
whose structural behaviour mimics wash-trader digit patterns.

References
----------
Nigrini, "Benford's Law" (2012), Chapter 5 — baseline calibration approach.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from detection.benford_engine import (
    BENFORD_EXPECTED,
    leading_digits,
    mad_score,
    observed_distribution,
)

_FALLBACK_DIST: dict[int, float] = dict(BENFORD_EXPECTED)
_N_REGIME_FEATURES = 4  # trades_per_hour, amount_cv, spread_bps, n_counterparties

_BUILD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "build_config.json")


class AssetLiquidityProfiler:
    """Cluster assets into liquidity regimes and calibrate Benford baselines.

    Regime features (k=4 k-means clusters on):
      - trades_per_hour (log-scaled)
      - amount_coefficient_of_variation
      - median_spread_bps
      - unique_counterparty_count

    After calling ``fit``, use ``calibrated_chi_square`` / ``calibrated_mad``
    in place of the raw Benford statistics.  Assets not seen during fit fall
    back to the theoretical Benford distribution.
    """

    def __init__(self, n_clusters: int = 4) -> None:
        self.n_clusters = n_clusters
        self._fitted = False
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._cluster_centers: np.ndarray | None = None
        self._asset_regime: dict[str, int] = {}
        self._regime_distributions: dict[int, dict[int, float]] = {}
        self._regime_baseline_mad: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, asset_histories: dict[str, pd.DataFrame]) -> None:
        """Fit regime clusters on known-clean trade data.

        Parameters
        ----------
        asset_histories:
            Mapping from asset identifier (e.g. ``"XLM:native"``) to a
            DataFrame of that asset's trade records.  Each DataFrame must
            have at minimum an ``amount`` column; ``ledger_close_time``,
            ``price``, ``base_account``, and ``counter_account`` are used
            when available.
        """
        from sklearn.cluster import KMeans

        feature_rows: list[list[float]] = []
        seen_assets: list[str] = []
        for asset, df in asset_histories.items():
            row = self._compute_asset_features(df)
            if row is not None:
                feature_rows.append(row)
                seen_assets.append(asset)

        if not feature_rows:
            return

        X = np.array(feature_rows, dtype=float)
        X[:, 0] = np.log1p(X[:, 0])  # log-scale trades_per_hour

        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-9
        X_scaled = (X - self._feature_mean) / self._feature_std

        n_actual_clusters = min(self.n_clusters, len(seen_assets))
        km = KMeans(n_clusters=n_actual_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        self._cluster_centers = km.cluster_centers_

        for asset, label in zip(seen_assets, labels, strict=True):
            self._asset_regime[asset] = int(label)

        # Build per-regime empirical distributions from all clean trades
        regime_amounts: dict[int, list[pd.Series]] = {i: [] for i in range(n_actual_clusters)}
        for asset, label in zip(seen_assets, labels, strict=True):
            df = asset_histories[asset]
            if "amount" in df.columns:
                regime_amounts[int(label)].append(df["amount"])

        for regime_id, amount_list in regime_amounts.items():
            if amount_list:
                combined = pd.concat(amount_list, ignore_index=True)
                self._regime_distributions[regime_id] = observed_distribution(combined)
                self._regime_baseline_mad[regime_id] = mad_score(combined)
            else:
                self._regime_distributions[regime_id] = dict(BENFORD_EXPECTED)
                self._regime_baseline_mad[regime_id] = 0.0

        self._fitted = True

    def _compute_asset_features(self, df: pd.DataFrame) -> list[float] | None:
        """Compute the four regime-clustering features for one asset."""
        if df.empty or "amount" not in df.columns:
            return None

        # 1. trades_per_hour
        if "ledger_close_time" in df.columns:
            ts = pd.to_datetime(df["ledger_close_time"])
            span_hours = (ts.max() - ts.min()).total_seconds() / 3600
            trades_per_hour = len(df) / max(span_hours, 1.0)
        else:
            trades_per_hour = float(len(df))

        # 2. amount coefficient of variation
        amounts = df["amount"][df["amount"] > 0]
        if len(amounts) > 1 and amounts.mean() > 0:
            amount_cv = float(amounts.std() / amounts.mean())
        else:
            amount_cv = 0.0

        # 3. median spread bps (price volatility proxy)
        if "price" in df.columns:
            prices = df["price"][df["price"] > 0]
            if len(prices) > 1 and prices.mean() > 0:
                spread_bps = float(prices.std() / prices.mean() * 10_000)
            else:
                spread_bps = 0.0
        else:
            spread_bps = 0.0

        # 4. unique counterparty count
        for col in ("counter_account", "base_account"):
            if col in df.columns:
                n_counterparties = float(df[col].nunique())
                break
        else:
            n_counterparties = 0.0

        return [trades_per_hour, amount_cv, spread_bps, n_counterparties]

    # ------------------------------------------------------------------
    # Regime lookup helpers
    # ------------------------------------------------------------------

    def get_regime_id(self, asset: str) -> int:
        """Return the regime cluster ID for an asset (-1 if not seen during fit)."""
        if not self._fitted:
            return -1
        return self._asset_regime.get(asset, -1)

    def get_baseline_distribution(self, asset: str) -> dict[int, float]:
        """Return the empirical digit distribution for the asset's regime.

        Falls back to the theoretical Benford distribution for unknown assets.
        """
        regime_id = self._asset_regime.get(asset, -1)
        return self._regime_distributions.get(regime_id, _FALLBACK_DIST)

    def get_baseline_mad(self, asset: str) -> float:
        """Return the regime's expected MAD (used for explainability features)."""
        regime_id = self._asset_regime.get(asset, -1)
        return self._regime_baseline_mad.get(regime_id, 0.0)

    # ------------------------------------------------------------------
    # Calibrated metrics
    # ------------------------------------------------------------------

    def calibrated_chi_square(self, amounts: pd.Series, asset: str) -> float:
        """Chi-square of `amounts` against the asset's regime baseline.

        Returns 0.0 when the observed distribution exactly matches the baseline
        (identical to how standard chi-square returns 0 vs. the theoretical
        distribution when the sample perfectly conforms).
        """
        baseline = self.get_baseline_distribution(asset)
        digits = leading_digits(amounts)
        n = len(digits)
        if n == 0:
            return 0.0

        observed_counts = digits.value_counts()
        chi_sq = 0.0
        for d in range(1, 10):
            expected_count = baseline.get(d, 0.0) * n
            observed_count = float(observed_counts.get(d, 0))
            if expected_count > 0:
                chi_sq += (observed_count - expected_count) ** 2 / expected_count

        return float(chi_sq)

    def calibrated_mad(self, amounts: pd.Series, asset: str) -> float:
        """MAD of `amounts` against the asset's regime baseline distribution."""
        positive = amounts[amounts > 0] if not amounts.empty else amounts
        if positive.empty:
            return 0.0
        baseline = self.get_baseline_distribution(asset)
        observed = observed_distribution(amounts)
        deviations = [abs(observed[d] - baseline.get(d, 0.0)) for d in range(1, 10)]
        return float(sum(deviations) / len(deviations))


# ---------------------------------------------------------------------------
# Thin-market wash trade detection (issue #273)
# ---------------------------------------------------------------------------

_THIN_MARKET_DEFAULTS = {
    "max_unique_traders_7d": 50,
    "min_liquidity_depth_usd": 1000.0,
    "min_amm_tvl_usd": 5000.0,
    "benford_chi_square_threshold_multiplier": 3.0,
    "alert_threshold": 60,
    "classification_cache_seconds": 3600,
}


def _load_thin_market_config() -> dict:
    try:
        with open(_BUILD_CONFIG_PATH) as f:
            cfg = json.load(f)
        raw = cfg.get("thin_market", {})
        merged = dict(_THIN_MARKET_DEFAULTS)
        merged.update(raw)
        return merged
    except Exception:
        return dict(_THIN_MARKET_DEFAULTS)


def _validate_thin_market_config(cfg: dict) -> None:
    """Raise ValueError on startup if any threshold is not a positive number."""
    numeric_keys = [
        "max_unique_traders_7d",
        "min_liquidity_depth_usd",
        "min_amm_tvl_usd",
        "benford_chi_square_threshold_multiplier",
        "alert_threshold",
        "classification_cache_seconds",
    ]
    for key in numeric_keys:
        val = cfg.get(key)
        if val is None or not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(
                f"thin_market config '{key}' must be a positive number, got: {val!r}"
            )


@dataclass
class ThinMarketClassification:
    """Result of a thin-market classification for one asset pair."""

    asset_pair: str
    is_thin: bool
    unique_traders_7d: int
    liquidity_depth_usd: float
    amm_tvl_usd: float
    reason: str


class ThinMarketDetector:
    """Detect thin-market conditions and emit a calibrated wash-risk signal.

    A pair is classified as thin market when ANY of the following holds:
      - Fewer than ``max_unique_traders_7d`` unique traders in the last 7 days
      - Liquidity depth below ``min_liquidity_depth_usd`` (USD equivalent)
      - AMM pool TVL below ``min_amm_tvl_usd``

    For thin-market pairs a calibrated ``thin_market_wash_risk`` composite
    score is computed (0–100).  For liquid pairs the feature is ``float('nan')``
    to avoid conflating thin and liquid market signals.

    Classifications are cached for ``classification_cache_seconds`` (default
    1 hour) to avoid per-request recomputation.

    Security: thresholds are validated on startup; invalid config raises
    ``ValueError`` immediately, preventing silent misconfiguration.
    """

    def __init__(self) -> None:
        self._cfg = _load_thin_market_config()
        _validate_thin_market_config(self._cfg)

        self._cache: dict[str, tuple[float, ThinMarketClassification]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        asset_pair: str,
        trades_7d: pd.DataFrame,
        liquidity_depth_usd: float = 0.0,
        amm_tvl_usd: float = 0.0,
    ) -> ThinMarketClassification:
        """Classify *asset_pair* as thin or liquid market.

        Results are cached for ``classification_cache_seconds``.

        Parameters
        ----------
        asset_pair:
            Canonical ``CODE:ISSUER/CODE:ISSUER`` pair identifier.
        trades_7d:
            DataFrame of trades in the last 7 days.  Must contain at least
            one of ``base_account`` or ``counter_account``.
        liquidity_depth_usd:
            Current order-book liquidity depth in USD equivalent.
        amm_tvl_usd:
            Current AMM pool TVL in USD equivalent.
        """
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(asset_pair)
            if cached and (now - cached[0]) < self._cfg["classification_cache_seconds"]:
                return cached[1]

        result = self._classify_uncached(asset_pair, trades_7d, liquidity_depth_usd, amm_tvl_usd)
        with self._lock:
            self._cache[asset_pair] = (now, result)
        return result

    def _classify_uncached(
        self,
        asset_pair: str,
        trades_7d: pd.DataFrame,
        liquidity_depth_usd: float,
        amm_tvl_usd: float,
    ) -> ThinMarketClassification:
        unique_traders = self._count_unique_traders(trades_7d)
        reasons = []

        if unique_traders < self._cfg["max_unique_traders_7d"]:
            reasons.append(f"only {unique_traders} unique traders in 7d")
        if liquidity_depth_usd < self._cfg["min_liquidity_depth_usd"]:
            reasons.append(f"depth=${liquidity_depth_usd:.0f} < ${self._cfg['min_liquidity_depth_usd']:.0f}")
        if amm_tvl_usd < self._cfg["min_amm_tvl_usd"]:
            reasons.append(f"TVL=${amm_tvl_usd:.0f} < ${self._cfg['min_amm_tvl_usd']:.0f}")

        is_thin = bool(reasons)
        return ThinMarketClassification(
            asset_pair=asset_pair,
            is_thin=is_thin,
            unique_traders_7d=unique_traders,
            liquidity_depth_usd=float(liquidity_depth_usd),
            amm_tvl_usd=float(amm_tvl_usd),
            reason="; ".join(reasons) if reasons else "liquid market",
        )

    @staticmethod
    def _count_unique_traders(trades_7d: pd.DataFrame) -> int:
        accounts: set = set()
        for col in ("base_account", "counter_account"):
            if col in trades_7d.columns:
                accounts.update(trades_7d[col].dropna().unique())
        return len(accounts)

    def thin_market_wash_risk(
        self,
        asset_pair: str,
        trades_7d: pd.DataFrame,
        liquidity_depth_usd: float = 0.0,
        amm_tvl_usd: float = 0.0,
        round_trip_frequency: float = 0.0,
    ) -> float:
        """Compute the thin-market wash-risk composite score (0–100).

        Returns ``float('nan')`` for liquid pairs to avoid conflating thin and
        liquid market signals.  Only emit this feature for confirmed thin-market
        pairs.

        The composite score combines:
          - Inverse liquidity score (higher = lower depth / TVL)
          - Trader concentration (fewer unique traders → higher risk)
          - Round-trip frequency (calibrated to thin-market base rates)
        """
        classification = self.classify(
            asset_pair, trades_7d, liquidity_depth_usd, amm_tvl_usd
        )
        if not classification.is_thin:
            return float("nan")

        # Inverse-liquidity component (0–1, higher = lower liquidity)
        depth_score = 1.0 - min(
            liquidity_depth_usd / self._cfg["min_liquidity_depth_usd"], 1.0
        )
        tvl_score = 1.0 - min(amm_tvl_usd / self._cfg["min_amm_tvl_usd"], 1.0)
        liquidity_component = 0.5 * depth_score + 0.5 * tvl_score

        # Trader concentration (0–1, higher = fewer traders)
        trader_component = 1.0 - min(
            classification.unique_traders_7d / self._cfg["max_unique_traders_7d"], 1.0
        )

        # Round-trip frequency (thin-market calibrated: already suspicious at lower rates)
        rt_component = min(round_trip_frequency * 2.0, 1.0)

        raw = (
            liquidity_component * 0.4
            + trader_component * 0.35
            + rt_component * 0.25
        )
        return float(round(raw * 100, 2))

    def check_and_dispatch_alert(
        self,
        asset_pair: str,
        trades_7d: pd.DataFrame,
        liquidity_depth_usd: float = 0.0,
        amm_tvl_usd: float = 0.0,
        round_trip_frequency: float = 0.0,
        dispatcher=None,
    ) -> float | None:
        """Compute thin-market wash risk and dispatch alert if threshold exceeded.

        Returns the composite score if the pair is thin-market, else None.

        Parameters
        ----------
        dispatcher:
            Optional :class:`streaming.alert_dispatcher.AlertDispatcher`.
            When ``None``, alert is logged to stderr only.
        """
        score = self.thin_market_wash_risk(
            asset_pair, trades_7d, liquidity_depth_usd, amm_tvl_usd, round_trip_frequency
        )
        if score is None or (isinstance(score, float) and score != score):  # NaN check
            return None

        if score >= self._cfg["alert_threshold"]:
            alert = {
                "type": "thin_market_wash_risk",
                "asset_pair": asset_pair,
                "score": score,
                "threshold": self._cfg["alert_threshold"],
            }
            if dispatcher is not None:
                try:
                    dispatcher.dispatch(asset_pair, round(score), alert)
                except Exception:
                    pass
            else:
                import sys
                print(f"THIN_MARKET_ALERT: {alert}", file=sys.stderr)

        return score
