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
