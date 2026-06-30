"""OHLCV-derived market microstructure features.

Builds candle aggregates from individual trade events at multiple resolutions
using vectorised pandas operations.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


_RESOLUTION_ALLOWLIST = ["1m", "5m", "15m", "1h", "4h"]


def _validate_resolutions(resolutions: Iterable[str]) -> list[str]:
    res = list(resolutions)
    invalid = [r for r in res if r not in _RESOLUTION_ALLOWLIST]
    if invalid:
        raise ValueError(
            "Invalid resolutions found: " + ", ".join(map(repr, invalid))
            + ". Allowed: "
            + ", ".join(map(repr, _RESOLUTION_ALLOWLIST))
        )
    return res


def compute_ohlcv_features(
    trades_df: pd.DataFrame, resolutions: list[str] | None = None
) -> dict[str, float]:
    """Compute OHLCV-derived candle features at each requested resolution.

    Expected trade columns:
      - ledger_close_time: timestamp of the trade (UTC-aware preferred)
      - price: trade price
      - amount: trade amount

    Returns:
      A flat dict containing per-resolution scalar features with the
      following keys:
        - price_range_ratio_{suffix}
        - vwap_deviation_{suffix}
        - candle_body_ratio_{suffix}
        - volume_spike_ratio_{suffix}

      Where suffix is one of the entries in the validated resolutions list
      (e.g., "1m").

    Sparse candles (fewer than 2 trades) yield NaN for candle-based features.
    """

    nan = float("nan")

    if resolutions is None:
        resolutions = ["1m", "5m", "1h"]

    resolutions = _validate_resolutions(resolutions)

    if trades_df is None or trades_df.empty:
        return {k: nan for k in _feature_keys_for_resolutions(resolutions)}

    required_cols = {"ledger_close_time", "price", "amount"}
    missing = required_cols - set(trades_df.columns)
    if missing:
        raise KeyError(f"Missing required trade columns: {sorted(missing)}")

    df = trades_df.copy()

    # Coerce & sort once; resample will be handled per-resolution.
    df["ledger_close_time"] = pd.to_datetime(df["ledger_close_time"], utc=True)
    df = df.sort_values("ledger_close_time")

    # Vectorised candle aggregates: resample and compute OHLCV in one go.
    out: dict[str, float] = {}

    for res in resolutions:
        # Window start labels; anchor to the trade timestamps.
        g = df.set_index("ledger_close_time").resample(res, label="left", closed="left")

        open_ = g["price"].first()
        high_ = g["price"].max()
        low_ = g["price"].min()
        close_ = g["price"].last()

        # Volume in this dataset is the traded amount.
        volume_ = g["amount"].sum()
        trade_count = g["price"].count()

        # VWAP: sum(price*amount)/sum(amount)
        pv = df["price"] * df["amount"]
        vwap_ = pv.groupby(pd.Grouper(freq=res, level=0)).sum() / volume_.replace(0.0, np.nan)

        # Features per candle.
        denom_range = (high_ - low_).replace(0.0, np.nan)
        price_range_ratio = (denom_range / open_.replace(0.0, np.nan))
        # candle_body_ratio ((close-open)/(high-low))
        candle_body_ratio = (close_ - open_) / denom_range
        # vwap_deviation (close - VWAP / VWAP)
        vwap_deviation = (close_ - vwap_) / vwap_.replace(0.0, np.nan)

        # volume_spike_ratio = candle volume / rolling(20)-avg volume (excluding current candle)
        rolling_avg_excl = (
            volume_
            .rolling(window=20, min_periods=1)
            .mean()
            .shift(1)
        )
        volume_spike_ratio = volume_ / rolling_avg_excl.replace(0.0, np.nan)

        # Sparse candle handling: fewer than 2 trades => NaN.
        sparse_mask = trade_count < 2
        for s in [price_range_ratio, candle_body_ratio]:
            s.loc[sparse_mask] = nan

        # vwap_deviation: with <2 trades, also NaN (standard).
        vwap_deviation.loc[sparse_mask] = nan

        # volume_spike_ratio for sparse candles: still definable, but spec says
        # "all candle features = NaN".
        volume_spike_ratio.loc[sparse_mask] = nan

        # The caller expects a single feature row; pick the latest candle.
        last_idx = volume_.last_valid_index()
        if last_idx is None:
            out.update({f"price_range_ratio_{res}": nan,
                        f"vwap_deviation_{res}": nan,
                        f"candle_body_ratio_{res}": nan,
                        f"volume_spike_ratio_{res}": nan})
            continue

        out[f"price_range_ratio_{res}"] = float(price_range_ratio.loc[last_idx])
        out[f"vwap_deviation_{res}"] = float(vwap_deviation.loc[last_idx])
        out[f"candle_body_ratio_{res}"] = float(candle_body_ratio.loc[last_idx])
        out[f"volume_spike_ratio_{res}"] = float(volume_spike_ratio.loc[last_idx])

    return out


def _feature_keys_for_resolutions(resolutions: list[str]) -> list[str]:
    keys: list[str] = []
    for r in resolutions:
        keys.extend(
            [
                f"price_range_ratio_{r}",
                f"vwap_deviation_{r}",
                f"candle_body_ratio_{r}",
                f"volume_spike_ratio_{r}",
            ]
        )
    return keys

