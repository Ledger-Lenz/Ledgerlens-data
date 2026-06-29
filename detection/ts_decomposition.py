"""Hierarchical time-series decomposition for Benford noise reduction.

Implements STL (Seasonal-Trend Decomposition via LOESS) on trade amount
time series to separate seasonal/trend components from anomaly residuals
before Benford scoring.  Residuals isolate genuine wash-trade signal from
cyclical market-maker behaviour that would otherwise inflate chi-square.

Reference: Cleveland et al., "STL: A Seasonal-Trend Decomposition Procedure
Based on Loess" (1990); statsmodels.tsa.seasonal.STL.
"""

import numpy as np
import pandas as pd

_MIN_OBS_FOR_STL = 48
_DEFAULT_PERIOD = 24  # 24-bin fallback (e.g. 24 minutes or 24 hours)


def to_amount_time_series(
    trades: pd.DataFrame,
    freq: str = "1min",
) -> pd.Series:
    """Convert a trade DataFrame to a time-indexed series of summed amounts.

    Amounts within each frequency bin are summed; missing bins are filled
    with 0.0.  Returns an empty Series when ``trades`` is empty or lacks
    ``ledger_close_time`` / ``amount`` columns.
    """
    if trades.empty or "ledger_close_time" not in trades.columns or "amount" not in trades.columns:
        return pd.Series(dtype=float)

    df = trades[["ledger_close_time", "amount"]].copy()
    df["_time"] = pd.to_datetime(df["ledger_close_time"], utc=True)
    df = df.set_index("_time").sort_index()
    return df["amount"].resample(freq).sum().fillna(0.0)


def detect_dominant_period(series: pd.Series) -> int | None:
    """Return the dominant periodic cycle length in bins via FFT periodogram.

    Analyses the power spectrum and returns the integer period (number of
    frequency bins) corresponding to the spectral peak.  Returns ``None``
    when the series is too short, flat, or shows no significant periodicity
    within a sensible range.
    """
    n = len(series)
    if n < 4:
        return None

    values = series.values.astype(float) - series.values.mean()
    if np.all(values == 0):
        return None

    fft_vals = np.fft.rfft(values)
    power = np.abs(fft_vals) ** 2
    power[0] = 0.0  # ignore DC component

    if power.max() == 0:
        return None

    freqs = np.fft.rfftfreq(n)
    dominant_idx = int(np.argmax(power))
    if dominant_idx == 0 or freqs[dominant_idx] == 0:
        return None

    period = int(round(1.0 / freqs[dominant_idx]))
    if period < 2 or period > n // 2:
        return None

    return period


def decompose_amounts(series: pd.Series, period: int | None = None):
    """Apply STL decomposition to an amount time series.

    ``period`` is the dominant cycle length in bins.  When *None*, it is
    auto-detected via :func:`detect_dominant_period` and defaults to
    ``_DEFAULT_PERIOD`` when detection fails.

    Returns a ``statsmodels`` ``DecomposeResult``-like object with
    ``trend``, ``seasonal``, and ``resid`` attributes.

    Raises ``ValueError`` when ``series`` has fewer than ``2 * period``
    observations (insufficient for STL).
    """
    from statsmodels.tsa.seasonal import STL

    if period is None:
        period = detect_dominant_period(series) or _DEFAULT_PERIOD

    min_required = 2 * period
    if len(series) < min_required:
        raise ValueError(
            f"Series has {len(series)} observations; need >= {min_required} for STL "
            f"with period={period}"
        )

    return STL(series, period=period, robust=True).fit()


def compute_volume_trend_slope(series: pd.Series) -> float:
    """Linear regression slope of volume over time (normalized by mean volume).
    Returns NaN for series with < 48 observations."""
    if len(series) < _MIN_OBS_FOR_STL:
        return float("nan")
    x = np.arange(len(series))
    slope = np.polyfit(x, series.values, 1)[0]
    return float(slope / (series.mean() + 1e-9))


def compute_volume_seasonality_strength(result) -> float:
    """Ratio of seasonal component variance to total variance (seasonal + residual).
    Returns NaN if decomposition result is None or insufficient data.
    High values (> 0.8) indicate mechanical/predictable trading."""
    if result is None:
        return float("nan")
    seasonal_var = np.var(result.seasonal)
    resid_var = np.var(result.resid)
    total = seasonal_var + resid_var
    return float(seasonal_var / total) if total > 0 else 0.0


def compute_volume_residual_anomaly(result) -> float:
    """Fraction of trade events with residual > 2 std devs above mean.
    Returns NaN if decomposition result is None."""
    if result is None:
        return float("nan")
    mean_r = np.mean(result.resid)
    std_r = np.std(result.resid)
    threshold = mean_r + 2 * std_r
    return float(np.mean(result.resid > threshold))


def compute_ts_features(wallet_trades: pd.DataFrame) -> dict:
    """Compute STL-based time-series features for a wallet.

    Converts trades to an hourly volume series, runs STL decomposition with
    period=24 (1-day seasonality), and returns trend slope, seasonality
    strength, and residual anomaly rate.

    Returns NaN for all features when fewer than 48 hourly observations are
    available.
    """
    nan_result = {
        "volume_trend_slope": float("nan"),
        "volume_seasonality_strength": float("nan"),
        "volume_residual_anomaly": float("nan"),
    }

    series = to_amount_time_series(wallet_trades, freq="1h")
    if len(series) < _MIN_OBS_FOR_STL:
        return nan_result

    try:
        result = decompose_amounts(series, period=_DEFAULT_PERIOD)
    except Exception:
        return nan_result

    return {
        "volume_trend_slope": compute_volume_trend_slope(series),
        "volume_seasonality_strength": compute_volume_seasonality_strength(result),
        "volume_residual_anomaly": compute_volume_residual_anomaly(result),
    }


def decompose_trade_amounts(
    trades: pd.DataFrame,
    freq: str = "1min",
) -> pd.Series | None:
    """Full pipeline: trades → 1-min bins → STL → residuals.

    Returns the *residual* component (time-indexed, same frequency as the
    input bins) after removing trend and seasonal components.  Returns
    ``None`` when:
    - the resulting time series has fewer than ``_MIN_OBS_FOR_STL`` bins, or
    - the detected period would require more data than available, or
    - STL raises any other error.
    """
    series = to_amount_time_series(trades, freq=freq)
    if len(series) < _MIN_OBS_FOR_STL:
        return None

    period = detect_dominant_period(series) or _DEFAULT_PERIOD
    if len(series) < 2 * period:
        return None

    try:
        result = decompose_amounts(series, period=period)
        return pd.Series(result.resid, index=series.index)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Volume TS feature extraction (issue #260)
# ---------------------------------------------------------------------------


def compute_volume_trend_slope(series: pd.Series) -> float:
    """Linear regression slope of volume, normalised by mean volume.

    Returns ``NaN`` for series with fewer than ``_MIN_OBS_FOR_STL`` observations.
    """
    if len(series) < _MIN_OBS_FOR_STL:
        return float("nan")
    x = np.arange(len(series), dtype=np.float64)
    y = series.values.astype(np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    mean_vol = float(series.mean())
    return slope / (mean_vol + 1e-9)


def compute_volume_seasonality_strength(result) -> float:
    """Ratio of seasonal component variance to (seasonal + residual) variance.

    Returns ``NaN`` when *result* is ``None``.
    """
    if result is None:
        return float("nan")
    seasonal_var = float(np.var(np.asarray(result.seasonal)))
    resid_var = float(np.var(np.asarray(result.resid)))
    total = seasonal_var + resid_var
    if total == 0.0:
        return 0.0
    return seasonal_var / total


def compute_volume_residual_anomaly(result) -> float:
    """Fraction of time bins whose residual exceeds mean + 2 * std.

    Returns ``NaN`` when *result* is ``None``.
    """
    if result is None:
        return float("nan")
    resid = np.asarray(result.resid, dtype=np.float64)
    mean_r = float(np.mean(resid))
    std_r = float(np.std(resid))
    if std_r == 0.0:
        return 0.0
    return float(np.mean(resid > mean_r + 2.0 * std_r))


def compute_ts_features(wallet_trades: pd.DataFrame) -> dict[str, float]:
    """Compute three volume time-series features from *wallet_trades*.

    Features
    --------
    volume_trend_slope
        Normalised linear regression slope of the 30-day hourly volume series.
    volume_seasonality_strength
        Fraction of variance explained by the seasonal component (STL,
        period=24 h).  High values (≥ 0.8) indicate mechanical, predictable
        trading.
    volume_residual_anomaly
        Fraction of hourly bins whose STL residual exceeds mean + 2 std.
        High values indicate bursty, unexplained volume spikes.

    Returns NaN for all three features when fewer than 48 hourly bins are
    available (< 48 h of data).  Uses STL with ``robust=True`` so outlier
    events do not distort the trend estimate.
    """
    nan_result: dict[str, float] = {
        "volume_trend_slope": float("nan"),
        "volume_seasonality_strength": float("nan"),
        "volume_residual_anomaly": float("nan"),
    }

    series = to_amount_time_series(wallet_trades, freq="1h")
    if len(series) < _MIN_OBS_FOR_STL:
        return nan_result

    try:
        result = decompose_amounts(series, period=_DEFAULT_PERIOD)
    except Exception:
        return nan_result

    return {
        "volume_trend_slope": compute_volume_trend_slope(series),
        "volume_seasonality_strength": compute_volume_seasonality_strength(result),
        "volume_residual_anomaly": compute_volume_residual_anomaly(result),
    }
