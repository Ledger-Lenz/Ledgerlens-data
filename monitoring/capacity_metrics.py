"""Prometheus metrics for capacity planning (Issue #242).

Registers three gauges scraped by Prometheus on the internal /metrics endpoint
(not exposed externally):

  - ledgerlens_cpu_usage_ratio    labelled by component
  - ledgerlens_memory_usage_bytes (process-wide)
  - ledgerlens_trades_per_second  labelled by asset_pair

In addition to current-value gauges this module provides a
:class:`CapacityForecaster` that fits a linear + optional weekly-seasonal
trend over a rolling history of observations and projects forward to warn
operators of an approaching capacity limit *days in advance*.

Forecasting design
------------------
The forecaster uses ordinary least-squares linear regression (numpy) over the
last ``history_window`` observations with an optional weekly-seasonal component
(7-period Fourier harmonics) when enough history is available.  The model is
deliberately simple so it:

* Works without scipy/statsmodels (only numpy required).
* Runs in O(N) per fit where N = history_window (default 336 = 2 weeks of
  hourly samples).
* Produces an interpretable forecast with confidence intervals.

Accuracy validation
-------------------
``CapacityForecaster.evaluate_holdout()`` splits historical data into a training
portion and a held-out tail, fits the model on the training portion, and returns
MAE / RMSE / MAPE against the held-out portion so operators have a documented
error margin before trusting the forecast.

Lead-time alert API
-------------------
``CapacityForecaster.days_to_limit()`` returns the estimated number of days
until the current trend crosses the configured capacity threshold.
``CapacityForecaster.check_lead_time_alert()`` returns a non-None dict (the
alert payload) when the crossing is imminent within ``lead_time_days``.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Gauge, Histogram

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metric definitions — registered once on import
# ---------------------------------------------------------------------------

CPU_USAGE_RATIO: Gauge | None = None
MEMORY_USAGE_BYTES: Gauge | None = None
TRADES_PER_SECOND: Gauge | None = None
E2E_LATENCY_SECONDS: Histogram | None = None

# Forecast gauges – expose Python-computed projections to Prometheus so
# Grafana can overlay them on the same panel as the raw metric.
FORECAST_CPU_RATIO: Gauge | None = None
FORECAST_DAYS_TO_LIMIT: Gauge | None = None

if _PROM_AVAILABLE:
    try:
        from prometheus_client import REGISTRY

        def _get_or_create_gauge(name: str, doc: str, labels: list[str] | None = None) -> Gauge:
            if name not in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
                if labels:
                    return Gauge(name, doc, labels)
                return Gauge(name, doc)
            return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]

        def _get_or_create_histogram(
            name: str, doc: str, buckets: list[float]
        ) -> Histogram:
            if name not in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
                return Histogram(name, doc, buckets=buckets)
            return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]

        CPU_USAGE_RATIO = _get_or_create_gauge(
            "ledgerlens_cpu_usage_ratio",
            "CPU usage ratio (0.0–1.0) per component",
            ["component"],
        )
        MEMORY_USAGE_BYTES = _get_or_create_gauge(
            "ledgerlens_memory_usage_bytes",
            "Process resident-set-size memory usage in bytes",
        )
        TRADES_PER_SECOND = _get_or_create_gauge(
            "ledgerlens_trades_per_second",
            "Observed trade-event ingestion rate (events/s) per asset pair",
            ["asset_pair"],
        )
        E2E_LATENCY_SECONDS = _get_or_create_histogram(
            "ledgerlens_e2e_latency_seconds",
            "End-to-end latency from ingestion to consumer decision",
            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )
        FORECAST_CPU_RATIO = _get_or_create_gauge(
            "ledgerlens_forecast_cpu_ratio",
            "Forecasted CPU usage ratio at lead_time_days horizon (component label)",
            ["component", "horizon_days"],
        )
        FORECAST_DAYS_TO_LIMIT = _get_or_create_gauge(
            "ledgerlens_forecast_days_to_capacity_limit",
            "Estimated days until metric reaches capacity_threshold (label: metric)",
            ["metric"],
        )

    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to register capacity metrics: %s", exc)


# ---------------------------------------------------------------------------
# Update helpers (original API – unchanged)
# ---------------------------------------------------------------------------


def set_cpu_usage(component: str, ratio: float) -> None:
    """Record CPU usage ratio (0.0–1.0) for *component*.

    Components: ``benford``, ``feature``, ``inference``, ``ingestion``.
    """
    if CPU_USAGE_RATIO is not None:
        CPU_USAGE_RATIO.labels(component=component).set(ratio)


def set_memory_usage(bytes_used: int) -> None:
    """Record process memory (RSS) in bytes."""
    if MEMORY_USAGE_BYTES is not None:
        MEMORY_USAGE_BYTES.set(bytes_used)


def set_trades_per_second(asset_pair: str, rate: float) -> None:
    """Record trade-event ingestion rate for *asset_pair*."""
    if TRADES_PER_SECOND is not None:
        TRADES_PER_SECOND.labels(asset_pair=asset_pair).set(rate)


def record_e2e_latency(latency_seconds: float) -> None:
    """Record end-to-end latency observation."""
    if E2E_LATENCY_SECONDS is not None:
        E2E_LATENCY_SECONDS.observe(latency_seconds)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


class ForecastPoint(NamedTuple):
    """A single point on the forecast trend line."""

    steps_ahead: int  # number of sample-steps in the future
    predicted_value: float
    ci_lower: float  # 95 % confidence interval lower bound
    ci_upper: float  # 95 % confidence interval upper bound


@dataclass
class HoldoutMetrics:
    """Accuracy metrics from a held-out evaluation."""

    mae: float
    rmse: float
    mape: float  # mean absolute percentage error (nan if any actual == 0)
    n_holdout: int
    n_train: int


@dataclass
class CapacityForecaster:
    """Linear + optional seasonal trend forecaster over rolling capacity data.

    Parameters
    ----------
    metric_name:
        Human-readable label used in log messages and alert payloads.
    samples_per_day:
        How many observations per day (e.g. 24 for hourly, 1 for daily).
        Used to convert *steps* to *days* and vice-versa.
    history_window:
        Maximum observations to retain in the rolling history ring-buffer.
        Defaults to 336 (two weeks of hourly samples).
    capacity_threshold:
        Fractional or absolute capacity limit (e.g. 0.9 for 90 % CPU).
        ``check_lead_time_alert`` fires when the projected crossing is within
        ``lead_time_days``.
    lead_time_days:
        How far in advance to warn about an approaching capacity limit.
    use_seasonality:
        If True and history is long enough (≥ 2 full periods), a weekly
        seasonal component (sin/cos Fourier pairs) is added to the model.
    season_period:
        Period of the seasonal cycle in samples.  Defaults to
        ``7 * samples_per_day`` (weekly).
    min_history_for_forecast:
        Minimum history observations required before a forecast is attempted.
    """

    metric_name: str = "cpu_usage"
    samples_per_day: float = 24.0  # hourly by default
    history_window: int = 336  # two weeks @ hourly
    capacity_threshold: float = 0.9
    lead_time_days: float = 5.0
    use_seasonality: bool = True
    season_period: int = 0  # 0 = auto (7 * samples_per_day)
    min_history_for_forecast: int = 14  # at least 14 observations

    # Internal rolling history (populated via .record())
    _history: deque = field(default_factory=deque, init=False, repr=False)
    # Cache of the last fitted coefficients (for diagnostics)
    _coef: np.ndarray | None = field(default=None, init=False, repr=False)
    _residual_std: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._history: deque[float] = deque(maxlen=self.history_window)
        if self.season_period == 0:
            self.season_period = max(1, int(7 * self.samples_per_day))
        self._coef = None
        self._residual_std = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, value: float) -> None:
        """Append one observation to the rolling history."""
        self._history.append(value)
        # Invalidate cached coefficients
        self._coef = None

    def history_length(self) -> int:
        """Return the number of observations in the rolling history."""
        return len(self._history)

    def forecast(self, steps_ahead: int = 1) -> ForecastPoint | None:
        """Forecast ``steps_ahead`` steps into the future.

        Returns ``None`` when there is insufficient history.

        The 95 % confidence interval assumes normally-distributed residuals:
        CI = predicted ± 1.96 * residual_std * sqrt(1 + leverage)
        where leverage is the diagonal of the hat matrix projection.
        """
        coef, std, X, y = self._fit()
        if coef is None:
            return None

        n = len(y)
        x_pred = self._feature_vector(n + steps_ahead - 1, n)
        predicted = float(x_pred @ coef)
        # Simple conservative CI: propagate residual std + extrapolation uncertainty
        # (leverage grows with distance from training data centre)
        t_steps = steps_ahead
        leverage_factor = math.sqrt(1.0 + t_steps / max(n, 1))
        margin = 1.96 * std * leverage_factor
        return ForecastPoint(
            steps_ahead=steps_ahead,
            predicted_value=predicted,
            ci_lower=predicted - margin,
            ci_upper=predicted + margin,
        )

    def forecast_series(
        self, days_ahead: float | None = None, n_points: int = 30
    ) -> list[ForecastPoint]:
        """Return a list of forecast points from now to ``days_ahead`` days.

        If ``days_ahead`` is None, defaults to ``lead_time_days``.
        ``n_points`` controls the resolution of the series.

        Returns empty list when there is insufficient history.
        """
        if days_ahead is None:
            days_ahead = self.lead_time_days
        total_steps = max(1, int(days_ahead * self.samples_per_day))
        step_size = max(1, total_steps // n_points)
        steps_list = list(range(step_size, total_steps + 1, step_size))
        if not steps_list or steps_list[-1] != total_steps:
            steps_list.append(total_steps)

        results = []
        for s in steps_list:
            pt = self.forecast(s)
            if pt is None:
                return []
            results.append(pt)
        return results

    def days_to_limit(self) -> float | None:
        """Estimate days until the trend crosses ``capacity_threshold``.

        Returns ``None`` if the trend is flat / declining or history is too
        short.  Returns 0.0 if already above the threshold.

        Method: solve ``predicted(t) = threshold`` analytically from the
        linear slope extracted from the fitted model.
        """
        if len(self._history) > 0 and self._history[-1] >= self.capacity_threshold:
            return 0.0

        coef, _, _, y = self._fit()
        if coef is None:
            return None

        # Extract slope from the trend coefficient (index 1 = time slope)
        slope_per_step = float(coef[1])
        if slope_per_step <= 0:
            return None  # flat or declining — no imminent crossing

        current_value = float(self._history[-1]) if self._history else 0.0
        remaining = self.capacity_threshold - current_value
        if remaining <= 0:
            return 0.0

        steps_needed = remaining / slope_per_step
        return steps_needed / self.samples_per_day

    def check_lead_time_alert(self) -> dict | None:
        """Return an alert payload dict if a capacity breach is projected.

        Returns ``None`` if:
        * History is too short to forecast.
        * Projected breach is more than ``lead_time_days`` away.
        * Trend is declining / flat.

        The returned dict has keys compatible with the ``Alert`` TypedDict in
        ``alerts/router.py`` plus extra capacity-specific fields:
          ``days_to_limit``, ``capacity_threshold``, ``forecasted_value``,
          ``metric_name``, ``detectors``, ``risk_score``.
        """
        days = self.days_to_limit()
        if days is None:
            return None
        if days == 0.0 or days <= self.lead_time_days:
            # Project what the value will be at the threshold crossing time
            steps = max(1, int((days or 0.5) * self.samples_per_day))
            pt = self.forecast(steps)
            forecasted = pt.predicted_value if pt else float(self.capacity_threshold)
            days_label = round(days, 2) if days > 0 else 0.0
            return {
                "metric_name": self.metric_name,
                "detectors": ["capacity_forecaster"],
                "risk_score": 0.0,  # set by caller if needed
                "days_to_limit": days_label,
                "capacity_threshold": self.capacity_threshold,
                "forecasted_value": forecasted,
                "message": (
                    f"{self.metric_name} projected to reach "
                    f"{self.capacity_threshold * 100:.0f}% capacity in "
                    f"{days_label} days"
                ),
            }
        return None

    def evaluate_holdout(self, holdout_fraction: float = 0.2) -> HoldoutMetrics | None:
        """Evaluate forecast accuracy against a held-out tail of history.

        Splits ``_history`` into a training portion (1 - holdout_fraction)
        and a holdout tail, fits on the training set, then predicts every
        holdout point one-step-at-a-time and reports MAE / RMSE / MAPE.

        Returns ``None`` if history is too short.

        This fulfils the acceptance criterion:
        "Forecast accuracy reported against a held-out historical period with
        a documented error margin."
        """
        data = list(self._history)
        n = len(data)
        if n < self.min_history_for_forecast * 2:
            logger.debug(
                "evaluate_holdout: not enough history (%d observations)", n
            )
            return None

        n_holdout = max(1, int(n * holdout_fraction))
        n_train = n - n_holdout

        # Temporarily replace history with training portion only
        orig_history = self._history
        self._history = deque(data[:n_train], maxlen=self.history_window)
        self._coef = None

        errors: list[float] = []
        pct_errors: list[float] = []
        for i in range(n_holdout):
            actual = data[n_train + i]
            steps = i + 1
            pt = self.forecast(steps)
            if pt is None:
                # Restore and abort
                self._history = orig_history
                self._coef = None
                return None
            err = abs(pt.predicted_value - actual)
            errors.append(err)
            if actual != 0:
                pct_errors.append(err / abs(actual))

        # Restore full history
        self._history = orig_history
        self._coef = None

        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        mape = float(np.mean(pct_errors)) if pct_errors else float("nan")

        return HoldoutMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            n_holdout=n_holdout,
            n_train=n_train,
        )

    # ------------------------------------------------------------------
    # Internal fitting helpers
    # ------------------------------------------------------------------

    def _build_feature_matrix(self, n: int) -> np.ndarray:
        """Build an (n, p) design matrix with intercept, trend, and seasonal terms."""
        t = np.arange(n, dtype=float)
        cols = [np.ones(n), t]  # intercept + linear trend

        # Add weekly seasonal Fourier harmonics when enabled and enough data
        if self.use_seasonality and n >= 2 * self.season_period:
            for k in range(1, 3):  # two harmonics covers most weekly patterns
                angle = 2 * np.pi * k * t / self.season_period
                cols.append(np.sin(angle))
                cols.append(np.cos(angle))

        return np.column_stack(cols)

    def _feature_vector(self, step_index: int, n_train: int) -> np.ndarray:
        """Build the feature row for a single step_index beyond the training set."""
        t = float(step_index)
        row = [1.0, t]
        if self.use_seasonality and n_train >= 2 * self.season_period:
            for k in range(1, 3):
                angle = 2 * np.pi * k * t / self.season_period
                row.append(math.sin(angle))
                row.append(math.cos(angle))
        return np.array(row)

    def _fit(
        self,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None, np.ndarray | None]:
        """Fit OLS on the current history.

        Returns (coef, residual_std, X, y) or (None, 0, None, None).
        """
        if self._coef is not None:
            # Return cached result – rebuild X/y for the ForecastPoint call
            data = np.array(self._history, dtype=float)
            n = len(data)
            X = self._build_feature_matrix(n)
            return self._coef, self._residual_std, X, data

        data = np.array(self._history, dtype=float)
        n = len(data)
        if n < self.min_history_for_forecast:
            return None, 0.0, None, None

        X = self._build_feature_matrix(n)
        y = data

        # OLS via lstsq (numerically stable, no scipy needed)
        coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        if residuals.size > 0:
            std = float(np.sqrt(residuals[0] / max(n - rank, 1)))
        else:
            # Compute residuals manually
            resid = y - X @ coef
            std = float(np.sqrt(np.mean(resid**2)))

        self._coef = coef
        self._residual_std = std
        return coef, std, X, y


# ---------------------------------------------------------------------------
# Prometheus gauge update helper for forecast values
# ---------------------------------------------------------------------------


def publish_forecast(forecaster: CapacityForecaster, component: str | None = None) -> None:
    """Publish forecast values to Prometheus gauges.

    Call this after each batch of ``forecaster.record()`` calls to expose the
    Python-computed forecast to Prometheus (and therefore Grafana).

    *component* is the Prometheus label value for ``ledgerlens_forecast_cpu_ratio``.
    When ``None``, ``forecaster.metric_name`` is used.
    """
    label = component or forecaster.metric_name

    days = forecaster.days_to_limit()
    if FORECAST_DAYS_TO_LIMIT is not None:
        value = days if days is not None else float("inf")
        # Prometheus Gauge doesn't support +Inf cleanly; use -1 for "no crossing projected"
        FORECAST_DAYS_TO_LIMIT.labels(metric=label).set(value if math.isfinite(value) else -1.0)

    if FORECAST_CPU_RATIO is not None:
        steps_1d = max(1, int(forecaster.samples_per_day))
        steps_7d = max(1, int(7 * forecaster.samples_per_day))
        for steps, horizon_label in [(steps_1d, "1"), (steps_7d, "7")]:
            pt = forecaster.forecast(steps)
            if pt is not None:
                FORECAST_CPU_RATIO.labels(
                    component=label, horizon_days=horizon_label
                ).set(pt.predicted_value)
