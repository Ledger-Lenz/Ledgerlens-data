"""Instrumental Variable (IV) estimation for causal effect of market maker activity.

Implements 2-Stage Least Squares (2SLS) to estimate the causal effect of
``counterparty_concentration_ratio`` on a risk score, using the Stellar DEX
liquidity incentive programme participation flag as an instrument.

The instrument is *exogenous*: programme participation affects market-making
intensity (first stage) but has no direct pathway to wash-trade risk
(exclusion restriction).

Usage
-----
    from detection.iv_estimator import IVEstimator, IVEstimateResult

    result = IVEstimator().estimate(
        endog=df["counterparty_concentration_ratio"],
        instrument=df["liquidity_programme_flag"],
        outcome=df["risk_score"],
        controls=df[["trade_count"]],   # optional
    )
    print(result)

The result is always presented as an *estimate with uncertainty*; it must not
be interpreted as a definitive causal fact without appropriate statistical
expertise (see docstring on IVEstimateResult).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEAK_INSTRUMENT_F_THRESHOLD = 10.0
IV_DISCLAIMER = (
    "IV estimates are statistical approximations subject to instrument validity assumptions. "
    "They must not be presented as definitive causal facts in legal proceedings without "
    "appropriate econometric expert review."
)


@dataclass
class IVEstimateResult:
    """Result of a 2SLS or OLS estimation.

    All numeric fields are floats. ``reliable`` is False when the instrument
    is weak (first-stage F < 10) or when OLS fallback was used.
    """

    method: str          # "2SLS" or "OLS_fallback"
    coef: float          # point estimate for endog variable
    ci_lower: float      # 95% confidence interval lower bound
    ci_upper: float      # 95% confidence interval upper bound
    first_stage_f: float | None   # None for OLS fallback
    reliable: bool       # False if weak instrument or OLS fallback
    warning: str | None  # human-readable warning when reliable=False
    disclaimer: str = IV_DISCLAIMER

    def __str__(self) -> str:
        bounds = f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        flag = "" if self.reliable else f" ⚠ {self.warning}"
        return f"{self.method} coef={self.coef:.4f} 95%CI={bounds}{flag}"


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return OLS coefficients and residuals."""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, resid


def _hc1_se(X: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Heteroskedasticity-robust (HC1) standard errors."""
    n, k = X.shape
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    bread = np.linalg.inv(X.T @ X)
    vcov = bread @ meat @ bread * n / (n - k)
    return np.sqrt(np.diag(vcov))


def _first_stage_f(X_first: np.ndarray, endog: np.ndarray, resid_first: np.ndarray) -> float:
    """Partial F-statistic for instrument relevance in the first stage.

    X_first layout: [controls..., instrument, constant]
    Restricted model drops the instrument column (index -2), keeping controls + constant.
    """
    n, k = X_first.shape
    # Drop instrument (second-to-last col), keep controls + constant
    instr_col = k - 2
    X_restricted = np.delete(X_first, instr_col, axis=1)
    if X_restricted.shape[1] == 0:
        X_restricted = np.ones((n, 1))
    _, resid_restricted = _ols(X_restricted, endog)
    rss_restricted = float(resid_restricted @ resid_restricted)
    rss_unrestricted = float(resid_first @ resid_first)
    if rss_unrestricted < 1e-12:
        return float("inf")
    f = ((rss_restricted - rss_unrestricted) / 1.0) / (rss_unrestricted / (n - k))
    return float(f)


class IVEstimator:
    """2SLS IV estimator with weak-instrument detection and OLS fallback."""

    def estimate(
        self,
        endog: pd.Series | np.ndarray,
        instrument: pd.Series | np.ndarray,
        outcome: pd.Series | np.ndarray,
        controls: pd.DataFrame | np.ndarray | None = None,
    ) -> IVEstimateResult:
        """Estimate the causal effect of ``endog`` on ``outcome`` using ``instrument``.

        Parameters
        ----------
        endog:
            Endogenous variable (e.g. counterparty_concentration_ratio).
        instrument:
            Binary or continuous instrument (e.g. liquidity_programme_flag).
        outcome:
            Outcome variable (e.g. risk_score).
        controls:
            Optional exogenous controls to include in both stages.

        Returns
        -------
        IVEstimateResult
        """
        endog_arr = np.asarray(endog, dtype=float)
        instr_arr = np.asarray(instrument, dtype=float)
        outcome_arr = np.asarray(outcome, dtype=float)
        n = len(endog_arr)

        if n < 4:
            raise ValueError("Need at least 4 observations for IV estimation.")

        # Check instrument has variation
        if np.std(instr_arr) < 1e-10:
            warn_msg = (
                "Instrument has zero variance: no wallets in the analysis participate "
                "in the liquidity programme. Falling back to OLS — estimates may be "
                "confounded by market maker activity."
            )
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
            logger.warning(warn_msg)
            return self._ols_fallback(endog_arr, outcome_arr, controls, warn_msg)

        # Build control matrix
        if controls is not None:
            ctrl = np.asarray(controls, dtype=float)
            if ctrl.ndim == 1:
                ctrl = ctrl[:, None]
        else:
            ctrl = np.empty((n, 0))

        # --- First stage: regress endog on instrument + controls + constant ---
        X_first = np.column_stack([ctrl, instr_arr, np.ones(n)])
        first_coef, first_resid = _ols(X_first, endog_arr)
        endog_hat = X_first @ first_coef

        f_stat = _first_stage_f(X_first, endog_arr, first_resid)

        weak = f_stat < WEAK_INSTRUMENT_F_THRESHOLD
        if weak:
            warn_msg = (
                f"Weak instrument: first-stage F={f_stat:.2f} < {WEAK_INSTRUMENT_F_THRESHOLD}. "
                "IV estimates are unreliable. Consider using a stronger instrument or "
                "reporting OLS estimates with a confounding caveat."
            )
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
            logger.warning(warn_msg)

        # --- Second stage: regress outcome on fitted endog + controls + constant ---
        X_second = np.column_stack([endog_hat, ctrl, np.ones(n)])
        second_coef, second_resid = _ols(X_second, outcome_arr)

        # HC1 standard errors
        se = _hc1_se(X_second, second_resid)
        coef_val = float(second_coef[0])
        se_val = float(se[0])
        z = 1.959964  # 97.5th percentile of N(0,1)
        ci_lower = coef_val - z * se_val
        ci_upper = coef_val + z * se_val

        return IVEstimateResult(
            method="2SLS",
            coef=coef_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            first_stage_f=f_stat,
            reliable=not weak,
            warning=(
                f"Weak instrument (F={f_stat:.2f} < {WEAK_INSTRUMENT_F_THRESHOLD}). "
                "Result flagged as unreliable."
            ) if weak else None,
        )

    def _ols_fallback(
        self,
        endog: np.ndarray,
        outcome: np.ndarray,
        controls: pd.DataFrame | np.ndarray | None,
        warn_msg: str,
    ) -> IVEstimateResult:
        n = len(endog)
        if controls is not None:
            ctrl = np.asarray(controls, dtype=float)
            if ctrl.ndim == 1:
                ctrl = ctrl[:, None]
            X = np.column_stack([endog, ctrl, np.ones(n)])
        else:
            X = np.column_stack([endog, np.ones(n)])

        coef, resid = _ols(X, outcome)
        se = _hc1_se(X, resid)
        coef_val = float(coef[0])
        se_val = float(se[0])
        z = 1.959964
        return IVEstimateResult(
            method="OLS_fallback",
            coef=coef_val,
            ci_lower=coef_val - z * se_val,
            ci_upper=coef_val + z * se_val,
            first_stage_f=None,
            reliable=False,
            warning=warn_msg,
        )
