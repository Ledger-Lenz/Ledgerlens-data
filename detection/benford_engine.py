"""Benford's Law anomaly metrics for transaction amount distributions.

Implements the three metrics described in the project README:
  - Chi-square statistic vs. the expected first-digit distribution
  - Per-digit Z-scores
  - Mean Absolute Deviation (MAD)

These are computed per wallet / asset / pair over rolling time windows
(see `config.BENFORD_WINDOWS_HOURS`) and feed into the Benford feature
group consumed by `feature_engineering.py`.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BenfordMetrics:
    """Standardized Benford anomaly metrics."""

    chi_square: float
    mad: float
    mad_nonconforming: bool
    z_scores: dict[int, float]
    sample_size: int

    def __getitem__(self, key: str) -> Any:
        if key == "z_max":
            return max(self.z_scores.values(), default=np.nan)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "z_max":
            return max(self.z_scores.values(), default=np.nan)
        return getattr(self, key, default)


# Benford's Law expected frequency for leading digits 1-9
BENFORD_EXPECTED = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

MAD_NONCONFORMITY_THRESHOLD = 0.015


def leading_digits(amounts: pd.Series) -> pd.Series:
    """Extract the leading (first significant) digit of each amount.

    Zero and negative amounts are dropped — Benford's Law applies to the
    magnitude of nonzero values.
    """
    amounts = amounts[amounts > 0]
    if amounts.empty:
        return amounts

    magnitudes = np.floor(np.log10(amounts)).astype(int)
    normalized = amounts / (10.0**magnitudes)
    return np.floor(normalized).astype(int).clip(1, 9)


def observed_distribution(amounts: pd.Series) -> dict[int, float]:
    """Observed frequency of each leading digit 1-9."""
    digits = leading_digits(amounts)
    if digits.empty:
        return {d: 0.0 for d in range(1, 10)}

    counts = digits.value_counts(normalize=True)
    return {d: float(counts.get(d, 0.0)) for d in range(1, 10)}


def chi_square_statistic(amounts: pd.Series) -> float:
    """Chi-square goodness-of-fit statistic against Benford's distribution."""
    digits = leading_digits(amounts)
    n = len(digits)
    if n == 0:
        return 0.0

    observed_counts = digits.value_counts()
    chi_sq = 0.0
    for d in range(1, 10):
        expected_count = BENFORD_EXPECTED[d] * n
        observed_count = observed_counts.get(d, 0)
        if expected_count > 0:
            chi_sq += (observed_count - expected_count) ** 2 / expected_count

    return float(chi_sq)


def z_scores(amounts: pd.Series) -> dict[int, float]:
    """Per-digit Z-score of the observed vs. expected Benford proportion."""
    digits = leading_digits(amounts)
    n = len(digits)
    if n == 0:
        return {d: 0.0 for d in range(1, 10)}

    observed = observed_distribution(amounts)
    scores = {}
    for d in range(1, 10):
        p = BENFORD_EXPECTED[d]
        # Standard error for a proportion under Benford's expected distribution,
        # with a continuity correction of 1/(2n) per Nigrini (2012).
        std_err = math.sqrt(p * (1 - p) / n)
        if std_err == 0:
            scores[d] = 0.0
            continue
        z = (abs(observed[d] - p) - (1 / (2 * n))) / std_err
        scores[d] = float(max(z, 0.0))

    return scores


def mad_score(amounts: pd.Series) -> float:
    """Mean Absolute Deviation between observed and expected distributions.

    Values above `MAD_NONCONFORMITY_THRESHOLD` (0.015) indicate the
    distribution does not conform to Benford's Law (Nigrini, 2012).
    """
    digits = leading_digits(amounts)
    if digits.empty:
        return 0.0

    observed = observed_distribution(amounts)
    deviations = [abs(observed[d] - BENFORD_EXPECTED[d]) for d in range(1, 10)]
    return float(sum(deviations) / len(deviations))


def compute_benford_metrics(amounts: pd.Series) -> BenfordMetrics:
    """Compute the full set of Benford metrics for a series of amounts.

    Returns a BenfordMetrics dataclass (backward compatible with dict access).
    """
    from config import config
    n = int((amounts > 0).sum())
    if n < config.MIN_TRADES_FOR_SCORING:
        return BenfordMetrics(
            chi_square=np.nan,
            mad=np.nan,
            mad_nonconforming=False,
            z_scores={d: np.nan for d in range(1, 10)},
            sample_size=n,
        )

    mad = mad_score(amounts)
    return BenfordMetrics(
        chi_square=chi_square_statistic(amounts),
        mad=mad,
        mad_nonconforming=mad > MAD_NONCONFORMITY_THRESHOLD,
        z_scores=z_scores(amounts),
        sample_size=n,
    )


def compute_benford_metrics_for_windows(
    df: pd.DataFrame,
    amount_col: str = "amount",
    time_col: str = "ledger_close_time",
    windows_hours: list[int] | None = None,
    reference_time: pd.Timestamp | None = None,
    asset: str | None = None,  # NEW: looks up per-asset windows
) -> dict[int, BenfordMetrics]:
    """Compute Benford metrics over multiple trailing windows ending at
    `reference_time` (defaults to the max timestamp in `df`).

    Returns a dict mapping window size (hours) -> metrics dict.
    """
    if windows_hours is None:
        from config import config
        # Infer asset if not provided
        if asset is None and not df.empty:
            for col in ["base_asset", "counter_asset"]:
                if col in df.columns:
                    unique_assets = df[col].dropna().unique()
                    for a in unique_assets:
                        if a in getattr(config, "ASSET_BENFORD_WINDOWS", {}):
                            asset = a
                            break
                    if asset:
                        break
            if asset is None and "base_asset" in df.columns:
                asset = df["base_asset"].mode().iloc[0] if not df["base_asset"].empty else None

        if asset and hasattr(config, "ASSET_BENFORD_WINDOWS") and asset in config.ASSET_BENFORD_WINDOWS:
            windows_hours = config.ASSET_BENFORD_WINDOWS[asset]
        else:
            windows_hours = config.BENFORD_WINDOWS_HOURS

    if df.empty:
        return {w: compute_benford_metrics(pd.Series(dtype=float)) for w in windows_hours}

    timestamps = pd.to_datetime(df[time_col])
    ref = reference_time or timestamps.max()

    results = {}
    for hours in windows_hours:
        window_start = ref - pd.Timedelta(hours=hours)
        window_df = df[(timestamps > window_start) & (timestamps <= ref)]
        results[hours] = compute_benford_metrics(window_df[amount_col])

    return results


def cross_pair_benford_consistency(per_pair_metrics: dict[str, BenfordMetrics]) -> float:
    """Compute cross-pair Benford MAD consistency.

    `per_pair_metrics` maps pair_id -> BenfordMetrics.
    Returns the standard deviation of MAD scores across pairs. Low values indicate
    all pairs have similar Benford conformity (consistent wash trading pattern).
    High values indicate mixed conformity (concentrated on specific pairs).
    """
    if not per_pair_metrics or len(per_pair_metrics) < 2:
        return 0.0

    mad_scores = [metrics.get("mad", 0.0) for metrics in per_pair_metrics.values() if metrics]
    if not mad_scores or len(mad_scores) < 2:
        return 0.0

    return float(np.std(mad_scores))


# ---------------------------------------------------------------------------
# Hardening measures
# ---------------------------------------------------------------------------


def leading_digits_log(amounts: pd.Series) -> pd.Series:
    """Extract leading digits after applying a log10 transform.

    Applying log10 before digit extraction defeats the AmountRounding attack:
    rounding to N significant figures collapses log10 values to a narrow
    range, which still reveals the deviation from Benford's Law.
    """
    amounts = amounts[amounts > 0]
    if amounts.empty:
        return amounts
    log_amounts = np.log10(amounts)
    # Shift so all values are > 1, preserving leading-digit semantics
    shift = math.floor(log_amounts.min()) - 1
    shifted = log_amounts - shift
    magnitudes = np.floor(np.log10(shifted)).astype(int)
    normalized = shifted / (10.0**magnitudes)
    return np.floor(normalized).astype(int).clip(1, 9)


def second_digits(amounts: pd.Series) -> pd.Series:
    """Extract the second significant digit of each amount (0–9).

    The Newcomb–Benford generalisation extends to second digits.  The
    expected distribution of the second digit is flatter but still
    non-uniform, and adversarial rounding typically produces a very
    different second-digit pattern.
    """
    amounts = amounts[amounts > 0]
    if amounts.empty:
        return amounts
    magnitudes = np.floor(np.log10(amounts)).astype(int)
    normalized = amounts / (10.0**magnitudes)  # first digit is floor(normalized)
    # Remove first digit contribution, scale up and take floor
    second = np.floor((normalized - np.floor(normalized)) * 10).astype(int).clip(0, 9)
    return second


def chi_square_log(amounts: pd.Series) -> float:
    """Chi-square statistic computed on log-transformed leading digits.

    Hardened against AmountRounding by working in log space.
    """
    digits = leading_digits_log(amounts)
    n = len(digits)
    if n == 0:
        return 0.0
    observed_counts = digits.value_counts()
    chi_sq = 0.0
    for d in range(1, 10):
        expected_count = BENFORD_EXPECTED[d] * n
        observed_count = observed_counts.get(d, 0)
        if expected_count > 0:
            chi_sq += (observed_count - expected_count) ** 2 / expected_count
    return float(chi_sq)


def compute_benford_confidence_intervals(
    amounts: pd.Series,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    wallet_id: str = "",
    window_hours: int = 0,
) -> dict:
    """Bootstrap confidence intervals for Benford chi-square, MAD, and z-scores.

    Gated behind ``config.BENFORD_CI_ENABLED`` (default False) because bootstrap
    resampling is computationally intensive — O(n_bootstrap × n) per call.

    The random seed is derived from (wallet_id, window_hours) so results are
    reproducible per wallet/window without touching the global random state.

    Args:
        amounts:      Series of positive trade amounts.
        n_bootstrap:  Number of bootstrap resamples (default 1000).
        alpha:        Significance level; 0.05 gives 95% CIs.
        wallet_id:    Used to derive a per-(wallet, window) RNG seed.
        window_hours: Used to derive a per-(wallet, window) RNG seed.

    Returns:
        Dict with keys:
          chi_square_lower, chi_square_upper, chi_square_ci_width
          mad_lower, mad_upper, mad_ci_width
          z_max_lower, z_max_upper, z_max_ci_width
          insufficient_data  — True when CI width > point estimate (< ~30 trades)
    """
    from config import config

    if not config.BENFORD_CI_ENABLED:
        return {
            "chi_square_lower": np.nan,
            "chi_square_upper": np.nan,
            "chi_square_ci_width": np.nan,
            "mad_lower": np.nan,
            "mad_upper": np.nan,
            "mad_ci_width": np.nan,
            "z_max_lower": np.nan,
            "z_max_upper": np.nan,
            "z_max_ci_width": np.nan,
            "insufficient_data": False,
        }

    amounts = amounts[amounts > 0].reset_index(drop=True)
    n = len(amounts)
    if n == 0:
        return {
            "chi_square_lower": 0.0,
            "chi_square_upper": 0.0,
            "chi_square_ci_width": 0.0,
            "mad_lower": 0.0,
            "mad_upper": 0.0,
            "mad_ci_width": 0.0,
            "z_max_lower": 0.0,
            "z_max_upper": 0.0,
            "z_max_ci_width": 0.0,
            "insufficient_data": True,
        }

    # Deterministic seed per (wallet_id, window_hours) — no global seed mutation
    seed = int(hashlib.sha256(f"{wallet_id}:{window_hours}".encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    chi_samples = []
    mad_samples = []
    zmax_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot = amounts.iloc[idx]
        chi_samples.append(chi_square_statistic(boot))
        mad_samples.append(mad_score(boot))
        zmax_boot = z_scores(boot)
        zmax_samples.append(max(zmax_boot.values(), default=0.0))

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    chi_lo = float(np.percentile(chi_samples, lo))
    chi_hi = float(np.percentile(chi_samples, hi))
    mad_lo = float(np.percentile(mad_samples, lo))
    mad_hi = float(np.percentile(mad_samples, hi))
    zm_lo = float(np.percentile(zmax_samples, lo))
    zm_hi = float(np.percentile(zmax_samples, hi))

    chi_point = chi_square_statistic(amounts)
    chi_width = chi_hi - chi_lo
    insufficient = (chi_point > 0) and (chi_width > chi_point)

    return {
        "chi_square_lower": chi_lo,
        "chi_square_upper": chi_hi,
        "chi_square_ci_width": chi_width,
        "mad_lower": mad_lo,
        "mad_upper": mad_hi,
        "mad_ci_width": mad_hi - mad_lo,
        "z_max_lower": zm_lo,
        "z_max_upper": zm_hi,
        "z_max_ci_width": zm_hi - zm_lo,
        "insufficient_data": insufficient,
    }


def bootstrap_chi_square_ci(
    amounts: pd.Series,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the Benford chi-square statistic.

    Returns ``(lower, upper)`` bounds.  A suspiciously *low* chi-square
    (upper bound near zero) can signal manufactured conformance — an
    adversary who over-tunes their distribution to match Benford's Law too
    precisely.
    """
    rng = rng or np.random.default_rng(0)
    amounts = amounts[amounts > 0].reset_index(drop=True)
    n = len(amounts)
    if n == 0:
        return (0.0, 0.0)

    samples = [
        chi_square_statistic(amounts.iloc[rng.choice(n, size=n, replace=True)])
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - ci
    lower = float(np.percentile(samples, alpha / 2 * 100))
    upper = float(np.percentile(samples, (1 - alpha / 2) * 100))
    return (lower, upper)
