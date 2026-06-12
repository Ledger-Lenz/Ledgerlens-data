"""Benford's Law anomaly metrics for transaction amount distributions.

Implements the three metrics described in the project README:
  - Chi-square statistic vs. the expected first-digit distribution
  - Per-digit Z-scores
  - Mean Absolute Deviation (MAD)

These are computed per wallet / asset / pair over rolling time windows
(see `config.BENFORD_WINDOWS_HOURS`) and feed into the Benford feature
group consumed by `feature_engineering.py`.
"""

import math

import numpy as np
import pandas as pd

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
    normalized = amounts / (10.0 ** magnitudes)
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


def compute_benford_metrics(amounts: pd.Series) -> dict:
    """Compute the full set of Benford metrics for a series of amounts.

    Returns a dict with `chi_square`, `mad`, `mad_nonconforming`, and
    `z_scores` (per-digit dict), suitable for use as a feature row.
    """
    return {
        "chi_square": chi_square_statistic(amounts),
        "mad": mad_score(amounts),
        "mad_nonconforming": mad_score(amounts) > MAD_NONCONFORMITY_THRESHOLD,
        "z_scores": z_scores(amounts),
        "sample_size": int((amounts > 0).sum()),
    }


def compute_benford_metrics_for_windows(
    df: pd.DataFrame,
    amount_col: str = "amount",
    time_col: str = "ledger_close_time",
    windows_hours: list[int] | None = None,
    reference_time: pd.Timestamp | None = None,
) -> dict[int, dict]:
    """Compute Benford metrics over multiple trailing windows ending at
    `reference_time` (defaults to the max timestamp in `df`).

    Returns a dict mapping window size (hours) -> metrics dict.
    """
    if windows_hours is None:
        from config import config

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
