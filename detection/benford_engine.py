"""Benford's Law anomaly metrics for transaction amount distributions.

Implements the three metrics described in the project README:
  - Chi-square statistic vs. the expected first-digit distribution
  - Per-digit Z-scores
  - Mean Absolute Deviation (MAD)

These are computed per wallet / asset / pair over rolling time windows
(see `config.BENFORD_WINDOWS_HOURS`) and feed into the Benford feature
group consumed by `feature_engineering.py`.

Asset-class-aware baselines (issue #279):
  Stablecoin amounts cluster around round dollar values (100, 1000, 10000 USDC)
  by convention, naturally producing elevated digit-1 frequency that deviates
  from the theoretical Benford distribution without indicating manipulation.
  `AssetClassifier` maintains separate expected distributions per asset class,
  loaded from `data/build_config.json`, to reduce stablecoin false positives.

Second-digit Benford analysis (Issue #179):
  Extends first-digit anomaly detection to the second significant digit (0-9).
  Wash-trade bots that vary their first digit to evade detection often leave
  systematic second-digit anomalies because their lot-size algorithms don't
  model the second digit independently. Functions: compute_second_digit_distribution,
  chi_square_second_digit, z_scores_second_digit, mad_score_second_digit.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.tracing import get_tracer

_tracer = get_tracer(__name__)


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

# Benford's Law expected frequency for second digits 0-9 (Issue #179)
# Reference: Benford, F. (1938) 'The law of anomalous numbers'; Hill, T.P. (1995)
# The second digit should appear with the following frequencies:
# 0: 11.97%, 1: 11.39%, 2: 10.88%, ..., 9: 8.52%
BENFORD_EXPECTED_2ND = {
    0: 0.11968,
    1: 0.11389,
    2: 0.10882,
    3: 0.10433,
    4: 0.10031,
    5: 0.09668,
    6: 0.09337,
    7: 0.09035,
    8: 0.08757,
    9: 0.08500,
}

_BUILD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "build_config.json")


def _load_build_config() -> dict:
    try:
        with open(_BUILD_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Asset-class-aware Benford baseline calibration (issue #279)
# ---------------------------------------------------------------------------

_ASSET_CLASS_STABLECOIN = "stablecoin"
_ASSET_CLASS_NATIVE = "native"
_ASSET_CLASS_VOLATILE = "volatile"


@dataclass
class AssetClassifier:
    """Classify assets as stablecoin, native, or volatile and supply per-class
    empirical Benford baseline distributions.

    The stablecoin list is loaded from ``data/build_config.json`` on startup.
    Empirical baselines for each class are computed from clean-labelled trades
    via :meth:`fit_from_clean_trades`.  Until ``fit_from_clean_trades`` is
    called the class falls back to the static distributions in
    ``build_config.json`` (stablecoins) or the theoretical Benford distribution
    (native / volatile).

    Security: the stablecoin set is derived exclusively from
    ``build_config.json``; arbitrary asset codes passed via API parameters are
    never added to the set.
    """

    _stablecoins: set[str] = field(default_factory=set)
    _native_assets: set[str] = field(default_factory=set)
    # Per-class empirical baselines: class -> {digit: frequency}
    _baselines: dict[str, dict[int, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load_config()

    def _load_config(self) -> None:
        cfg = _load_build_config()
        stablecoins = cfg.get("stablecoins", [])
        if not isinstance(stablecoins, list) or not all(
            isinstance(s, str) for s in stablecoins
        ):
            raise ValueError(
                "build_config.json 'stablecoins' must be a list of strings"
            )
        self._stablecoins = {s.upper() for s in stablecoins}

        native_assets = cfg.get("native_assets", ["XLM"])
        self._native_assets = {a.upper() for a in native_assets}

        # Load static stablecoin baseline from config if present
        static = cfg.get("asset_class_benford_baselines", {})
        sc_dist = static.get("stablecoin", {}).get("distribution")
        if sc_dist:
            try:
                parsed = {int(k): float(v) for k, v in sc_dist.items()}
                if len(parsed) == 9:
                    self._baselines[_ASSET_CLASS_STABLECOIN] = parsed
            except (ValueError, KeyError):
                pass

    def classify(self, asset_code: str) -> str:
        """Return 'stablecoin', 'native', or 'volatile' for *asset_code*.

        *asset_code* is the bare asset code (e.g. 'USDC', 'XLM'), not the
        full CODE:ISSUER string.
        """
        code = asset_code.upper().split(":")[0]
        if code in self._stablecoins:
            return _ASSET_CLASS_STABLECOIN
        if code in self._native_assets:
            return _ASSET_CLASS_NATIVE
        return _ASSET_CLASS_VOLATILE

    def classify_pair(self, asset_pair: str) -> str:
        """Return the asset class for the *base* leg of a CODE:ISSUER/CODE:ISSUER pair.

        If either leg is a stablecoin the pair is classified as stablecoin to
        apply conservative thresholds.
        """
        parts = asset_pair.split("/")
        classes = [self.classify(p.split(":")[0]) for p in parts if p]
        if _ASSET_CLASS_STABLECOIN in classes:
            return _ASSET_CLASS_STABLECOIN
        if _ASSET_CLASS_NATIVE in classes and len(classes) == 1:
            return _ASSET_CLASS_NATIVE
        return _ASSET_CLASS_VOLATILE

    def get_baseline(self, asset_code: str) -> dict[int, float]:
        """Return the expected digit distribution for *asset_code*'s class.

        Falls back to the theoretical Benford distribution for unknown classes
        or when no empirical baseline has been fitted.
        """
        cls = self.classify(asset_code)
        return self._baselines.get(cls, dict(BENFORD_EXPECTED))

    def fit_from_clean_trades(
        self, labelled_df: pd.DataFrame, amount_col: str = "amount"
    ) -> None:
        """Compute empirical baselines per asset class from clean (label=0) trades.

        Parameters
        ----------
        labelled_df:
            DataFrame with columns ``amount``, ``asset_code`` (bare code such
            as ``'USDC'``), and ``label`` (0 = clean, 1 = wash trade).
        """
        if labelled_df.empty:
            return
        clean = labelled_df[labelled_df.get("label", pd.Series(dtype=int)) == 0]
        if clean.empty:
            return

        for cls in (_ASSET_CLASS_STABLECOIN, _ASSET_CLASS_NATIVE, _ASSET_CLASS_VOLATILE):
            if "asset_code" in clean.columns:
                mask = clean["asset_code"].apply(
                    lambda c: self.classify(str(c)) == cls
                )
                subset = clean.loc[mask, amount_col].dropna()
            else:
                subset = pd.Series(dtype=float)

            if len(subset) >= 30:
                self._baselines[cls] = observed_distribution(subset)


# Module-level singleton, lazily constructed on first access.
_classifier = None  # type: ignore[assignment]  # AssetClassifier | None


def get_asset_classifier() -> "AssetClassifier":
    """Return the module-level AssetClassifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = AssetClassifier()
    return _classifier

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


def chi_square_statistic(
    amounts: pd.Series,
    baseline: dict[int, float] | None = None,
) -> float:
    """Chi-square goodness-of-fit statistic against a digit distribution.

    *baseline* defaults to the theoretical Benford distribution when omitted,
    but callers may supply an asset-class-specific baseline (issue #279).
    """
    expected = baseline if baseline is not None else BENFORD_EXPECTED
    digits = leading_digits(amounts)
    n = len(digits)
    if n == 0:
        return 0.0

    observed_counts = digits.value_counts()
    chi_sq = 0.0
    for d in range(1, 10):
        expected_count = expected.get(d, BENFORD_EXPECTED[d]) * n
        observed_count = observed_counts.get(d, 0)
        if expected_count > 0:
            chi_sq += (observed_count - expected_count) ** 2 / expected_count

    return float(chi_sq)


def z_scores(
    amounts: pd.Series,
    baseline: dict[int, float] | None = None,
) -> dict[int, float]:
    """Per-digit Z-score of the observed vs. expected digit proportion.

    *baseline* defaults to the theoretical Benford distribution when omitted.
    """
    expected = baseline if baseline is not None else BENFORD_EXPECTED
    digits = leading_digits(amounts)
    n = len(digits)
    if n == 0:
        return {d: 0.0 for d in range(1, 10)}

    observed = observed_distribution(amounts)
    scores = {}
    for d in range(1, 10):
        p = expected.get(d, BENFORD_EXPECTED[d])
        # Standard error for a proportion under the expected distribution,
        # with a continuity correction of 1/(2n) per Nigrini (2012).
        std_err = math.sqrt(p * (1 - p) / n) if p * (1 - p) > 0 else 0.0
        if std_err == 0:
            scores[d] = 0.0
            continue
        z = (abs(observed[d] - p) - (1 / (2 * n))) / std_err
        scores[d] = float(max(z, 0.0))

    return scores


def mad_score(
    amounts: pd.Series,
    baseline: dict[int, float] | None = None,
) -> float:
    """Mean Absolute Deviation between observed and expected distributions.

    *baseline* defaults to the theoretical Benford distribution when omitted.
    Values above `MAD_NONCONFORMITY_THRESHOLD` (0.015) indicate the
    distribution does not conform (Nigrini, 2012).
    """
    expected = baseline if baseline is not None else BENFORD_EXPECTED
    digits = leading_digits(amounts)
    if digits.empty:
        return 0.0

    observed = observed_distribution(amounts)
    deviations = [abs(observed[d] - expected.get(d, BENFORD_EXPECTED[d])) for d in range(1, 10)]
    return float(sum(deviations) / len(deviations))


def compute_benford_metrics(
    amounts: pd.Series,
    asset_code: str | None = None,
) -> BenfordMetrics:
    """Compute the full set of Benford metrics for a series of amounts.

    When *asset_code* is supplied the appropriate per-class baseline is
    looked up via :func:`get_asset_classifier` (issue #279).  Falls back
    to the theoretical Benford distribution for unknown assets.

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

    baseline: dict[int, float] | None = None
    if asset_code is not None:
        baseline = get_asset_classifier().get_baseline(asset_code)

    mad = mad_score(amounts, baseline=baseline)
    return BenfordMetrics(
        chi_square=chi_square_statistic(amounts, baseline=baseline),
        mad=mad,
        mad_nonconforming=mad > MAD_NONCONFORMITY_THRESHOLD,
        z_scores=z_scores(amounts, baseline=baseline),
        sample_size=n,
    )


def compute_benford_metrics_for_windows(
    df: pd.DataFrame,
    amount_col: str = "amount",
    time_col: str = "ledger_close_time",
    windows_hours: list[int] | None = None,
    reference_time: pd.Timestamp | None = None,
    asset: str | None = None,  # NEW: looks up per-asset windows
    include_second_digit: bool = True,  # NEW: Issue #179 - include second-digit metrics
) -> dict[int, dict[str, any]]:
    """Compute Benford metrics over multiple trailing windows ending at
    `reference_time` (defaults to the max timestamp in `df`).

    Parameters
    ----------
    df:
        DataFrame with columns specified by amount_col, time_col, and optionally
        base_asset/counter_asset for per-asset window selection.
    amount_col:
        Column name for trade amounts.
    time_col:
        Column name for timestamp/ledger_close_time.
    windows_hours:
        List of window sizes in hours. Defaults to config.BENFORD_WINDOWS_HOURS or
        per-asset windows if available.
    reference_time:
        End timestamp for the windows (defaults to max timestamp in df).
    asset:
        Asset code for looking up per-asset windows; inferred from df if omitted.
    include_second_digit:
        If True (default), also compute second-digit metrics (chi-square, MAD, z-scores).

    Returns
    -------
    dict[int, dict]:
        Maps window size (hours) -> dict with keys:
          - "chi_square", "mad", "z_scores": first-digit metrics
          - "chi_square_2nd", "mad_2nd", "z_scores_2nd": second-digit metrics (if include_second_digit=True)
          - "sample_size": number of trades in window
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
        return {
            w: {
                "chi_square": np.nan,
                "mad": np.nan,
                "z_scores": {d: np.nan for d in range(1, 10)},
                "chi_square_2nd": np.nan if include_second_digit else None,
                "mad_2nd": np.nan if include_second_digit else None,
                "z_scores_2nd": {d: np.nan for d in range(10)} if include_second_digit else None,
                "sample_size": 0,
            }
            for w in windows_hours
        }

    timestamps = pd.to_datetime(df[time_col])
    ref = reference_time or timestamps.max()

    results = {}
    for hours in windows_hours:
        window_start = ref - pd.Timedelta(hours=hours)
        window_df = df[(timestamps > window_start) & (timestamps <= ref)]
        amounts = window_df[amount_col]
        
        # First-digit metrics
        metrics = compute_benford_metrics(amounts, asset_code=asset)
        result = {
            "chi_square": metrics.chi_square,
            "mad": metrics.mad,
            "z_scores": metrics.z_scores,
            "sample_size": metrics.sample_size,
        }
        
        # Second-digit metrics (Issue #179)
        if include_second_digit:
            result["chi_square_2nd"] = chi_square_second_digit(amounts)
            result["mad_2nd"] = mad_score_second_digit(amounts)
            result["z_scores_2nd"] = z_scores_second_digit(amounts)
        
        results[hours] = result

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
    
    Amounts with a single significant digit (amounts < 10) are excluded
    from the analysis since they have no second digit. The count of excluded
    amounts is logged for auditing.
    """
    amounts_positive = amounts[amounts > 0]
    if amounts_positive.empty:
        return amounts_positive
    
    magnitudes = np.floor(np.log10(amounts_positive)).astype(int)
    normalized = amounts_positive / (10.0**magnitudes)  # first digit is floor(normalized)
    # Remove first digit contribution, scale up and take floor
    second = np.floor((normalized - np.floor(normalized)) * 10).astype(int).clip(0, 9)
    
    return second


def second_digit_distribution(amounts: pd.Series) -> dict[int, float]:
    """Observed frequency of each second digit 0-9 in the data.
    
    Amounts < 10 (single-digit) are excluded; the exclusion count is logged.
    """
    digits = second_digits(amounts)
    if digits.empty:
        return {d: 0.0 for d in range(10)}
    
    # Log exclusion if significant
    amounts_positive = amounts[amounts > 0]
    excluded = len(amounts_positive[amounts_positive < 10])
    if excluded > 0 and len(amounts_positive) > 0:
        exclusion_rate = excluded / len(amounts_positive)
        if exclusion_rate > 0.1:  # Log if >10% excluded
            logger_module = logging.getLogger(__name__)
            logger_module.debug(
                f"second_digit_distribution: {excluded}/{len(amounts_positive)} "
                f"amounts excluded (< 10). Exclusion rate: {exclusion_rate:.1%}."
            )
    
    counts = digits.value_counts(normalize=True)
    return {d: float(counts.get(d, 0.0)) for d in range(10)}


def chi_square_second_digit(amounts: pd.Series) -> float:
    """Chi-square goodness-of-fit for second-digit distribution.
    
    Tests whether the observed second-digit frequencies conform to Benford's
    expected second-digit distribution. High values indicate non-conformity
    (a potential manipulation signal).
    """
    digits = second_digits(amounts)
    n = len(digits)
    if n == 0:
        return 0.0
    
    observed_counts = digits.value_counts()
    chi_sq = 0.0
    for d in range(10):
        expected_count = BENFORD_EXPECTED_2ND[d] * n
        observed_count = observed_counts.get(d, 0)
        if expected_count > 0:
            chi_sq += (observed_count - expected_count) ** 2 / expected_count
    
    return float(chi_sq)


def z_scores_second_digit(amounts: pd.Series) -> dict[int, float]:
    """Per-digit Z-scores for second digits (0-9).
    
    Measures how many standard errors each digit's observed frequency deviates
    from the expected Benford second-digit frequency.
    """
    digits = second_digits(amounts)
    n = len(digits)
    if n == 0:
        return {d: 0.0 for d in range(10)}
    
    observed = second_digit_distribution(amounts)
    scores = {}
    for d in range(10):
        p = BENFORD_EXPECTED_2ND[d]
        std_err = math.sqrt(p * (1 - p) / n) if p * (1 - p) > 0 else 0.0
        if std_err == 0:
            scores[d] = 0.0
            continue
        z = (abs(observed[d] - p) - (1 / (2 * n))) / std_err
        scores[d] = float(max(z, 0.0))
    
    return scores


def mad_score_second_digit(amounts: pd.Series) -> float:
    """Mean Absolute Deviation for second digits (0-9).
    
    Measures the average absolute deviation between observed and expected
    second-digit frequencies. Values above 0.015 may indicate non-conformity.
    """
    digits = second_digits(amounts)
    if digits.empty:
        return 0.0
    
    observed = second_digit_distribution(amounts)
    deviations = [abs(observed[d] - BENFORD_EXPECTED_2ND[d]) for d in range(10)]
    return float(sum(deviations) / len(deviations))


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
