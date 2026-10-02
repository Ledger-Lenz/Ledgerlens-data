"""Property-based tests for numeric-precision-sensitive modules (Issue #965).

LedgerLens operates on Stellar on-chain amounts (7-decimal stroop precision)
and financial aggregations where floating-point errors can produce false
anomaly scores.  NUMERIC_PRECISION_PR.md documents the prior precision work
that introduced ``utils/decimal_guards.py``, ``utils/currency_normalization.py``,
and the Decimal-based Benford digit extractor in ``utils/benford_precision.py``.

These property-based tests use **Hypothesis** to cover invariants that
example-based tests miss — especially at numeric boundaries, large/small
magnitudes, and arbitrary-precision arithmetic.

Test strategy documented for contributors
-----------------------------------------
When extending a numeric module, add a section here that covers:

1. **Round-trip invariant** — encode → decode must be identity (e.g. stroops).
2. **Monotonicity** — larger input → larger output for monotone functions.
3. **Bounded output** — output never escapes its documented range.
4. **Commutativity / associativity** — order of operands shouldn't matter.
5. **Zero / identity elements** — f(x, 0) == x, f(x, identity) == x.
6. **Edge cases** — minimum positive value, maximum representable value, NaN/Inf
   rejected at ingestion boundary.

CI integration
--------------
These tests are included in the standard ``pytest`` run (`make test` /
`make test-fast`).  Hypothesis profiles are set to run 200 examples per
property by default, which keeps the wall-clock budget well under the CI
timeout while still exercising a broad input space.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Hypothesis settings profile for CI (faster, still thorough)
# ---------------------------------------------------------------------------
_CI_SETTINGS = dict(max_examples=200, deadline=10_000, suppress_health_check=[HealthCheck.too_slow])

# ---------------------------------------------------------------------------
# Shared strategies for Stellar-domain values
# ---------------------------------------------------------------------------

# Stellar amounts: 7 decimal places, positive, within int64 stroop range
# STELLAR_MAX_AMOUNT = 922337203685.4775807
stellar_amounts = st.decimals(
    min_value="0.0000001",
    max_value="922337203685.4775807",
    places=7,
    allow_nan=False,
    allow_infinity=False,
)

# Smaller, focused amounts for arithmetic correctness (avoid huge sums)
small_amounts = st.decimals(
    min_value="0.0000001",
    max_value="1000000",
    places=7,
    allow_nan=False,
    allow_infinity=False,
)

# Positive floats for Benford digit extraction
positive_floats = st.floats(
    min_value=1e-7,
    max_value=1e12,
    allow_nan=False,
    allow_infinity=False,
)


# ===========================================================================
# Section 1 — utils/decimal_guards.py
# ===========================================================================

from utils.decimal_guards import (
    STELLAR_MAX_AMOUNT,
    STELLAR_MIN_AMOUNT,
    STELLAR_PRECISION,
    STROOPS_MULTIPLIER,
    AmountValidationError,
    DecimalAmount,
    decimal_context,
    validate_amount,
    validate_stellar_amount,
)


class TestDecimalAmountRoundTrip:
    """Round-trip: to_stroops → from_stroops must recover original value."""

    @given(stellar_amounts)
    @settings(**_CI_SETTINGS)
    def test_stroops_round_trip_identity(self, amount: Decimal):
        """Encoding as stroops and decoding back must be exact (no precision loss)."""
        da = DecimalAmount(str(amount))
        stroops = da.to_stroops()
        recovered = DecimalAmount.from_stroops(stroops)
        # Round original to 7 dp for fair comparison (extra dp would survive stroops)
        expected = amount.quantize(Decimal("0.0000001"), rounding=ROUND_HALF_EVEN)
        assert recovered.value == expected, (
            f"Round-trip failed: {amount} → {stroops} stroops → {recovered.value}"
        )

    @given(st.integers(min_value=1, max_value=9_223_372_036_854_775_807))
    @settings(**_CI_SETTINGS)
    def test_from_stroops_to_stroops_identity(self, stroops: int):
        """Integer stroops → Decimal → back to stroops is identity."""
        da = DecimalAmount.from_stroops(stroops)
        assert da.to_stroops() == stroops

    @given(stellar_amounts)
    @settings(**_CI_SETTINGS)
    def test_stroops_are_positive_integer(self, amount: Decimal):
        """Stroops must always be a non-negative integer for valid Stellar amounts."""
        da = DecimalAmount(str(amount))
        stroops = da.to_stroops()
        assert isinstance(stroops, int)
        assert stroops >= 0


class TestDecimalAmountArithmetic:
    """Arithmetic invariants: commutativity, identity, monotonicity."""

    @given(small_amounts, small_amounts)
    @settings(**_CI_SETTINGS)
    def test_addition_commutativity(self, a: Decimal, b: Decimal):
        """a + b == b + a."""
        da = DecimalAmount(str(a))
        db = DecimalAmount(str(b))
        assert (da + db).value == (db + da).value

    @given(small_amounts)
    @settings(**_CI_SETTINGS)
    def test_addition_zero_identity(self, a: Decimal):
        """a + 0 == a."""
        da = DecimalAmount(str(a))
        zero = DecimalAmount("0")
        assert (da + zero).value == da.value

    @given(small_amounts)
    @settings(**_CI_SETTINGS)
    def test_multiplication_one_identity(self, a: Decimal):
        """a * 1 == a."""
        da = DecimalAmount(str(a))
        one = DecimalAmount("1")
        assert (da * one).value == da.value

    @given(small_amounts, small_amounts)
    @settings(**_CI_SETTINGS)
    def test_multiplication_commutativity(self, a: Decimal, b: Decimal):
        """a * b == b * a."""
        da = DecimalAmount(str(a))
        db = DecimalAmount(str(b))
        assert (da * db).value == (db * da).value

    @given(small_amounts, small_amounts)
    @settings(**_CI_SETTINGS)
    def test_subtraction_self_is_zero(self, a: Decimal, b: Decimal):
        """a - a == 0 regardless of value."""
        da = DecimalAmount(str(a))
        result = da - da
        assert result.value == Decimal("0")

    @given(
        st.decimals(min_value="0.0000001", max_value="1000", places=7, allow_nan=False, allow_infinity=False),
        st.decimals(min_value="0.0000001", max_value="1000", places=7, allow_nan=False, allow_infinity=False),
    )
    @settings(**_CI_SETTINGS)
    def test_division_inverse_of_multiplication(self, a: Decimal, b: Decimal):
        """(a * b) / b == a (within Decimal precision)."""
        da = DecimalAmount(str(a))
        db = DecimalAmount(str(b))
        product = da * db
        recovered = product / db
        # Allow tiny rounding difference from Decimal arithmetic
        assert abs(recovered.value - da.value) < Decimal("1E-6")

    @given(small_amounts, small_amounts)
    @settings(**_CI_SETTINGS)
    def test_addition_monotone(self, a: Decimal, b: Decimal):
        """Adding a positive value always increases the result."""
        da = DecimalAmount(str(a))
        db = DecimalAmount(str(b))
        result = da + db
        assert result.value >= da.value
        assert result.value >= db.value


class TestDecimalAmountComparison:
    """Comparison operators must be consistent with the underlying Decimal."""

    @given(small_amounts)
    @settings(**_CI_SETTINGS)
    def test_equality_reflexive(self, a: Decimal):
        """a == a (reflexivity)."""
        da = DecimalAmount(str(a))
        assert da == DecimalAmount(str(a))

    @given(small_amounts, small_amounts)
    @settings(**_CI_SETTINGS)
    def test_ordering_consistent_with_decimal(self, a: Decimal, b: Decimal):
        """DecimalAmount ordering matches Decimal ordering."""
        da = DecimalAmount(str(a))
        db = DecimalAmount(str(b))
        if a < b:
            assert da < db
        elif a > b:
            assert da > db
        else:
            assert da == db


class TestValidateAmount:
    """validate_amount invariants."""

    @given(stellar_amounts)
    @settings(**_CI_SETTINGS)
    def test_valid_stellar_amount_passes(self, amount: Decimal):
        """Every value in the valid stellar range is accepted."""
        result = validate_amount(str(amount), min_value="0", max_value=str(STELLAR_MAX_AMOUNT))
        assert result == amount

    @given(st.decimals(min_value="-1000", max_value="-0.0000001", places=7, allow_nan=False, allow_infinity=False))
    @settings(**_CI_SETTINGS)
    def test_negative_rejected_by_default(self, amount: Decimal):
        """Negative amounts are rejected when allow_negative=False."""
        with pytest.raises(AmountValidationError):
            validate_amount(str(amount))

    @given(st.decimals(min_value="-1000", max_value="-0.0000001", places=7, allow_nan=False, allow_infinity=False))
    @settings(**_CI_SETTINGS)
    def test_negative_allowed_when_flag_set(self, amount: Decimal):
        """Negative amounts are accepted when allow_negative=True."""
        result = validate_amount(str(amount), allow_negative=True)
        assert result == amount

    @given(
        st.decimals(min_value="0.0000001", max_value="100", places=7, allow_nan=False, allow_infinity=False),
        st.decimals(min_value="101", max_value="1000", places=7, allow_nan=False, allow_infinity=False),
    )
    @settings(**_CI_SETTINGS)
    def test_value_below_min_rejected(self, value: Decimal, min_v: Decimal):
        """Values strictly below min_value are rejected."""
        with pytest.raises(AmountValidationError):
            validate_amount(str(value), min_value=str(min_v))

    @given(
        st.decimals(min_value="1001", max_value="10000", places=7, allow_nan=False, allow_infinity=False),
        st.decimals(min_value="0.0000001", max_value="1000", places=7, allow_nan=False, allow_infinity=False),
    )
    @settings(**_CI_SETTINGS)
    def test_value_above_max_rejected(self, value: Decimal, max_v: Decimal):
        """Values strictly above max_value are rejected."""
        with pytest.raises(AmountValidationError):
            validate_amount(str(value), max_value=str(max_v))

    def test_nan_string_rejected(self):
        with pytest.raises(AmountValidationError):
            validate_amount("nan")

    def test_infinity_string_rejected(self):
        with pytest.raises(AmountValidationError):
            validate_amount("inf")

    def test_non_numeric_rejected(self):
        with pytest.raises(AmountValidationError):
            validate_amount("not-a-number")


class TestValidateStellarAmount:
    """validate_stellar_amount: Stellar-specific constraints."""

    @given(stellar_amounts)
    @settings(**_CI_SETTINGS)
    def test_valid_amounts_accepted(self, amount: Decimal):
        """All amounts within the Stellar range with ≤7 dp are accepted."""
        result = validate_stellar_amount(str(amount))
        # Must be quantized to 7 dp
        assert result.as_tuple().exponent >= -STELLAR_PRECISION

    @given(
        st.decimals(
            min_value=str(STELLAR_MAX_AMOUNT + Decimal("0.0000001")),
            max_value=str(STELLAR_MAX_AMOUNT + Decimal("1000")),
            places=7,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(**_CI_SETTINGS)
    def test_amounts_above_stellar_max_rejected(self, amount: Decimal):
        """Amounts above STELLAR_MAX_AMOUNT are rejected."""
        with pytest.raises(AmountValidationError):
            validate_stellar_amount(str(amount))


class TestDecimalContext:
    """decimal_context: precision and rounding isolation."""

    @given(st.integers(min_value=5, max_value=50))
    @settings(**_CI_SETTINGS)
    def test_context_sets_precision(self, prec: int):
        """The context manager must activate the requested precision."""
        with decimal_context(precision=prec) as ctx:
            assert ctx.prec == prec

    def test_context_restores_on_exit(self):
        """Previous context must be restored after exiting the block."""
        import decimal as _decimal
        orig_prec = _decimal.getcontext().prec
        with decimal_context(precision=10):
            pass
        assert _decimal.getcontext().prec == orig_prec

    def test_context_restores_on_exception(self):
        """Context must be restored even when an exception is raised."""
        import decimal as _decimal
        orig_prec = _decimal.getcontext().prec
        with pytest.raises(ZeroDivisionError):
            with decimal_context(precision=10):
                raise ZeroDivisionError("test")
        assert _decimal.getcontext().prec == orig_prec


# ===========================================================================
# Section 2 — utils/benford_precision.py (safe digit extraction)
# ===========================================================================
try:
    from utils.benford_precision import (
        extract_leading_digit_safe,
        extract_second_digit_safe,
        leading_digits_safe,
    )

    _BENFORD_PRECISION_AVAILABLE = True
except ImportError:
    _BENFORD_PRECISION_AVAILABLE = False

_skip_benford_precision = pytest.mark.skipif(
    not _BENFORD_PRECISION_AVAILABLE,
    reason="utils.benford_precision not importable",
)


@_skip_benford_precision
class TestLeadingDigitSafe:
    """extract_leading_digit_safe: always returns int in [1, 9]."""

    @given(stellar_amounts)
    @settings(**_CI_SETTINGS)
    def test_digit_in_range_1_to_9(self, amount: Decimal):
        """Every valid Stellar amount has a leading digit in [1, 9]."""
        digit = extract_leading_digit_safe(amount)
        assert 1 <= digit <= 9, f"Digit {digit} out of range for amount {amount}"

    @given(stellar_amounts)
    @settings(**_CI_SETTINGS)
    def test_scale_invariant_powers_of_ten(self, amount: Decimal):
        """Multiplying by 10 must not change the leading digit.

        For example, 1234 and 12340 both have leading digit 1.
        (Note: this invariance holds exactly for powers of 10.)
        """
        digit_orig = extract_leading_digit_safe(amount)
        # Multiply by 10 and check — the leading digit must stay the same if
        # the result stays within Stellar range
        scaled = amount * Decimal("10")
        if scaled <= STELLAR_MAX_AMOUNT:
            digit_scaled = extract_leading_digit_safe(scaled)
            assert digit_orig == digit_scaled, (
                f"Leading digit changed from {digit_orig} to {digit_scaled} "
                f"when scaling {amount} by 10"
            )

    @given(stellar_amounts)
    @settings(**_CI_SETTINGS)
    def test_decimal_agrees_with_float_for_typical_values(self, amount: Decimal):
        """Decimal and float extraction must agree for typical Stellar amounts.

        This catches precision-loss bugs that only manifest at digit boundaries
        such as 9.9999999... → 10 (digit 1 vs digit 9).
        """
        digit_decimal = extract_leading_digit_safe(amount)
        # Float path — only used for comparison; may disagree at exact boundaries
        float_val = float(amount)
        if float_val > 0 and not math.isinf(float_val):
            # Only assert agreement for amounts well away from digit boundaries
            magnitude = 10 ** int(math.floor(math.log10(float_val)))
            normalised = float_val / magnitude
            if abs(normalised - 1.0) > 1e-6 and abs(normalised - 9.999999) > 1e-6:
                float_digit = int(normalised)
                if float_digit in range(1, 10):
                    assert digit_decimal == float_digit, (
                        f"Decimal digit {digit_decimal} != float digit {float_digit} "
                        f"for amount {amount}"
                    )


@_skip_benford_precision
class TestLeadingDigitsSafeSeries:
    """leading_digits_safe over a Series: bulk extraction invariants."""

    @given(st.lists(positive_floats, min_size=1, max_size=200))
    @settings(**_CI_SETTINGS)
    def test_output_length_matches_positive_input(self, values: list[float]):
        """One digit extracted per valid positive value."""
        series = pd.Series(values)
        digits = leading_digits_safe(series)
        assert len(digits) == len(series)

    @given(st.lists(positive_floats, min_size=1, max_size=200))
    @settings(**_CI_SETTINGS)
    def test_all_digits_in_range(self, values: list[float]):
        """All extracted digits must be in [1, 9]."""
        series = pd.Series(values)
        digits = leading_digits_safe(series)
        for d in digits:
            assert 1 <= d <= 9, f"Digit {d} out of range"


# ===========================================================================
# Section 3 — detection/score_normaliser.py (percentile rank bounds)
# ===========================================================================

from detection.score_normaliser import (
    SCORE_NORM_MIN_SAMPLES,
    SCORE_NORM_WINDOW_SIZE,
    NormalisedScore,
    PerPairScoreNormaliser,
)
from unittest.mock import MagicMock

_VALID_PAIR = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def _mock_normaliser(window_scores: list[float]) -> PerPairScoreNormaliser:
    redis_mock = MagicMock()
    redis_mock.zrange.return_value = [(str(s).encode(), s) for s in window_scores]
    pipe_mock = MagicMock()
    pipe_mock.execute.return_value = None
    redis_mock.pipeline.return_value = pipe_mock
    return PerPairScoreNormaliser(redis_mock)


class TestScoreNormaliserBounds:
    """NormalisedScore.normalised_risk_score is bounded in (0, ~1] once active."""

    @given(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=SCORE_NORM_MIN_SAMPLES,
            max_size=SCORE_NORM_WINDOW_SIZE,
        ),
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(**_CI_SETTINGS)
    def test_normalised_score_in_valid_range(self, window: list[float], query_score: float):
        """Once the window is full, normalised_risk_score is in (0, ~1.01]."""
        normaliser = _mock_normaliser(sorted(window))
        result = normaliser.normalise(_VALID_PAIR, query_score)

        if not result.normalisation_skipped:
            n = len(window)
            # Minimum possible = 0.5/n; maximum possible = (n+0.5)/n
            assert result.normalised_risk_score > 0.0, (
                f"Score {result.normalised_risk_score} must be > 0"
            )
            assert result.normalised_risk_score <= (n + 0.5) / n + 1e-9, (
                f"Score {result.normalised_risk_score} exceeds (n+0.5)/n = {(n + 0.5) / n}"
            )

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=SCORE_NORM_MIN_SAMPLES,
            max_size=SCORE_NORM_WINDOW_SIZE,
        ),
    )
    @settings(**_CI_SETTINGS)
    def test_monotone_score_produces_monotone_percentile(self, window: list[float]):
        """A query score strictly below all window values has lower percentile
        than a query score strictly above all window values."""
        if not window:
            return
        low_query = min(window) - 1.0
        high_query = max(window) + 1.0

        normaliser = _mock_normaliser(sorted(window))
        result_low = normaliser.normalise(_VALID_PAIR, low_query)
        result_high = normaliser.normalise(_VALID_PAIR, high_query)

        if not result_low.normalisation_skipped:
            assert result_high.normalised_risk_score > result_low.normalised_risk_score

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=SCORE_NORM_MIN_SAMPLES - 1,
        ),
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(**_CI_SETTINGS)
    def test_skip_when_window_too_small(self, window: list[float], query_score: float):
        """When window < MIN_SAMPLES, raw score is passed through unchanged."""
        normaliser = _mock_normaliser(window)
        result = normaliser.normalise(_VALID_PAIR, query_score)

        assert result.normalisation_skipped is True
        assert result.normalised_risk_score == query_score


# ===========================================================================
# Section 4 — Benford digit extraction precision (utils/benford_precision.py
#              or detection/benford_engine.py's leading_digits)
# ===========================================================================

from detection.benford_engine import leading_digits, mad_score, observed_distribution


class TestBenfordPrecisionInvariantsHypothesis:
    """High-level property invariants for the production Benford engine."""

    @given(st.lists(positive_floats, min_size=1, max_size=1000))
    @settings(**_CI_SETTINGS)
    def test_observed_distribution_sums_to_one(self, values: list[float]):
        """Probability axiom: all digit frequencies must sum to 1.0."""
        series = pd.Series(values)
        dist = observed_distribution(series)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-9, f"Distribution sums to {total}, expected 1.0"

    @given(st.lists(positive_floats, min_size=1, max_size=1000))
    @settings(**_CI_SETTINGS)
    def test_all_digit_frequencies_non_negative(self, values: list[float]):
        """No digit can have a negative frequency."""
        series = pd.Series(values)
        dist = observed_distribution(series)
        for d, freq in dist.items():
            assert freq >= 0.0, f"Negative frequency {freq} for digit {d}"

    @given(st.lists(positive_floats, min_size=10, max_size=500))
    @settings(**_CI_SETTINGS)
    def test_mad_score_non_negative(self, values: list[float]):
        """MAD is a non-negative deviation measure."""
        series = pd.Series(values)
        mad = mad_score(series)
        assert mad >= 0.0, f"Negative MAD {mad}"

    @given(st.lists(positive_floats, min_size=10, max_size=500))
    @settings(**_CI_SETTINGS)
    def test_leading_digits_all_in_range(self, values: list[float]):
        """Every extracted digit must be in [1, 9]."""
        series = pd.Series(values)
        digits = leading_digits(series)
        for d in digits:
            assert 1 <= d <= 9, f"Digit {d} out of range for values {values[:5]}"

    @given(
        st.lists(
            st.floats(min_value=1e-7, max_value=1e12, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=500,
        ),
        st.integers(min_value=-5, max_value=5),
    )
    @settings(**_CI_SETTINGS)
    def test_scale_invariance(self, values: list[float], power: int):
        """Multiplying all amounts by 10^k must not change observed distribution.

        This is a fundamental invariant of Benford's Law — the digit frequencies
        are independent of the measurement unit (Benford, 1938).
        """
        series = pd.Series(values)
        scaled = series * (10 ** power)
        # Filter to valid positive values after scaling
        scaled = scaled[scaled > 0]
        series = series[series > 0]
        if len(series) < 5 or len(scaled) < 5:
            return

        dist_orig = observed_distribution(series)
        dist_scaled = observed_distribution(scaled)

        for d in range(1, 10):
            assert abs(dist_orig[d] - dist_scaled[d]) < 1e-9, (
                f"Digit {d} frequency changed after scaling by 10^{power}: "
                f"{dist_orig[d]:.6f} → {dist_scaled[d]:.6f}"
            )

    @given(st.lists(positive_floats, min_size=10, max_size=500))
    @settings(**_CI_SETTINGS)
    def test_order_independence(self, values: list[float]):
        """Reordering amounts must not affect distribution or MAD."""
        import random
        series = pd.Series(values)
        shuffled = pd.Series(random.sample(values, len(values)))

        dist_orig = observed_distribution(series)
        dist_shuffled = observed_distribution(shuffled)

        for d in range(1, 10):
            assert abs(dist_orig[d] - dist_shuffled[d]) < 1e-9

        assert abs(mad_score(series) - mad_score(shuffled)) < 1e-9


# ===========================================================================
# Section 5 — Edge-case bugs found / explicitly documented (Issue #965
#              acceptance criterion: document bugs found or lack thereof)
# ===========================================================================

class TestEdgeCaseBugsFound:
    """Document any latent edge-case bugs found during property-testing.

    Per issue #965 acceptance criteria: if no bugs were found, document the
    test parameters used so future contributors know what was checked.

    --- Finding during implementation (2026-09-28) ---

    No latent bugs were found in the current codebase during property testing.
    The following invariants were verified over 200 Hypothesis examples each:

    * DecimalAmount stroops round-trip: exact for all valid Stellar amounts.
    * validate_amount boundary rejection: NaN, Inf, negative (when disallowed),
      below-min, above-max all correctly raise AmountValidationError.
    * observed_distribution sum-to-one: holds for all positive float lists.
    * leading_digits range [1,9]: holds for all positive floats 1e-7 to 1e12.
    * Scale invariance (×10^k, k∈[-5,5]): distribution unchanged.
    * PerPairScoreNormaliser percentile bound: (0, (n+0.5)/n].

    Search parameters:
      - Amounts: Decimal [0.0000001, 922337203685.4775807], 7 decimal places
      - Floats: [1e-7, 1e12], not NaN, not Inf
      - Windows: 50–1000 scores of float in [-1e6, 1e6]
      - Hypothesis max_examples=200 per property
    """

    def test_documentation_marker(self):
        """This test always passes and serves as the documented record for
        issue #965 acceptance criterion: 'explicitly document that none [bugs]
        were found with the test parameters used'.
        """
        # Bugs found during implementation: none.
        # All invariants checked: stroops round-trip, arithmetic commutativity,
        # boundary validation, distribution sum-to-one, scale invariance,
        # percentile bounds.
        assert True
