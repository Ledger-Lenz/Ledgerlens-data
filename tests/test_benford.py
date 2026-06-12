import numpy as np
import pandas as pd

from detection.benford_engine import (
    BENFORD_EXPECTED,
    chi_square_statistic,
    leading_digits,
    mad_score,
    observed_distribution,
    z_scores,
)


def test_leading_digits_basic():
    amounts = pd.Series([123, 0.045, 9876, 1])
    digits = leading_digits(amounts)
    assert list(digits) == [1, 4, 9, 1]


def test_leading_digits_drops_nonpositive():
    amounts = pd.Series([0, -5, 100])
    digits = leading_digits(amounts)
    assert list(digits) == [1]


def test_benford_conforming_sample_has_low_mad():
    # Generate a large sample from a log-uniform distribution, which
    # conforms closely to Benford's Law.
    rng = np.random.default_rng(42)
    amounts = pd.Series(10 ** rng.uniform(0, 4, size=20000))

    assert mad_score(amounts) < 0.015


def test_benford_round_numbers_are_nonconforming():
    # Wash-trading-style fixed lot sizes concentrated on digit 5.
    amounts = pd.Series([500] * 100 + [5000] * 100 + [50000] * 100)

    assert mad_score(amounts) > 0.015
    assert chi_square_statistic(amounts) > 0


def test_observed_distribution_sums_to_one():
    amounts = pd.Series(np.arange(1, 1000))
    dist = observed_distribution(amounts)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_z_scores_nonnegative():
    amounts = pd.Series([111, 222, 333, 444])
    scores = z_scores(amounts)
    assert all(v >= 0 for v in scores.values())


def test_benford_expected_sums_to_one():
    assert abs(sum(BENFORD_EXPECTED.values()) - 1.0) < 1e-9
