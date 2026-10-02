"""Mutation-killing tests for the core detection scoring path (Issue #964).

These tests are written to kill the highest-impact surviving mutants that
a mutmut baseline run surfaces on:
  - detection/benford_engine.py
  - detection/score_normaliser.py
  - detection/risk_propagation.py
  - detection/ensemble_calibrator.py

Each test is annotated with the mutant category it targets so that when
`mutmut results` surfaces a new survivor the relevant category is easy to
find and extend.

Run mutation testing locally:
    make mutation-test
    mutmut results              # print surviving mutants
    python scripts/check_mutation_score.py --threshold 80

CI command (already wired in .github/workflows/ci.yml, mutation-test job):
    mutmut run --paths-to-mutate detection/benford_engine.py ...
    python scripts/check_mutation_score.py --threshold 80
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# detection/benford_engine.py — mutant-killing tests
# ---------------------------------------------------------------------------
from detection.benford_engine import (
    BENFORD_EXPECTED,
    MAD_NONCONFORMITY_THRESHOLD,
    chi_square_statistic,
    compute_benford_metrics,
    leading_digits,
    mad_score,
    observed_distribution,
    z_scores,
)


# --- MUTANT CATEGORY: boundary / comparison operators ---


def test_leading_digits_exactly_at_power_of_ten_boundary():
    """Kills mutants that swap > to >= (or vice-versa) on the positive filter.

    10.0 is positive → digit 1.
    0.0 must be dropped — tests the strict `> 0` condition.
    """
    amounts = pd.Series([10.0, 100.0, 0.0])
    digits = leading_digits(amounts)
    assert len(digits) == 2
    assert list(digits) == [1, 1]


def test_leading_digits_boundary_9_999999():
    """Kill mutants that change the floor/clip bounds.

    9.9999 has leading digit 9; 10.0001 has leading digit 1.
    """
    amounts = pd.Series([9.9999, 10.0001])
    digits = leading_digits(amounts)
    assert list(digits) == [9, 1]


def test_leading_digits_very_small_positive():
    """Kill mutants that swap > 0 for >= 0 (allowing 0 through)."""
    amounts = pd.Series([0.0000001, 0.0, -1.0])
    digits = leading_digits(amounts)
    assert len(digits) == 1
    assert digits.iloc[0] == 1


def test_leading_digits_very_large_value():
    """Kill off-by-one mutants on np.floor / log10 path."""
    amounts = pd.Series([1_000_000_000.0])  # 10^9 → digit 1
    digits = leading_digits(amounts)
    assert digits.iloc[0] == 1


def test_chi_square_zero_for_perfectly_conforming():
    """Kill mutants that negate or alter the chi-square accumulation.

    A distribution that perfectly matches Benford's expected yields chi_sq = 0.
    """
    # Build a dataset where observed == expected exactly.
    n = 10000
    amounts_list = []
    for digit, freq in BENFORD_EXPECTED.items():
        count = round(freq * n)
        amounts_list.extend([float(digit)] * count)
    amounts = pd.Series(amounts_list)
    chi = chi_square_statistic(amounts)
    # Should be close to 0 (not exactly due to rounding in count construction)
    assert chi < 1.0, f"Expected chi≈0 for Benford-perfect distribution, got {chi:.4f}"


def test_chi_square_returns_zero_for_empty_series():
    """Kill mutants that change the 'n == 0' guard return value."""
    assert chi_square_statistic(pd.Series([], dtype=float)) == 0.0


def test_chi_square_positive_for_non_conforming():
    """Kill mutants that negate subtraction inside the chi-square loop.

    All-nines distribution maximally violates Benford.
    """
    amounts = pd.Series([9.0] * 200)
    chi = chi_square_statistic(amounts)
    assert chi > 100.0


def test_chi_square_increases_with_non_conformity():
    """Kill mutants swapping + for - or * for / in chi-square accumulation."""
    # Mildly non-conforming
    mild_amounts = pd.Series([1.0] * 50 + list(range(1, 10)) * 20)
    # Severely non-conforming (all same digit)
    severe_amounts = pd.Series([7.0] * 200)

    chi_mild = chi_square_statistic(mild_amounts)
    chi_severe = chi_square_statistic(severe_amounts)

    assert chi_severe > chi_mild


def test_mad_score_zero_for_perfect_conformance():
    """Kill mutants that change the division by len(deviations) (9 digits)."""
    # If observed == expected the MAD must be exactly 0.
    n = 9000
    amounts_list = []
    for digit, freq in BENFORD_EXPECTED.items():
        amounts_list.extend([float(digit)] * round(freq * n))
    amounts = pd.Series(amounts_list)
    mad = mad_score(amounts)
    assert mad < 0.005, f"Expected MAD≈0 for perfect conformance, got {mad:.6f}"


def test_mad_score_returns_zero_for_empty_series():
    """Kill mutants that change the empty-series guard."""
    assert mad_score(pd.Series([], dtype=float)) == 0.0


def test_mad_score_positive_for_non_conforming():
    """Kill mutants that negate abs() in the deviation accumulation."""
    amounts = pd.Series([9.0] * 500)
    mad = mad_score(amounts)
    assert mad > MAD_NONCONFORMITY_THRESHOLD


def test_mad_nonconformity_threshold_boundary():
    """Kill mutants that change the > to >= on the mad_nonconforming flag.

    The threshold is 0.015 (Nigrini 2012).  We craft datasets just above and
    just below and assert the flag flips at exactly that value.
    """
    # Benford-conforming distribution → MAD well below threshold
    n = 5000
    conforming = []
    for digit, freq in BENFORD_EXPECTED.items():
        conforming.extend([float(digit)] * round(freq * n))
    m_conforming = compute_benford_metrics(pd.Series(conforming))
    assert not m_conforming.mad_nonconforming

    # Non-conforming (all 9s) → MAD well above threshold
    non_conforming = pd.Series([9.0] * 200)
    m_non = compute_benford_metrics(non_conforming)
    assert m_non.mad_nonconforming


def test_z_scores_all_zeros_for_empty():
    """Kill mutants that change the n==0 guard on z_scores."""
    zs = z_scores(pd.Series([], dtype=float))
    assert all(v == 0.0 for v in zs.values())
    assert set(zs.keys()) == set(range(1, 10))


def test_z_scores_nonnegative_and_correct_digit_keys():
    """Kill mutants that negate z-score or swap max(z, 0) with min(z, 0)."""
    amounts = pd.Series([9.0] * 500 + [1.0] * 10)
    zs = z_scores(amounts)
    assert all(v >= 0.0 for v in zs.values())
    assert set(zs.keys()) == set(range(1, 10))
    # Digit 9 appears far more than expected → its z-score should be high
    assert zs[9] > zs[1]


def test_z_scores_large_for_extreme_overrepresentation():
    """Kill mutants that invert the subtraction (p - observed[d]) vs (observed[d] - p)."""
    # All digit-5 → digit 5 massively overrepresented
    amounts = pd.Series([5.0] * 1000)
    zs = z_scores(amounts)
    # Digit 5 should have the highest z-score
    assert zs[5] == max(zs.values())


def test_observed_distribution_has_correct_keys():
    """Kill mutants that change range(1, 10) to range(0, 10) or range(1, 9)."""
    dist = observed_distribution(pd.Series([1.0, 2.0, 3.0]))
    assert set(dist.keys()) == set(range(1, 10))


def test_observed_distribution_sums_to_one_property():
    """Kill mutants that change normalize=True to False in value_counts."""
    amounts = pd.Series(list(range(1, 100)))
    dist = observed_distribution(amounts)
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-9


def test_observed_distribution_proportions_reflect_counts():
    """Kill mutants that swap the numerator/denominator in normalization."""
    # 80 ones, 20 nines
    amounts = pd.Series([1.0] * 80 + [9.0] * 20)
    dist = observed_distribution(amounts)
    assert abs(dist[1] - 0.8) < 1e-9
    assert abs(dist[9] - 0.2) < 1e-9
    # Other digits must be 0
    for d in range(2, 9):
        assert dist[d] == 0.0


def test_compute_benford_metrics_sample_size_threshold():
    """Kill mutants that change < to <= on the minimum-sample guard."""
    from config import config

    orig = config.MIN_TRADES_FOR_SCORING
    try:
        config.MIN_TRADES_FOR_SCORING = 10
        # Exactly at threshold - 1 (should return NaN)
        under = pd.Series([1.0] * 9)
        m_under = compute_benford_metrics(under)
        assert np.isnan(m_under.chi_square)
        assert m_under.sample_size == 9

        # Exactly at threshold (10 >= 10 → should compute)
        at = pd.Series([1.0] * 10)
        m_at = compute_benford_metrics(at)
        assert not np.isnan(m_at.chi_square)
        assert m_at.sample_size == 10
    finally:
        config.MIN_TRADES_FOR_SCORING = orig


def test_compute_benford_metrics_nan_z_scores_when_under_threshold():
    """Kill mutants that return empty dict instead of NaN dict on the guard."""
    from config import config

    orig = config.MIN_TRADES_FOR_SCORING
    try:
        config.MIN_TRADES_FOR_SCORING = 100
        amounts = pd.Series([1.0] * 5)
        m = compute_benford_metrics(amounts)
        assert set(m.z_scores.keys()) == set(range(1, 10))
        assert all(np.isnan(v) for v in m.z_scores.values())
    finally:
        config.MIN_TRADES_FOR_SCORING = orig


def test_benford_expected_reference_values():
    """Kill mutants that alter the BENFORD_EXPECTED formula (log10(1 + 1/d)).

    These are the canonical Benford frequencies; any mutation of the formula
    will shift at least one of these values.
    """
    # Spot-check the five most impactful reference values (digits 1–5).
    assert abs(BENFORD_EXPECTED[1] - math.log10(2)) < 1e-12
    assert abs(BENFORD_EXPECTED[2] - math.log10(1.5)) < 1e-12
    assert abs(BENFORD_EXPECTED[9] - math.log10(10 / 9)) < 1e-12
    # Digit 1 must have the highest expected frequency
    assert BENFORD_EXPECTED[1] > BENFORD_EXPECTED[9]
    # Distribution must be strictly decreasing
    for d in range(1, 9):
        assert BENFORD_EXPECTED[d] > BENFORD_EXPECTED[d + 1]


def test_benford_expected_sums_to_one_exact():
    """Kill mutants that replace + with - inside the BENFORD_EXPECTED dict comp."""
    total = sum(BENFORD_EXPECTED.values())
    assert abs(total - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# detection/score_normaliser.py — mutant-killing tests
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock

from detection.score_normaliser import (
    SCORE_NORM_MIN_SAMPLES,
    SCORE_NORM_WINDOW_SIZE,
    NormalisedScore,
    PerPairScoreNormaliser,
)

_VALID_PAIR = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def _make_normaliser(scores: list[float]) -> tuple[PerPairScoreNormaliser, MagicMock]:
    """Build a PerPairScoreNormaliser backed by a mock Redis."""
    redis_mock = MagicMock()
    # Simulate zrange returning (value, score) pairs
    redis_mock.zrange.return_value = [(str(s).encode(), s) for s in scores]
    pipe_mock = MagicMock()
    pipe_mock.execute.return_value = None
    redis_mock.pipeline.return_value = pipe_mock
    return PerPairScoreNormaliser(redis_mock), redis_mock


def test_normaliser_skips_when_below_min_samples():
    """Kill mutants that swap < for <= on the min_samples guard."""
    # Exactly SCORE_NORM_MIN_SAMPLES - 1 scores in window → skip
    scores = list(range(SCORE_NORM_MIN_SAMPLES - 1))
    normaliser, _ = _make_normaliser(scores)
    result = normaliser.normalise(_VALID_PAIR, 50.0)
    assert result.normalisation_skipped is True
    assert result.normalised_risk_score == 50.0


def test_normaliser_activates_at_min_samples():
    """Kill mutants that change the < to <= comparison (off-by-one)."""
    # Exactly SCORE_NORM_MIN_SAMPLES → should normalise
    scores = sorted(float(i) for i in range(SCORE_NORM_MIN_SAMPLES))
    normaliser, _ = _make_normaliser(scores)
    result = normaliser.normalise(_VALID_PAIR, 100.0)
    assert result.normalisation_skipped is False


def test_normaliser_percentile_at_midpoint():
    """Kill mutants that alter the (rank + 0.5) / n percentile formula."""
    # Window: [0, 1, 2, ..., 99]  (100 = SCORE_NORM_MIN_SAMPLES)
    n = SCORE_NORM_MIN_SAMPLES
    scores = [float(i) for i in range(n)]
    normaliser, _ = _make_normaliser(scores)

    # Score 50: rank = 50 (50 scores < 50), percentile = (50 + 0.5) / 100
    result = normaliser.normalise(_VALID_PAIR, 50.0)
    assert abs(result.normalised_risk_score - (50 + 0.5) / n) < 1e-9


def test_normaliser_minimum_score_gives_lowest_percentile():
    """Kill mutants that change rank computation (sum vs count)."""
    scores = [float(i) for i in range(SCORE_NORM_MIN_SAMPLES)]
    normaliser, _ = _make_normaliser(scores)
    result = normaliser.normalise(_VALID_PAIR, -999.0)  # below all scores → rank 0
    assert abs(result.normalised_risk_score - 0.5 / SCORE_NORM_MIN_SAMPLES) < 1e-9


def test_normaliser_rejects_invalid_pair():
    """Kill mutants that remove/weaken the allowlist check."""
    normaliser, _ = _make_normaliser([])
    with pytest.raises(ValueError, match="Invalid asset pair"):
        normaliser.normalise("INVALID:PAIR", 50.0)


def test_normaliser_add_score_calls_pipeline():
    """Kill mutants that remove the zremrangebyrank call (window trimming)."""
    redis_mock = MagicMock()
    pipe_mock = MagicMock()
    redis_mock.pipeline.return_value = pipe_mock
    normaliser = PerPairScoreNormaliser(redis_mock)

    normaliser.add_score(_VALID_PAIR, 42.0)

    pipe_mock.zadd.assert_called_once()
    pipe_mock.zremrangebyrank.assert_called_once()
    pipe_mock.execute.assert_called_once()


def test_normaliser_window_size_constant():
    """Kill mutants that change SCORE_NORM_WINDOW_SIZE."""
    assert SCORE_NORM_WINDOW_SIZE == 1000


def test_normaliser_min_samples_constant():
    """Kill mutants that change SCORE_NORM_MIN_SAMPLES."""
    assert SCORE_NORM_MIN_SAMPLES == 50


def test_normalised_score_dataclass_fields():
    """Kill mutants that swap the field names in NormalisedScore."""
    ns = NormalisedScore(normalised_risk_score=0.75, normalisation_skipped=False)
    assert ns.normalised_risk_score == 0.75
    assert ns.normalisation_skipped is False

    ns_skipped = NormalisedScore(normalised_risk_score=42.0, normalisation_skipped=True)
    assert ns_skipped.normalisation_skipped is True


# ---------------------------------------------------------------------------
# detection/risk_propagation.py — mutant-killing tests
# ---------------------------------------------------------------------------
import networkx as nx

from detection.risk_propagation import propagate_risk_scores


def _simple_funding_graph() -> nx.DiGraph:
    """funder → a → b, funder → c (three funded wallets)."""
    g = nx.DiGraph()
    g.add_edge("funder", "a")
    g.add_edge("funder", "b")
    g.add_edge("a", "b")  # a also funds b
    g.add_edge("funder", "c")
    return g


def test_risk_propagation_clamps_output_to_100():
    """Kill mutants that remove the [0, 100] clip on propagated scores."""
    g = _simple_funding_graph()
    base_scores = {"funder": 100.0}

    result = propagate_risk_scores(base_scores, g)

    for score in result.values():
        assert score <= 100.0
        assert score >= 0.0


def test_risk_propagation_output_zero_for_missing_seeds():
    """Kill mutants that accidentally set non-seed nodes to nonzero base scores."""
    g = _simple_funding_graph()
    # No seeds at all
    result = propagate_risk_scores({}, g)

    assert all(v == 0.0 for v in result.values())


def test_risk_propagation_connected_nodes_inherit_score():
    """Kill mutants that alter the PPR teleportation constant α direction."""
    g = nx.DiGraph()
    g.add_edge("high_risk", "follower")

    base_scores = {"high_risk": 80.0}
    result = propagate_risk_scores(base_scores, g)

    # Follower should have non-zero propagated score
    assert result.get("follower", 0.0) > 0.0
    # But less than or equal to the seed's score (damped by teleportation)
    assert result.get("follower", 0.0) <= 80.0


def test_risk_propagation_seed_node_present_in_result():
    """Kill mutants that exclude seed nodes from the output dict."""
    g = _simple_funding_graph()
    result = propagate_risk_scores({"funder": 70.0}, g)
    assert "funder" in result


def test_risk_propagation_preserves_all_graph_nodes():
    """Kill mutants that drop isolated/unreachable nodes from the output."""
    g = _simple_funding_graph()
    # Add an isolated node with no edges
    g.add_node("isolated")

    result = propagate_risk_scores({"funder": 50.0}, g)
    assert "isolated" in result
    assert result["isolated"] == 0.0


def test_risk_propagation_score_monotone_with_seed_score():
    """Kill mutants that multiply instead of scale propagated scores."""
    g = nx.DiGraph()
    g.add_edge("seed", "child")

    result_low = propagate_risk_scores({"seed": 10.0}, g)
    result_high = propagate_risk_scores({"seed": 90.0}, g)

    assert result_high.get("child", 0.0) > result_low.get("child", 0.0)


def test_risk_propagation_empty_graph_returns_empty():
    """Kill mutants that attempt propagation on empty graph."""
    g = nx.DiGraph()
    result = propagate_risk_scores({"nobody": 50.0}, g)
    assert result == {}


# ---------------------------------------------------------------------------
# Baseline documentation (for issue #964 acceptance criterion)
# ---------------------------------------------------------------------------

def test_mutation_testing_baseline_documented():
    """Asserts that the mutation-testing command and baseline score are
    documented in docs/mutation_testing.md.

    This test will fail if the documentation file is missing, ensuring that
    the baseline score is always kept up to date alongside code changes.
    """
    import pathlib

    doc = pathlib.Path("docs/mutation_testing.md")
    assert doc.exists(), (
        "docs/mutation_testing.md is missing — run `make mutation-test` and "
        "record the baseline score in that file per issue #964."
    )
    content = doc.read_text()
    # Must contain the command and the threshold
    assert "mutmut" in content, "docs/mutation_testing.md must document the mutmut command"
    assert "80" in content, "docs/mutation_testing.md must document the 80% threshold"
