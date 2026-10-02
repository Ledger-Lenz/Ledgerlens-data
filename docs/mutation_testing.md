# Mutation Testing — Issue #964

LedgerLens uses **mutmut** to measure how well the test suite catches real logic
errors in the core detection scoring path.  This document records the baseline
mutation score and explains how to run, interpret, and improve it.

## What is Mutation Testing?

Mutation testing automatically injects small code changes (mutants) — for
example, replacing `>` with `>=`, `+` with `-`, or `True` with `False` — and
checks whether the test suite detects each change.  A mutant that the tests
*do not* catch is called a **surviving mutant** and indicates a gap in test
coverage.

The **mutation score** is:

```
killed / (killed + survived) × 100
```

A score of 100 % means every injected defect was caught.

## Baseline Score

| Module | Mutation Score | Survived | Killed | Date |
|---|---|---|---|---|
| `detection/benford_engine.py` | ≥ 80 % | TBD | TBD | recorded on first CI run |

> Run `make mutation-test` to populate `.mutmut-cache` and then
> `python scripts/check_mutation_score.py --threshold 80` to print the current
> score against the baseline.

## Running Mutation Tests

### Quick run (CI scope)

```bash
# Enforces ≥80% threshold on benford_engine.py — mirrors the CI job exactly
make mutation-test
```

### Extended run (all core scoring modules)

```bash
mutmut run \
  --paths-to-mutate "detection/benford_engine.py,detection/score_normaliser.py,detection/risk_propagation.py,detection/ensemble_calibrator.py" \
  --runner "python -m pytest -x -q --timeout=30 -m 'not integration and not slow' \
    tests/test_benford.py \
    tests/test_benford_ci.py \
    tests/test_mutation_coverage.py" \
  --no-progress || true

mutmut results
python scripts/check_mutation_score.py --threshold 80
```

### Threshold override (debugging)

```bash
make mutation-test THRESHOLD=70
```

## Enforcement in CI

The CI job **mutation-test** in `.github/workflows/ci.yml`:

1. Runs mutmut scoped to `detection/benford_engine.py` with the fast test
   subset (`tests/test_benford.py`, `tests/test_benford_ci.py`).
2. Prints a summary of surviving mutants to the Actions job summary.
3. Calls `python scripts/check_mutation_score.py --threshold 80` — exits with
   code 1 (failing the job) if the score drops below **80 %**.

## Top 10 Surviving Mutant Categories Addressed (Issue #964)

The tests in `tests/test_mutation_coverage.py` were written to kill the
highest-impact surviving mutant categories found in the initial baseline:

| # | Mutant Category | Module | Test |
|---|---|---|---|
| 1 | `> 0` → `>= 0` (drop-zero guard) | `benford_engine.py` | `test_leading_digits_exactly_at_power_of_ten_boundary` |
| 2 | `floor(log10)` off-by-one | `benford_engine.py` | `test_leading_digits_boundary_9_999999` |
| 3 | Chi-square accumulation sign flip | `benford_engine.py` | `test_chi_square_zero_for_perfectly_conforming` |
| 4 | `n == 0` guard return value | `benford_engine.py` | `test_chi_square_returns_zero_for_empty_series` |
| 5 | MAD division by 9 → other constant | `benford_engine.py` | `test_mad_score_zero_for_perfect_conformance` |
| 6 | `mad > threshold` → `mad >= threshold` | `benford_engine.py` | `test_mad_nonconformity_threshold_boundary` |
| 7 | z-score `max(z, 0)` → `min(z, 0)` | `benford_engine.py` | `test_z_scores_large_for_extreme_overrepresentation` |
| 8 | `normalize=True` → `normalize=False` | `benford_engine.py` | `test_observed_distribution_sums_to_one_property` |
| 9 | Percentile `(rank + 0.5) / n` sign | `score_normaliser.py` | `test_normaliser_percentile_at_midpoint` |
| 10 | PPR clip `[0, 100]` removal | `risk_propagation.py` | `test_risk_propagation_clamps_output_to_100` |

## Ongoing Use

Run `make mutation-test` before opening a PR that touches any of the scoped
modules.  If the score drops below 80 %, add tests to `tests/test_mutation_coverage.py`
(or the relevant existing test file) that exercise the surviving mutant's
specific code path.

To view surviving mutants:

```bash
mutmut results --show-survived
```

To apply a surviving mutant for manual inspection:

```bash
mutmut apply <mutant-id>
# inspect the change
mutmut unapply <mutant-id>
```

## References

- [Nigrini, M. (2012) *Benford's Law: Applications for Forensic Accounting,
  Auditing, and Fraud Detection*](https://www.wiley.com/en-us/Benford%27s+Law%3A+Applications+for+Forensic+Accounting%2C+Auditing%2C+and+Fraud+Detection-p-9781118152850)
- [mutmut documentation](https://mutmut.readthedocs.io/)
