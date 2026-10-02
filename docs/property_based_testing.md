# Property-Based Testing for Numeric Precision — Issue #965

LedgerLens processes Stellar on-chain amounts that require exact Decimal
arithmetic (7-decimal stroop precision).  This document describes the
property-based testing strategy used to verify numeric-precision-sensitive
modules and explains how contributors should extend it.

## Why Property-Based Testing?

Example-based tests check specific inputs.  Financial arithmetic fails at
*boundaries*: a value of `9.9999999` vs `10.0000000` can change the leading
digit and silently corrupt Benford statistics.  Hypothesis generates hundreds
of inputs from a declared *strategy* (e.g. "any valid Stellar amount") and
shrinks failing cases to the smallest reproducer.

Key invariants that example tests miss:

| Property | What it catches |
|---|---|
| Stroops round-trip | Off-by-one at the `0.0000001` boundary |
| Scale invariance | Values near digit boundaries (9.9999 × 10) |
| Distribution sums to 1 | Rounding errors in `value_counts(normalize=True)` |
| MAD non-negative | `abs()` removal mutations |
| Percentile bound | `(n+0.5)/n` off-by-one |

## Modules Covered

| Module | Test file | Invariants |
|---|---|---|
| `utils/decimal_guards.py` | `tests/test_numeric_precision_properties.py` | Stroops round-trip, arithmetic commutativity/identity, validation boundaries |
| `utils/benford_precision.py` | `tests/test_numeric_precision_properties.py` | Digit in [1,9], scale invariance, Decimal/float agreement |
| `detection/benford_engine.py` | `tests/test_numeric_precision_properties.py` | Distribution sum-to-one, MAD ≥ 0, order independence, scale invariance |
| `detection/score_normaliser.py` | `tests/test_numeric_precision_properties.py` | Percentile in (0, (n+0.5)/n], monotonicity, skip-when-small |

## Running the Tests

```bash
# All property-based precision tests
pytest tests/test_numeric_precision_properties.py -v

# Included in the standard fast suite
make test-fast

# More examples for thorough local checking
HYPOTHESIS_MAX_EXAMPLES=500 pytest tests/test_numeric_precision_properties.py
```

## Edge Cases Found During Implementation

No latent bugs were found in the current codebase during initial property
testing (2026-09-28, Hypothesis `max_examples=200`).  All invariants verified:

- `DecimalAmount` stroops round-trip: exact for all valid Stellar amounts.
- `validate_amount` boundary rejection: NaN, Inf, negative (when disallowed), below-min, above-max.
- `observed_distribution` sum-to-one: holds for all positive float lists tested.
- `leading_digits` range [1,9]: holds for all positive floats 1e-7 to 1e12.
- Scale invariance (× 10^k, k ∈ [−5, 5]): distribution unchanged.
- `PerPairScoreNormaliser` percentile: bounded in (0, (n+0.5)/n].

## How to Extend — Checklist for Contributors

When you add or modify a numeric module, add a section to
`tests/test_numeric_precision_properties.py` covering:

1. **Round-trip** — encode → decode is identity (e.g. stroops ↔ Decimal).
2. **Monotonicity** — larger input → larger output for monotone functions.
3. **Bounded output** — output stays within its documented range.
4. **Commutativity** — order of operands doesn't affect the result.
5. **Identity elements** — `f(x, 0) == x`, `f(x, 1) == x`.
6. **Invalid inputs rejected** — NaN, Inf, out-of-range, wrong type raise the
   correct exception at the ingestion boundary.

### Recommended Hypothesis strategies

| Data type | Strategy |
|---|---|
| Stellar amount | `st.decimals(min_value="0.0000001", max_value="922337203685.4775807", places=7)` |
| Positive financial float | `st.floats(min_value=1e-7, max_value=1e12, allow_nan=False, allow_infinity=False)` |
| Wallet ID | `tests/strategies.stellar_ids` |
| Asset code | `tests/strategies.asset_codes` |
| Trade record | `tests/strategies.trades()` |

### CI settings

Use `_CI_SETTINGS` from the test module to keep new tests within the CI
time budget:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

_CI_SETTINGS = dict(max_examples=200, deadline=10_000)

@given(st.decimals(...))
@settings(**_CI_SETTINGS)
def test_my_invariant(amount):
    ...
```

## References

- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- `docs/numeric_precision.md` — prior Decimal precision work (Issue #483)
- `NUMERIC_PRECISION_PR.md` — design rationale for `utils/decimal_guards.py`
- `utils/decimal_guards.py` — the precision guard system
- `utils/benford_precision.py` — Decimal-based Benford digit extractor
