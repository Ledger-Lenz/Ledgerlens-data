# Contributor Guide: Adding a New ML Feature

This guide covers everything you need to add a new feature to the LedgerLens
ML pipeline end-to-end: from writing the computation function all the way
through registration, validation, dataset documentation, and testing.

**Before you start** — read the [Security Threat Model](security_threat_model.md)
and the [Privacy Policy](privacy.md).  Raw wallet addresses must never be used
as ML features directly; see the [Security section](#security) below.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Where to add your computation function](#2-where-to-add-your-computation-function)
3. [Function signature](#3-function-signature)
4. [Naming conventions](#4-naming-conventions)
5. [Registering the feature in `build_feature_vector`](#5-registering-the-feature-in-build_feature_vector)
6. [Per-feature range validation](#6-per-feature-range-validation)
7. [Adding a description to `FEATURE_DESCRIPTIONS`](#7-adding-a-description-to-feature_descriptions)
8. [Updating `data/dataset_card.md`](#8-updating-datadataset_cardmd)
9. [SHAP integration](#9-shap-integration)
10. [Required tests](#10-required-tests)
11. [Security](#11-security)
12. [Worked example — `counterparty_variance`](#12-worked-example--counterparty_variance)

---

## 1. Architecture overview

```
detection/feature_engineering.py
│
├── compute_<group>_features(...)   ← one function per feature group
│       └── returns dict[str, float]
│
├── build_feature_vector(...)       ← assembles the full row for one wallet
│       └── calls every compute_* function in order
│
└── build_feature_matrix(...)       ← maps build_feature_vector over all wallets
```

Each `compute_*` function is responsible for a **group** of related features
(e.g. Benford statistics, volume/timing, wallet graph).  A new feature that
naturally belongs to an existing group goes inside that function.  A new
feature that introduces a genuinely new data dependency gets its own
`compute_<new_group>_features` function.

---

## 2. Where to add your computation function

Open `detection/feature_engineering.py`.

| Situation | Where to add |
|---|---|
| New signal, fits an existing group | Add the computation inside the relevant `compute_*` function and add the key to its return dict |
| New signal, new data dependency | Add a new `compute_<group>_features(wallet, <data>, ...) -> dict` function near the other `compute_*` functions |
| Benford / statistical signal | Add inside `compute_benford_features` or a new `compute_<name>_features` that calls `compute_benford_metrics` |

Keep all feature computation in `detection/feature_engineering.py`.  Do not
scatter feature logic across ingestion or scoring modules.

---

## 3. Function signature

Every public `compute_*_features` function must follow this exact signature
pattern so that `build_feature_vector` can call it uniformly:

```python
def compute_<group>_features(
    wallet: str,
    wallet_trades: pd.DataFrame,        # always required
    <optional_data>: <Type> | None = None,  # only what your feature needs
) -> dict[str, float]:
    """One-line summary.

    Longer description of what this function computes and why it is a
    wash-trade signal.

    Args:
        wallet: Stellar account ID being scored.  Must not be stored directly
            as an ML feature — see Security section of the contributor guide.
        wallet_trades: Trades involving ``wallet`` as base_account or
            counter_account.  Columns: ledger_close_time, base_account,
            counter_account, amount, base_asset, counter_asset.
        <optional_data>: Description and where it comes from.

    Returns:
        A dict mapping snake_case feature name(s) to float values.
        All values must be finite (no NaN, no Inf).
        Return 0.0 (or another safe sentinel) when data is insufficient.

    Raises:
        KeyError: If a required column is missing from wallet_trades.
    """
    if wallet_trades.empty:
        return {"<group>_<signal_name>": 0.0}

    # ... computation ...

    return {"<group>_<signal_name>": float(result)}
```

Key rules:
- **First parameter is always `wallet: str`**, even if your function does not
  use the wallet ID directly.  This keeps call-site symmetry.
- **Second parameter is always `wallet_trades: pd.DataFrame`** (already
  filtered to the target wallet — do not re-filter inside your function).
- **All optional inputs default to `None`** so `build_feature_vector` can
  call it unconditionally and the function degrades gracefully.
- **Return `dict[str, float]`** always.  Use `float(...)` to coerce numpy
  scalars.  Use `0.0` as the safe fallback when data is absent.
- **Never return `NaN` or `Inf`**.  `build_feature_vector` replaces `NaN`
  floats with `0.0`, but it is better to never produce them.

---

## 4. Naming conventions

All feature names use **snake_case** with a **category prefix** matching their
feature group:

| Category prefix | Feature group | Example |
|---|---|---|
| `benford_` | Benford's Law statistics | `benford_mad_24h` |
| `counterparty_` | Trade-partner signals | `counterparty_concentration_ratio` |
| `round_trip_` | Self-trade / closed-cycle signals | `round_trip_frequency` |
| `volume_` | Volume anomalies | `volume_spike_frequency` |
| `intra_` | Intra-minute/intra-bucket clustering | `intra_minute_clustering` |
| `off_hours_` | Temporal/time-of-day signals | `off_hours_activity_ratio` |
| `funding_` | Wallet funding graph | `funding_source_similarity` |
| `network_` | Graph centrality | `network_centrality` |
| `account_` | Account metadata | `account_age_days` |
| `cross_pair_` | Multi-pair coordination | `cross_pair_trade_synchrony` |
| `cross_venue_` | SDEX vs AMM coordination | `cross_venue_volume_correlation` |
| `bot_` | Bot behaviour detection | `bot_interval_regularity` |
| `gnn_` | GNN embedding dimensions | `gnn_0` … `gnn_31` |
| `temporal_kge_` | Temporal knowledge graph | `temporal_kge_collab_score` |
| `path_` | Payment path analysis | `path_payment_round_trip_frequency` |

If your feature does not fit any existing prefix, introduce a new one that
describes the signal category (not the computation method).

---

## 5. Registering the feature in `build_feature_vector`

`build_feature_vector` in `detection/feature_engineering.py` is the single
assembly point for the full feature row.  You must add a `features.update(...)`
call here.

### Adding to an existing group

Find the `features.update(compute_<existing_group>_features(...))` call and
verify your new key is returned by that function — nothing else to do here.

### Adding a new compute function

Add a new `features.update(...)` call in the correct position inside
`build_feature_vector`.  The order is:

```python
features: dict[str, float | str] = {"wallet": wallet}
features.update(compute_benford_features(...))          # 1. Benford
features.update(compute_trade_pattern_features(...))    # 2. Trade patterns
features.update(compute_volume_timing_features(...))    # 3. Volume / timing
features.update(compute_wallet_graph_features(...))     # 4. Graph
# cross-asset (optional — only when all_pairs_df is provided)
features.update(compute_payment_path_features(...))     # 5. Payment paths
features.update(compute_temporal_kge_features(...))     # 6. Temporal KGE
features.update(compute_hardening_features(...))        # 7. Adversarial hardening
features.update(compute_ts_decomposition_features(...)) # 8. TS decomposition
# cross-venue (optional — only when amm_trades is provided)
# GNN embeddings (optional — only when encoder present)
# >>> your new group goes here <<<
```

Also add the same optional parameter to `build_feature_vector`'s signature
(defaulting to `None`) and thread it through to your new function.

### Updating `build_feature_matrix`

`build_feature_matrix` calls `build_feature_vector` for every wallet.  If your
new function needs a dataset-level input (e.g. a pre-built graph), add a
matching parameter to `build_feature_matrix` and pass it through to
`build_feature_vector`.

---

## 6. Per-feature range validation

Add a range assertion inside your `compute_*` function.  Use this pattern,
which matches the project's existing style:

```python
result = _my_computation(wallet_trades)

# Range validation — catch miscalculations before they reach the model
if not (0.0 <= result <= 1.0):
    logger.warning(
        "counterparty_variance out of expected range [0, 1]: %.4f for wallet %s — clamping",
        result,
        wallet,
    )
    result = max(0.0, min(1.0, result))

return {"counterparty_variance": float(result)}
```

The `logger.warning` on an out-of-range value makes silent regressions
visible in the log stream without crashing the scoring pipeline.  Always
**clamp** rather than raise, because a single bad feature should not abort
scoring for the whole wallet.

Document the expected range in the docstring and in
`FEATURE_DESCRIPTIONS` (see next section).

Common ranges by feature type:

| Signal type | Expected range |
|---|---|
| Ratio / fraction / frequency | `[0.0, 1.0]` |
| Entropy (nats/bits) | `[0.0, ∞)` — no upper bound, document typical max |
| Correlation (Pearson) | `[-1.0, 1.0]` |
| Age in days | `[0.0, ∞)` |
| Log-normalised amounts | document explicitly |
| Risk scores (external) | `[0.0, 100.0]` |

---

## 7. Adding a description to `FEATURE_DESCRIPTIONS`

`FEATURE_DESCRIPTIONS` is a module-level dict near the top of
`detection/feature_engineering.py`.  It maps every feature name to a
human-readable description used in SHAP reports and forensic output.

Add your feature there **before** `compute_benford_features`:

```python
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # ... existing entries ...

    # Trade pattern features
    "counterparty_variance": (
        "Variance of per-counterparty trade volume, normalised by mean volume. "
        "High variance indicates one or a few counterparties dominate volume "
        "while others receive negligible flow — a pattern consistent with "
        "coordinated wash trading using decoy counterparties."
        " Range: [0, 1] (clipped). Higher = more suspicious."
    ),
}
```

Description requirements:
1. What the feature measures (one sentence).
2. Why it is a wash-trade signal (one sentence).
3. Expected range and direction (higher/lower = more suspicious).

---

## 8. Updating `data/dataset_card.md`

`data/dataset_card.md` documents every column in the labelled dataset Parquet
file.  Add a row to the appropriate table under **Schema → Feature columns**:

```markdown
| `counterparty_variance` | float | Variance of per-counterparty volume, normalised; range [0, 1] |
```

If the feature introduces a new feature group that doesn't map to any existing
table heading, add a new sub-table with a heading matching the group prefix.

---

## 9. SHAP integration

SHAP values are computed automatically for every column in the feature matrix
by `detection/model_inference.py`.  You do **not** need to register your
feature explicitly — it will appear in the SHAP waterfall chart and the
forensic report as soon as it is returned by `build_feature_matrix`.

Two things to verify:

1. **Your feature name is in `FEATURE_DESCRIPTIONS`** — the forensic report
   uses this dict to generate human-readable explanations.  A missing entry
   will produce `"Unknown feature"` in the report.

2. **The feature has a finite value for every wallet row** — `NaN` or `Inf`
   in the feature matrix will cause `shap.TreeExplainer` to raise.
   `build_feature_vector` replaces `NaN` floats with `0.0` automatically,
   but you should still return finite values from your function.

---

## 10. Required tests

Every new feature needs at minimum three tests.  Place them in
`tests/test_feature_engineering.py` (or a dedicated
`tests/test_<group>_features.py` for large groups).

### 10.1 — Unit test: empty DataFrame returns safe defaults

```python
def test_counterparty_variance_empty():
    features = compute_trade_pattern_features("W", pd.DataFrame())
    assert features["counterparty_variance"] == 0.0
```

### 10.2 — Unit test: known input produces expected output

```python
def test_counterparty_variance_known_value():
    df = _make_trades_with_two_counterparties(vol_a=900.0, vol_b=100.0)
    features = compute_trade_pattern_features("W", df)
    # vol = [900, 100], mean = 500, var = 160000 / 500^2 = 0.64 → clamp → 0.64
    assert abs(features["counterparty_variance"] - 0.64) < 1e-6
```

### 10.3 — Hypothesis property test: output always in valid range

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(
    n=st.integers(min_value=0, max_value=200),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=200)
def test_counterparty_variance_always_in_range(n, seed):
    rng = np.random.default_rng(seed)
    if n == 0:
        df = pd.DataFrame()
    else:
        times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        cps = [f"CP_{rng.integers(0, 5)}" for _ in range(n)]
        df = pd.DataFrame({
            "ledger_close_time": times,
            "base_account": "W",
            "counter_account": cps,
            "amount": rng.uniform(1, 10_000, n),
            "base_asset": "A:B", "counter_asset": "C:D",
        })
    features = compute_trade_pattern_features("W", df)
    val = features["counterparty_variance"]
    assert 0.0 <= val <= 1.0, f"Out of range: {val}"
    assert math.isfinite(val)
```

### 10.4 — Integration test: feature appears in `build_feature_matrix` output

```python
def test_build_feature_matrix_includes_counterparty_variance():
    df = pd.DataFrame(make_clean_trades(n=20))
    matrix = build_feature_matrix(df)
    assert "counterparty_variance" in matrix.columns
    assert matrix["counterparty_variance"].notna().all()
```

Run tests with:

```bash
make test
# or directly:
pytest tests/test_feature_engineering.py -v
```

---

## 11. Security

> **Raw wallet addresses must never be used directly as ML features.**

Stellar public keys (G-prefixed 56-character strings) are pseudonymous but
persistent identifiers.  Using them directly as model inputs would:

- Allow the model to memorise specific wallets rather than learning general
  patterns.
- Leak address-level information through SHAP explanations.
- Create linkability risks if explanations are shared externally.

**Required mitigations for any address-derived feature:**

| ❌ Forbidden | ✅ Required instead |
|---|---|
| Store the raw wallet address as a feature | Hash it: `hashlib.sha256(wallet.encode()).hexdigest()` and use as a lookup key, never as a feature value |
| Use wallet address as a categorical feature column | Derive continuous or boolean signals (e.g., age, centrality, cluster membership) |
| Include raw counterparty addresses in the feature vector | Compute aggregate statistics (concentration ratio, unique count, Jaccard similarity) |
| Log raw wallet addresses in feature computation | Use the wallet ID only for filtering, never in the returned dict |

The feature dict returned by `compute_*` must contain **only numeric
(float) values**.  The `"wallet"` key in `build_feature_vector` is the one
deliberate exception — it is excluded from the feature matrix by
`detection/model_training.py::FEATURE_COLUMNS_EXCLUDE`.

---

## 12. Worked example — `counterparty_variance`

This section demonstrates every file that changes when adding a new feature.
`counterparty_variance` is a fictional feature that measures the variance of
per-counterparty trade volume (normalised), as a proxy for coordinated
wash-trade rings that use decoy counterparties to obscure concentration.

### 12.1 — Feature computation (`detection/feature_engineering.py`)

Add the computation inside `compute_trade_pattern_features`.  The function
already computes per-counterparty volume — `counterparty_variance` is a
natural addition to the same group:

```python
def compute_trade_pattern_features(
    wallet: str,
    wallet_trades: pd.DataFrame,
    orderbook_events: pd.DataFrame | None = None,
) -> dict:
    # ... existing logic ...

    # --- NEW: counterparty_variance ---
    # Variance of per-counterparty volume, normalised by mean^2 so the value
    # is dimensionless and comparable across wallets of different sizes.
    if len(volume_by_counterparty) >= 2:
        mean_vol = float(volume_by_counterparty.mean())
        var_vol = float(volume_by_counterparty.var(ddof=0))
        counterparty_variance = (var_vol / (mean_vol ** 2)) if mean_vol > 0 else 0.0
        # Clamp to [0, 1] — values above 1 indicate extreme concentration;
        # treat as the maximum suspicious signal.
        counterparty_variance = max(0.0, min(1.0, counterparty_variance))
    else:
        counterparty_variance = 0.0  # safe default: single counterparty or no data

    # Range validation
    if not (0.0 <= counterparty_variance <= 1.0):
        logger.warning(
            "counterparty_variance out of expected range [0, 1]: %.4f for wallet %s — clamping",
            counterparty_variance,
            wallet,
        )
        counterparty_variance = max(0.0, min(1.0, counterparty_variance))

    return {
        "counterparty_concentration_ratio": float(concentration),
        "round_trip_frequency": float(round_trip_frequency),
        "net_roundtrip_ratio": float(round_trip_frequency),
        "self_matching_rate": float(self_matching_rate),
        "order_cancellation_rate": order_cancellation_rate,
        "counterparty_variance": counterparty_variance,  # ← new key
    }
```

### 12.2 — Feature description (`detection/feature_engineering.py`)

Add to `FEATURE_DESCRIPTIONS`:

```python
"counterparty_variance": (
    "Variance of per-counterparty trade volume, normalised by mean volume squared "
    "(coefficient of variation squared). High values indicate a small number of "
    "counterparties dominate while others receive minimal volume — consistent with "
    "wash-trade networks that use decoy accounts to reduce concentration_ratio. "
    "Range: [0, 1] (clipped). Higher = more suspicious."
),
```

### 12.3 — Dataset card (`data/dataset_card.md`)

Add a row to the **Feature columns** table under the trade-pattern section:

```markdown
| `counterparty_variance` | float | Normalised variance of per-counterparty volume; range [0, 1] |
```

### 12.4 — Tests (`tests/test_feature_engineering.py`)

```python
import math
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detection.feature_engineering import (
    build_feature_matrix,
    compute_trade_pattern_features,
)
from tests.factories import make_clean_trades


# ---------------------------------------------------------------------------
# counterparty_variance
# ---------------------------------------------------------------------------

def _trades_with_counterparty_volumes(vol_map: dict[str, float]) -> pd.DataFrame:
    """Build a minimal trades DataFrame with exact per-counterparty volumes."""
    rows = []
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    i = 0
    for cp, vol in vol_map.items():
        rows.append({
            "ledger_close_time": t0 + pd.Timedelta(minutes=i),
            "base_account": "W",
            "counter_account": cp,
            "amount": vol,
            "base_asset": "USDC:GA5Z",
            "counter_asset": "XLM:native",
        })
        i += 1
    return pd.DataFrame(rows)


def test_counterparty_variance_empty_returns_zero():
    """Empty DataFrame must return 0.0 without raising."""
    features = compute_trade_pattern_features("W", pd.DataFrame())
    assert features["counterparty_variance"] == 0.0


def test_counterparty_variance_equal_volumes_returns_zero():
    """Equal volume for each counterparty means zero variance."""
    df = _trades_with_counterparty_volumes({"CP_A": 500.0, "CP_B": 500.0, "CP_C": 500.0})
    features = compute_trade_pattern_features("W", df)
    assert features["counterparty_variance"] == pytest.approx(0.0, abs=1e-9)


def test_counterparty_variance_single_counterparty_returns_zero():
    """Single counterparty — variance is undefined, should return 0.0."""
    df = _trades_with_counterparty_volumes({"CP_A": 1000.0})
    features = compute_trade_pattern_features("W", df)
    assert features["counterparty_variance"] == 0.0


def test_counterparty_variance_known_value():
    """Verify the normalised variance calculation against a hand-computed value.

    vol = [900, 100], mean = 500, var = ((900-500)^2 + (100-500)^2) / 2 = 160000
    normalised = 160000 / 500^2 = 0.64
    """
    df = _trades_with_counterparty_volumes({"CP_A": 900.0, "CP_B": 100.0})
    features = compute_trade_pattern_features("W", df)
    assert features["counterparty_variance"] == pytest.approx(0.64, abs=1e-6)


def test_counterparty_variance_always_in_range_property():
    """Hypothesis: counterparty_variance is always in [0.0, 1.0]."""
    @given(
        n=st.integers(min_value=0, max_value=200),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=200)
    def _property(n, seed):
        rng = np.random.default_rng(seed)
        if n == 0:
            df = pd.DataFrame()
        else:
            times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
            cps = [f"CP_{rng.integers(0, 5)}" for _ in range(n)]
            df = pd.DataFrame({
                "ledger_close_time": times,
                "base_account": "W",
                "counter_account": cps,
                "amount": rng.uniform(1.0, 10_000.0, n),
                "base_asset": "USDC:GA5Z",
                "counter_asset": "XLM:native",
            })
        features = compute_trade_pattern_features("W", df)
        val = features["counterparty_variance"]
        assert 0.0 <= val <= 1.0, f"Out of range [{val}] for n={n} seed={seed}"
        assert math.isfinite(val), f"Non-finite value {val} for n={n} seed={seed}"

    _property()


def test_build_feature_matrix_includes_counterparty_variance():
    """counterparty_variance must appear in every row of build_feature_matrix."""
    df = pd.DataFrame(make_clean_trades(n=20))
    matrix = build_feature_matrix(df)
    assert "counterparty_variance" in matrix.columns
    assert matrix["counterparty_variance"].notna().all()
    assert (matrix["counterparty_variance"].between(0.0, 1.0)).all()
```

### 12.5 — SHAP integration (no code change required)

SHAP is computed automatically.  To verify your feature appears correctly:

```python
from detection.model_inference import RiskScorer

scorer = RiskScorer()
report = scorer.explain(wallet="GABC...", trades_df=my_trades_df)
# report.shap_explanations will contain an entry for "counterparty_variance"
# report.feature_descriptions["counterparty_variance"] will show your docstring
```

In the forensic HTML report, `counterparty_variance` will appear in the SHAP
waterfall chart with the description from `FEATURE_DESCRIPTIONS`.

### 12.6 — Checklist

Before opening a PR for a new feature, verify:

- [ ] `compute_trade_pattern_features` (or new `compute_*_features`) returns the new key
- [ ] Key added to `FEATURE_DESCRIPTIONS` with range and direction documented
- [ ] Range validation + clamp inside the compute function
- [ ] `build_feature_vector` updated (only if a new `compute_*` function was added)
- [ ] `build_feature_matrix` updated (only if a new dataset-level input is needed)
- [ ] Row added to `data/dataset_card.md`
- [ ] Unit test: empty DataFrame → `0.0`
- [ ] Unit test: known input → known output
- [ ] Hypothesis property test: output always in `[lo, hi]` and `isfinite`
- [ ] Integration test: column present in `build_feature_matrix` output
- [ ] `make test` passes
- [ ] No raw wallet addresses in the returned feature dict
