# Thin Market Wash Trade Detection

## Overview

Thin markets — asset pairs with very few natural participants — are especially vulnerable to wash trading because:

1. **Low capital requirement**: Small trade volumes produce large price movements, so wash traders need less capital to create convincing activity.
2. **High price impact**: Even minor round-trip trades shift the visible price, amplifying the apparent volume and attracting further (or spoofing) activity.
3. **Small sample bias**: With fewer than ~50 trades per Benford window, the chi-square statistic has high variance; standard thresholds produce many false positives and false negatives.

Thin-market wash trading therefore requires **distinct feature thresholds and a separate risk signal** that is meaningless (NaN) for liquid pairs, preventing signal conflation.

## Classification Criteria

A pair is classified as thin market when **any** of the following hold:

| Criterion | Default threshold | Source |
|---|---|---|
| Unique traders in last 7 days | < 50 | `build_config.json` → `thin_market.max_unique_traders_7d` |
| Liquidity depth (USD equivalent) | < $1,000 | `thin_market.min_liquidity_depth_usd` |
| AMM pool TVL | < $5,000 | `thin_market.min_amm_tvl_usd` |

Classifications are cached for 1 hour (configurable via `thin_market.classification_cache_seconds`) to avoid per-request recomputation.

## Adjusted Benford Thresholds

For thin markets the Benford chi-square threshold is multiplied by `benford_chi_square_threshold_multiplier` (default 3.0) before flagging.  This accounts for the high sampling variance when fewer than 50 trades are available in a window — low sample sizes naturally produce chi-square values well above the standard threshold even for conforming distributions.

The multiplier is sourced from `build_config.json` and validated on startup.

**Rationale**: For n = 20 trades, the expected chi-square under the null (true Benford distribution) is approximately 8 d.f. = 8; the 95th-percentile critical value is ~15.5.  The standard alert threshold (chi-square > 15) has a 50% false-positive rate at this sample size.  A 3× multiplier (threshold > 45) reduces the false-positive rate to < 5%.

## The `thin_market_wash_risk` Feature

```
thin_market_wash_risk = (
    liquidity_component × 0.40
    + trader_concentration × 0.35
    + round_trip_frequency × 0.25
)  × 100
```

Where:
- **liquidity_component** = 0.5 × (1 − depth/min_depth) + 0.5 × (1 − TVL/min_TVL), clamped to [0, 1]
- **trader_concentration** = 1 − (unique_traders / max_unique_traders), clamped to [0, 1]
- **round_trip_frequency** = min(rt_frequency × 2.0, 1.0) — calibrated for thin markets where any round-trip activity is more suspicious

This feature is **NaN for liquid pairs** (unique_traders ≥ 50, depth ≥ $1,000, TVL ≥ $5,000).  Using NaN instead of 0 ensures that liquid-pair rows do not suppress thin-market signals in ensemble models.

## Alert Dispatching

When `thin_market_wash_risk` exceeds `THIN_MARKET_ALERT_THRESHOLD` (default 60), a `thin_market_wash_risk` alert is dispatched via `streaming.alert_dispatcher.AlertDispatcher`.  This alert is separate from the standard Benford and ML flags and carries the pair's thin-market classification metadata.

## Configuration

All thresholds live in `data/build_config.json` under the `thin_market` key:

```json
{
  "thin_market": {
    "max_unique_traders_7d": 50,
    "min_liquidity_depth_usd": 1000,
    "min_amm_tvl_usd": 5000,
    "benford_chi_square_threshold_multiplier": 3.0,
    "alert_threshold": 60,
    "classification_cache_seconds": 3600
  }
}
```

All values are validated on startup by `_validate_thin_market_config()`; a non-positive value raises `ValueError` immediately.

## Implementation

- `detection/liquidity_profiler.py`: `ThinMarketDetector` class
- `data/build_config.json`: `thin_market` section (thresholds)
- `streaming/alert_dispatcher.py`: `AlertDispatcher.dispatch()` (reused)
- `tests/test_liquidity_profiler.py`: unit tests for all acceptance criteria
