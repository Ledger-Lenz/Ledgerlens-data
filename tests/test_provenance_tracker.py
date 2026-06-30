"""Unit tests for ProvenanceTracker (Issue #244)."""

import json
import datetime

import pandas as pd
import pytest

from detection.feature_engineering import ProvenanceTracker, compute_benford_features


def _make_trades(n: int, trade_ids: list[str] | None = None) -> pd.DataFrame:
    """Build a minimal trade DataFrame suitable for compute_benford_features."""
    if trade_ids is None:
        trade_ids = [f"tok_{i}" for i in range(n)]
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    return pd.DataFrame(
        {
            "trade_id": trade_ids,
            "amount": [float(i + 1) * 10.0 for i in range(n)],
            "ledger_close_time": [
                now - datetime.timedelta(minutes=10 * (n - i)) for i in range(n)
            ],
        }
    )


# ---------------------------------------------------------------------------
# 1. Provenance disabled by default — no writes to DB
# ---------------------------------------------------------------------------


def test_provenance_disabled_returns_none():
    tracker = ProvenanceTracker(enabled=False)
    tracker.record("benford_chi_square_24h", ["t1", "t2"])
    assert tracker.to_json() is None


def test_provenance_disabled_records_nothing():
    tracker = ProvenanceTracker(enabled=False)
    tracker.record("benford_mad_24h", ["t1"])
    assert tracker.get("benford_mad_24h") == []


# ---------------------------------------------------------------------------
# 2. Provenance enabled — records correct trade IDs
# ---------------------------------------------------------------------------


def test_provenance_records_known_trade_ids():
    """For 5 known trades, provenance must record exactly those 5 IDs."""
    trade_ids = ["tok_a", "tok_b", "tok_c", "tok_d", "tok_e"]
    trades = _make_trades(5, trade_ids=trade_ids)

    tracker = ProvenanceTracker(enabled=True)
    compute_benford_features(trades, decompose=False, provenance=tracker)

    recorded = tracker.get("benford_chi_square_24h")
    assert set(recorded) == set(trade_ids), (
        f"Expected {set(trade_ids)}, got {set(recorded)}"
    )


def test_provenance_json_round_trips():
    """to_json() must produce valid JSON that round-trips correctly."""
    trades = _make_trades(5)
    tracker = ProvenanceTracker(enabled=True)
    compute_benford_features(trades, decompose=False, provenance=tracker)

    blob = tracker.to_json()
    assert blob is not None
    parsed = json.loads(blob)
    assert isinstance(parsed, dict)
    # At least one benford window must be present
    assert any(k.startswith("benford_") for k in parsed)


# ---------------------------------------------------------------------------
# 3. Derived features are not tracked
# ---------------------------------------------------------------------------


def test_derived_features_not_tracked():
    """Non-base-window features (calibrated, residual) must not appear in provenance."""
    trades = _make_trades(5)
    tracker = ProvenanceTracker(enabled=True)
    tracker.record("benford_deviation_from_regime", ["t1"])  # derived — should be ignored
    assert tracker.get("benford_deviation_from_regime") == []


# ---------------------------------------------------------------------------
# 4. Provenance disabled — zero DB writes (no provenance_json)
# ---------------------------------------------------------------------------


def test_provenance_disabled_compute_benford_no_side_effects():
    """compute_benford_features with disabled tracker must return same features as without tracker."""
    trades = _make_trades(8)
    tracker_disabled = ProvenanceTracker(enabled=False)

    feats_with = compute_benford_features(trades, decompose=False, provenance=tracker_disabled)
    feats_without = compute_benford_features(trades, decompose=False, provenance=None)

    assert feats_with.keys() == feats_without.keys()
    for key in feats_with:
        v1, v2 = feats_with[key], feats_without[key]
        # NaN-safe comparison
        if v1 != v1:  # NaN
            assert v2 != v2
        else:
            assert abs(float(v1) - float(v2)) < 1e-9
