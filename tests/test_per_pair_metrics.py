"""Unit tests for per-asset-pair Prometheus metrics (issue #276)."""

import pytest


def test_canonical_pair_sorts_alphabetically():
    from detection.per_pair_metrics import canonical_pair

    # B/A should be normalised to A/B
    result = canonical_pair(
        "XLM:native/USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
    )
    expected = canonical_pair(
        "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"
    )
    assert result == expected


def test_canonical_pair_format():
    from detection.per_pair_metrics import canonical_pair

    pair = "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"
    result = canonical_pair(pair)
    # Must contain exactly one "/" separator
    assert result.count("/") == 1
    # Must not contain wallet addresses (no G... account IDs without CODE: prefix)
    parts = result.split("/")
    assert all(":" in p for p in parts), f"Non-canonical part in {result}"


def test_canonical_pair_idempotent():
    from detection.per_pair_metrics import canonical_pair

    pair = "AQUA:GBNZILSTVQZ4R7IKQDGHYGY2QXL5QOFJYQMXPKWRRM5PAV7Y4M67AQUA/XLM:native"
    assert canonical_pair(canonical_pair(pair)) == canonical_pair(pair)


def test_record_scoring_duration_does_not_raise():
    """record_scoring_duration must never raise even if prometheus_client is absent."""
    from detection.per_pair_metrics import record_scoring_duration

    with record_scoring_duration("USDC:GABC/XLM:native"):
        pass  # no exception


def test_record_benford_computation_does_not_raise():
    from detection.per_pair_metrics import record_benford_computation

    record_benford_computation("USDC:GABC/XLM:native", status="ok")
    record_benford_computation("USDC:GABC/XLM:native", status="false_positive")


def test_record_risk_score_does_not_raise():
    from detection.per_pair_metrics import record_risk_score

    record_risk_score("BTC:GAUT/XLM:native", 75.0)
    record_risk_score("BTC:GAUT/XLM:native", 0.0)
    record_risk_score("BTC:GAUT/XLM:native", 100.0)


def test_metrics_carry_asset_pair_label():
    """When prometheus_client is available, all three metrics must emit asset_pair label."""
    prometheus = pytest.importorskip("prometheus_client")
    from detection.per_pair_metrics import (
        ledgerlens_score_duration_seconds,
        ledgerlens_benford_computation_total,
        ledgerlens_risk_score_distribution,
    )

    pair = "USDC:GABC123/XLM:native"

    if ledgerlens_score_duration_seconds is not None:
        from detection.per_pair_metrics import canonical_pair, record_scoring_duration
        with record_scoring_duration(pair):
            pass
        # Verify the label is registered
        canon = canonical_pair(pair)
        sample = ledgerlens_score_duration_seconds.labels(asset_pair=canon)
        assert sample is not None

    if ledgerlens_benford_computation_total is not None:
        from detection.per_pair_metrics import record_benford_computation, canonical_pair
        record_benford_computation(pair, status="ok")
        canon = canonical_pair(pair)
        sample = ledgerlens_benford_computation_total.labels(asset_pair=canon, status="ok")
        assert sample is not None

    if ledgerlens_risk_score_distribution is not None:
        from detection.per_pair_metrics import record_risk_score, canonical_pair
        record_risk_score(pair, 55.0)
        canon = canonical_pair(pair)
        sample = ledgerlens_risk_score_distribution.labels(asset_pair=canon)
        assert sample is not None
