"""Tests for StubDriftDetector (Issue #953).

Tests that the drift detector:
1. Catches an intentionally introduced mismatch between stub and "real" results.
2. Passes (no exception) when stub and real results are aligned.
"""

from __future__ import annotations

import pytest

from integrations.offline_stubs import (
    StubContractClient,
    StubDriftDetector,
    StubDriftError,
    run_contract_test_scenarios,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aligned_results() -> tuple[dict, dict]:
    """Return (stub_results, real_results) that are structurally identical."""
    stub = run_contract_test_scenarios(client=StubContractClient())
    # Simulate real results by running against a second, independent stub instance
    # (in a full integration test this would be replaced by a live Testnet call)
    real = run_contract_test_scenarios(client=StubContractClient())
    return stub, real


# ---------------------------------------------------------------------------
# Test: drift detector raises on intentional mismatch
# ---------------------------------------------------------------------------


def test_drift_detector_catches_mismatch():
    """StubDriftDetector.compare_scenarios raises StubDriftError when results diverge."""
    stub_results, _ = _make_aligned_results()

    # Introduce a deliberate mismatch: the real client now returns a different
    # result for 'submit_and_get_score' (extra key that stub doesn't return).
    real_results = {
        scenario: dict(result) for scenario, result in stub_results.items()
    }
    # Real client adds a 'ledger_sequence' field that the stub doesn't return
    real_results["submit_and_get_score"] = {
        "ok": True,
        "result_keys": sorted(
            list(stub_results["submit_and_get_score"].get("result_keys") or [])
            + ["ledger_sequence"]
        ),
        "error_type": None,
    }

    detector = StubDriftDetector()
    with pytest.raises(StubDriftError) as exc_info:
        detector.compare_scenarios(stub_results, real_results)

    err = str(exc_info.value)
    assert "submit_and_get_score" in err
    assert "ledger_sequence" in err
    assert "drifted" in err.lower() or "divergence" in err.lower()


def test_drift_detector_catches_ok_flag_mismatch():
    """StubDriftError raised when stub succeeds but real client raises."""
    stub_results, real_results = _make_aligned_results()

    # Simulate real client raising for get_score_missing_raises differently
    real_results["get_score_missing_raises"] = {
        "ok": True,  # real returns empty dict rather than raising
        "result_keys": [],
        "error_type": None,
    }

    detector = StubDriftDetector()
    with pytest.raises(StubDriftError) as exc_info:
        detector.compare_scenarios(stub_results, real_results)

    err = str(exc_info.value)
    assert "get_score_missing_raises" in err


def test_drift_detector_catches_missing_scenario():
    """StubDriftError raised when a scenario is present in real but missing from stub."""
    stub_results, real_results = _make_aligned_results()
    # Real client exposes a new scenario that the stub doesn't implement yet
    real_results["new_scenario_v2"] = {
        "ok": True,
        "result_keys": ["foo", "bar"],
        "error_type": None,
    }

    detector = StubDriftDetector()
    with pytest.raises(StubDriftError) as exc_info:
        detector.compare_scenarios(stub_results, real_results)

    err = str(exc_info.value)
    assert "new_scenario_v2" in err


def test_drift_detector_catches_extra_stub_scenario():
    """StubDriftError raised when a scenario is in stub but missing from real."""
    stub_results, real_results = _make_aligned_results()
    # Stub implements a scenario that was removed from the real client
    stub_results["deprecated_scenario"] = {
        "ok": True,
        "result_keys": ["old_field"],
        "error_type": None,
    }

    detector = StubDriftDetector()
    with pytest.raises(StubDriftError) as exc_info:
        detector.compare_scenarios(stub_results, real_results)

    err = str(exc_info.value)
    assert "deprecated_scenario" in err


# ---------------------------------------------------------------------------
# Test: drift detector passes when aligned
# ---------------------------------------------------------------------------


def test_drift_detector_passes_when_aligned():
    """compare_scenarios does NOT raise when stub and real results are identical."""
    stub_results, real_results = _make_aligned_results()

    detector = StubDriftDetector()
    # Must not raise
    detector.compare_scenarios(stub_results, real_results)


def test_run_contract_test_scenarios_returns_expected_scenarios():
    """run_contract_test_scenarios returns all required scenario keys."""
    results = run_contract_test_scenarios()

    expected_scenarios = {
        "submit_and_get_score",
        "submit_score_with_uncertainty",
        "get_score_missing_raises",
        "propose_threshold_change",
    }
    assert expected_scenarios.issubset(results.keys()), (
        f"Missing scenarios: {expected_scenarios - results.keys()}"
    )


def test_run_contract_test_scenarios_submit_and_get_score_succeeds():
    """submit_and_get_score scenario succeeds and returns expected keys."""
    results = run_contract_test_scenarios()
    r = results["submit_and_get_score"]
    assert r["ok"] is True
    assert r["error_type"] is None
    # Score and basic fields must be present
    assert "score" in (r["result_keys"] or [])


def test_run_contract_test_scenarios_get_score_missing_raises():
    """get_score for an unknown wallet raises (ok=False) and error_type is set."""
    results = run_contract_test_scenarios()
    r = results["get_score_missing_raises"]
    assert r["ok"] is False
    assert r["error_type"] is not None


def test_stub_drift_error_message_contains_update_guidance():
    """StubDriftError message includes instructions to update the stub."""
    stub_results, real_results = _make_aligned_results()
    real_results["submit_and_get_score"]["ok"] = False
    real_results["submit_and_get_score"]["error_type"] = "SomeNewError"

    with pytest.raises(StubDriftError) as exc_info:
        StubDriftDetector().compare_scenarios(stub_results, real_results)

    assert "offline_stubs" in str(exc_info.value)
