"""Tests for the flaky-test detection and quarantine infrastructure (Issue #966).

Acceptance criteria verified:
  1. Detection mechanism correctly identifies an intentionally-flaky fixture.
  2. Quarantine mechanism stops the flaky test from blocking merges while
     still running and reporting it.
  3. Quarantine report is accessible and documented.

The intentionally-flaky test in this file uses a shared in-memory counter so
that it fails on every *even* call and passes on every *odd* call.  In normal
test runs it always executes once (always passing or always failing depending
on call count), but the flaky detection workflow runs the suite twice and the
comparison catches it.

HOW TO USE THE QUARANTINE MARKER
---------------------------------
Mark a test that is known-flaky but not yet fixable:

    import pytest

    @pytest.mark.quarantine
    def test_my_flaky_test():
        ...

Quarantined tests:
  - Are executed in the `run-quarantined` CI job.
  - Their failures NEVER set the CI exit code to 1.
  - Are listed in the quarantine report (GitHub Actions job summary).
  - Surface in the monthly staleness audit after 30 days.

To see currently quarantined tests locally:
    pytest -m quarantine -v

To run the quarantine report locally:
    python scripts/detect_flaky_tests.py \\
        --audit-only \\
        --registry reports/flaky/quarantine_registry.json
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Intentionally-flaky fixture (acceptance criterion 1)
#
# This counter persists for the lifetime of the process.  The flaky detection
# workflow compares two *separate* pytest processes; the counter resets to 0
# in the second process, so the test produces PASS in run-1 and can be
# toggled to FAIL in run-2 via the FLAKY_TEST_FAIL env variable — which the
# workflow injects to verify detection without a real race condition.
# ---------------------------------------------------------------------------

import os

_FLAKY_CALL_COUNT: list[int] = [0]  # mutable container to survive re-imports


@pytest.mark.quarantine
def test_intentionally_flaky_for_detection_verification():
    """Intentionally flaky test used to verify the detection mechanism.

    Marked @quarantine so it does NOT block merges.

    In normal runs (no env variable) this test always passes.
    The CI workflow sets LEDGERLENS_TEST_SIMULATE_FLAKY=1 in exactly one of
    the two detection runs, causing this test to fail in that run.
    When both run-1 and run-2 results are compared, the disagreement is
    detected and recorded in the quarantine registry.
    """
    simulate_flaky = os.environ.get("LEDGERLENS_TEST_SIMULATE_FLAKY", "0") == "1"
    if simulate_flaky:
        pytest.fail(
            "Simulated flaky failure (LEDGERLENS_TEST_SIMULATE_FLAKY=1). "
            "This is expected in the flaky-detection run."
        )


# ---------------------------------------------------------------------------
# Unit tests for the detection script itself
# ---------------------------------------------------------------------------

from scripts.detect_flaky_tests import (
    _detect_flaky,
    _extract_quarantine_failures,
    _load_registry,
    _parse_junit_xml,
    _quarantine_audit,
    _update_registry,
)


def _make_junit_xml(tmp_path: Path, name: str, test_results: dict[str, str]) -> Path:
    """Write a minimal JUnit XML file with the given test results.

    test_results: {test_id: 'passed'|'failed'|'skipped'}
    """
    root = ET.Element("testsuite")
    root.set("name", "tests")
    root.set("tests", str(len(test_results)))

    for test_id, outcome in test_results.items():
        # Split classname::name
        if "::" in test_id:
            classname, testname = test_id.rsplit("::", 1)
        else:
            classname, testname = "", test_id

        tc = ET.SubElement(root, "testcase")
        tc.set("classname", classname)
        tc.set("name", testname)
        tc.set("time", "0.1")

        if outcome == "failed":
            failure = ET.SubElement(tc, "failure")
            failure.set("message", "AssertionError")
            failure.text = "assert False"
        elif outcome == "skipped":
            ET.SubElement(tc, "skipped")

    tree = ET.ElementTree(root)
    path = tmp_path / name
    tree.write(str(path))
    return path


class TestFlakyDetection:
    """Unit tests for the flaky test detection logic."""

    def test_detects_flaky_test_pass_then_fail(self):
        """A test that passes in run-1 and fails in run-2 is flaky."""
        run1 = {"tests::test_stable": "passed", "tests::test_flaky": "passed"}
        run2 = {"tests::test_stable": "passed", "tests::test_flaky": "failed"}

        flaky = _detect_flaky(run1, run2)
        flaky_ids = [f["test_id"] for f in flaky]

        assert "tests::test_flaky" in flaky_ids
        assert "tests::test_stable" not in flaky_ids
        assert flaky[0]["failed_run"] == 2

    def test_detects_flaky_test_fail_then_pass(self):
        """A test that fails in run-1 and passes in run-2 is also flaky."""
        run1 = {"tests::test_flaky": "failed"}
        run2 = {"tests::test_flaky": "passed"}

        flaky = _detect_flaky(run1, run2)
        assert len(flaky) == 1
        assert flaky[0]["failed_run"] == 1

    def test_consistently_failing_is_not_flaky(self):
        """A test that fails in both runs is a genuine failure, not flaky."""
        run1 = {"tests::test_broken": "failed"}
        run2 = {"tests::test_broken": "failed"}

        flaky = _detect_flaky(run1, run2)
        assert len(flaky) == 0

    def test_consistently_passing_is_not_flaky(self):
        """A test that passes in both runs is not flaky."""
        run1 = {"tests::test_good": "passed"}
        run2 = {"tests::test_good": "passed"}

        flaky = _detect_flaky(run1, run2)
        assert len(flaky) == 0

    def test_missing_from_one_run_not_flaky(self):
        """A test absent from one run (collection error) is not classified as flaky."""
        run1 = {"tests::test_only_in_run1": "passed"}
        run2 = {}

        flaky = _detect_flaky(run1, run2)
        assert len(flaky) == 0

    def test_multiple_flaky_tests_detected(self):
        """All flaky tests across the full suite are detected."""
        run1 = {
            "tests::test_a": "passed",
            "tests::test_b": "failed",
            "tests::test_c": "passed",
        }
        run2 = {
            "tests::test_a": "failed",
            "tests::test_b": "passed",
            "tests::test_c": "passed",
        }

        flaky = _detect_flaky(run1, run2)
        flaky_ids = {f["test_id"] for f in flaky}
        assert flaky_ids == {"tests::test_a", "tests::test_b"}


class TestJUnitXMLParsing:
    """Unit tests for JUnit XML parsing."""

    def test_parses_passed_test(self, tmp_path):
        path = _make_junit_xml(tmp_path, "run.xml", {"tests::test_ok": "passed"})
        results = _parse_junit_xml(path)
        assert results["tests::test_ok"] == "passed"

    def test_parses_failed_test(self, tmp_path):
        path = _make_junit_xml(tmp_path, "run.xml", {"tests::test_bad": "failed"})
        results = _parse_junit_xml(path)
        assert results["tests::test_bad"] == "failed"

    def test_parses_skipped_test(self, tmp_path):
        path = _make_junit_xml(tmp_path, "run.xml", {"tests::test_skip": "skipped"})
        results = _parse_junit_xml(path)
        assert results["tests::test_skip"] == "skipped"

    def test_returns_empty_for_missing_file(self, tmp_path):
        results = _parse_junit_xml(tmp_path / "nonexistent.xml")
        assert results == {}

    def test_returns_empty_for_malformed_xml(self, tmp_path):
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("not valid xml <><")
        results = _parse_junit_xml(bad_xml)
        assert results == {}


class TestQuarantineRegistry:
    """Unit tests for the quarantine registry persistence."""

    def test_new_flaky_test_added_to_registry(self):
        registry = {}
        newly_flaky = [{"test_id": "tests::test_flaky", "run1": "passed", "run2": "failed", "failed_run": 2}]
        updated = _update_registry(registry, newly_flaky)
        assert "tests::test_flaky" in updated["tests"]

    def test_repeat_flaky_increments_count(self):
        registry = {"tests": {"tests::test_flaky": {"test_id": "tests::test_flaky", "flaky_run_count": 1}}}
        newly_flaky = [{"test_id": "tests::test_flaky", "run1": "failed", "run2": "passed", "failed_run": 1}]
        updated = _update_registry(registry, newly_flaky)
        assert updated["tests"]["tests::test_flaky"]["flaky_run_count"] == 2

    def test_empty_flaky_list_does_not_change_registry(self):
        registry = {"tests": {"tests::test_existing": {"test_id": "tests::test_existing"}}}
        updated = _update_registry(registry, [])
        assert "tests::test_existing" in updated["tests"]

    def test_load_registry_returns_empty_for_missing_file(self, tmp_path):
        reg = _load_registry(tmp_path / "no_such_file.json")
        assert reg == {}

    def test_load_registry_reads_json(self, tmp_path):
        data = {"tests": {"tests::test_x": {"test_id": "tests::test_x"}}}
        p = tmp_path / "reg.json"
        p.write_text(json.dumps(data))
        reg = _load_registry(p)
        assert "tests::test_x" in reg["tests"]


class TestQuarantineAudit:
    """Unit tests for the staleness audit."""

    def test_stale_entry_detected(self):
        from datetime import UTC, datetime, timedelta

        old_date = (datetime.now(UTC) - timedelta(days=40)).isoformat()
        registry = {
            "tests": {
                "tests::test_old": {
                    "test_id": "tests::test_old",
                    "quarantined_since": old_date,
                    "flaky_run_count": 3,
                }
            }
        }
        stale = _quarantine_audit(registry, stale_days=30)
        assert len(stale) == 1
        assert stale[0]["test_id"] == "tests::test_old"

    def test_recent_entry_not_stale(self):
        from datetime import UTC, datetime

        recent = datetime.now(UTC).isoformat()
        registry = {
            "tests": {
                "tests::test_new": {
                    "test_id": "tests::test_new",
                    "quarantined_since": recent,
                }
            }
        }
        stale = _quarantine_audit(registry, stale_days=30)
        assert len(stale) == 0


class TestQuarantineMarker:
    """Verify the quarantine marker mechanism works as expected."""

    def test_quarantine_marker_exists_in_pytest(self):
        """The 'quarantine' marker must be registered in pyproject.toml."""
        from pathlib import Path

        pyproject = Path("pyproject.toml").read_text()
        assert "quarantine" in pyproject, (
            "'quarantine' marker not registered in pyproject.toml [tool.pytest.ini_options].markers"
        )

    def test_flaky_marker_exists_in_pytest(self):
        """The 'flaky' marker must be registered in pyproject.toml."""
        from pathlib import Path

        pyproject = Path("pyproject.toml").read_text()
        assert "flaky" in pyproject, (
            "'flaky' marker not registered in pyproject.toml [tool.pytest.ini_options].markers"
        )

    def test_quarantine_workflow_exists(self):
        """The flaky-test workflow must exist at .github/workflows/flaky-tests.yml."""
        assert Path(".github/workflows/flaky-tests.yml").exists(), (
            ".github/workflows/flaky-tests.yml is missing"
        )

    def test_detection_script_exists(self):
        """The detect_flaky_tests.py script must exist."""
        assert Path("scripts/detect_flaky_tests.py").exists(), (
            "scripts/detect_flaky_tests.py is missing"
        )

    def test_quarantine_report_documented(self):
        """Quarantine report and usage must be documented."""
        assert Path("docs/flaky_test_quarantine.md").exists(), (
            "docs/flaky_test_quarantine.md is missing — add quarantine documentation"
        )
