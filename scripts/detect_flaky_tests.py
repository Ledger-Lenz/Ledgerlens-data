#!/usr/bin/env python3
"""Detect flaky tests by comparing two JUnit XML test result files.

A test is considered **flaky** if it FAILED in one run and PASSED in another.
Flaky tests are added to the quarantine registry with a timestamp so the
monthly audit can surface stale entries.

Usage:
    # Compare two runs and update quarantine registry
    python scripts/detect_flaky_tests.py \\
        --run1 reports/flaky/run1.xml \\
        --run2 reports/flaky/run2.xml \\
        --registry reports/flaky/quarantine_registry.json \\
        --output reports/flaky/flaky_report.json

    # Also include quarantine run results in the report
    python scripts/detect_flaky_tests.py \\
        --run1 reports/flaky/run1.xml \\
        --run2 reports/flaky/run2.xml \\
        --quarantine-xml reports/flaky/quarantine.xml \\
        --registry reports/flaky/quarantine_registry.json \\
        --output reports/flaky/flaky_report.json

    # Monthly audit — show stale quarantined tests (>30 days)
    python scripts/detect_flaky_tests.py \\
        --audit-only \\
        --registry reports/flaky/quarantine_registry.json \\
        --stale-days 30

Exit codes:
    0  Detection completed (flakiness may have been found but is tracked).
    1  Fatal error (missing required inputs, malformed XML, etc.).
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _parse_junit_xml(path: Path) -> dict[str, str]:
    """Parse a JUnit XML file and return {test_id: outcome}.

    Outcome is one of: 'passed', 'failed', 'error', 'skipped'.
    Test ID format: ``classname.testname`` (mirrors pytest's JUnit output).
    """
    if not path.exists():
        return {}

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        print(f"WARNING: Could not parse {path}: {exc}", file=sys.stderr)
        return {}

    root = tree.getroot()

    # JUnit XML can have either <testsuites><testsuite> or just <testsuite>
    testcases = root.iter("testcase")
    results: dict[str, str] = {}

    for tc in testcases:
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        test_id = f"{classname}::{name}" if classname else name

        if tc.find("failure") is not None or tc.find("error") is not None:
            outcome = "failed"
        elif tc.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"

        results[test_id] = outcome

    return results


def _detect_flaky(
    run1: dict[str, str],
    run2: dict[str, str],
) -> list[dict]:
    """Return newly detected flaky tests (pass in one run, fail in the other).

    A test is only flagged if it appeared in *both* runs (not just missing
    from one due to collection errors).
    """
    flaky: list[dict] = []
    common = set(run1.keys()) & set(run2.keys())

    for test_id in sorted(common):
        r1 = run1[test_id]
        r2 = run2[test_id]

        if r1 == "passed" and r2 == "failed":
            flaky.append({"test_id": test_id, "run1": r1, "run2": r2, "failed_run": 2})
        elif r1 == "failed" and r2 == "passed":
            flaky.append({"test_id": test_id, "run1": r1, "run2": r2, "failed_run": 1})

    return flaky


def _load_registry(path: Path) -> dict:
    """Load the quarantine registry JSON, returning an empty dict if missing."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_registry(registry: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, default=str))


def _update_registry(registry: dict, newly_flaky: list[dict]) -> dict:
    """Add newly detected flaky tests to the registry.

    Each entry records: test_id, quarantined_since, flaky_run_count, last_seen.
    """
    now = datetime.now(UTC).isoformat()

    if "tests" not in registry:
        registry["tests"] = {}

    for item in newly_flaky:
        tid = item["test_id"]
        if tid in registry["tests"]:
            registry["tests"][tid]["flaky_run_count"] = (
                registry["tests"][tid].get("flaky_run_count", 0) + 1
            )
            registry["tests"][tid]["last_seen"] = now
        else:
            registry["tests"][tid] = {
                "test_id": tid,
                "quarantined_since": now,
                "flaky_run_count": 1,
                "last_seen": now,
            }

    registry["updated_at"] = now
    return registry


def _quarantine_audit(registry: dict, stale_days: int) -> list[dict]:
    """Return registry entries older than *stale_days*."""
    now = datetime.now(UTC)
    stale: list[dict] = []

    for entry in registry.get("tests", {}).values():
        since_str = entry.get("quarantined_since")
        if since_str:
            try:
                since = datetime.fromisoformat(since_str)
                age = (now - since).days
                if age >= stale_days:
                    stale.append({**entry, "age_days": age})
            except ValueError:
                pass

    return stale


def _extract_quarantine_failures(path: Path) -> list[str]:
    """Return list of test IDs that failed in the quarantine run."""
    results = _parse_junit_xml(path)
    return [tid for tid, outcome in results.items() if outcome == "failed"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect flaky tests by comparing two JUnit XML result files."
    )
    parser.add_argument("--run1", type=Path, help="JUnit XML for run 1")
    parser.add_argument("--run2", type=Path, help="JUnit XML for run 2")
    parser.add_argument("--quarantine-xml", type=Path, help="JUnit XML for the quarantine run")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("reports/flaky/quarantine_registry.json"),
        help="Path to persistent quarantine registry JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/flaky/flaky_report.json"),
        help="Path to write the flaky detection report JSON",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only run the staleness audit (no XML comparison needed)",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=30,
        help="Days before a quarantined test is considered stale (default: 30)",
    )
    args = parser.parse_args()

    registry = _load_registry(args.registry)

    if args.audit_only:
        stale = _quarantine_audit(registry, args.stale_days)
        total = len(registry.get("tests", {}))

        print(f"\nQuarantine registry: {total} entries")
        print(f"Stale (>{args.stale_days} days): {len(stale)}\n")

        if stale:
            print("Stale quarantined tests — these need attention:")
            for entry in sorted(stale, key=lambda e: e.get("age_days", 0), reverse=True):
                print(
                    f"  [{entry['age_days']:3d} days]  {entry['test_id']}"
                    f"  (flaky count: {entry.get('flaky_run_count', '?')})"
                )
        else:
            print("No stale quarantined tests. All within the staleness window.")

        return 0

    # Normal mode: compare two runs
    if not args.run1 or not args.run2:
        print("ERROR: --run1 and --run2 are required in normal mode.", file=sys.stderr)
        return 1

    run1_results = _parse_junit_xml(args.run1)
    run2_results = _parse_junit_xml(args.run2)

    newly_flaky = _detect_flaky(run1_results, run2_results)
    registry = _update_registry(registry, newly_flaky)
    _save_registry(registry, args.registry)

    # Summarise quarantine run if provided
    quarantine_failures: list[str] = []
    if args.quarantine_xml:
        quarantine_failures = _extract_quarantine_failures(args.quarantine_xml)

    # Build report
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run1_tests": len(run1_results),
        "run2_tests": len(run2_results),
        "newly_detected_flaky": newly_flaky,
        "currently_quarantined": list(registry.get("tests", {}).values()),
        "quarantine_run_failures": quarantine_failures,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))

    # Print summary
    print(f"\nFlaky-test detection summary")
    print(f"  Run 1 tests:           {len(run1_results)}")
    print(f"  Run 2 tests:           {len(run2_results)}")
    print(f"  Newly detected flaky:  {len(newly_flaky)}")
    print(f"  Total quarantined:     {len(registry.get('tests', {}))}")
    print(f"  Quarantine failures:   {len(quarantine_failures)}")

    if newly_flaky:
        print("\nNewly detected flaky tests:")
        for item in newly_flaky:
            print(f"  {item['test_id']}  (failed in run {item['failed_run']})")
        print(
            "\nAction: Add @pytest.mark.quarantine to these tests and open a tracking issue."
        )

    if quarantine_failures:
        print("\nQuarantined tests that failed this run (non-blocking):")
        for tid in quarantine_failures:
            print(f"  {tid}")

    print(f"\nRegistry saved to: {args.registry}")
    print(f"Report saved to:   {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
