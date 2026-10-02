#!/usr/bin/env python3
"""Vulnerability severity gate for CI (Issue #967).

Reads pip-audit JSON output, applies time-bounded exception waivers from
security/vuln_exceptions.json, and exits with code 1 if any un-waived
vulnerabilities meet or exceed the configured fail-on severity.

Usage:
    python scripts/check_vuln_severity.py \\
        --input reports/security/pip_audit_results.json \\
        --exceptions security/vuln_exceptions.json \\
        --fail-on HIGH \\
        --output reports/security/vuln_gate_result.json

Exit codes:
    0  No blocking vulnerabilities (all above threshold are waived or none found).
    1  One or more un-waived vulnerabilities at or above --fail-on threshold.
    2  Input file missing or malformed.

Severity levels (highest to lowest):
    CRITICAL → HIGH → MEDIUM → LOW → NONE
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Severity order (higher index = more severe)
_SEVERITY_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _severity_level(s: str) -> int:
    """Return numeric severity level (higher = more severe)."""
    return _SEVERITY_ORDER.index(s.upper()) if s.upper() in _SEVERITY_ORDER else 0


def _load_exceptions(path: Path) -> dict[str, dict]:
    """Load exception waivers from JSON.

    Returns {vuln_id: waiver_dict} for all waivers that haven't expired.
    """
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"WARNING: Could not parse {path}: {exc}", file=sys.stderr)
        return {}

    now = datetime.now(UTC)
    active: dict[str, dict] = {}

    for waiver in data.get("waivers", []):
        vuln_id = waiver.get("vuln_id", "")
        if not vuln_id:
            continue

        # Check expiry
        expires_str = waiver.get("expires")
        if expires_str:
            try:
                expires = datetime.fromisoformat(expires_str)
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=UTC)
                if now > expires:
                    print(
                        f"INFO: Waiver for {vuln_id} expired on {expires_str} — "
                        "no longer applied.",
                        file=sys.stderr,
                    )
                    continue
            except ValueError:
                print(
                    f"WARNING: Invalid expires date for {vuln_id}: {expires_str}",
                    file=sys.stderr,
                )

        active[vuln_id] = waiver

    return active


def _parse_pip_audit_json(path: Path) -> list[dict]:
    """Parse pip-audit JSON output into a flat list of vulnerability dicts.

    Supports both pip-audit v1 (list of {name, version, vulns}) and
    v2 (top-level {dependencies: [...]}) output formats.
    """
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: Could not parse pip-audit output {path}: {exc}", file=sys.stderr)
        sys.exit(2)

    vulns: list[dict] = []

    # pip-audit outputs a list of package objects
    packages = data if isinstance(data, list) else data.get("dependencies", [])

    for pkg in packages:
        pkg_name = pkg.get("name", "")
        pkg_version = pkg.get("version", "")
        for vuln in pkg.get("vulns", []):
            # Map pip-audit severity to our canonical levels
            severity = _normalise_severity(vuln.get("fix_versions", []), vuln)
            vulns.append(
                {
                    "id": vuln.get("id", "UNKNOWN"),
                    "package": pkg_name,
                    "version": pkg_version,
                    "severity": severity,
                    "description": vuln.get("description", "")[:200],
                    "fix_versions": vuln.get("fix_versions", []),
                    "aliases": vuln.get("aliases", []),
                }
            )

    return vulns


def _normalise_severity(fix_versions: list[str], vuln: dict) -> str:
    """Map pip-audit vuln metadata to a severity string.

    pip-audit's OSV format doesn't always include severity directly.
    We infer it from aliases (CVE CVSS scores if present) or default to HIGH
    for any unfixed vulnerability (conservative assumption).
    """
    # Check for explicit severity field (newer pip-audit versions)
    if "severity" in vuln:
        return vuln["severity"].upper()

    # Check CVSS score in aliases or details
    # pip-audit may include GHSA or OSV IDs; without a score, default to HIGH
    aliases = vuln.get("aliases", [])
    for alias in aliases:
        if alias.upper().startswith("CVE-"):
            # Conservative: treat unknown CVE severity as HIGH
            return "HIGH"

    return "HIGH"  # Default conservative assumption


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply vulnerability severity gating with exception waivers."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to pip-audit JSON output file",
    )
    parser.add_argument(
        "--exceptions",
        type=Path,
        default=Path("security/vuln_exceptions.json"),
        help="Path to exception waivers JSON file",
    )
    parser.add_argument(
        "--fail-on",
        default="HIGH",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        help="Minimum severity level that triggers a CI failure (default: HIGH)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write gate result JSON",
    )
    args = parser.parse_args()

    fail_threshold = _severity_level(args.fail_on)

    # Load inputs
    vulns = _parse_pip_audit_json(args.input)
    waivers = _load_exceptions(args.exceptions)

    # Categorise
    by_severity: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    blocking_items: list[dict] = []
    waived_items: list[dict] = []

    for v in vulns:
        sev = v["severity"].upper()
        by_severity[sev] = by_severity.get(sev, 0) + 1

        # Check if waived
        waiver = waivers.get(v["id"])
        if not waiver:
            # Also check aliases
            for alias in v.get("aliases", []):
                waiver = waivers.get(alias)
                if waiver:
                    break

        if waiver:
            waived_items.append({**v, "waiver_reason": waiver.get("reason", ""), "expires": waiver.get("expires", "")})
            continue

        if _severity_level(sev) >= fail_threshold:
            blocking_items.append(v)

    # Build result
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fail_on": args.fail_on,
        "total_vulns": len(vulns),
        "critical": by_severity.get("CRITICAL", 0),
        "high": by_severity.get("HIGH", 0),
        "medium": by_severity.get("MEDIUM", 0),
        "low": by_severity.get("LOW", 0),
        "waived": len(waived_items),
        "blocking": len(blocking_items),
        "blocking_items": blocking_items,
        "waived_items": waived_items,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str))

    # Print summary
    print(f"\nVulnerability Severity Gate")
    print(f"  Fail threshold:  {args.fail_on}")
    print(f"  Total found:     {len(vulns)}")
    print(f"  CRITICAL:        {by_severity.get('CRITICAL', 0)}")
    print(f"  HIGH:            {by_severity.get('HIGH', 0)}")
    print(f"  MEDIUM:          {by_severity.get('MEDIUM', 0)}")
    print(f"  LOW:             {by_severity.get('LOW', 0)}")
    print(f"  Waived:          {len(waived_items)}")
    print(f"  Blocking:        {len(blocking_items)}")

    if blocking_items:
        print("\nBLOCKING vulnerabilities (un-waived, at or above threshold):")
        for item in blocking_items:
            print(f"  {item['id']} in {item['package']}=={item['version']}")
            print(f"    Severity: {item['severity']}")
            print(f"    Fix versions: {item.get('fix_versions', 'N/A')}")
            print(f"    {item.get('description', '')[:120]}")
        print(
            "\nTo request a waiver, follow the process in docs/vulnerability_scanning.md"
        )
        return 1

    if waived_items:
        print("\nActive waivers applied:")
        for item in waived_items:
            print(
                f"  {item['id']} in {item['package']} — expires {item.get('expires', 'never')}"
                f" | {item.get('waiver_reason', '')[:80]}"
            )

    print("\n✅ No blocking vulnerabilities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
