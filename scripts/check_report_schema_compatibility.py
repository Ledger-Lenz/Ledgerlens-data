"""Enforce backward-compatible evolution for LedgerLens report schemas (Issue #947).

Compares every JSON schema under ``reporting/schemas/`` in the working tree
against the same files at the git baseline ref (the PR target branch), and
fails when a change would break downstream consumers (regulators, exchange
partners) who are parsing exported reports.

Compatibility rules for JSON-Schema report files:

* **Required field removed** — consumers that expect the field will break.
* **Required field type narrowed** — a previously valid value may now be invalid.
* **New field added as required** — consumers that populate reports from the old
  schema will be missing the field.
* **Enum values removed from an existing field** — consumers relying on a removed
  value will produce invalid exports.
* **Schema title/id changed** — downstream consumers that identify the schema by
  title will break.

Safe (backward-compatible) changes:

* Adding a new *optional* property.
* Relaxing a type constraint (e.g. ``"integer"`` → ``["integer", "null"]``).
* Adding a new enum value.
* Adding / updating ``description`` fields.
* Version bump in ``$id`` url (only the final path component).

The ``schema_version`` field in every schema document tracks the semantic
version of the report format and must follow the policy defined in
``docs/report_schema_deprecation_policy.md``.

Usage::

    python scripts/check_report_schema_compatibility.py
    python scripts/check_report_schema_compatibility.py --baseline-ref origin/main
    python scripts/check_report_schema_compatibility.py --schemas-dir reporting/schemas
    python scripts/check_report_schema_compatibility.py --dry-run

Exit codes:
    0  All report schemas are compatible with (or identical to) the baseline.
    1  One or more schemas have breaking changes — violations are printed.
    2  The baseline git ref could not be resolved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMAS_DIR = "reporting/schemas"
DEFAULT_BASELINE_REF = "origin/main"

EXIT_OK = 0
EXIT_INCOMPATIBLE = 1
EXIT_BASELINE_UNRESOLVED = 2


class BaselineUnavailable(Exception):
    """The baseline ref exists but does not contain the schema file."""


class BaselineRefError(Exception):
    """The baseline ref itself could not be resolved."""


def default_baseline_ref() -> str:
    base = os.environ.get("GITHUB_BASE_REF", "").strip()
    return f"origin/{base}" if base else DEFAULT_BASELINE_REF


def read_baseline_schema(ref: str, schema_path: str) -> dict:
    """Return the JSON schema as it exists at *ref* in git.

    Raises:
        BaselineUnavailable: The ref exists but has no such file (new schema).
        BaselineRefError: The ref itself cannot be resolved.
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise BaselineRefError(f"cannot resolve baseline ref {ref!r}") from exc

    result = subprocess.run(
        ["git", "show", f"{ref}:{schema_path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BaselineUnavailable(f"{schema_path} does not exist at {ref}")
    return json.loads(result.stdout)


def _required_fields(schema: dict) -> set[str]:
    """Return top-level required field names."""
    return set(schema.get("required", []))


def _properties(schema: dict) -> dict:
    return schema.get("properties", {})


def _schema_id(schema: dict) -> str | None:
    return schema.get("$id") or schema.get("id")


def _schema_title(schema: dict) -> str | None:
    return schema.get("title")


def check_report_schema_compatibility(
    baseline: dict, current: dict, schema_name: str
) -> list[str]:
    """Return a list of compatibility violation descriptions.

    An empty list means the change is backward-compatible.
    """
    violations: list[str] = []

    # Rule 1: title must not change
    if _schema_title(baseline) and _schema_title(baseline) != _schema_title(current):
        violations.append(
            f"{schema_name}: schema title changed from "
            f"{_schema_title(baseline)!r} to {_schema_title(current)!r} — "
            "downstream consumers that identify this schema by title will break."
        )

    # Rule 2: required fields must not be removed
    baseline_required = _required_fields(baseline)
    current_required = _required_fields(current)
    removed_required = baseline_required - current_required
    if removed_required:
        violations.append(
            f"{schema_name}: required field(s) removed — consumers expecting "
            f"these fields will fail validation: {sorted(removed_required)}"
        )

    # Rule 3: new required fields must not be added (would break old producers)
    added_required = current_required - baseline_required
    if added_required:
        violations.append(
            f"{schema_name}: new required field(s) added — exports produced "
            f"by old code will be missing these fields and fail validation: "
            f"{sorted(added_required)}"
        )

    # Rule 4: existing property types must not be narrowed
    baseline_props = _properties(baseline)
    current_props = _properties(current)
    for field_name, baseline_spec in baseline_props.items():
        if field_name not in current_props:
            # Field removed — only a violation if it was required (covered above)
            continue
        current_spec = current_props[field_name]
        b_type = baseline_spec.get("type")
        c_type = current_spec.get("type")
        if b_type is None or c_type is None:
            continue
        # Narrowing: was a list/union, now a single type
        b_types = set(b_type) if isinstance(b_type, list) else {b_type}
        c_types = set(c_type) if isinstance(c_type, list) else {c_type}
        removed_types = b_types - c_types
        if removed_types:
            violations.append(
                f"{schema_name}: field {field_name!r} type narrowed — "
                f"type(s) {sorted(removed_types)} were valid before but are now "
                "rejected. This is a breaking change for existing exports."
            )

        # Enum narrowing: values removed from an enum
        b_enum = set(baseline_spec.get("enum", []))
        c_enum = set(current_spec.get("enum", []))
        removed_enum = b_enum - c_enum
        if removed_enum:
            violations.append(
                f"{schema_name}: field {field_name!r} enum values removed: "
                f"{sorted(str(v) for v in removed_enum)} — existing exports "
                "containing these values will now fail validation."
            )

    # Rule 5: schema_version field must be present
    if "schema_version" not in current_props and "schemaVersion" not in current_props:
        # Check nested payloadMetadata.schemaVersion (IVMS101 pattern)
        payload_meta = current_props.get("payloadMetadata", {})
        payload_props = (
            payload_meta.get("properties", {}) if isinstance(payload_meta, dict) else {}
        )
        if "schemaVersion" not in payload_props:
            violations.append(
                f"{schema_name}: 'schema_version' (or 'schemaVersion' in "
                "'payloadMetadata') is missing. All report schemas must carry "
                "an explicit version field so downstream consumers can detect "
                "schema changes. Add it as a top-level 'schema_version' property "
                "following the semver pattern documented in "
                "docs/report_schema_deprecation_policy.md."
            )

    return violations


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce backward-compatible evolution for LedgerLens report schemas.",
    )
    parser.add_argument(
        "--schemas-dir",
        default=DEFAULT_SCHEMAS_DIR,
        help=f"Directory of JSON schema files (default: {DEFAULT_SCHEMAS_DIR}).",
    )
    parser.add_argument(
        "--baseline-ref",
        default=None,
        help="Git ref to compare against (default: GITHUB_BASE_REF, else origin/main).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report violations but always exit 0.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    baseline_reader=read_baseline_schema,
) -> int:
    """Entry point. *baseline_reader* is injectable so tests need no git."""
    args = _parse_args(argv)
    ref = args.baseline_ref or default_baseline_ref()
    schemas_dir = REPO_ROOT / args.schemas_dir

    if not schemas_dir.is_dir():
        print(f"Schemas directory not found: {schemas_dir}")
        return EXIT_OK if args.dry_run else EXIT_BASELINE_UNRESOLVED

    schema_files = sorted(schemas_dir.glob("*.json"))
    if not schema_files:
        print(f"No JSON schema files found in {schemas_dir}")
        return EXIT_OK

    all_violations: list[str] = []
    new_schemas: list[str] = []
    unchanged: list[str] = []

    for schema_path in schema_files:
        try:
            relative_path = str(schema_path.relative_to(REPO_ROOT))
        except ValueError:
            # Path is outside REPO_ROOT (e.g. a temp dir in tests)
            relative_path = str(schema_path)
        schema_name = schema_path.stem
        current = json.loads(schema_path.read_text(encoding="utf-8"))

        try:
            baseline = baseline_reader(ref, relative_path)
        except BaselineUnavailable:
            new_schemas.append(schema_name)
            continue
        except BaselineRefError as exc:
            print(f"Cannot resolve baseline ref {ref!r}: {exc}")
            print(
                "Hint: run with fetch-depth: 0 and fetch the base branch before "
                "running this check."
            )
            return EXIT_OK if args.dry_run else EXIT_BASELINE_UNRESOLVED

        if baseline == current:
            unchanged.append(schema_name)
            continue

        violations = check_report_schema_compatibility(baseline, current, schema_name)
        all_violations.extend(violations)

    # Summary
    if new_schemas:
        print(f"New schemas (no baseline): {new_schemas}")
    if unchanged:
        print(f"Unchanged schemas: {unchanged}")

    if not all_violations:
        print(
            f"Report schema compatibility check passed "
            f"({len(schema_files)} schema(s) checked)."
        )
        return EXIT_OK

    print(f"\nReport schema compatibility violations ({len(all_violations)}):")
    for v in all_violations:
        print(f"  {v}")
    print()
    print(
        "See docs/report_schema_deprecation_policy.md for the allowed change "
        "categories and the version-bump procedure."
    )

    if args.dry_run:
        print("(dry run — not failing)")
        return EXIT_OK
    return EXIT_INCOMPATIBLE


if __name__ == "__main__":
    raise SystemExit(main())
