"""Tests for the report schema compatibility check (Issue #947).

Verifies that:
1. Breaking changes (removed required field, new required field, type narrowing,
   enum removal, title change) are correctly detected.
2. Backward-compatible changes (new optional field, description update, new
   enum value) are not flagged.
3. The check passes cleanly against the current schemas.
4. Every current schema carries a schema_version (or schemaVersion) field.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_report_schema_compatibility import (
    EXIT_INCOMPATIBLE,
    EXIT_OK,
    BaselineUnavailable,
    BaselineRefError,
    check_report_schema_compatibility,
    main,
)

SCHEMAS_DIR = REPO_ROOT / "reporting" / "schemas"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_schema(name: str = "TestSchema") -> dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": name,
        "type": "object",
        "required": ["wallet", "risk_score"],
        "properties": {
            "schema_version": {"type": "string"},
            "wallet": {"type": "string"},
            "risk_score": {"type": "number"},
            "verdict": {"type": "string", "enum": ["clean", "suspicious", "flagged"]},
        },
    }


def _reader(baseline: dict):
    """Return an injectable baseline reader that always returns *baseline*."""

    def read(ref: str, path: str) -> dict:
        return copy.deepcopy(baseline)

    return read


# ---------------------------------------------------------------------------
# Unit tests for check_report_schema_compatibility()
# ---------------------------------------------------------------------------


class TestBreakingChanges:
    def test_removed_required_field_is_violation(self):
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        current["required"].remove("wallet")
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert any("removed" in v and "wallet" in v for v in violations), violations

    def test_new_required_field_is_violation(self):
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        current["required"].append("new_mandatory")
        current["properties"]["new_mandatory"] = {"type": "string"}
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert any("new_mandatory" in v for v in violations), violations

    def test_type_narrowing_is_violation(self):
        baseline = _base_schema()
        baseline["properties"]["risk_score"]["type"] = ["number", "null"]
        current = copy.deepcopy(baseline)
        current["properties"]["risk_score"]["type"] = "number"  # removed null
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert any("narrowed" in v for v in violations), violations

    def test_enum_value_removal_is_violation(self):
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        current["properties"]["verdict"]["enum"] = ["clean", "flagged"]  # removed "suspicious"
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert any("suspicious" in v for v in violations), violations

    def test_title_change_is_violation(self):
        baseline = _base_schema("OriginalTitle")
        current = copy.deepcopy(baseline)
        current["title"] = "RenamedTitle"
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert any("title" in v.lower() for v in violations), violations

    def test_missing_schema_version_field_is_violation(self):
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        # Remove schema_version
        current["properties"].pop("schema_version", None)
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert any("schema_version" in v for v in violations), violations


class TestCompatibleChanges:
    def test_new_optional_field_is_not_a_violation(self):
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        current["properties"]["new_optional"] = {"type": "string"}
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert violations == []

    def test_description_update_is_not_a_violation(self):
        baseline = _base_schema()
        baseline["properties"]["wallet"]["description"] = "Stellar account"
        current = copy.deepcopy(baseline)
        current["properties"]["wallet"]["description"] = "Stellar account ID (G...)"
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert violations == []

    def test_new_enum_value_is_not_a_violation(self):
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        current["properties"]["verdict"]["enum"] = [
            "clean", "suspicious", "flagged", "inconclusive"
        ]
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert violations == []

    def test_type_relaxation_is_not_a_violation(self):
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        current["properties"]["risk_score"]["type"] = ["number", "null"]  # relaxed
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert violations == []

    def test_identical_schemas_produce_no_violations(self):
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        violations = check_report_schema_compatibility(baseline, current, "test")
        assert violations == []


# ---------------------------------------------------------------------------
# Integration tests running main() against real schema files
# ---------------------------------------------------------------------------


class TestCurrentSchemasPassCheck:
    """CI gate: the current schemas must be self-consistent and carry version."""

    def test_current_schemas_have_schema_version(self):
        """Every schema file must carry a schema_version or schemaVersion field."""
        for schema_file in sorted(SCHEMAS_DIR.glob("*.json")):
            schema = json.loads(schema_file.read_text())
            props = schema.get("properties", {})

            # Check top-level schema_version
            has_top_level = "schema_version" in props

            # Check nested payloadMetadata.schemaVersion (IVMS101 pattern)
            payload_meta = props.get("payloadMetadata", {})
            payload_props = (
                payload_meta.get("properties", {}) if isinstance(payload_meta, dict) else {}
            )
            has_nested = "schemaVersion" in payload_props

            assert has_top_level or has_nested, (
                f"{schema_file.name}: missing 'schema_version' (or "
                "'payloadMetadata.schemaVersion') — see "
                "docs/report_schema_deprecation_policy.md"
            )

    def test_main_passes_against_identical_baseline(self):
        """Identical baseline → no violations → exit 0."""
        # Build a reader that returns the *current* file as the baseline too
        def reader(ref: str, path: str) -> dict:
            full_path = REPO_ROOT / path
            if full_path.is_file():
                return json.loads(full_path.read_text())
            raise BaselineUnavailable(path)

        exit_code = main([], baseline_reader=reader)
        assert exit_code == EXIT_OK

    def test_breaking_change_caught_by_main(self):
        """Deliberately break the model_metadata schema — main() returns exit 1."""
        # Load the current model_metadata schema as the baseline
        model_meta_path = SCHEMAS_DIR / "model_metadata.json"
        baseline = json.loads(model_meta_path.read_text())

        # Remove a required field from the "current" version
        current = copy.deepcopy(baseline)
        current["required"].remove("model_name")

        # The "current" files on disk are the originals; we inject a broken
        # current by monkeypatching the schemas_dir to a temp directory.
        def reader(ref: str, path: str) -> dict:
            # Return baseline for model_metadata, not-found for others
            if "model_metadata" in path:
                return baseline
            raise BaselineUnavailable(path)

        import tempfile, shutil

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tmp_schemas = tmp_path / "schemas"
            tmp_schemas.mkdir()
            # Write the broken schema
            (tmp_schemas / "model_metadata.json").write_text(
                json.dumps(current, indent=2)
            )

            exit_code = main(
                [f"--schemas-dir={tmp_schemas}"],
                baseline_reader=reader,
            )

        assert exit_code == EXIT_INCOMPATIBLE, (
            "Expected exit code 1 when a required field is removed, "
            f"but got {exit_code}"
        )

    def test_new_schema_no_baseline_exits_ok(self):
        """A schema with no baseline (new file) should exit 0."""
        def reader(ref: str, path: str) -> dict:
            raise BaselineUnavailable(path)

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            new_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "NewReportType",
                "type": "object",
                "required": ["schema_version"],
                "properties": {"schema_version": {"type": "string"}},
            }
            (tmp_path / "new_report.json").write_text(json.dumps(new_schema))

            exit_code = main(
                [f"--schemas-dir={tmp_path}"],
                baseline_reader=reader,
            )

        assert exit_code == EXIT_OK

    def test_dry_run_returns_ok_despite_violations(self):
        """--dry-run always exits 0 even with violations."""
        baseline = _base_schema()
        current = copy.deepcopy(baseline)
        current["required"].remove("wallet")  # breaking

        def reader(ref: str, path: str) -> dict:
            return baseline

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.json"
            broken.write_text(json.dumps(current))
            exit_code = main(
                [f"--schemas-dir={tmp}", "--dry-run"],
                baseline_reader=reader,
            )

        assert exit_code == EXIT_OK
