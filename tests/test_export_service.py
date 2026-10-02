"""Tests for `reporting.export_service` — typed report export contracts and
data-minimization profiles (Issue #945)."""

from __future__ import annotations

import csv
import io
import json

import pytest

from reporting.export_service import (
    DEFAULT_PROFILE_NAME,
    EXPORT_REGISTRY,
    PROFILE_PARTNER_AGGREGATE,
    PROFILE_REGULATOR_FULL,
    PROFILE_REGISTRY,
    CSVExporter,
    ExportProfile,
    ExportResult,
    FieldSpec,
    JSONExporter,
    NDJSONExporter,
    ReportSchema,
    SchemaValidationError,
    UnknownProfileError,
    UnsupportedFormatError,
    export_report,
    register_exporter,
    register_profile,
)

SCHEMA = ReportSchema(
    fields=[
        FieldSpec("wallet", str, required=True),
        FieldSpec("risk_score", (int, float), required=True),
        FieldSpec("note", str, required=False),
    ]
)

RECORDS = [
    {"wallet": "GA1", "risk_score": 85, "note": "flagged"},
    {"wallet": "GA2", "risk_score": 12.5},
]


class TestReportSchema:
    def test_valid_batch_passes(self):
        SCHEMA.validate_batch(RECORDS)  # should not raise

    def test_missing_required_field_raises_with_row_and_field(self):
        bad = [{"wallet": "GA1"}]
        with pytest.raises(SchemaValidationError) as exc:
            SCHEMA.validate_batch(bad)
        assert exc.value.row_index == 0
        assert exc.value.field_name == "risk_score"

    def test_wrong_type_raises(self):
        bad = [{"wallet": "GA1", "risk_score": "not-a-number"}]
        with pytest.raises(SchemaValidationError):
            SCHEMA.validate_batch(bad)

    def test_optional_field_may_be_absent(self):
        ok = [{"wallet": "GA1", "risk_score": 1}]
        SCHEMA.validate_batch(ok)  # should not raise

    def test_field_names(self):
        assert SCHEMA.field_names() == ["wallet", "risk_score", "note"]


class TestJSONExporter:
    def test_exports_valid_json_array(self):
        result = JSONExporter().export(RECORDS, schema=SCHEMA)
        assert isinstance(result, ExportResult)
        assert result.content_type == "application/json"
        parsed = json.loads(result.content)
        assert len(parsed) == 2

    def test_validates_against_schema(self):
        with pytest.raises(SchemaValidationError):
            JSONExporter().export([{"wallet": "GA1"}], schema=SCHEMA)

    def test_checksum_is_deterministic(self):
        r1 = JSONExporter().export(RECORDS)
        r2 = JSONExporter().export(RECORDS)
        assert r1.checksum == r2.checksum


class TestNDJSONExporter:
    def test_one_json_object_per_line(self):
        result = NDJSONExporter().export(RECORDS, schema=SCHEMA)
        lines = result.content.decode("utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["wallet"] == "GA1"

    def test_empty_batch_produces_empty_body(self):
        result = NDJSONExporter().export([])
        assert result.content == b""
        assert result.record_count == 0


class TestCSVExporter:
    def test_uses_schema_column_order(self):
        result = CSVExporter().export(RECORDS, schema=SCHEMA)
        reader = csv.reader(io.StringIO(result.content.decode("utf-8")))
        header = next(reader)
        assert header == ["wallet", "risk_score", "note"]

    def test_missing_optional_value_is_blank_cell(self):
        result = CSVExporter().export(RECORDS, schema=SCHEMA)
        rows = list(csv.DictReader(io.StringIO(result.content.decode("utf-8"))))
        assert rows[1]["note"] == ""

    def test_infers_columns_without_schema(self):
        result = CSVExporter().export([{"b": 1, "a": 2}])
        reader = csv.reader(io.StringIO(result.content.decode("utf-8")))
        header = next(reader)
        assert header == ["a", "b"]  # sorted for determinism


class TestExportReport:
    def test_dispatches_by_format(self):
        result = export_report(RECORDS, fmt="csv", schema=SCHEMA, profile="regulator_full")
        assert result.content_type == "text/csv"

    def test_unknown_format_lists_available(self):
        with pytest.raises(UnsupportedFormatError) as exc:
            export_report(RECORDS, fmt="xml")
        assert "json" in str(exc.value)

    def test_register_custom_exporter(self):
        class UpperCsvExporter:
            format_name = "csv_upper"
            content_type = "text/csv"

            def export(self, records, schema=None):
                from reporting.export_service import _make_result

                body = "\n".join(str(r).upper() for r in records)
                return _make_result(body, self.content_type, "report.csv", len(records))

        register_exporter(UpperCsvExporter())
        try:
            result = export_report(RECORDS, fmt="csv_upper", profile="regulator_full")
            assert b"GA1" in result.content
        finally:
            del EXPORT_REGISTRY["csv_upper"]


# ---------------------------------------------------------------------------
# Issue #945 — Data-minimization profile tests
# ---------------------------------------------------------------------------

#: Records with sensitive fields that should be excluded in aggregate profiles.
SENSITIVE_RECORDS = [
    {
        "wallet": "GAABC123",
        "risk_score": 85,
        "score_band": "high",
        "shap_values": {"benford_mad_1h": 0.42, "counterparty_concentration_ratio": 0.31},
        "trades": [{"id": "t1", "amount": 1000}],
        "feature_vector": [0.1, 0.2, 0.3],
        "benford_flag": True,
        "trade_count": 150,
        "avg_trade_size": 2340.5,
    },
    {
        "wallet": "GBDEF456",
        "risk_score": 22,
        "score_band": "low",
        "shap_values": {"benford_mad_1h": 0.01},
        "trades": [{"id": "t2", "amount": 500}],
        "feature_vector": [0.4, 0.5],
        "benford_flag": False,
        "trade_count": 30,
        "avg_trade_size": 750.0,
    },
]


class TestExportProfile:
    """Unit tests for ExportProfile.apply()."""

    def test_aggregate_profile_removes_wallet_field(self):
        result = PROFILE_PARTNER_AGGREGATE.apply(SENSITIVE_RECORDS)
        for row in result:
            assert "wallet" not in row, (
                "partner_aggregate profile must not leak wallet identifiers"
            )

    def test_aggregate_profile_removes_shap_values(self):
        result = PROFILE_PARTNER_AGGREGATE.apply(SENSITIVE_RECORDS)
        for row in result:
            assert "shap_values" not in row, (
                "partner_aggregate profile must not leak raw SHAP values"
            )

    def test_aggregate_profile_removes_trades(self):
        result = PROFILE_PARTNER_AGGREGATE.apply(SENSITIVE_RECORDS)
        for row in result:
            assert "trades" not in row, (
                "partner_aggregate profile must not leak individual trade records"
            )

    def test_aggregate_profile_removes_feature_vector(self):
        result = PROFILE_PARTNER_AGGREGATE.apply(SENSITIVE_RECORDS)
        for row in result:
            assert "feature_vector" not in row

    def test_aggregate_profile_retains_aggregate_fields(self):
        result = PROFILE_PARTNER_AGGREGATE.apply(SENSITIVE_RECORDS)
        for row in result:
            assert "risk_score" in row
            assert "score_band" in row
            assert "benford_flag" in row
            assert "trade_count" in row

    def test_regulator_profile_retains_all_fields(self):
        result = PROFILE_REGULATOR_FULL.apply(SENSITIVE_RECORDS)
        for original, filtered in zip(SENSITIVE_RECORDS, result):
            assert set(filtered.keys()) == set(original.keys()), (
                "regulator_full profile must retain all fields"
            )

    def test_regulator_profile_retains_wallet(self):
        result = PROFILE_REGULATOR_FULL.apply(SENSITIVE_RECORDS)
        assert result[0]["wallet"] == "GAABC123"

    def test_regulator_profile_retains_shap_values(self):
        result = PROFILE_REGULATOR_FULL.apply(SENSITIVE_RECORDS)
        assert "shap_values" in result[0]

    def test_excluded_fields_config(self):
        """Custom excluded_fields list is enforced."""
        profile = ExportProfile(
            name="test_custom",
            description="custom",
            excluded_fields=frozenset({"score_band", "benford_flag"}),
        )
        result = profile.apply(SENSITIVE_RECORDS)
        for row in result:
            assert "score_band" not in row
            assert "benford_flag" not in row
            assert "risk_score" in row  # not in exclusion list

    def test_allowlist_keeps_only_allowed_fields(self):
        """When allowed_fields is set, only those fields are retained."""
        profile = ExportProfile(
            name="test_allowlist",
            description="allowlist only",
            allowed_fields=frozenset({"risk_score", "score_band"}),
        )
        result = profile.apply(SENSITIVE_RECORDS)
        for row in result:
            assert set(row.keys()) <= {"risk_score", "score_band"}

    def test_empty_records_pass_through(self):
        result = PROFILE_PARTNER_AGGREGATE.apply([])
        assert result == []


class TestDataMinimizationProfiles:
    """Integration tests for profiles applied through export_report()."""

    def test_default_profile_is_most_restrictive(self):
        """No profile argument → partner_aggregate (most restrictive) is applied."""
        result = export_report(SENSITIVE_RECORDS, fmt="json")
        data = json.loads(result.content)
        for row in data:
            assert "wallet" not in row, (
                "Default export must not include wallet IDs (partner_aggregate profile)"
            )
            assert "shap_values" not in row

    def test_regulator_full_profile_includes_wallet(self):
        result = export_report(SENSITIVE_RECORDS, fmt="json", profile="regulator_full")
        data = json.loads(result.content)
        assert data[0]["wallet"] == "GAABC123"

    def test_partner_aggregate_profile_excludes_wallet(self):
        result = export_report(SENSITIVE_RECORDS, fmt="json", profile="partner_aggregate")
        data = json.loads(result.content)
        for row in data:
            assert "wallet" not in row

    def test_partner_aggregate_excludes_shap(self):
        result = export_report(SENSITIVE_RECORDS, fmt="json", profile="partner_aggregate")
        data = json.loads(result.content)
        for row in data:
            assert "shap_values" not in row

    def test_unknown_profile_raises(self):
        with pytest.raises(UnknownProfileError) as exc:
            export_report(SENSITIVE_RECORDS, fmt="json", profile="nonexistent_profile")
        assert "nonexistent_profile" in str(exc.value)
        assert "available profiles" in str(exc.value)

    def test_regulator_full_csv_contains_wallet_column(self):
        result = export_report(SENSITIVE_RECORDS, fmt="csv", profile="regulator_full")
        reader = csv.DictReader(io.StringIO(result.content.decode("utf-8")))
        rows = list(reader)
        assert "wallet" in rows[0]

    def test_partner_aggregate_csv_no_wallet_column(self):
        result = export_report(SENSITIVE_RECORDS, fmt="csv", profile="partner_aggregate")
        reader = csv.DictReader(io.StringIO(result.content.decode("utf-8")))
        for row in reader:
            assert "wallet" not in row

    def test_register_custom_profile(self):
        """register_profile() adds a new profile to the global registry."""
        custom = ExportProfile(
            name="test_internal",
            description="Internal only",
            excluded_fields=frozenset({"score_band"}),
        )
        register_profile(custom)
        try:
            result = export_report(SENSITIVE_RECORDS, fmt="json", profile="test_internal")
            data = json.loads(result.content)
            for row in data:
                assert "score_band" not in row
                assert "wallet" in row  # not excluded by this profile
        finally:
            del PROFILE_REGISTRY["test_internal"]

    def test_all_profiles_defined_in_registry(self):
        """Both built-in profiles are registered and retrievable."""
        assert "regulator_full" in PROFILE_REGISTRY
        assert "partner_aggregate" in PROFILE_REGISTRY

    def test_default_profile_name_matches_registry(self):
        assert DEFAULT_PROFILE_NAME in PROFILE_REGISTRY
