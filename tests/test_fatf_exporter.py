"""Tests for reporting/fatf_exporter.py and reporting/fatf_risk_codes.py."""

import os

import pytest

from reporting.fatf_exporter import (
    ExportValidationError,
    _mask_wallet,
    _unavailable,
    export_batch_ivms101,
    export_ivms101,
)
from reporting.fatf_risk_codes import (
    RISK_CODES,
    Severity,
    map_to_risk_codes,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_HIGH_SCORE_REPORT = {
    "report_id": "aaaaaaaa-0000-0000-0000-000000000001",
    "generated_at": "2024-06-01T12:00:00+00:00",
    "wallet": "GABC123456789STELLAR",
    "asset_pair": "USDC:GA5ZSEJY/XLM:native",
    "risk_score": 92,
    "score_lower": 82,
    "score_upper": 100,
    "verdict": "wash_trade",
    "top_shap_features": [
        {"feature": "round_trip_frequency", "contribution": 18.5, "value": 0.91},
        {"feature": "counterparty_concentration_ratio", "contribution": 12.0, "value": 0.87},
        {"feature": "benford_mad_24h", "contribution": 9.0, "value": 0.14},
    ],
    "benford_analysis": {
        "24": {"chi_square": 42.1, "mad": 0.14, "mad_nonconforming": True, "sample_size": 300}
    },
    "trade_evidence": [],
    "model_metadata": {"name": "LedgerLens Ensemble", "version": "1.0"},
    "report_sha256": "abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
    "soroban_anchor_tx": None,
}

_LOW_SCORE_REPORT = {
    **_HIGH_SCORE_REPORT,
    "report_id": "aaaaaaaa-0000-0000-0000-000000000002",
    "wallet": "GLOW123456789STELLAR",
    "risk_score": 50,
    "score_lower": 40,
    "score_upper": 60,
    "verdict": "clean",
    "top_shap_features": [],
}

_MINIMAL_REPORT = {
    "report_id": "aaaaaaaa-0000-0000-0000-000000000003",
    "generated_at": "2024-06-01T13:00:00+00:00",
    "wallet": "GMIN123456789STELLAR",
    "asset_pair": "",       # intentionally absent/empty
    "risk_score": 88,
    "score_lower": None,    # intentionally missing confidence interval
    "score_upper": None,
    "verdict": "suspicious",
    "top_shap_features": [],
    "benford_analysis": {},
    "trade_evidence": [],
    "model_metadata": {},
    "report_sha256": None,  # intentionally absent
    "soroban_anchor_tx": None,
}


# ---------------------------------------------------------------------------
# _mask_wallet
# ---------------------------------------------------------------------------


class TestMaskWallet:
    def test_masks_address(self):
        masked = _mask_wallet("GABC123")
        assert masked.startswith("REDACTED-")
        assert "GABC123" not in masked

    def test_same_address_same_token(self):
        assert _mask_wallet("GTEST") == _mask_wallet("GTEST")

    def test_different_addresses_differ(self):
        assert _mask_wallet("GA") != _mask_wallet("GB")

    def test_token_length(self):
        # REDACTED- (9 chars) + 12 hex chars = 21 chars total
        assert len(_mask_wallet("GABC")) == 21


# ---------------------------------------------------------------------------
# _unavailable
# ---------------------------------------------------------------------------


class TestUnavailable:
    def test_structure(self):
        sentinel = _unavailable()
        assert sentinel["value"] is None
        assert sentinel["dataUnavailable"] is True

    def test_with_field_name(self):
        sentinel = _unavailable("countryOfResidence")
        assert sentinel["fieldName"] == "countryOfResidence"

    def test_without_field_name_no_field_name_key(self):
        sentinel = _unavailable()
        assert "fieldName" not in sentinel


# ---------------------------------------------------------------------------
# map_to_risk_codes
# ---------------------------------------------------------------------------


class TestMapToRiskCodes:
    def test_wash_trade_verdict_maps_critical_codes(self):
        codes = map_to_risk_codes(_HIGH_SCORE_REPORT)
        code_ids = [rc.code for rc in codes]
        assert "VA-002" in code_ids
        assert "VA-009" in code_ids

    def test_suspicious_verdict_maps_va_001(self):
        report = {**_HIGH_SCORE_REPORT, "verdict": "suspicious", "top_shap_features": []}
        codes = map_to_risk_codes(report)
        assert any(rc.code == "VA-001" for rc in codes)

    def test_clean_verdict_no_primary_code(self):
        codes = map_to_risk_codes({**_LOW_SCORE_REPORT, "verdict": "clean"})
        code_ids = [rc.code for rc in codes]
        assert "VA-001" not in code_ids
        assert "VA-002" not in code_ids

    def test_shap_feature_adds_supplementary_code(self):
        report = {
            **_HIGH_SCORE_REPORT,
            "verdict": "suspicious",
            "top_shap_features": [
                {"feature": "round_trip_frequency", "contribution": 10.0, "value": 0.9}
            ],
        }
        codes = map_to_risk_codes(report)
        code_ids = [rc.code for rc in codes]
        assert "VA-005" in code_ids  # round-trip cycling

    def test_benford_shap_adds_va004(self):
        report = {
            **_HIGH_SCORE_REPORT,
            "verdict": "suspicious",
            "top_shap_features": [
                {"feature": "benford_mad_24h", "contribution": 8.0, "value": 0.15}
            ],
        }
        codes = map_to_risk_codes(report)
        assert any(rc.code == "VA-004" for rc in codes)

    def test_no_duplicate_codes(self):
        codes = map_to_risk_codes(_HIGH_SCORE_REPORT)
        code_ids = [rc.code for rc in codes]
        assert len(code_ids) == len(set(code_ids))

    def test_negative_contribution_ignored(self):
        report = {
            **_HIGH_SCORE_REPORT,
            "verdict": "clean",
            "top_shap_features": [
                {"feature": "round_trip_frequency", "contribution": -5.0, "value": 0.1}
            ],
        }
        codes = map_to_risk_codes(report)
        assert all(rc.code != "VA-005" for rc in codes)

    def test_all_defined_codes_have_valid_severity(self):
        for code_id, rc in RISK_CODES.items():
            assert isinstance(rc.severity, Severity)

    def test_sorted_by_severity(self):
        codes = map_to_risk_codes(_HIGH_SCORE_REPORT)
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        values = [severity_order[rc.severity.value] for rc in codes]
        assert values == sorted(values)


# ---------------------------------------------------------------------------
# export_ivms101 — unit tests (spec required)
# ---------------------------------------------------------------------------


class TestExportIvms101:
    """Core behavioural tests for the IVMS101 single-report export."""

    def test_high_scoring_report_produces_valid_structure(self):
        """A high-scoring report must map to a valid IVMS101 structure passing schema."""
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        # Top-level required fields
        assert doc["@context"] == "https://intervasp.org/ivms101"
        assert doc["@type"] == "ivms101:IdentityPayload"
        assert "payloadMetadata" in doc
        assert "originator" in doc
        assert "riskIndicators" in doc
        assert "transactionReference" in doc

    def test_payload_metadata_fields(self):
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        meta = doc["payloadMetadata"]
        assert meta["reportId"] == _HIGH_SCORE_REPORT["report_id"]
        assert meta["reportingEntity"] == "LedgerLens"
        assert meta["schemaVersion"] == "1.0"
        assert meta["generatedAt"] == _HIGH_SCORE_REPORT["generated_at"]

    def test_risk_score_normalised_to_0_1(self):
        """risk_score (0–100) must be normalised to [0, 1] in the export."""
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        assert doc["transactionReference"]["riskScore"] == pytest.approx(0.92)

    def test_confidence_interval_normalised(self):
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        ci = doc["transactionReference"]["scoreConfidenceInterval"]
        assert ci["lower"] == pytest.approx(0.82)
        assert ci["upper"] == pytest.approx(1.0)

    def test_risk_indicators_populated_for_wash_trade(self):
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        codes = [ri["code"] for ri in doc["riskIndicators"]]
        assert "VA-002" in codes

    def test_risk_indicator_severity_valid(self):
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for ri in doc["riskIndicators"]:
            assert ri["severity"] in valid_severities

    def test_wallet_masked_by_default(self):
        """Wallet address must be pseudonymised when reveal_addresses=False."""
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        account_number = doc["originator"]["accountNumber"][0]
        assert "GABC123456789STELLAR" not in account_number
        assert account_number.startswith("REDACTED-")

    def test_wallet_revealed_with_admin_auth(self, monkeypatch):
        monkeypatch.setenv("FATF_ADMIN_TOKEN", "test-token-abc")
        doc = export_ivms101(_HIGH_SCORE_REPORT, reveal_addresses=True)
        account_number = doc["originator"]["accountNumber"][0]
        assert account_number == "GABC123456789STELLAR"

    def test_reveal_addresses_without_auth_raises(self, monkeypatch):
        monkeypatch.delenv("FATF_ADMIN_TOKEN", raising=False)
        with pytest.raises(PermissionError, match="FATF_ADMIN_TOKEN"):
            export_ivms101(_HIGH_SCORE_REPORT, reveal_addresses=True)

    # --- Missing optional fields produce data_unavailable, not missing keys ---

    def test_missing_asset_pair_produces_unavailable_sentinel(self):
        """Empty asset_pair must produce the data-unavailable sentinel, not a missing key."""
        doc = export_ivms101(_MINIMAL_REPORT)
        ref = doc["transactionReference"]
        assert "assetPair" in ref
        ap = ref["assetPair"]
        assert isinstance(ap, dict)
        assert ap["value"] is None
        assert ap["dataUnavailable"] is True

    def test_missing_confidence_interval_produces_unavailable_sentinel(self):
        """Missing score_lower/score_upper must produce the sentinel, not a missing key."""
        doc = export_ivms101(_MINIMAL_REPORT)
        ci = doc["transactionReference"]["scoreConfidenceInterval"]
        assert isinstance(ci, dict)
        assert ci["value"] is None
        assert ci["dataUnavailable"] is True

    def test_missing_sha256_produces_unavailable_sentinel(self):
        doc = export_ivms101(_MINIMAL_REPORT)
        sha = doc["payloadMetadata"]["sourceReportSha256"]
        assert isinstance(sha, dict)
        assert sha["value"] is None
        assert sha["dataUnavailable"] is True

    def test_natural_person_name_always_present_as_unavailable(self):
        """naturalPerson.name must always be present (as sentinel) — never a missing key."""
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        person = doc["originator"]["originatorPersons"][0]["naturalPerson"]
        assert "name" in person
        name = person["name"]
        assert name["value"] is None
        assert name["dataUnavailable"] is True

    def test_beneficiary_vasp_always_present(self):
        """beneficiaryVASP must be in the document (as unavailable sentinel)."""
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        assert "beneficiaryVASP" in doc
        bvasp = doc["beneficiaryVASP"]
        assert bvasp["value"] is None
        assert bvasp["dataUnavailable"] is True

    def test_schema_validation_runs_on_valid_report(self):
        """export_ivms101 must not raise ExportValidationError for a valid report."""
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        assert isinstance(doc, dict)

    def test_invalid_document_raises_export_validation_error(self, monkeypatch):
        """Corrupting the assembled doc before validation triggers ExportValidationError."""
        import reporting.fatf_exporter as exporter

        real_validate = exporter._validate

        def bad_validate(doc):
            import jsonschema
            raise jsonschema.ValidationError("injected failure")

        monkeypatch.setattr(exporter, "_validate", bad_validate)

        with pytest.raises(ExportValidationError, match="schema validation"):
            export_ivms101(_HIGH_SCORE_REPORT)

    def test_source_report_sha256_included(self):
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        sha = doc["payloadMetadata"]["sourceReportSha256"]
        assert sha == _HIGH_SCORE_REPORT["report_sha256"]

    def test_originator_person_is_list(self):
        doc = export_ivms101(_HIGH_SCORE_REPORT)
        assert isinstance(doc["originator"]["originatorPersons"], list)
        assert len(doc["originator"]["originatorPersons"]) == 1


# ---------------------------------------------------------------------------
# export_batch_ivms101 — unit tests (spec required)
# ---------------------------------------------------------------------------


class TestExportBatchIvms101:
    def test_high_score_included(self):
        """Reports at or above threshold must appear in batch output."""
        results = export_batch_ivms101([_HIGH_SCORE_REPORT])
        assert len(results) == 1

    def test_low_score_excluded(self):
        """Reports below FATF_EXPORT_THRESHOLD (default 0.85) must be excluded."""
        results = export_batch_ivms101([_LOW_SCORE_REPORT])
        assert len(results) == 0

    def test_mixed_batch_filters_correctly(self):
        results = export_batch_ivms101([_HIGH_SCORE_REPORT, _LOW_SCORE_REPORT])
        assert len(results) == 1
        assert results[0]["payloadMetadata"]["reportId"] == _HIGH_SCORE_REPORT["report_id"]

    def test_empty_input_returns_empty(self):
        assert export_batch_ivms101([]) == []

    def test_report_without_risk_score_excluded(self):
        no_score = {**_HIGH_SCORE_REPORT, "risk_score": None}
        assert export_batch_ivms101([no_score]) == []

    def test_exact_threshold_boundary_included(self, monkeypatch):
        """risk_score == 85 maps to 0.85 which exactly meets the default threshold."""
        monkeypatch.setattr("reporting.fatf_exporter.config.FATF_EXPORT_THRESHOLD", 0.85)
        boundary_report = {**_HIGH_SCORE_REPORT, "risk_score": 85}
        results = export_batch_ivms101([boundary_report])
        assert len(results) == 1

    def test_just_below_threshold_excluded(self, monkeypatch):
        monkeypatch.setattr("reporting.fatf_exporter.config.FATF_EXPORT_THRESHOLD", 0.85)
        boundary_report = {**_HIGH_SCORE_REPORT, "risk_score": 84}
        results = export_batch_ivms101([boundary_report])
        assert len(results) == 0

    def test_reveal_addresses_propagated(self, monkeypatch):
        monkeypatch.setenv("FATF_ADMIN_TOKEN", "test-token")
        results = export_batch_ivms101([_HIGH_SCORE_REPORT], reveal_addresses=True)
        account = results[0]["originator"]["accountNumber"][0]
        assert account == "GABC123456789STELLAR"

    def test_all_outputs_pass_schema_validation(self):
        """Every doc in a batch export must pass schema validation."""
        reports = [_HIGH_SCORE_REPORT, _MINIMAL_REPORT]
        results = export_batch_ivms101(reports)
        # Both have risk_score >= 85
        assert len(results) == 2
        for doc in results:
            assert "@context" in doc
            assert "riskIndicators" in doc
