"""Unit tests for ForensicReport CSV and JSON export helpers.

Covers (per acceptance criteria #97):
- to_dict() returns a valid JSON-serialisable dict
- to_dict() includes the propagation_path section when present
- to_csv_rows() returns one row per SHAP feature with correct columns
- to_csv_rows() emits a single placeholder row when shap_explanations is empty
- write_csv_report() produces a parseable CSV with the expected header
- write_report_secure() writes mode 0o600 (existing test coverage kept here
  for completeness; canonical tests live in test_forensic_report.py)
"""

from __future__ import annotations

import csv
import io
import json
import os
import stat
import sys
from unittest.mock import MagicMock

import pytest

sys.modules["detection.causal_attribution"] = MagicMock()
sys.modules["detection.model_inference"] = MagicMock()
sys.modules["detection.shap_explainer"] = MagicMock()
sys.modules["detection.risk_propagation"] = MagicMock()

from detection.forensic_report import (
    CSV_COLUMNS,
    CausalAttribution,
    ForensicReport,
    PropagationContributor,
    PropagationPath,
    write_csv_report,
    write_report_secure,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

WALLET = "GABC1234567890123456789012345678901234567890123456789012"
PAIR = "USDC:GA5Z/XLM:native"
RISK_SCORE = {"score": 83, "benford_flag": True, "ml_flag": True, "confidence": 76}
SHAP = [
    {"feature": "benford_mad_24h", "contribution": 0.34, "value": 0.047},
    {"feature": "counterparty_concentration_ratio", "contribution": 0.29, "value": 0.98},
]


@pytest.fixture()
def basic_report() -> ForensicReport:
    """A ForensicReport with SHAP values but no causal/propagation sections."""
    return ForensicReport(
        wallet=WALLET,
        asset_pair=PAIR,
        risk_score=RISK_SCORE,
        shap_explanations=SHAP,
    )


@pytest.fixture()
def report_with_propagation() -> ForensicReport:
    """A ForensicReport that includes a PropagationPath."""
    pp = PropagationPath(
        propagated_risk=0.42,
        contributors=[
            PropagationContributor(
                source_wallet="GSRC" + "X" * 52,
                base_score=70.0,
                ppr_weight=0.85,
                contribution=59.5,
                fraction=0.62,
            ),
        ],
    )
    return ForensicReport(
        wallet=WALLET,
        asset_pair=PAIR,
        risk_score=RISK_SCORE,
        shap_explanations=SHAP,
        propagation_path=pp,
    )


@pytest.fixture()
def report_no_shap() -> ForensicReport:
    """A ForensicReport with an empty shap_explanations list."""
    return ForensicReport(
        wallet=WALLET,
        asset_pair=PAIR,
        risk_score=RISK_SCORE,
        shap_explanations=[],
    )


# ---------------------------------------------------------------------------
# to_dict() — JSON export
# ---------------------------------------------------------------------------


class TestToDict:
    def test_returns_dict(self, basic_report):
        assert isinstance(basic_report.to_dict(), dict)

    def test_json_serialisable(self, basic_report):
        """to_dict() output must round-trip through json.dumps/loads without error."""
        raw = json.dumps(basic_report.to_dict())
        d = json.loads(raw)
        assert d["wallet"] == WALLET

    def test_top_level_keys(self, basic_report):
        d = basic_report.to_dict()
        required = {"wallet", "asset_pair", "risk_score", "shap_explanations"}
        assert required.issubset(d.keys())

    def test_wallet_and_pair(self, basic_report):
        d = basic_report.to_dict()
        assert d["wallet"] == WALLET
        assert d["asset_pair"] == PAIR

    def test_risk_score_is_dict(self, basic_report):
        d = basic_report.to_dict()
        assert isinstance(d["risk_score"], dict)
        assert d["risk_score"]["score"] == RISK_SCORE["score"]

    def test_shap_explanations_preserved(self, basic_report):
        d = basic_report.to_dict()
        assert len(d["shap_explanations"]) == len(SHAP)
        assert d["shap_explanations"][0]["feature"] == SHAP[0]["feature"]

    def test_causal_attribution_none_when_absent(self, basic_report):
        d = basic_report.to_dict()
        assert d["causal_attribution"] is None

    def test_propagation_path_none_when_absent(self, basic_report):
        d = basic_report.to_dict()
        assert d["propagation_path"] is None

    def test_propagation_path_included_when_present(self, report_with_propagation):
        """Acceptance criterion: propagation_path section included in JSON export."""
        d = report_with_propagation.to_dict()
        assert d["propagation_path"] is not None
        pp = d["propagation_path"]
        assert pp["propagated_risk"] == pytest.approx(0.42)
        assert len(pp["contributors"]) == 1
        contributor = pp["contributors"][0]
        assert "source_wallet" in contributor
        assert "base_score" in contributor
        assert "ppr_weight" in contributor
        assert "contribution" in contributor
        assert "fraction" in contributor

    def test_propagation_path_json_serialisable(self, report_with_propagation):
        raw = json.dumps(report_with_propagation.to_dict())
        d = json.loads(raw)
        assert d["propagation_path"]["propagated_risk"] == pytest.approx(0.42)

    def test_causal_attribution_included_when_present(self):
        ca = CausalAttribution(
            minimal_exonerating_trades=["trade-1"],
            counterfactual_score=40,
            root_cause_wallet=WALLET,
            causal_chain=[{"hop": 1, "wallet": WALLET, "role": "initiator"}],
            interventional_score_if_no_wash=25,
        )
        report = ForensicReport(
            wallet=WALLET,
            asset_pair=PAIR,
            risk_score=RISK_SCORE,
            shap_explanations=SHAP,
            causal_attribution=ca,
        )
        d = report.to_dict()
        assert d["causal_attribution"] is not None
        assert d["causal_attribution"]["counterfactual_score"] == 40
        assert d["causal_attribution"]["minimal_exonerating_trades"] == ["trade-1"]


# ---------------------------------------------------------------------------
# to_csv_rows() — CSV export
# ---------------------------------------------------------------------------


class TestToCsvRows:
    def test_returns_list(self, basic_report):
        assert isinstance(basic_report.to_csv_rows(), list)

    def test_one_row_per_shap_feature(self, basic_report):
        """Acceptance criterion: one row per SHAP feature."""
        rows = basic_report.to_csv_rows()
        assert len(rows) == len(SHAP)

    def test_columns_match_spec(self, basic_report):
        """Every row must contain exactly the columns in CSV_COLUMNS."""
        for row in basic_report.to_csv_rows():
            assert set(row.keys()) == set(CSV_COLUMNS)

    def test_wallet_and_pair_in_every_row(self, basic_report):
        for row in basic_report.to_csv_rows():
            assert row["wallet"] == WALLET
            assert row["asset_pair"] == PAIR

    def test_risk_score_value_is_numeric_score(self, basic_report):
        for row in basic_report.to_csv_rows():
            assert row["risk_score"] == RISK_SCORE["score"]

    def test_feature_name_correct(self, basic_report):
        rows = basic_report.to_csv_rows()
        assert rows[0]["feature"] == SHAP[0]["feature"]
        assert rows[1]["feature"] == SHAP[1]["feature"]

    def test_shap_value_and_contribution(self, basic_report):
        rows = basic_report.to_csv_rows()
        assert rows[0]["shap_value"] == pytest.approx(SHAP[0]["value"])
        assert rows[0]["shap_contribution"] == pytest.approx(SHAP[0]["contribution"])

    def test_no_shap_emits_single_placeholder_row(self, report_no_shap):
        """When shap_explanations is empty one placeholder row must be emitted."""
        rows = report_no_shap.to_csv_rows()
        assert len(rows) == 1
        row = rows[0]
        assert row["wallet"] == WALLET
        assert row["feature"] == ""
        assert row["shap_value"] == ""
        assert row["shap_contribution"] == ""

    def test_no_shap_placeholder_has_correct_columns(self, report_no_shap):
        rows = report_no_shap.to_csv_rows()
        assert set(rows[0].keys()) == set(CSV_COLUMNS)


# ---------------------------------------------------------------------------
# write_csv_report() — I/O helper
# ---------------------------------------------------------------------------


class TestWriteCsvReport:
    def test_creates_file(self, basic_report, tmp_path):
        out = str(tmp_path / "report.csv")
        write_csv_report(out, basic_report)
        assert os.path.exists(out)

    def test_valid_csv_with_header(self, basic_report, tmp_path):
        out = str(tmp_path / "report.csv")
        write_csv_report(out, basic_report)
        with open(out, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert reader.fieldnames == CSV_COLUMNS
        assert len(rows) == len(SHAP)

    def test_csv_values_correct(self, basic_report, tmp_path):
        out = str(tmp_path / "report.csv")
        write_csv_report(out, basic_report)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["wallet"] == WALLET
        assert rows[0]["feature"] == SHAP[0]["feature"]

    def test_file_mode_0o600(self, basic_report, tmp_path):
        out = str(tmp_path / "secure.csv")
        write_csv_report(out, basic_report)
        file_mode = stat.S_IMODE(os.stat(out).st_mode)
        assert file_mode == 0o600, f"Expected 0o600, got {oct(file_mode)}"

    def test_creates_parent_dirs(self, basic_report, tmp_path):
        out = str(tmp_path / "a" / "b" / "c" / "report.csv")
        write_csv_report(out, basic_report)
        assert os.path.exists(out)

    def test_no_shap_still_writes_header_and_one_row(self, report_no_shap, tmp_path):
        out = str(tmp_path / "empty_shap.csv")
        write_csv_report(out, report_no_shap)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["feature"] == ""


# ---------------------------------------------------------------------------
# write_report_secure() — mode guard (supplement to test_forensic_report.py)
# ---------------------------------------------------------------------------


def test_write_report_secure_mode(tmp_path):
    out = str(tmp_path / "report.json")
    write_report_secure(out, '{"ok": true}')
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600


def test_write_report_secure_creates_nested_dirs(tmp_path):
    out = str(tmp_path / "x" / "y" / "report.json")
    write_report_secure(out, "hello")
    assert os.path.exists(out)
