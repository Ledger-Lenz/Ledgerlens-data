"""Tests for the Forensic Reporting Engine.

Covers:
- SHA-256 integrity after round-trip serialisation
- Tamper-evidence (any field change alters the hash)
- to_markdown() section headers all present
- trade_evidence capped at 20 entries
- horizon_url format validation
- anchor_report called only when --anchor flag is set
- Report file written with mode 0o600
"""

import hashlib
import json
import os
import re
import stat
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from detection.forensic_report import (
    ForensicReportGenerator,
    write_report_secure,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WALLET = "GABC1234567890123456789012345678901234567890123456789012"

SHAP_VALUES = [
    {"feature": "benford_mad_24h", "contribution": 0.34, "value": 0.047},
    {"feature": "counterparty_concentration_ratio", "contribution": 0.29, "value": 0.98},
]

RISK_SCORE_DICT = {
    "score": 83,
    "benford_flag": True,
    "ml_flag": True,
    "confidence": 76,
}


def _make_trades(n: int = 25) -> pd.DataFrame:
    """Return a minimal trades DataFrame with n rows."""
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "trade_id": [f"trade-{i}" for i in range(n)],
            "id": [f"trade-{i}" for i in range(n)],
            "ledger": rng.integers(1_000_000, 2_000_000, n),
            "base_account": [WALLET] * n,
            "counter_account": ["GBBB" + "X" * 52 for _ in range(n)],
            "base_amount": rng.uniform(10, 10000, n),
            "counter_amount": rng.uniform(10, 10000, n),
            "amount": rng.uniform(10, 10000, n),
            "ledger_close_time": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "pair_id": ["XLM:native/USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"]
            * n,
        }
    )


@pytest.fixture
def generator():
    return ForensicReportGenerator()


@pytest.fixture
def sample_report(generator):
    return generator.generate(
        wallet=WALLET,
        wallet_trades=_make_trades(),
        risk_score_dict=RISK_SCORE_DICT,
        shap_values=SHAP_VALUES,
        asset_pair="XLM:native/USDC:issuer",
    )


# ---------------------------------------------------------------------------
# SHA-256 integrity
# ---------------------------------------------------------------------------


def test_sha256_matches_independent_recomputation(sample_report):
    """report_sha256 must equal independently recomputed hash from to_dict()."""
    d = sample_report.to_dict()
    stored_hash = d.pop("report_sha256")
    computed = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    assert computed == stored_hash


def test_sha256_round_trip_after_deserialisation(sample_report):
    """Serialise to JSON, deserialise, recompute hash — must still match."""
    raw = json.dumps(sample_report.to_dict())
    d = json.loads(raw)
    stored_hash = d.pop("report_sha256")
    computed = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    assert computed == stored_hash


def test_verify_integrity_method(sample_report):
    assert sample_report.verify_integrity() is True


# ---------------------------------------------------------------------------
# Tamper-evidence
# ---------------------------------------------------------------------------


def test_tampering_with_risk_score_changes_sha256(sample_report):
    original_hash = sample_report.report_sha256
    d = sample_report.to_dict()
    d["risk_score"] = 0  # tamper
    d.pop("report_sha256")
    tampered_hash = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    assert tampered_hash != original_hash


def test_tampering_with_wallet_changes_sha256(sample_report):
    original_hash = sample_report.report_sha256
    d = sample_report.to_dict()
    d["wallet"] = "GMALICIOUS" + "X" * 46
    d.pop("report_sha256")
    tampered_hash = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    assert tampered_hash != original_hash


# ---------------------------------------------------------------------------
# to_markdown()
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "## Executive Summary",
    "## Risk Score Summary",
    "## SHAP Feature Attribution",
    "## Benford's Law Analysis",
    "## Trade Evidence",
    "## On-Chain Anchor & Verification",
]


def test_to_markdown_contains_all_sections(sample_report):
    md = sample_report.to_markdown()
    for section in REQUIRED_SECTIONS:
        assert section in md, f"Missing section: {section}"


def test_to_markdown_contains_disclaimer(sample_report):
    md = sample_report.to_markdown()
    # The disclaimer may wrap across lines in the blockquote, so search flexibly
    assert "legal advice" in md.lower()


def test_to_markdown_contains_wallet(sample_report):
    md = sample_report.to_markdown()
    assert WALLET in md


# ---------------------------------------------------------------------------
# trade_evidence capped at 20
# ---------------------------------------------------------------------------


def test_trade_evidence_capped_at_20(generator):
    """With > 20 trades, only the 20 most anomalous are included."""
    report = generator.generate(
        wallet=WALLET,
        wallet_trades=_make_trades(50),
        risk_score_dict=RISK_SCORE_DICT,
        shap_values=[],
        asset_pair="XLM:native/USDC:issuer",
    )
    assert len(report.trade_evidence) == 20


def test_trade_evidence_all_trades_if_fewer_than_20(generator):
    """With < 20 trades, all are included."""
    report = generator.generate(
        wallet=WALLET,
        wallet_trades=_make_trades(5),
        risk_score_dict=RISK_SCORE_DICT,
        shap_values=[],
        asset_pair="XLM:native/USDC:issuer",
    )
    assert len(report.trade_evidence) == 5


def test_trade_evidence_empty_trades(generator):
    report = generator.generate(
        wallet=WALLET,
        wallet_trades=pd.DataFrame(),
        risk_score_dict=RISK_SCORE_DICT,
        shap_values=[],
        asset_pair="XLM:native/USDC:issuer",
    )
    assert report.trade_evidence == []


# ---------------------------------------------------------------------------
# horizon_url format
# ---------------------------------------------------------------------------


def test_horizon_url_format(sample_report):
    from config import config

    horizon_base = config.HORIZON_URL.rstrip("/")
    pattern = re.compile(rf"^{re.escape(horizon_base)}/trades/[A-Za-z0-9\-]+$")
    for t in sample_report.trade_evidence:
        assert pattern.match(
            t.horizon_url
        ), f"horizon_url does not match expected pattern: {t.horizon_url}"


# ---------------------------------------------------------------------------
# anchor_report called only with --anchor
# ---------------------------------------------------------------------------


def test_anchor_report_not_called_without_flag(tmp_path):
    """_generate_report must not call anchor_report when --anchor is not set."""
    with (
        patch("scripts.score_wallet.RiskScorer") as MockScorer,
        patch("scripts.score_wallet.load_trades", return_value=iter([])),
        patch("scripts.score_wallet.load_orderbook_events", return_value=iter([])),
        patch("scripts.score_wallet.ShapExplainer") as MockExplainer,
        patch("scripts.score_wallet.ForensicReportGenerator") as MockGen,
        patch("scripts.score_wallet.write_report_secure") as _mock_write,
        patch("integrations.contract_client.LedgerLensContractClient") as MockClient,
    ):
        scorer = MockScorer.return_value
        scorer.score.return_value = RISK_SCORE_DICT
        scorer.models = {}
        scorer.metadata = {}
        MockExplainer.return_value.explain_ensemble.return_value = []

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"report_sha256": "abc", "soroban_anchor_tx": None}
        MockGen.return_value.generate.return_value = mock_report

        import argparse

        from scripts.score_wallet import _generate_report

        args = argparse.Namespace(
            wallet=WALLET,
            pair="XLM:native",
            report=True,
            report_format="json",
            anchor=False,  # no anchor
        )
        _generate_report(args, RISK_SCORE_DICT, [], pd.DataFrame(), pd.Series(dtype=float), scorer)

        MockClient.return_value.anchor_report.assert_not_called()


def test_anchor_report_called_with_flag():
    """_generate_report must call anchor_report when --anchor is set."""
    with (
        patch("scripts.score_wallet.RiskScorer") as MockScorer,
        patch("scripts.score_wallet.load_trades", return_value=iter([])),
        patch("scripts.score_wallet.load_orderbook_events", return_value=iter([])),
        patch("scripts.score_wallet.ShapExplainer") as MockExplainer,
        patch("scripts.score_wallet.ForensicReportGenerator") as MockGen,
        patch("scripts.score_wallet.write_report_secure"),
        # Patch where the name is looked up (inside _generate_report's local import)
        patch("integrations.contract_client.LedgerLensContractClient") as MockClient,
    ):
        scorer = MockScorer.return_value
        scorer.score.return_value = RISK_SCORE_DICT
        scorer.models = {}
        scorer.metadata = {}
        MockExplainer.return_value.explain_ensemble.return_value = []

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"report_sha256": "abc", "soroban_anchor_tx": None}
        MockGen.return_value.generate.return_value = mock_report

        import argparse

        from scripts.score_wallet import _generate_report

        args = argparse.Namespace(
            wallet=WALLET,
            pair="XLM:native",
            report=True,
            report_format="json",
            anchor=True,
        )

        # Patch the LedgerLensContractClient in the integrations module so the
        # local import inside _generate_report picks up the mock
        with patch("integrations.contract_client.LedgerLensContractClient", MockClient):
            _generate_report(
                args, RISK_SCORE_DICT, [], pd.DataFrame(), pd.Series(dtype=float), scorer
            )

        MockClient.return_value.anchor_report.assert_called_once_with(mock_report)


# ---------------------------------------------------------------------------
# File mode 0o600
# ---------------------------------------------------------------------------


def test_write_report_secure_mode(tmp_path):
    out_path = str(tmp_path / "report.json")
    write_report_secure(out_path, '{"test": 1}')
    file_mode = stat.S_IMODE(os.stat(out_path).st_mode)
    assert file_mode == 0o600, f"Expected 0o600, got {oct(file_mode)}"


def test_write_report_secure_creates_parent_dirs(tmp_path):
    nested = str(tmp_path / "a" / "b" / "c" / "report.json")
    write_report_secure(nested, "hello")
    assert os.path.exists(nested)
    assert stat.S_IMODE(os.stat(nested).st_mode) == 0o600
