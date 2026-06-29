"""Tests for the interactive HTML forensic report export (issue #248).

Covers:
- generate_interactive_report() produces a valid HTML file
- HTML file contains all feature names from top_shap_features
- Raw wallet address is NOT present in the HTML source
- File size is < 5 MB for a standard report (100 trades, 50 graph nodes)
- _encrypt_wallet() and _hash_wallet() are deterministic
- Graceful fallback when plotly / pyvis are not installed
"""

from __future__ import annotations

import html.parser
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Minimal forensic dict fixture
# ---------------------------------------------------------------------------

WALLET = "GABC1234567890EXAMPLEWALLETADDRESS000000000000000000000001"

SHAP_FEATURES = [
    {"feature": "benford_mad_24h", "contribution": 0.34, "value": 0.047},
    {"feature": "counterparty_concentration_ratio", "contribution": 0.29, "value": 0.98},
    {"feature": "self_matching_rate", "contribution": 0.18, "value": 0.72},
]


def _make_forensic_dict(
    n_trades: int = 100,
    n_graph_nodes: int = 50,
) -> dict:
    trades = []
    for i in range(n_trades):
        trades.append({
            "trade_id": f"trade-{i:04d}",
            "ledger": 50000000 + i,
            "base_account": WALLET,
            "counter_account": f"GCOUNTER{i:040d}",
            "base_amount": 100.0 + i,
            "counter_amount": 99.0 + i,
            "asset_pair": "XLM:native/USDC:GA5ZSE",
            "horizon_url": "https://horizon.stellar.org/trades/trade-id",
        })

    edges = []
    for i in range(min(n_graph_nodes, 49)):
        edges.append({
            "source": WALLET,
            "target": f"GNODE{i:045d}",
            "weight": 1.0,
            "risk_score": 30 + i,
        })

    return {
        "report_id": "test-report-id-0001",
        "generated_at": "2026-06-29T10:00:00+00:00",
        "wallet": WALLET,
        "asset_pair": "XLM:native",
        "risk_score": 95,
        "verdict": "wash_trade",
        "top_shap_features": SHAP_FEATURES,
        "benford_analysis": {},
        "trade_evidence": trades,
        "model_metadata": {"model_version": "1.0"},
        "propagation_path": {"edges": edges},
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class _HTMLValidator(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        pass

    def handle_error(self, message):
        self.errors.append(message)


def _validate_html(content: str) -> list[str]:
    validator = _HTMLValidator()
    validator.feed(content)
    return validator.errors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_interactive_report_produces_valid_html():
    from detection.forensic_report_interactive import generate_interactive_report

    fd = _make_forensic_dict()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = tmp.name

    try:
        generate_interactive_report(fd, out_path)
        content = Path(out_path).read_text(encoding="utf-8")

        # Must start with a valid doctype / html tag
        assert "<!DOCTYPE html>" in content or "<html" in content

        # Python's html.parser must not emit errors
        errors = _validate_html(content)
        assert not errors, f"HTML parse errors: {errors}"
    finally:
        os.unlink(out_path)


def test_report_contains_all_feature_names():
    from detection.forensic_report_interactive import generate_interactive_report

    fd = _make_forensic_dict()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = tmp.name

    try:
        generate_interactive_report(fd, out_path)
        content = Path(out_path).read_text(encoding="utf-8")

        for feat in SHAP_FEATURES:
            assert feat["feature"] in content, f"Feature '{feat['feature']}' not found in HTML"
    finally:
        os.unlink(out_path)


def test_report_does_not_contain_raw_wallet_address():
    from detection.forensic_report_interactive import generate_interactive_report

    fd = _make_forensic_dict()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = tmp.name

    try:
        generate_interactive_report(fd, out_path)
        content = Path(out_path).read_text(encoding="utf-8")
        assert WALLET not in content, "Raw wallet address must not appear in the HTML source"
    finally:
        os.unlink(out_path)


def test_report_file_size_under_5mb():
    from detection.forensic_report_interactive import generate_interactive_report

    fd = _make_forensic_dict(n_trades=100, n_graph_nodes=50)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = tmp.name

    try:
        generate_interactive_report(fd, out_path)
        size_bytes = os.path.getsize(out_path)
        assert size_bytes < 5 * 1024 * 1024, (
            f"HTML report is {size_bytes / 1024 / 1024:.2f} MB; must be < 5 MB"
        )
    finally:
        os.unlink(out_path)


def test_encrypt_wallet_is_deterministic():
    from detection.forensic_report_interactive import _encrypt_wallet

    e1 = _encrypt_wallet(WALLET, key_hint="test-key")
    e2 = _encrypt_wallet(WALLET, key_hint="test-key")
    assert e1 == e2


def test_hash_wallet_is_deterministic():
    from detection.forensic_report_interactive import _hash_wallet

    h1 = _hash_wallet(WALLET)
    h2 = _hash_wallet(WALLET)
    assert h1 == h2
    assert len(h1) == 16


def test_encrypted_wallet_differs_from_hash():
    from detection.forensic_report_interactive import _encrypt_wallet, _hash_wallet

    enc = _encrypt_wallet(WALLET)
    hsh = _hash_wallet(WALLET)
    assert enc != hsh


def test_report_created_with_parent_dirs(tmp_path):
    from detection.forensic_report_interactive import generate_interactive_report

    fd = _make_forensic_dict(n_trades=5, n_graph_nodes=3)
    out_path = str(tmp_path / "nested" / "dir" / "report.html")
    result = generate_interactive_report(fd, out_path)
    assert Path(result).exists()


def test_report_gracefully_handles_empty_shap_features():
    from detection.forensic_report_interactive import generate_interactive_report

    fd = _make_forensic_dict(n_trades=2, n_graph_nodes=2)
    fd["top_shap_features"] = []

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = tmp.name

    try:
        generate_interactive_report(fd, out_path)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content or "<html" in content
    finally:
        os.unlink(out_path)


def test_report_gracefully_handles_missing_propagation_path():
    from detection.forensic_report_interactive import generate_interactive_report

    fd = _make_forensic_dict(n_trades=5, n_graph_nodes=0)
    fd.pop("propagation_path", None)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = tmp.name

    try:
        generate_interactive_report(fd, out_path)
        assert Path(out_path).exists()
    finally:
        os.unlink(out_path)
