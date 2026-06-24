"""Tests for detection/audit_trail.py (issue #126)."""

from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from config import config
from detection.audit_trail import (
    AuditTrailWriter,
    commitment_hash,
    hash_features,
    read_audit_log,
    verify_audit_log,
)
from detection.forensic_report import ForensicReport
from detection.persistence import ModelIntegrityError


@pytest.fixture
def signing_key_paths(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "signing_key.pem"
    public_path = tmp_path / "signing_key_pub.pem"
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path.write_bytes(private_bytes)
    public_path.write_bytes(public_bytes)
    return str(private_path), str(public_path)


@pytest.fixture
def sample_report():
    return ForensicReport(
        report_id="test-report-id",
        generated_at="2026-06-22T12:00:00Z",
        wallet="GABC1234567890123456789012345678901234567890123456789012",
        asset_pair="XLM:native/USDC:issuer",
        risk_score=83,
        score_lower=73,
        score_upper=93,
        verdict="wash_trade",
        top_shap_features=[
            {"feature": "benford_mad_24h", "contribution": 0.34, "value": 0.047},
        ],
        benford_analysis={},
        trade_evidence=[],
        model_metadata={"name": "test", "version": "v1"},
    )


def test_commit_writes_signed_append_only_entry(tmp_path, signing_key_paths, sample_report):
    private_path, public_path = signing_key_paths
    log_path = tmp_path / "audit.ndjson"
    writer = AuditTrailWriter(log_path=str(log_path), private_key_path=private_path)

    digest = writer.commit(
        sample_report,
        "ensemble-v1",
        features={"benford_mad_24h": 0.04, "trade_count": 120},
        timestamp="2026-06-22T12:00:00Z",
    )

    assert digest == commitment_hash(
        writer.build_payload(
            sample_report,
            features={"benford_mad_24h": 0.04, "trade_count": 120},
            model_version="ensemble-v1",
            timestamp="2026-06-22T12:00:00Z",
        )
    )
    assert log_path.exists()

    writer.commit(
        sample_report,
        "ensemble-v2",
        features={"benford_mad_24h": 0.05},
        timestamp="2026-06-22T13:00:00Z",
    )
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    valid, failures = verify_audit_log(str(log_path), public_key_path=public_path)
    assert failures == []
    assert valid == 2


def test_verify_detects_tampered_signature(tmp_path, signing_key_paths, sample_report):
    private_path, public_path = signing_key_paths
    log_path = tmp_path / "audit.ndjson"
    writer = AuditTrailWriter(log_path=str(log_path), private_key_path=private_path)
    writer.commit(sample_report, "v1", features={"x": 1.0})

    entries = read_audit_log(str(log_path))
    entries[0]["sig"] = "00" * 64
    log_path.write_text(json.dumps(entries[0]) + "\n", encoding="utf-8")

    valid, failures = verify_audit_log(str(log_path), public_key_path=public_path)
    assert valid == 0
    assert failures == [1]


def test_verify_detects_tampered_payload(tmp_path, signing_key_paths, sample_report):
    private_path, public_path = signing_key_paths
    log_path = tmp_path / "audit.ndjson"
    writer = AuditTrailWriter(log_path=str(log_path), private_key_path=private_path)
    writer.commit(sample_report, "v1", features={"x": 1.0})

    entries = read_audit_log(str(log_path))
    entries[0]["payload"]["score"] = 0
    log_path.write_text(json.dumps(entries[0]) + "\n", encoding="utf-8")

    valid, failures = verify_audit_log(str(log_path), public_key_path=public_path)
    assert valid == 0
    assert failures == [1]


def test_hash_features_stable():
    assert hash_features({"b": 2, "a": 1}) == hash_features({"a": 1, "b": 2})


def test_commit_requires_signing_key(sample_report):
    writer = AuditTrailWriter(
        log_path="data/test_audit.ndjson", private_key_path="/nonexistent/key.pem"
    )
    with pytest.raises(ModelIntegrityError):
        writer.commit(sample_report, "v1", features={"x": 1})


def test_audit_log_path_in_config():
    assert config.AUDIT_LOG_PATH


def test_verify_script_cli(tmp_path, signing_key_paths, sample_report, monkeypatch):
    private_path, public_path = signing_key_paths
    log_path = tmp_path / "audit.ndjson"
    writer = AuditTrailWriter(log_path=str(log_path), private_key_path=private_path)
    writer.commit(sample_report, "v1", features={"x": 1.0})

    from scripts.verify_audit_trail import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_audit_trail",
            "--log-path",
            str(log_path),
            "--public-key-path",
            public_path,
        ],
    )
    assert main() == 0
