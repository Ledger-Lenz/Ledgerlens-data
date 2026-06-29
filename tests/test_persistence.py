"""Tests for detection/persistence.py — RiskScoreRecord + ModelArtifact."""

import hashlib
import json
import os

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from detection.persistence import (
    ModelArtifact,
    ModelIntegrityError,
    get_engine,
    get_session_factory,
    sign_metrics,
)
from detection.risk_score_store import RiskScoreStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store() -> RiskScoreStore:
    engine = get_engine("sqlite:///:memory:")
    return RiskScoreStore(get_session_factory(engine))


def _gen_keypair():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def _write_keypair(tmp_path, private_key):
    key_path = str(tmp_path / "signing_key.pem")
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    return key_path


def _setup_valid_artifact(tmp_path):
    """Create a minimal but fully signed artifact directory. Returns (public_key, model_dir)."""
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir)

    # Fake model file
    artifact = tmp_path / "models" / "rf.joblib"
    artifact.write_bytes(b"fake-model-data")

    sha = hashlib.sha256(b"fake-model-data").hexdigest()
    metrics = {"rf": {"artifact_sha256": sha}}
    metrics_path = str(tmp_path / "models" / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    private_key, public_key = _gen_keypair()
    key_path = _write_keypair(tmp_path, private_key)
    sign_metrics(metrics_path, key_path)

    return public_key, model_dir


# ---------------------------------------------------------------------------
# Existing persistence tests
# ---------------------------------------------------------------------------


def test_upsert_creates_record():
    store = make_store()
    record = store.upsert(
        "GABC",
        "USDC:issuer/XLM:native",
        {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 80},
    )
    assert record.wallet == "GABC"
    assert record.score == 80


def test_upsert_updates_existing_record():
    store = make_store()
    pair = "USDC:issuer/XLM:native"
    store.upsert(
        "GABC", pair, {"score": 50, "benford_flag": False, "ml_flag": False, "confidence": 50}
    )
    store.upsert(
        "GABC", pair, {"score": 90, "benford_flag": True, "ml_flag": True, "confidence": 90}
    )

    record = store.get("GABC", pair)
    assert record.score == 90
    assert record.benford_flag is True


def test_to_risk_score_shape():
    store = make_store()
    pair = "USDC:issuer/XLM:native"
    store.upsert(
        "GABC", pair, {"score": 75, "benford_flag": True, "ml_flag": False, "confidence": 60}
    )

    risk_score = store.get("GABC", pair).to_risk_score()
    assert set(risk_score) == {"score", "benford_flag", "ml_flag", "timestamp", "confidence"}
    assert risk_score["score"] == 75


def test_list_flagged_filters_by_threshold():
    store = make_store()
    pair = "USDC:issuer/XLM:native"
    store.upsert(
        "GABC", pair, {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 80}
    )
    store.upsert(
        "GXYZ", pair, {"score": 20, "benford_flag": False, "ml_flag": False, "confidence": 20}
    )

    flagged = store.list_flagged(70)
    assert [r.wallet for r in flagged] == ["GABC"]


# ---------------------------------------------------------------------------
# ModelArtifact / verify_chain tests
# ---------------------------------------------------------------------------


def test_verify_chain_passes_for_valid_artifact(tmp_path):
    public_key, model_dir = _setup_valid_artifact(tmp_path)
    artifact = ModelArtifact(model_dir)
    artifact.verify_chain("rf", public_key=public_key)  # must not raise


def test_verify_chain_raises_on_wrong_sha256(tmp_path):
    public_key, model_dir = _setup_valid_artifact(tmp_path)
    # Tamper the model file so its SHA-256 no longer matches
    with open(os.path.join(model_dir, "rf.joblib"), "wb") as f:
        f.write(b"tampered-model-data")

    artifact = ModelArtifact(model_dir)
    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        artifact.verify_chain("rf", public_key=public_key)


def test_verify_chain_raises_on_invalid_signature(tmp_path):
    public_key, model_dir = _setup_valid_artifact(tmp_path)
    # Corrupt the signature file
    sig_path = os.path.join(model_dir, "metrics.json.sig")
    with open(sig_path, "wb") as f:
        f.write(b"\x00" * 64)

    artifact = ModelArtifact(model_dir)
    with pytest.raises(ModelIntegrityError, match="signature verification failed"):
        artifact.verify_chain("rf", public_key=public_key)


def test_verify_chain_raises_on_wrong_signing_key_fingerprint(tmp_path):
    public_key, model_dir = _setup_valid_artifact(tmp_path)
    wrong_fingerprint = "a" * 64

    artifact = ModelArtifact(model_dir)
    with pytest.raises(ModelIntegrityError, match="fingerprint mismatch"):
        artifact.verify_chain("rf", public_key=public_key, trusted_fingerprint=wrong_fingerprint)


def test_verify_chain_raises_on_mismatched_training_data_sha256(tmp_path):
    public_key, model_dir = _setup_valid_artifact(tmp_path)

    artifact = ModelArtifact(model_dir)
    with pytest.raises(ModelIntegrityError, match="Training data SHA-256 mismatch"):
        artifact.verify_chain(
            "rf",
            public_key=public_key,
            expected_data_sha256="deadbeef" * 8,
        )


# ---------------------------------------------------------------------------
# Issue #277 — Supply-chain transparency log + ModelArtifactVerifier
# ---------------------------------------------------------------------------


def _make_transparency_log(tmp_path):
    from detection.persistence import TransparencyLog, get_engine, get_session_factory

    engine = get_engine("sqlite:///:memory:")
    sf = get_session_factory(engine)
    return TransparencyLog(sf), sf


def _setup_verifier_artifact(tmp_path):
    """Full artifact setup: valid file + signed metrics + transparency log entry."""
    from detection.persistence import TransparencyLog, ModelArtifactVerifier, get_engine, get_session_factory

    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir)

    artifact_bytes = b"fake-model-data"
    artifact_path = os.path.join(model_dir, "rf.joblib")
    with open(artifact_path, "wb") as f:
        f.write(artifact_bytes)

    sha = hashlib.sha256(artifact_bytes).hexdigest()
    metrics = {"rf": {"artifact_sha256": sha}}
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    private_key, public_key = _gen_keypair()
    key_path = _write_keypair(tmp_path, private_key)
    sign_metrics(metrics_path, key_path)

    engine = get_engine("sqlite:///:memory:")
    sf = get_session_factory(engine)
    log = TransparencyLog(sf)
    log.append("rf", sha)

    verifier = ModelArtifactVerifier(transparency_log=log, model_dir=model_dir)
    return verifier, public_key, sha


def test_model_artifact_verifier_passes_for_valid_artifact(tmp_path):
    verifier, public_key, sha = _setup_verifier_artifact(tmp_path)
    result = verifier.verify("rf", public_key=public_key)
    assert result == sha


def test_model_artifact_verifier_fails_on_tampered_file(tmp_path):
    from detection.persistence import ModelIntegrityError

    verifier, public_key, _ = _setup_verifier_artifact(tmp_path)
    # Tamper the artifact (one byte changed)
    model_dir = verifier._model_dir
    artifact_path = os.path.join(model_dir, "rf.joblib")
    with open(artifact_path, "rb") as f:
        data = bytearray(f.read())
    data[0] ^= 0xFF
    with open(artifact_path, "wb") as f:
        f.write(bytes(data))

    with pytest.raises(ModelIntegrityError):
        verifier.verify("rf", public_key=public_key)


def test_model_artifact_verifier_fails_on_unsigned_artifact(tmp_path):
    from detection.persistence import ModelIntegrityError, TransparencyLog, ModelArtifactVerifier, get_engine, get_session_factory

    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir)

    artifact_bytes = b"unsigned-model"
    artifact_path = os.path.join(model_dir, "rf.joblib")
    with open(artifact_path, "wb") as f:
        f.write(artifact_bytes)

    sha = hashlib.sha256(artifact_bytes).hexdigest()
    metrics = {"rf": {"artifact_sha256": sha}}
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    # No metrics.json.sig written — simulates unsigned artifact

    engine = get_engine("sqlite:///:memory:")
    sf = get_session_factory(engine)
    log = TransparencyLog(sf)
    log.append("rf", sha)

    verifier = ModelArtifactVerifier(transparency_log=log, model_dir=model_dir)
    _, public_key = _gen_keypair()

    with pytest.raises(ModelIntegrityError, match="Signature file not found"):
        verifier.verify("rf", public_key=public_key)


def test_model_artifact_verifier_fails_when_not_in_transparency_log(tmp_path):
    from detection.persistence import ModelIntegrityError, TransparencyLog, ModelArtifactVerifier, get_engine, get_session_factory

    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir)

    artifact_bytes = b"valid-but-not-logged"
    artifact_path = os.path.join(model_dir, "rf.joblib")
    with open(artifact_path, "wb") as f:
        f.write(artifact_bytes)

    sha = hashlib.sha256(artifact_bytes).hexdigest()
    metrics = {"rf": {"artifact_sha256": sha}}
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    private_key, public_key = _gen_keypair()
    key_path = _write_keypair(tmp_path, private_key)
    sign_metrics(metrics_path, key_path)

    # Transparency log is empty — artifact hash not registered
    engine = get_engine("sqlite:///:memory:")
    sf = get_session_factory(engine)
    log = TransparencyLog(sf)

    verifier = ModelArtifactVerifier(transparency_log=log, model_dir=model_dir)
    with pytest.raises(ModelIntegrityError, match="not in the transparency log"):
        verifier.verify("rf", public_key=public_key)


def test_transparency_log_is_append_only(tmp_path):
    """Appending the same hash twice must be idempotent (no duplicate rows)."""
    from detection.persistence import TransparencyLog, get_engine, get_session_factory

    engine = get_engine("sqlite:///:memory:")
    sf = get_session_factory(engine)
    log = TransparencyLog(sf)

    sha = "a" * 64
    log.append("rf", sha)
    log.append("rf", sha)  # idempotent

    assert log.all_hashes().count(sha) == 1
    assert log.contains(sha) is True
