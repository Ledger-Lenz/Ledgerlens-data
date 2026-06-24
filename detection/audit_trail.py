"""Cryptographically committed audit trail for forensic reports."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from config import config
from detection.forensic_report import ForensicReport
from detection.persistence import ModelIntegrityError
from utils.logging import get_logger

logger = get_logger(__name__)


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def hash_features(features: dict[str, Any]) -> str:
    """SHA-256 of sorted feature key/value pairs."""
    material = json.dumps(sorted(features.items()), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()


def hash_shap_explanations(shap_explanations: list[dict]) -> str:
    """SHA-256 of SHAP explanation records in stable order."""
    material = json.dumps(shap_explanations, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()


def commitment_hash(payload: dict[str, Any]) -> str:
    """Return the SHA-256 commitment over the canonical audit payload."""
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def load_signing_key(path: str | None = None) -> Ed25519PrivateKey:
    key_path = path or config.MODEL_SIGNING_PRIVATE_KEY_PATH
    if not key_path or not os.path.exists(key_path):
        raise ModelIntegrityError(
            "MODEL_SIGNING_PRIVATE_KEY_PATH is not set or does not exist — "
            "cannot sign audit trail entries"
        )
    with open(key_path, "rb") as handle:
        private_key = serialization.load_pem_private_key(handle.read(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ModelIntegrityError("Signing key is not an Ed25519 private key")
    return private_key


def load_verification_key(
    public_key_path: str | None = None,
    private_key_path: str | None = None,
) -> Ed25519PublicKey:
    if public_key_path and os.path.exists(public_key_path):
        with open(public_key_path, "rb") as handle:
            public_key = serialization.load_pem_public_key(handle.read())
        if not isinstance(public_key, Ed25519PublicKey):
            raise ModelIntegrityError("Verification key is not an Ed25519 public key")
        return public_key

    private_key = load_signing_key(private_key_path)
    return private_key.public_key()


@dataclass
class AuditTrailEntry:
    payload: dict[str, Any]
    signature_hex: str
    commitment_hash: str


class AuditTrailWriter:
    """Append-only, signed NDJSON audit log for forensic report commitments."""

    def __init__(
        self,
        log_path: str | None = None,
        private_key_path: str | None = None,
    ) -> None:
        self.log_path = log_path or config.AUDIT_LOG_PATH
        self._private_key_path = private_key_path

    def _private_key(self) -> Ed25519PrivateKey:
        return load_signing_key(self._private_key_path)

    def build_payload(
        self,
        report: ForensicReport,
        *,
        features: dict[str, Any],
        model_version: str,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        risk_score = report.risk_score
        score_value = risk_score.get("score") if isinstance(risk_score, dict) else risk_score
        return {
            "wallet": report.wallet,
            "asset_pair": report.asset_pair,
            "score": score_value,
            "risk_score": risk_score,
            "features_hash": hash_features(features),
            "shap_hash": hash_shap_explanations(report.shap_explanations),
            "model_version": model_version,
            "timestamp": timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

    def commit(
        self,
        report: ForensicReport,
        model_version: str,
        *,
        features: dict[str, Any],
        timestamp: str | None = None,
    ) -> str:
        """Append a signed entry and return the payload commitment hash."""
        payload = self.build_payload(
            report,
            features=features,
            model_version=model_version,
            timestamp=timestamp,
        )
        digest = commitment_hash(payload)
        signature = self._private_key().sign(_canonical_json(payload))

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        entry = {
            "payload": payload,
            "commitment_hash": digest,
            "sig": signature.hex(),
        }
        with open(self.log_path, "ab") as handle:
            handle.write((json.dumps(entry, sort_keys=True) + "\n").encode())

        logger.info("Audit trail entry committed for %s (%s)", report.wallet, digest)
        return digest

    def verify_entry(
        self,
        entry: dict[str, Any],
        public_key: Ed25519PublicKey,
    ) -> bool:
        payload = entry["payload"]
        expected = commitment_hash(payload)
        if entry.get("commitment_hash") != expected:
            return False
        signature = bytes.fromhex(entry["sig"])
        public_key.verify(signature, _canonical_json(payload))
        return True


def read_audit_log(log_path: str | None = None) -> list[dict[str, Any]]:
    path = log_path or config.AUDIT_LOG_PATH
    if not os.path.exists(path):
        return []
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ModelIntegrityError(f"Invalid JSON on line {line_number} of {path}") from exc
    return entries


def verify_audit_log(
    log_path: str | None = None,
    public_key_path: str | None = None,
) -> tuple[int, list[int]]:
    """Verify every entry; return (valid_count, failing_line_numbers)."""
    path = log_path or config.AUDIT_LOG_PATH
    public_key = load_verification_key(public_key_path)
    writer = AuditTrailWriter(log_path=path)
    failures: list[int] = []
    valid = 0
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            try:
                if writer.verify_entry(entry, public_key):
                    valid += 1
                else:
                    failures.append(line_number)
            except Exception:
                failures.append(line_number)
    return valid, failures


def commit_forensic_report(
    report: ForensicReport,
    features: dict[str, Any],
    model_version: str,
    *,
    timestamp: str | None = None,
) -> str | None:
    """Append a signed audit entry when model signing is configured."""
    if not config.MODEL_SIGNING_PRIVATE_KEY_PATH:
        logger.debug("MODEL_SIGNING_PRIVATE_KEY_PATH unset — skipping audit trail commit")
        return None
    return AuditTrailWriter().commit(
        report,
        model_version,
        features=features,
        timestamp=timestamp,
    )
