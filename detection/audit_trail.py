"""Cryptographically committed audit trail for forensic reports.

Security note: all content hashing uses SHA-256 exclusively (not MD5 or SHA-1),
per the project security policy documented in docs/security.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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


# ---------------------------------------------------------------------------
# Merkle-chain tamper detection (Issue #196)
# ---------------------------------------------------------------------------

class TamperDetectedError(Exception):
    """Raised when AuditMerkleChain.verify_chain detects a modified entry."""


class _MerkleBase(DeclarativeBase):
    pass


class _MerkleRootRecord(_MerkleBase):
    """Separate append-only table storing Merkle roots independently."""

    __tablename__ = "audit_merkle_roots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_index = Column(Integer, nullable=False, unique=True)
    merkle_root = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _leaf_hash(index: int, content_hash: str) -> str:
    return _sha256(f"{index}:{content_hash}")


def _node_hash(left: str, right: str) -> str:
    return _sha256(left + right)


def _merkle_root(leaf_hashes: list[str]) -> str:
    """Compute Merkle root from leaf hashes in O(n) time."""
    if not leaf_hashes:
        return _sha256("empty")
    nodes = list(leaf_hashes)
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])  # duplicate last leaf when odd
        nodes = [_node_hash(nodes[i], nodes[i + 1]) for i in range(0, len(nodes), 2)]
    return nodes[0]


@dataclass
class MerkleAuditEntry:
    """A single entry in the Merkle audit chain."""

    index: int
    content_hash: str          # SHA-256 of entry content
    prev_merkle_root: str      # root before this entry
    merkle_root: str           # root after including this entry
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class AuditMerkleChain:
    """Merkle tree commitment scheme over audit log entries.

    Each new entry extends a running Merkle root so any retrospective
    modification to a log entry is detectable via :meth:`verify_chain`.

    Merkle roots are stored in a separate ``audit_merkle_roots`` table,
    making root tampering independently detectable from in-entry tampering.

    Security: SHA-256 is used for all hashing (not MD5 or SHA-1).
    """

    def __init__(self, session_factory=None) -> None:
        if session_factory is None:
            engine = create_engine(config.RISK_SCORE_DB_URL)
            _MerkleBase.metadata.create_all(engine, checkfirst=True)
            if str(engine.url).startswith("sqlite"):
                @event.listens_for(engine, "connect")
                def _wal(conn, _rec):
                    conn.execute("PRAGMA journal_mode=WAL")
            self._session_factory = sessionmaker(bind=engine, future=True)
        else:
            self._session_factory = session_factory
        self._entries: list[MerkleAuditEntry] = []

    def _load_roots_from_db(self) -> dict[int, str]:
        with self._session_factory() as session:
            rows = session.query(_MerkleRootRecord).order_by(_MerkleRootRecord.entry_index).all()
            return {r.entry_index: r.merkle_root for r in rows}

    def _save_root(self, session: Session, index: int, root: str) -> None:
        record = _MerkleRootRecord(
            entry_index=index,
            merkle_root=root,
            created_at=datetime.now(UTC),
        )
        session.add(record)

    def append(self, content: dict[str, Any]) -> MerkleAuditEntry:
        """Append a new audit entry and extend the Merkle chain.

        Parameters
        ----------
        content:
            Arbitrary dict representing the audit event.  Its canonical
            JSON SHA-256 hash becomes the leaf node in the chain.

        Returns the constructed :class:`MerkleAuditEntry`.
        """
        content_bytes = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        index = len(self._entries)
        prev_root = self._entries[-1].merkle_root if self._entries else _sha256("genesis")

        leaf_hashes = [_leaf_hash(e.index, e.content_hash) for e in self._entries]
        leaf_hashes.append(_leaf_hash(index, content_hash))
        new_root = _merkle_root(leaf_hashes)

        entry = MerkleAuditEntry(
            index=index,
            content_hash=content_hash,
            prev_merkle_root=prev_root,
            merkle_root=new_root,
        )
        self._entries.append(entry)

        with self._session_factory() as session:
            self._save_root(session, index, new_root)
            session.commit()

        return entry

    def verify_chain(self, start_index: int = 0, end_index: int | None = None) -> bool:
        """Re-compute Merkle roots and confirm they match recorded roots.

        Runs in O(n) where n = end_index - start_index.

        Raises
        ------
        TamperDetectedError
            When any entry's re-computed root does not match the stored root.
        """
        stop = end_index if end_index is not None else len(self._entries)
        db_roots = self._load_roots_from_db()

        leaf_hashes: list[str] = []
        for i in range(stop):
            if i < start_index:
                # Still need to build up hashes before the window
                leaf_hashes.append(_leaf_hash(i, self._entries[i].content_hash))
                continue
            if i >= len(self._entries):
                raise TamperDetectedError(f"Entry index {i} is missing from in-memory chain")
            entry = self._entries[i]
            leaf_hashes.append(_leaf_hash(i, entry.content_hash))
            recomputed = _merkle_root(leaf_hashes[:])

            if recomputed != entry.merkle_root:
                raise TamperDetectedError(
                    f"In-entry Merkle root mismatch at index {i}: "
                    f"expected {entry.merkle_root!r}, got {recomputed!r}"
                )

            stored_root = db_roots.get(i)
            if stored_root is None:
                raise TamperDetectedError(
                    f"No Merkle root found in separate table for index {i}"
                )
            if stored_root != recomputed:
                raise TamperDetectedError(
                    f"Separate-table Merkle root mismatch at index {i}: "
                    f"db={stored_root!r}, computed={recomputed!r}"
                )

        return True
