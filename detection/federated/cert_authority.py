"""Certificate Authority for federated learning participants.

Issues, revokes, and rotates X.509 client certificates signed by the
LedgerLens CA.  The CA private key must be stored in an HSM or encrypted
secrets manager (see docs/security.md); this module expects the key material
to be passed in at runtime, never stored on disk by this code.

Certificate CN convention: the Common Name (CN) of each participant certificate
is its opaque participant identifier (e.g. ``participant-A``).  The SAN
extension encodes the allowed model IDs as a comma-separated string in the
``organizationalUnitName`` (OU) field for easy extraction during auth.

Revocation is stored in a SQLite table (same DB as the rest of LedgerLens) and
is reloaded by the coordinator every ≤60 seconds.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from sqlalchemy import Boolean, Column, DateTime, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from utils.logging import get_logger

logger = get_logger(__name__)

_DB_URL = os.getenv("RISK_SCORE_DB_URL", "sqlite:///ledgerlens.db")

# ---------------------------------------------------------------------------
# ORM
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class ParticipantCertRecord(_Base):
    __tablename__ = "federated_participant_certs"

    cn: str = Column(String, primary_key=True)
    allowed_models: str = Column(String, nullable=False)  # comma-separated
    issued_at: datetime.datetime = Column(DateTime, nullable=False)
    expires_at: datetime.datetime = Column(DateTime, nullable=False)
    revoked: bool = Column(Boolean, nullable=False, default=False)
    revoked_at: datetime.datetime | None = Column(DateTime, nullable=True)
    cert_pem: str = Column(String, nullable=False)


def _get_session_factory(db_url: str = _DB_URL):
    engine = create_engine(db_url)
    _Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# CA helpers
# ---------------------------------------------------------------------------


def generate_ca_keypair() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Generate a new ECDSA P-256 CA key and self-signed certificate.

    Returns (ca_private_key, ca_cert).  The private key must be stored in an
    HSM or encrypted secrets manager — never write it to plaintext files.
    """
    ca_key = ec.generate_private_key(ec.SECP256R1())
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LedgerLens"),
            x509.NameAttribute(NameOID.COMMON_NAME, "LedgerLens Federated CA"),
        ]
    )
    now = datetime.datetime.now(datetime.UTC)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    return ca_key, ca_cert


def issue_certificate(
    cn: str,
    allowed_models: Sequence[str],
    ca_key: ec.EllipticCurvePrivateKey,
    ca_cert: x509.Certificate,
    validity_days: int = 365,
    db_url: str = _DB_URL,
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Issue a new participant certificate signed by the CA.

    The participant private key is generated here and returned to the caller.
    It must be transmitted to the participant over a secure channel — the
    coordinator NEVER stores or sees it after this call returns.

    The allowed model IDs are encoded in the OU field for retrieval during auth.

    Returns (participant_private_key, participant_cert).
    """
    part_key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.UTC)
    expires_at = now + datetime.timedelta(days=validity_days)

    models_str = ",".join(allowed_models)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LedgerLens Participant"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, models_str),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(part_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(expires_at)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        existing = session.get(ParticipantCertRecord, cn)
        if existing is not None:
            session.delete(existing)
            session.flush()
        session.add(
            ParticipantCertRecord(
                cn=cn,
                allowed_models=models_str,
                issued_at=now,
                expires_at=expires_at,
                revoked=False,
                revoked_at=None,
                cert_pem=cert_pem,
            )
        )
        session.commit()

    logger.info("Issued certificate for CN=%r (models=%s, expires=%s)", cn, models_str, expires_at.date())
    return part_key, cert


def revoke_certificate(cn: str, db_url: str = _DB_URL) -> None:
    """Mark a participant certificate as revoked in the DB.

    The coordinator reloads the revocation list every ≤60 s, so revocation
    takes effect within 60 seconds.
    """
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        record = session.get(ParticipantCertRecord, cn)
        if record is None:
            raise KeyError(f"No certificate found for CN={cn!r}")
        record.revoked = True
        record.revoked_at = datetime.datetime.now(datetime.UTC)
        session.commit()
    logger.info("Revoked certificate for CN=%r", cn)


def rotate_certificate(
    cn: str,
    allowed_models: Sequence[str],
    ca_key: ec.EllipticCurvePrivateKey,
    ca_cert: x509.Certificate,
    validity_days: int = 365,
    db_url: str = _DB_URL,
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Revoke the existing certificate for *cn* and issue a fresh one.

    Returns (new_private_key, new_cert).
    """
    try:
        revoke_certificate(cn, db_url)
    except KeyError:
        pass  # first issuance — nothing to revoke
    return issue_certificate(cn, allowed_models, ca_key, ca_cert, validity_days, db_url)


def list_expiring_soon(
    within_days: int = 30, db_url: str = _DB_URL
) -> list[ParticipantCertRecord]:
    """Return participant records whose certificate expires within *within_days* days."""
    threshold = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=within_days)
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        return (
            session.query(ParticipantCertRecord)
            .filter(
                ParticipantCertRecord.expires_at <= threshold,
                ParticipantCertRecord.revoked == False,  # noqa: E712
            )
            .all()
        )


def list_all_participants(db_url: str = _DB_URL) -> list[ParticipantCertRecord]:
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        return session.query(ParticipantCertRecord).all()


# ---------------------------------------------------------------------------
# RevocationCache
# ---------------------------------------------------------------------------


class RevocationCache:
    """In-memory revocation list, refreshed from the DB every *refresh_interval_seconds*.

    The coordinator instantiates one cache and calls ``is_revoked()`` on every
    request.  The background refresh ensures revocation takes effect ≤60 s after
    ``revoke_certificate()`` is called.
    """

    def __init__(
        self,
        refresh_interval_seconds: int = 30,
        db_url: str = _DB_URL,
    ) -> None:
        self._db_url = db_url
        self._refresh_interval = refresh_interval_seconds
        self._revoked_cns: set[str] = set()
        self._lock = threading.RLock()
        self._refresh()

    def is_revoked(self, cn: str) -> bool:
        with self._lock:
            return cn in self._revoked_cns

    def get_allowed_models(self, cn: str, db_url: str | None = None) -> list[str] | None:
        """Return the allowed model list for *cn*, or None if unknown."""
        url = db_url or self._db_url
        SessionFactory = _get_session_factory(url)
        with SessionFactory() as session:
            record = session.get(ParticipantCertRecord, cn)
            if record is None or record.revoked:
                return None
            return [m.strip() for m in record.allowed_models.split(",") if m.strip()]

    def refresh(self) -> None:
        """Force an immediate reload from the DB."""
        self._refresh()

    def _refresh(self) -> None:
        try:
            SessionFactory = _get_session_factory(self._db_url)
            with SessionFactory() as session:
                revoked = {
                    r.cn
                    for r in session.query(ParticipantCertRecord)
                    .filter(ParticipantCertRecord.revoked == True)  # noqa: E712
                    .all()
                }
            with self._lock:
                self._revoked_cns = revoked
        except Exception as exc:
            logger.warning("RevocationCache refresh failed: %s", exc)
        finally:
            import threading as _threading

            t = _threading.Timer(self._refresh_interval, self._refresh)
            t.daemon = True
            t.start()


import threading  # noqa: E402 — used by RevocationCache above
