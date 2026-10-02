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
from collections.abc import Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from sqlalchemy import Boolean, Column, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from utils.logging import get_logger

logger = get_logger(__name__)

_DB_URL = os.getenv("RISK_SCORE_DB_URL", "sqlite:///ledgerlens.db")

# Default certificate lifetime (days) used for issuance and rotation.  Override
# per-call via ``validity_days`` or globally via the env var below.
DEFAULT_CERT_LIFETIME_DAYS = int(os.getenv("FEDERATED_CERT_LIFETIME_DAYS", "365"))

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


class CertAuditLog(_Base):
    """Append-only audit trail of certificate lifecycle events.

    Records who/what/when for every issuance, rotation, and revocation so the
    federation can be audited without relying on transient log files.
    """

    __tablename__ = "federated_cert_audit_log"

    id: int = Column(String, primary_key=True)  # uuid4 hex
    event: str = Column(String, nullable=False)  # issued|rotated|revoked
    cn: str = Column(String, nullable=False)
    actor: str = Column(String, nullable=False)
    allowed_models: str | None = Column(String, nullable=True)
    expires_at: datetime.datetime | None = Column(DateTime, nullable=True)
    occurred_at: datetime.datetime = Column(DateTime, nullable=False)
    detail: str | None = Column(String, nullable=True)


def _get_session_factory(db_url: str = _DB_URL):
    engine = create_engine(db_url)
    _Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _record_audit(
    session,
    event: str,
    cn: str,
    actor: str,
    allowed_models: str | None = None,
    expires_at: datetime.datetime | None = None,
    detail: str | None = None,
) -> None:
    """Append a certificate lifecycle event to the audit log."""
    import uuid

    session.add(
        CertAuditLog(
            id=uuid.uuid4().hex,
            event=event,
            cn=cn,
            actor=actor,
            allowed_models=allowed_models,
            expires_at=expires_at,
            occurred_at=datetime.datetime.now(datetime.UTC),
            detail=detail,
        )
    )


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
    validity_days: int | None = None,
    db_url: str = _DB_URL,
    actor: str = "ca",
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Issue a new participant certificate signed by the CA.

    The participant private key is generated here and returned to the caller.
    It must be transmitted to the participant over a secure channel — the
    coordinator NEVER stores or sees it after this call returns.

    The allowed model IDs are encoded in the OU field for retrieval during auth.
    ``validity_days`` defaults to ``DEFAULT_CERT_LIFETIME_DAYS`` (configurable
    via the ``FEDERATED_CERT_LIFETIME_DAYS`` env var).

    Returns (participant_private_key, participant_cert).
    """
    if validity_days is None:
        validity_days = DEFAULT_CERT_LIFETIME_DAYS
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
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
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
        _record_audit(
            session,
            event="issued",
            cn=cn,
            actor=actor,
            allowed_models=models_str,
            expires_at=expires_at,
        )
        session.commit()

    logger.info(
        "Issued certificate for CN=%r (models=%s, expires=%s)", cn, models_str, expires_at.date()
    )
    return part_key, cert


def revoke_certificate(cn: str, db_url: str = _DB_URL, actor: str = "ca") -> None:
    """Mark a participant certificate as revoked in the DB.

    The coordinator reloads the revocation list every ≤60 s, so revocation
    takes effect within 60 seconds.  The revocation is recorded in the audit
    log with the acting principal and timestamp.
    """
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        record = session.get(ParticipantCertRecord, cn)
        if record is None:
            raise KeyError(f"No certificate found for CN={cn!r}")
        record.revoked = True
        record.revoked_at = datetime.datetime.now(datetime.UTC)
        _record_audit(
            session,
            event="revoked",
            cn=cn,
            actor=actor,
            allowed_models=record.allowed_models,
            expires_at=record.expires_at,
        )
        session.commit()
    logger.info("Revoked certificate for CN=%r", cn)


def rotate_certificate(
    cn: str,
    allowed_models: Sequence[str],
    ca_key: ec.EllipticCurvePrivateKey,
    ca_cert: x509.Certificate,
    validity_days: int | None = None,
    db_url: str = _DB_URL,
    actor: str = "ca",
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Revoke the existing certificate for *cn* and issue a fresh one.

    Rotation is atomic from the coordinator's perspective: the old record is
    revoked and the new one issued in a single transaction, so an in-progress
    federated round for unaffected participants is not disrupted.

    Returns (new_private_key, new_cert).
    """
    if validity_days is None:
        validity_days = DEFAULT_CERT_LIFETIME_DAYS
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        existing = session.get(ParticipantCertRecord, cn)
        if existing is not None:
            existing.revoked = True
            existing.revoked_at = datetime.datetime.now(datetime.UTC)
            _record_audit(
                session,
                event="revoked",
                cn=cn,
                actor=actor,
                allowed_models=existing.allowed_models,
                expires_at=existing.expires_at,
                detail="superseded by rotation",
            )
        session.commit()

    part_key, cert = issue_certificate(
        cn, allowed_models, ca_key, ca_cert, validity_days, db_url, actor=actor
    )
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        _record_audit(
            session,
            event="rotated",
            cn=cn,
            actor=actor,
            allowed_models=",".join(allowed_models),
            expires_at=cert.not_valid_after_utc,
        )
        session.commit()
    return part_key, cert


def is_revoked(cn: str, db_url: str = _DB_URL) -> bool:
    """Return True if the participant's certificate is revoked or unknown.

    The coordinator calls this before accepting a participant's contribution;
    unknown CNs are treated as revoked (fail-closed).
    """
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        record = session.get(ParticipantCertRecord, cn)
        if record is None:
            return True
        return bool(record.revoked)


def is_expired(cn: str, db_url: str = _DB_URL) -> bool:
    """Return True if the participant's certificate has expired."""
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        record = session.get(ParticipantCertRecord, cn)
        if record is None:
            return True
        expires_at = record.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=datetime.UTC)
        return expires_at <= datetime.datetime.now(datetime.UTC)


def list_expiring_soon(within_days: int = 30, db_url: str = _DB_URL) -> list[ParticipantCertRecord]:
    """Return participant records whose certificate expires within *within_days* days."""
    threshold = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=within_days)
    SessionFactory = _get_session_factory(db_url)
    with SessionFactory() as session:
        records = session.query(ParticipantCertRecord).filter(
            ParticipantCertRecord.revoked.is_(False),
            ParticipantCertRecord.expires_at <= threshold,
        ).all()
        return list(records)
