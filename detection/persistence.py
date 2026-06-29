"""SQLAlchemy persistence model for `RiskScore` records, plus model artifact
integrity verification (Ed25519 trust chain).
"""

import hashlib
import json
import os
import threading
from datetime import UTC, datetime

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from sqlalchemy import DateTime, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import QueuePool

from config import config

_table_init_lock = threading.Lock()


class Base(DeclarativeBase):
    pass


class RiskScoreRecord(Base):
    """Mirrors the on-chain/API `RiskScore` shape documented in the README."""

    __tablename__ = "risk_scores"
    __table_args__ = (UniqueConstraint("wallet", "asset_pair", name="uq_wallet_asset_pair"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet: Mapped[str] = mapped_column(String, index=True, nullable=False)
    asset_pair: Mapped[str] = mapped_column(String, index=True, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    benford_flag: Mapped[bool] = mapped_column(nullable=False, default=False)
    ml_flag: Mapped[bool] = mapped_column(nullable=False, default=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Non-breaking addition: NULL means propagation has not been run yet.
    propagated_risk: Mapped[float | None] = mapped_column(nullable=True, default=None)
    # Stable wash-trading ring id ("ring_<hash>") grouping wallets in the same
    # detected community; NULL when the wallet is not part of any ring.
    ring_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True, default=None)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    def to_risk_score(self) -> dict:
        result = {
            "score": self.score,
            "benford_flag": self.benford_flag,
            "ml_flag": self.ml_flag,
            "timestamp": int(self.updated_at.timestamp()),
            "confidence": self.confidence,
        }
        if self.propagated_risk is not None:
            result["propagated_risk"] = self.propagated_risk
        return result


class EnsembleWeightRecord(Base):
    """Persists per-model dynamic weight adjustment history (issue #268)."""

    __tablename__ = "ensemble_weight_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    model_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    weight: Mapped[float] = mapped_column(nullable=False)
    fp_rate: Mapped[float] = mapped_column(nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    is_systemic_reset: Mapped[bool] = mapped_column(nullable=False, default=False)


class ShapQueryCount(Base):
    """Per-wallet SHAP explanation query counter used for Rényi DP composition.

    Each call to the differentially-private explanation endpoint increments the
    wallet's count; once it exceeds the configured threshold the Gaussian noise
    is scaled up to bound cumulative privacy leakage across repeated queries.
    """

    __tablename__ = "shap_query_counts"

    wallet: Mapped[str] = mapped_column(String, primary_key=True)
    query_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


def get_engine(db_url: str | None = None) -> Engine:
    """Create SQLAlchemy engine with connection pooling.

    Uses QueuePool for better concurrency support, preventing
    'database is locked' errors when multiple threads write simultaneously.

    Args:
        db_url: Database URL (defaults to config.RISK_SCORE_DB_URL)

    Returns:
        SQLAlchemy Engine with connection pooling configured
    """
    effective_db_url = db_url or config.RISK_SCORE_DB_URL

    # Enable WAL mode for SQLite to improve concurrent access
    connect_args = {}
    if effective_db_url.startswith("sqlite"):
        connect_args = {
            "check_same_thread": False,
            # Enable WAL mode for better concurrent access
            "timeout": 20,
        }

    return create_engine(
        effective_db_url,
        future=True,
        poolclass=QueuePool,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW,
        pool_timeout=config.DB_POOL_TIMEOUT,
        pool_pre_ping=True,  # Verify connections before use
        connect_args=connect_args,
    )


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Create session factory with properly configured engine.

    Args:
        engine: Optional engine instance (creates new one if not provided)

    Returns:
        SQLAlchemy sessionmaker bound to the engine
    """
    engine = engine or get_engine()
    with _table_init_lock:
        Base.metadata.create_all(engine, checkfirst=True)

    # Configure SQLite for better concurrent access
    if str(engine.url).startswith("sqlite"):
        _configure_sqlite_for_concurrency(engine)

    return sessionmaker(bind=engine, future=True)


def _configure_sqlite_for_concurrency(engine: Engine) -> None:
    """Configure SQLite database for optimal concurrent access.

    Enables WAL mode and adjusts pragmas for better concurrent performance.

    Args:
        engine: SQLAlchemy engine connected to SQLite database
    """
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        # Enable WAL mode for better concurrent access
        cursor = dbapi_connection.cursor()

        # WAL mode allows concurrent readers with one writer
        cursor.execute("PRAGMA journal_mode=WAL")

        # Increase timeout to reduce contention errors
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds

        # Optimize for concurrent access
        cursor.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL, still safe in WAL mode
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables

        cursor.close()


# ---------------------------------------------------------------------------
# Model artifact integrity
# ---------------------------------------------------------------------------


class ModelIntegrityError(Exception):
    """Raised when any step of the artifact trust chain fails."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _key_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def sign_metrics(metrics_path: str, private_key_path: str) -> str:
    """Sign *metrics_path* with the Ed25519 private key at *private_key_path*.

    Writes a detached signature to ``<metrics_path>.sig`` and returns that
    path.  The private key is never logged or stored anywhere else.
    """
    with open(private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ModelIntegrityError("Signing key is not an Ed25519 private key")

    with open(metrics_path, "rb") as f:
        payload = f.read()

    signature = private_key.sign(payload)
    sig_path = metrics_path + ".sig"
    with open(sig_path, "wb") as f:
        f.write(signature)
    return sig_path


class ModelArtifact:
    """Wraps a model directory and performs end-to-end trust-chain verification."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or config.MODEL_DIR

    def _metrics_path(self) -> str:
        return os.path.join(self.model_dir, "metrics.json")

    def verify_chain(
        self,
        model_name: str,
        public_key: Ed25519PublicKey | None = None,
        trusted_fingerprint: str | None = None,
        expected_data_sha256: str | None = None,
    ) -> None:
        """Verify the complete trust chain for *model_name*.

        Checks (in order):
        1. SHA-256 of the .joblib file matches ``metrics.json``
        2. ``metrics.json`` signature (``metrics.json.sig``) is valid
        3. The signing key fingerprint matches *trusted_fingerprint*
           (falls back to ``config.TRUSTED_SIGNING_KEY_FINGERPRINT``)
        4. If *expected_data_sha256* is given, it matches the value recorded
           in ``metrics.json``

        Raises :class:`ModelIntegrityError` with a descriptive reason on any
        failure.
        """
        metrics_path = self._metrics_path()
        if not os.path.exists(metrics_path):
            raise ModelIntegrityError(f"metrics.json not found in {self.model_dir}")

        with open(metrics_path) as f:
            metrics = json.load(f)

        # 1 — artifact SHA-256
        artifact_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        if not os.path.exists(artifact_path):
            raise ModelIntegrityError(f"Model artifact not found: {artifact_path}")

        actual_sha = _sha256_file(artifact_path)
        expected_sha = (metrics.get(model_name) or {}).get("artifact_sha256")
        if expected_sha is None:
            raise ModelIntegrityError(
                f"No artifact_sha256 entry for '{model_name}' in metrics.json"
            )
        if actual_sha != expected_sha:
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {model_name}: expected {expected_sha}, got {actual_sha}"
            )

        # 2 — metrics.json signature
        sig_path = metrics_path + ".sig"
        if not os.path.exists(sig_path):
            raise ModelIntegrityError(f"Signature file not found: {sig_path}")

        if public_key is None:
            raise ModelIntegrityError(
                "A public key must be supplied to verify_chain (no default public key configured)"
            )

        with open(metrics_path, "rb") as f:
            payload = f.read()
        with open(sig_path, "rb") as f:
            signature = f.read()

        from cryptography.exceptions import InvalidSignature

        try:
            public_key.verify(signature, payload)
        except InvalidSignature:
            raise ModelIntegrityError("metrics.json signature verification failed") from None

        # 3 — signing key fingerprint
        fp = trusted_fingerprint or config.TRUSTED_SIGNING_KEY_FINGERPRINT
        if fp:
            actual_fp = _key_fingerprint(public_key)
            if actual_fp != fp:
                raise ModelIntegrityError(
                    f"Signing key fingerprint mismatch: expected {fp}, got {actual_fp}"
                )

        # 4 — training data SHA-256 (optional)
        if expected_data_sha256 is not None:
            recorded = metrics.get("training_data_sha256")
            if recorded != expected_data_sha256:
                raise ModelIntegrityError(
                    f"Training data SHA-256 mismatch: expected {expected_data_sha256}, "
                    f"got {recorded}"
                )


# ---------------------------------------------------------------------------
# Supply-chain transparency log (issue #277)
# ---------------------------------------------------------------------------


class TransparencyLogRecord(Base):
    """Append-only log of known-good model artifact hashes.

    Each row records one published artifact: its SHA-256 hash, the model name,
    and the timestamp at which it was registered.  Rows are never updated or
    deleted; the log is strictly append-only (enforced by the application layer
    — there is no UPDATE/DELETE path exposed).

    Backup requirement: this table must be backed up separately from the main
    DB so that a coordinated attack cannot modify both the artifact and the log.
    The signing key must be stored in an HSM or encrypted secrets manager.
    """

    __tablename__ = "transparency_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class TransparencyLog:
    """Append-only store for known-good model artifact hashes.

    Usage::

        log = TransparencyLog(session_factory)
        log.append("rf", "<sha256>")           # publish_model_artifact.py
        log.contains("rf", "<sha256>")         # ModelArtifactVerifier
    """

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        # Ensure the transparency_log table exists using the bound engine
        with session_factory() as session:
            Base.metadata.create_all(session.get_bind(), checkfirst=True)

    def append(self, model_name: str, artifact_sha256: str) -> None:
        """Register a new known-good artifact hash (idempotent for same hash)."""
        if len(artifact_sha256) != 64 or not all(c in "0123456789abcdef" for c in artifact_sha256):
            raise ValueError(f"artifact_sha256 must be a 64-char lowercase hex string, got: {artifact_sha256!r}")
        with self._session_factory() as session:
            existing = (
                session.query(TransparencyLogRecord)
                .filter_by(artifact_sha256=artifact_sha256)
                .first()
            )
            if existing is None:
                session.add(
                    TransparencyLogRecord(
                        model_name=model_name,
                        artifact_sha256=artifact_sha256,
                    )
                )
                session.commit()

    def contains(self, artifact_sha256: str) -> bool:
        """Return True if *artifact_sha256* is in the transparency log."""
        with self._session_factory() as session:
            return (
                session.query(TransparencyLogRecord)
                .filter_by(artifact_sha256=artifact_sha256)
                .first()
            ) is not None

    def all_hashes(self) -> list[str]:
        """Return all registered hashes (for auditing)."""
        with self._session_factory() as session:
            rows = session.query(TransparencyLogRecord).order_by(
                TransparencyLogRecord.registered_at
            ).all()
            return [r.artifact_sha256 for r in rows]


class ModelArtifactVerifier:
    """Supply-chain verifier for model artifacts (issue #277).

    Performs three checks in < 1 second regardless of model file size:

    1. SHA-256 hash of the artifact matches the expected value.
    2. Ed25519 cryptographic signature on metrics.json is valid.
    3. Artifact hash is present in the append-only transparency log.

    Any failure raises :class:`ModelIntegrityError`.

    Security note: The signing key must be stored in an HSM or encrypted
    secrets manager (e.g. AWS Secrets Manager, HashiCorp Vault).  This class
    only handles verification; signing is done by
    ``scripts/publish_model_artifact.py`` in a controlled environment.
    """

    def __init__(
        self,
        transparency_log: "TransparencyLog",
        model_dir: str | None = None,
    ) -> None:
        self._log = transparency_log
        self._model_dir = model_dir or config.MODEL_DIR

    def verify(
        self,
        model_name: str,
        public_key: "Ed25519PublicKey",
        expected_sha256: str | None = None,
    ) -> str:
        """Verify *model_name* artifact passes all supply-chain checks.

        Returns the artifact's SHA-256 hex digest on success.
        Raises :class:`ModelIntegrityError` on any failure.

        Parameters
        ----------
        model_name:
            Bare model name without extension (e.g. ``"rf"``).
        public_key:
            Ed25519 public key used to verify the ``metrics.json`` signature.
        expected_sha256:
            If supplied, the artifact SHA-256 must equal this value in addition
            to the transparency log check.
        """
        artifact_path = os.path.join(self._model_dir, f"{model_name}.joblib")
        if not os.path.exists(artifact_path):
            raise ModelIntegrityError(f"Artifact not found: {artifact_path}")

        # 1 — SHA-256 (fast: hash-only, no model parsing)
        actual_sha = _sha256_file(artifact_path)
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {model_name}: "
                f"expected {expected_sha256}, got {actual_sha}"
            )

        # 2 — Ed25519 signature on metrics.json
        metrics_path = os.path.join(self._model_dir, "metrics.json")
        sig_path = metrics_path + ".sig"
        if not os.path.exists(metrics_path):
            raise ModelIntegrityError(f"metrics.json not found in {self._model_dir}")
        if not os.path.exists(sig_path):
            raise ModelIntegrityError(f"Signature file not found: {sig_path}")

        with open(metrics_path, "rb") as f:
            payload = f.read()
        with open(sig_path, "rb") as f:
            signature = f.read()

        from cryptography.exceptions import InvalidSignature

        try:
            public_key.verify(signature, payload)
        except InvalidSignature:
            raise ModelIntegrityError(
                f"metrics.json signature verification failed for {model_name}"
            ) from None

        # 3 — Transparency log check
        if not self._log.contains(actual_sha):
            raise ModelIntegrityError(
                f"Artifact {model_name} (sha256={actual_sha[:16]}…) "
                "is not in the transparency log — refusing to load"
            )

        return actual_sha
