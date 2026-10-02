"""SQLAlchemy persistence model for `RiskScore` records, plus model artifact
integrity verification (Ed25519 trust chain).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Optional

try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone
    UTC = timezone.utc  # type: ignore

import numpy as np
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except Exception:
    serialization = None
    Ed25519PrivateKey = None
    Ed25519PublicKey = None
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
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
    propagated_risk: Mapped[Optional[float]] = mapped_column(nullable=True, default=None)
    # Stable wash-trading ring id ("ring_<hash>") grouping wallets in the same
    # detected community; NULL when the wallet is not part of any ring.
    ring_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True, default=None)
    # JSON blob mapping feature_name → [trade_id, ...] for provenance tracking
    # (Issue #244). NULL when FEATURE_PROVENANCE_ENABLED=False or not computed.
    provenance_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    # True when this score has been certified robust via IBP at the standard
    # evaluation epsilons (ε=0.01 and ε=0.05) — Issue #245. Internal only.
    certified_robust: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    # "provisional" (default): written by the continuous streaming/SSE path,
    # which has no window-close event and may still change as more trades
    # arrive. "final": written by a completed batch pipeline run or a
    # completed stream-replay run over a closed, bounded time window
    # (Issue #670). See docs/adr/0001-unified-idempotency-finality.md.
    finality: Mapped[str] = mapped_column(String(16), nullable=False, default="provisional")
    # Planned for Issue #856: nullable `fused_score_lower`, `fused_score_upper`
    # (Float) and `fusion_strategy` (String(32)) columns, added by migration
    # 0008. NULL = fusion not computed. See the plan in
    # detection/risk_score_store.py (RiskScoreStore) for read/write rules.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    @property
    def propagated_risk_score(self) -> float | None:
        """Alias for propagated_risk (used by WeightedRiskPropagation cache)."""
        return self.propagated_risk

    def to_risk_score(self) -> dict:
        # NOTE: `finality` and `certified_robust` are deliberately excluded —
        # this dict mirrors the on-chain/API RiskScore shape shared with
        # ledgerlens-core, and unilaterally changing that wire shape from
        # this repo would silently break cross-repo ABI compatibility. Read
        # `finality` via the record attribute or `RiskScoreStore` instead.
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


class ModelVersionRecord(Base):
    """Tracks every trained model version with its shadow deployment lifecycle.

    ``status`` transitions: shadow → production | rolled_back | archived, or
    shadow/production → quarantined (Issue #871: a backdoor scan flagged
    this candidate; ``quarantine_reason``/``quarantined_at`` are populated
    and ``detection.model_governance.promote_candidate`` refuses to ever
    promote a quarantined ``model_artifact_path`` again, pending human
    review — see ``detection.model_governance.quarantine_candidate``).
    ``training_metadata`` stores a JSON blob (metrics, feature hash, etc.).
    ``artifact_signature`` is the hex Ed25519 signature of the model directory
    produced by :class:`ModelArtifact` — verified on rollback before loading.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    model_artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    artifact_signature: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="shadow", index=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    shadow_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    rolled_back_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    shadow_drift_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shadow_total_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    shadow_drift_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    promotion_blocked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Actor identity that approved the promotion/rollback (Grand 2 — issue
    # #671). Populated exclusively by detection.model_governance so every row
    # with status in (production, rolled_back) has a queryable approving
    # actor; NULL only for legacy rows written before this column existed.
    promoted_by: Mapped[str | None] = mapped_column(String, nullable=True)
    rolled_back_by: Mapped[str | None] = mapped_column(String, nullable=True)
    # version_id of the ModelVersionRecord this row supersedes (its immediate
    # predecessor in the shadow -> production -> rolled_back chain), used to
    # walk the promotion history and to find the rollback target.
    parent_version_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # Issue #871: populated only when status == "quarantined". quarantine_reason
    # is a short human-readable summary (the full structured backdoor report is
    # stored in training_metadata); quarantined_at records when the scan flagged it.
    quarantined_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    quarantine_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


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


class ModelInversionQueryTracker(Base):
    """Track API score queries per (caller_id, wallet_id) pair to defend against
    model inversion attacks via repeated queries (Issue #264).

    When a caller exceeds MODEL_INVERSION_QUERY_LIMIT queries on a wallet,
    subsequent API requests return 429 Too Many Requests.
    """

    __tablename__ = "model_inversion_query_tracker"
    __table_args__ = (UniqueConstraint("caller_id", "wallet_id", name="uq_caller_wallet"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    caller_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    wallet_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    query_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    last_query_at: Mapped[datetime] = mapped_column(
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


def load_trusted_public_key(path: str | None = None) -> Ed25519PublicKey:
    """Load the Ed25519 public key used to verify model artifacts at load time.

    Args:
        path: PEM public-key path. Defaults to
            ``config.TRUSTED_SIGNING_PUBLIC_KEY_PATH``.

    Raises:
        ModelIntegrityError: if no path is configured, the file is missing,
            or it does not contain an Ed25519 public key. This is
            deliberately a hard failure — a production process with no
            configured trust anchor must never fall back to loading
            artifacts unverified.
    """
    key_path = path or config.TRUSTED_SIGNING_PUBLIC_KEY_PATH
    if not key_path:
        raise ModelIntegrityError(
            "TRUSTED_SIGNING_PUBLIC_KEY_PATH is not configured — cannot verify model "
            "artifacts. Set it to the PEM Ed25519 public key matching the key used by "
            "MODEL_SIGNING_PRIVATE_KEY_PATH, or pass public_key= explicitly."
        )
    if not os.path.exists(key_path):
        raise ModelIntegrityError(f"Trusted public key file not found: {key_path}")
    with open(key_path, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())
    if not isinstance(public_key, Ed25519PublicKey):
        raise ModelIntegrityError(f"{key_path} does not contain an Ed25519 public key")
    return public_key


def get_default_transparency_log(
    session_factory: "sessionmaker | None" = None,
) -> "TransparencyLog":
    """Build a :class:`TransparencyLog` bound to ``config.RISK_SCORE_DB_URL``.

    Convenience for production call sites that don't already hold a
    session factory (e.g. ``RiskScorer`` constructed with no explicit
    ``transparency_log=``).
    """
    sf = session_factory or get_session_factory(get_engine())
    return TransparencyLog(sf)


def sign_and_register_artifact(
    model_name: str,
    model_dir: str,
    private_key_path: str,
    transparency_log: "TransparencyLog",
) -> str:
    """Compute the artifact's SHA-256, sign ``metrics.json``, and append the
    hash to the transparency log — in that order, atomically from the
    caller's perspective.

    This is the single implementation of "sign + publish" shared by the
    manual ``scripts/publish_model_artifact.py`` CLI and the automated
    promotion path in ``detection.model_governance``, so both go through
    byte-identical signing logic and neither can silently diverge from the
    other. Returns the artifact's SHA-256 hex digest.
    """
    artifact_path = os.path.join(model_dir, f"{model_name}.joblib")
    if not os.path.exists(artifact_path):
        raise ModelIntegrityError(f"Artifact not found: {artifact_path}")

    metrics_path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise ModelIntegrityError(f"metrics.json not found in {model_dir}")

    with open(private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ModelIntegrityError("Signing key is not an Ed25519 private key")

    artifact_sha = _sha256_file(artifact_path)

    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics.setdefault(model_name, {})["artifact_sha256"] = artifact_sha
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    sign_metrics(metrics_path, private_key_path)
    transparency_log.append(model_name, artifact_sha)
    return artifact_sha


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
                f"SHA-256 mismatch for {model_name}: expected {expected_sha}, got {actual_sha}. "
                "Remediation: restore the model artifact from version control or re-run "
                "`scripts/publish_model_artifact.py` to re-register the current file's hash."
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
            raise ModelIntegrityError(
                "metrics.json signature verification failed. "
                "Remediation: re-run `scripts/publish_model_artifact.py` to regenerate "
                "metrics.json.sig with the correct signing key."
            ) from None

        # 3 — signing key fingerprint
        fp = trusted_fingerprint or config.TRUSTED_SIGNING_KEY_FINGERPRINT
        if fp:
            actual_fp = _key_fingerprint(public_key)
            if actual_fp != fp:
                raise ModelIntegrityError(
                    f"Signing key fingerprint mismatch: expected {fp}, got {actual_fp}. "
                    "Remediation: set TRUSTED_SIGNING_KEY_FINGERPRINT to the key that actually "
                    "signed this artifact, or re-run `scripts/publish_model_artifact.py` with the "
                    "expected key."
                )

        # 4 — training data SHA-256 (optional)
        if expected_data_sha256 is not None:
            recorded = metrics.get("training_data_sha256")
            if recorded != expected_data_sha256:
                raise ModelIntegrityError(
                    f"Training data SHA-256 mismatch: expected {expected_data_sha256}, "
                    f"got {recorded}. "
                    "Remediation: re-run `scripts/publish_model_artifact.py` after regenerating "
                    "the training dataset so metrics.json's training_data_sha256 matches the data "
                    "this artifact was trained on."
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


class FederatedAuditRecord(Base):
    """Append-only audit trail row for one federated learning round (issue #227).

    See `detection.federated.coordinator.FederatedAuditTrail` for the writer.
    Rows are never updated or deleted; ``prev_hash`` chains each record to its
    predecessor so retroactive tampering is detectable.
    """

    __tablename__ = "federated_audit_trail"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    round_timestamp: Mapped[str] = mapped_column(String, nullable=False)
    participant_fingerprints: Mapped[str] = mapped_column(Text, nullable=False)
    gradient_norms: Mapped[str] = mapped_column(Text, nullable=False)
    aggregation_algorithm: Mapped[str] = mapped_column(String, nullable=False)
    aggregate_model_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    round_outcome: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    participant_count: Mapped[int] = mapped_column(Integer, nullable=False)
    prev_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PromotionAuditRecord(Base):
    """Append-only audit trail of every model promotion/rollback attempt.

    Written exclusively by :mod:`detection.model_governance` — one row per
    call to ``promote_candidate``/``rollback_production``, regardless of
    outcome, so a denied or failed attempt is exactly as durable as a
    successful one. Never updated or deleted after insertion.
    """

    __tablename__ = "promotion_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String, nullable=False, index=True)
    action: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )


class PromotionAuditLog:
    """Append-only writer/reader for :class:`PromotionAuditRecord` rows."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        with session_factory() as session:
            Base.metadata.create_all(session.get_bind(), checkfirst=True)

    def record(
        self,
        *,
        actor: str,
        action: str,
        model_name: str,
        success: bool,
        version_id: str | None = None,
        reason: str | None = None,
        detail: str | None = None,
    ) -> PromotionAuditRecord:
        with self._session_factory() as session:
            row = PromotionAuditRecord(
                actor=actor,
                action=action,
                model_name=model_name,
                version_id=version_id,
                success=success,
                reason=reason,
                detail=detail,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def recent(self, limit: int = 100) -> list[PromotionAuditRecord]:
        with self._session_factory() as session:
            return list(
                session.query(PromotionAuditRecord)
                .order_by(PromotionAuditRecord.created_at.desc())
                .limit(limit)
                .all()
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
            raise ValueError(
                f"artifact_sha256 must be a 64-char lowercase hex string, got: {artifact_sha256!r}"
            )
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
            rows = (
                session.query(TransparencyLogRecord)
                .order_by(TransparencyLogRecord.registered_at)
                .all()
            )
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


# ---------------------------------------------------------------------------
# Model watermark verification (#200)
# ---------------------------------------------------------------------------


def verify_watermark(
    model,
    trigger_set: "np.ndarray",
    target_label: int = 1,
    agreement_threshold: float = 0.9,
) -> dict:
    """Measure how strongly *model* has learned the watermark trigger vectors.

    The watermark is a backdoor injected during training: *trigger_set* rows
    should be classified as *target_label* by any model that was trained on
    (or distilled from) a watermarked model.

    Args:
        model: A fitted scikit-learn–compatible classifier with ``predict()``.
        trigger_set: Array of shape (n_triggers, n_features) — the secret
            trigger feature vectors.  **Never log or expose these.**
        target_label: The expected output label for every trigger (default 1).
        agreement_threshold: Fraction of triggers that must agree with
            *target_label* to consider the watermark present (default 0.90).

    Returns:
        {
            "agreement": float,          # fraction of triggers → target_label
            "n_triggers": int,
            "watermark_detected": bool,  # agreement >= agreement_threshold
            "threshold": float,
        }
    """
    preds = model.predict(trigger_set)
    agreement = float(np.mean(preds == target_label))
    return {
        "agreement": agreement,
        "n_triggers": len(trigger_set),
        "watermark_detected": agreement >= agreement_threshold,
        "threshold": agreement_threshold,
    }
