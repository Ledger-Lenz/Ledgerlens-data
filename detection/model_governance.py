"""Single authoritative gate for promoting/rolling back production ML models.

Grand 2 (issue #671) found that LedgerLens had a real cryptographic trust
chain (``detection.persistence.ModelArtifact`` / ``ModelArtifactVerifier``)
that production inference silently bypassed, a separate JSON-manifest
lifecycle state machine (``detection.artifact_lifecycle.ModelArtifactRegistry``)
that could reach ``PROMOTED`` without ever calling the trust chain, and at
least two more code paths (``training/train.py`` and
``scripts/retrain_if_drifted.py``) that could write directly into
``config.MODEL_DIR`` with no regression gate at all.

This module is the fix: **every** path that promotes a trained candidate to
production, or rolls production back to a prior version, must call
:func:`promote_candidate` or :func:`rollback_production`. Both:

1. Authorize the calling actor (``authorize_actor``).
2. Run the regression gate (``evaluate_regression_gate``) before anything is
   written to disk.
3. Sign the candidate and verify its full Ed25519 + transparency-log trust
   chain (the *same* :class:`~detection.persistence.ModelArtifactVerifier`
   code path used at inference time) before anything is written to disk.
4. Only then copy files into ``model_dir`` and update the two lifecycle
   records that must never disagree: ``detection.artifact_lifecycle
   .ModelArtifactRegistry`` (per-artifact-name state machine) and
   ``detection.persistence.ModelVersionRecord`` (per-training-run shadow ->
   production -> rolled_back history).
5. Write an audit-log row (``detection.persistence.PromotionAuditRecord``)
   for the attempt regardless of outcome — a denied or failed promotion is
   exactly as durable as a successful one.

No filesystem write to ``model_dir`` happens unless steps 1-3 all succeed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from config import config
from detection.production_write_guard import (
    UngatedProductionWriteError,  # noqa: F401 - re-exported public entry point
    guard_production_write,  # noqa: F401 - re-exported public entry point
)
from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import Counter

    _promotion_denied_total: Counter | None = Counter(
        "ledgerlens_promotion_denied_total",
        "Number of denied model promotion/rollback attempts (unauthorized, regression, or trust-chain failure)",
        ["action"],
    )
except Exception:  # pragma: no cover
    _promotion_denied_total = None


def _record_denied_metric(action: str) -> None:
    if _promotion_denied_total is not None:
        try:
            _promotion_denied_total.labels(action=action).inc()
        except Exception:  # pragma: no cover
            pass


class PromotionError(Exception):
    """Base class for every governance-path failure."""


class UnauthorizedPromotionError(PromotionError):
    """Raised when the calling actor/credential is not authorized."""


class RegressionGateError(PromotionError):
    """Raised when a candidate's metrics regress beyond tolerance."""


class ArtifactTrustError(PromotionError):
    """Raised when a candidate fails the signature/transparency-log trust chain."""


class NoRollbackTargetError(PromotionError):
    """Raised when there is no prior known-good version to roll back to."""


class QuarantinedModelError(PromotionError):
    """Raised when a candidate is blocked because a backdoor scan flagged it
    (Issue #871) — see `quarantine_candidate` and `_existing_quarantine_reason`."""


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reason: str


# ---------------------------------------------------------------------------
# A. Single write gate for config.MODEL_DIR (Task C)
# ---------------------------------------------------------------------------
#
# `promote_candidate`/`rollback_production` write to production via a raw
# `shutil.copy2` file copy (see `_publish_files_to_production` below) and
# never call `detection.model_training.save_models`/`save_training_artifacts`
# — those two functions are the only other place in the codebase that writes
# `.joblib`/`metrics.json` files, and both call `guard_production_write()`
# below before touching disk. This makes it structurally impossible for a
# training run (however it was invoked) to overwrite a live production
# artifact without going through this module.


@contextmanager
def _promotion_lock(model_dir: str):
    """Advisory, process-wide exclusive lock serializing every promotion/
    rollback that targets the same *model_dir*.

    Two concurrent ``promote_candidate``/``rollback_production`` calls for
    the same production directory must not interleave their archive/publish/
    registry-update steps — one full attempt (success or failure) always
    completes before the next one's critical section begins. POSIX
    ``fcntl.flock`` on a sidecar lock file, matching the existing pattern in
    ``ci_metrics/store.py``; a no-op on Windows (single-writer deployments
    only there, same limitation as that module).
    """
    os.makedirs(model_dir, exist_ok=True)
    lock_path = os.path.join(model_dir, ".promotion.lock")
    fh = open(lock_path, "a+")  # noqa: SIM115 - held for the `with` block's lifetime
    try:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


# guard_production_write / UngatedProductionWriteError live in
# detection.production_write_guard (imported above) and are re-exported here
# as the documented public entry point — see that module's docstring for why.


# ---------------------------------------------------------------------------
# B. Authorization (Task E)
# ---------------------------------------------------------------------------


def _authorized_actors() -> set[str]:
    raw = config.MODEL_PROMOTION_AUTHORIZED_ACTORS or ""
    return {a.strip() for a in raw.split(",") if a.strip()}


def expected_credential(actor: str, secret: str | None = None) -> str:
    """Deterministic HMAC-SHA256 credential for *actor* given the shared
    promotion secret. Used both to issue credentials to authorized actors
    (e.g. the automated retraining pipeline derives its own at call time)
    and to verify one presented by a caller.
    """
    key = (secret if secret is not None else config.MODEL_PROMOTION_SECRET).encode()
    return hmac.new(key, actor.encode(), hashlib.sha256).hexdigest()


def authorize_actor(actor: str, credential: str) -> None:
    """Raise :class:`UnauthorizedPromotionError` unless *actor* is on the
    configured allowlist and *credential* is the correct HMAC for it.

    Fails closed: an empty allowlist or empty secret means nothing is
    authorized, not that authorization is skipped.
    """
    if not actor:
        raise UnauthorizedPromotionError("actor must not be empty")
    if not config.MODEL_PROMOTION_SECRET:
        raise UnauthorizedPromotionError(
            "MODEL_PROMOTION_SECRET is not configured — refusing every promotion/"
            "rollback until an operator provisions a promotion secret."
        )
    allowed = _authorized_actors()
    if actor not in allowed:
        raise UnauthorizedPromotionError(
            f"actor {actor!r} is not in MODEL_PROMOTION_AUTHORIZED_ACTORS"
        )
    expected = expected_credential(actor)
    if not hmac.compare_digest(expected, credential or ""):
        raise UnauthorizedPromotionError(f"invalid credential for actor {actor!r}")


def system_actor_credential() -> tuple[str, str]:
    """Return ``(actor, credential)`` for the automated retraining pipeline.

    Lets ``scripts/retrain_if_drifted.py`` authenticate itself from
    ``MODEL_PROMOTION_SECRET`` alone (no interactive credential), while still
    going through the same ``authorize_actor`` check as a human operator.
    """
    actor = config.MODEL_PROMOTION_SYSTEM_ACTOR
    return actor, expected_credential(actor)


# ---------------------------------------------------------------------------
# C. Regression gate (Task C / F, invariant 6)
# ---------------------------------------------------------------------------


def evaluate_regression_gate(
    old_metrics: dict[str, dict] | None,
    new_metrics: dict[str, dict] | None,
    model_names: list[str] | None = None,
    tolerance: float | None = None,
) -> PromotionDecision:
    """Decide whether *new_metrics* is safe to promote over *old_metrics*.

    Requires ``auc_roc >= old - tolerance`` AND ``f1 >= old - tolerance`` for
    every model named in *model_names* (defaults to
    ``detection.model_training.MODEL_REGISTRY``). No prior metrics (first-ever
    promotion) always approves — there is nothing to regress against.
    """
    if tolerance is None:
        tolerance = config.MODEL_PROMOTION_REGRESSION_TOLERANCE
    if not old_metrics:
        return PromotionDecision(True, "No prior production metrics — nothing to regress against.")
    if not new_metrics:
        return PromotionDecision(False, "Candidate has no metrics to evaluate.")

    if model_names is None:
        from detection.model_training import MODEL_REGISTRY

        model_names = list(MODEL_REGISTRY)

    reasons: list[str] = []
    for name in model_names:
        if name not in old_metrics:
            continue  # nothing to regress against for a brand-new model family
        if name not in new_metrics:
            reasons.append(f"{name}: missing from candidate metrics")
            continue

        old, new = old_metrics[name], new_metrics[name]
        old_auc, new_auc = old.get("auc_roc"), new.get("auc_roc")
        old_f1, new_f1 = old.get("f1"), new.get("f1")

        if old_auc is not None and new_auc is not None and new_auc < old_auc - tolerance:
            reasons.append(
                f"{name}: AUC-ROC {new_auc:.4f} < {old_auc:.4f} - {tolerance} "
                f"(delta {new_auc - old_auc:+.4f})"
            )
        if old_f1 is not None and new_f1 is not None and new_f1 < old_f1 - tolerance:
            reasons.append(
                f"{name}: F1 {new_f1:.4f} < {old_f1:.4f} - {tolerance} (delta {new_f1 - old_f1:+.4f})"
            )

    if reasons:
        return PromotionDecision(False, "; ".join(reasons))
    return PromotionDecision(True, "All model metrics within tolerance.")


# Backward-compatible alias for scripts/retrain_if_drifted.py's original name.
should_promote = evaluate_regression_gate


# ---------------------------------------------------------------------------
# D. Trust-chain verification (Task A / B)
# ---------------------------------------------------------------------------


def _candidate_model_names(candidate_dir: str, model_names: list[str] | None) -> list[str]:
    if model_names is not None:
        return model_names
    from detection.model_training import MODEL_REGISTRY

    return [n for n in MODEL_REGISTRY if os.path.exists(os.path.join(candidate_dir, f"{n}.joblib"))]


def sign_and_verify_candidate(
    candidate_dir: str,
    model_names: list[str],
    *,
    signing_key_path: str | None = None,
    public_key: Any = None,
    transparency_log: Any = None,
) -> dict[str, str]:
    """Sign every model in *candidate_dir*, publish to the transparency log,
    then re-verify the full trust chain through
    :class:`~detection.persistence.ModelArtifactVerifier` — the identical
    check ``RiskScorer`` runs at load time. Returns ``{model_name: sha256}``.

    Raises :class:`ArtifactTrustError` (wrapping the underlying
    ``ModelIntegrityError``) on any failure; nothing is copied to production
    if this raises.
    """
    from detection.persistence import (
        ModelArtifactVerifier,
        ModelIntegrityError,
        get_default_transparency_log,
        load_trusted_public_key,
        sign_and_register_artifact,
    )

    signing_key_path = signing_key_path or config.MODEL_SIGNING_PRIVATE_KEY_PATH
    if not signing_key_path:
        raise ArtifactTrustError(
            "MODEL_SIGNING_PRIVATE_KEY_PATH is not configured — cannot sign the "
            "candidate for promotion. Pass signing_key_path= explicitly or configure "
            "the signing key (HSM/secrets-manager mounted path)."
        )

    transparency_log = transparency_log or get_default_transparency_log()
    public_key = public_key or load_trusted_public_key()

    shas: dict[str, str] = {}
    try:
        for name in model_names:
            sha = sign_and_register_artifact(
                name, candidate_dir, signing_key_path, transparency_log
            )
            verified_sha = ModelArtifactVerifier(transparency_log, candidate_dir).verify(
                name, public_key=public_key, expected_sha256=sha
            )
            shas[name] = verified_sha
    except ModelIntegrityError as exc:
        raise ArtifactTrustError(str(exc)) from exc
    return shas


def make_trust_verifier(
    model_dir: str,
    *,
    public_key: Any = None,
    transparency_log: Any = None,
):
    """Build a :class:`~detection.artifact_lifecycle.ModelArtifactRegistry`
    ``trust_verifier`` callback that re-runs the full Ed25519 +
    transparency-log chain via :class:`~detection.persistence.ModelArtifactVerifier`
    for the artifact directory holding the record being promoted, and
    confirms the on-disk hash still matches the hash recorded at
    ``registry.register()`` time.

    Used both internally by :func:`promote_candidate` (defense in depth: the
    registry's own gate is never trusted to have been pre-satisfied by its
    caller) and by any other code that constructs a ``ModelArtifactRegistry``
    directly, such as ``scripts/manage_artifact_lifecycle.py``.
    """
    from detection.persistence import (
        ModelArtifactVerifier,
        ModelIntegrityError,
        get_default_transparency_log,
        load_trusted_public_key,
    )

    def _verify(record: Any) -> None:
        log = transparency_log or get_default_transparency_log()
        key = public_key or load_trusted_public_key()
        artifact_dir = os.path.dirname(os.path.abspath(record.artifact_path)) or model_dir
        try:
            verified_sha = ModelArtifactVerifier(log, artifact_dir).verify(
                record.name, public_key=key, expected_sha256=record.sha256
            )
        except ModelIntegrityError as exc:
            raise ArtifactTrustError(str(exc)) from exc
        if verified_sha != record.sha256:
            raise ArtifactTrustError(
                f"Registry record sha256 {record.sha256} does not match verified "
                f"artifact sha256 {verified_sha} for {record.name}:{record.version}"
            )

    return _verify


def verify_candidate_compatibility(candidate_dir: str, model_names: list[str]) -> None:
    """Raise :class:`~detection.artifact_compatibility.ArtifactCompatibilityError`
    if any model in *candidate_dir* fails its compatibility contract."""
    from detection.artifact_compatibility import (
        ArtifactCompatibilityError,
        ArtifactCompatibilityGate,
    )

    gate = ArtifactCompatibilityGate(candidate_dir)
    for name in model_names:
        report = gate.check(name)
        if not report.passed:
            raise ArtifactCompatibilityError(
                f"Compatibility check failed for '{name}': " + "; ".join(report.errors)
            )


# ---------------------------------------------------------------------------
# E. Archival helper (shared by promote/rollback)
# ---------------------------------------------------------------------------


def archive_production(model_dir: str, archive_root: str | None = None) -> str:
    """Copy every file currently in *model_dir* into a timestamped archive
    directory. Returns the archive path (empty archive dir if model_dir has
    no files yet, e.g. first-ever promotion)."""
    archive_root = archive_root or os.path.join(model_dir, "archive")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    archive_path = os.path.join(archive_root, timestamp)
    os.makedirs(archive_path, exist_ok=True)
    if os.path.isdir(model_dir):
        for item in os.listdir(model_dir):
            src = os.path.join(model_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(archive_path, item))
    os.chmod(archive_path, 0o750)
    return archive_path


def _publish_files_to_production(candidate_dir: str, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    for item in os.listdir(candidate_dir):
        src = os.path.join(candidate_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(model_dir, item))


# ---------------------------------------------------------------------------
# F. The gated promotion function (Task C, unifies A/B/E)
# ---------------------------------------------------------------------------


def promote_candidate(
    *,
    candidate_dir: str,
    model_dir: str,
    actor: str,
    credential: str,
    old_metrics: dict | None = None,
    new_metrics: dict | None = None,
    reason: str = "",
    model_names: list[str] | None = None,
    signing_key_path: str | None = None,
    public_key: Any = None,
    transparency_log: Any = None,
    audit_log: Any = None,
    session_factory: Any = None,
    registry: Any = None,
    backdoor_report: dict | None = None,
) -> ModelVersionRecordLike:
    """The single gated path from a trained candidate directory to production.

    Order of operations (any failure raises before touching *model_dir*):

    1. ``authorize_actor(actor, credential)``.
    2. Auto-quarantine gate (Issue #871): refuse if *this* call's
       ``backdoor_report`` recommends quarantine, or if `candidate_dir` was
       already quarantined by an earlier call to `quarantine_candidate` —
       see `_existing_quarantine_reason`.
    3. ``evaluate_regression_gate(old_metrics, new_metrics)``.
    4. ``sign_and_verify_candidate`` — sign + Ed25519 + transparency-log
       verification, byte-identical to the ``RiskScorer`` load path.
    5. ``verify_candidate_compatibility``.

    Only once all five pass: archive the current production directory, copy
    the candidate's files into ``model_dir``, register+validate+promote the
    candidate in ``ModelArtifactRegistry`` (superseding any prior promoted
    version), and insert a ``ModelVersionRecord`` row with
    ``status="production"``. Every attempt — successful or not — is written
    to the promotion audit log.
    """
    from detection.artifact_lifecycle import ModelArtifactRegistry
    from detection.persistence import (
        ModelVersionRecord,
        PromotionAuditLog,
        get_default_transparency_log,
        get_engine,
        get_session_factory,
    )

    # Resolve the session factory ONCE and reuse it for every DB write in
    # this call (audit log, ModelVersionRecord) — building a second,
    # independently-configured factory from config.RISK_SCORE_DB_URL here
    # would silently point the audit trail at a different database than the
    # one the caller explicitly passed in.
    sf = session_factory or get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
    audit_log = audit_log or PromotionAuditLog(sf)
    model_names = _candidate_model_names(candidate_dir, model_names)
    if not model_names:
        raise PromotionError(f"No model artifacts found in candidate directory {candidate_dir}")

    def _deny(exc: Exception, detail: str = "") -> None:
        _record_denied_metric("promote")
        audit_log.record(
            actor=actor,
            action="promote",
            model_name=",".join(model_names),
            success=False,
            reason=str(exc),
            detail=detail,
        )

    try:
        authorize_actor(actor, credential)
    except UnauthorizedPromotionError as exc:
        _deny(exc)
        raise

    if backdoor_report is not None and backdoor_report.get("quarantine_recommended"):
        quarantine_candidate(
            candidate_dir=candidate_dir,
            reason=backdoor_report.get("reason", "Backdoor scan flagged this candidate."),
            report=backdoor_report,
            actor=actor,
            session_factory=sf,
        )

    quarantine_reason = _existing_quarantine_reason(candidate_dir, sf)
    if quarantine_reason is not None:
        exc = QuarantinedModelError(
            f"Candidate {candidate_dir} is quarantined pending review: {quarantine_reason}"
        )
        _deny(exc, detail=quarantine_reason)
        raise exc

    try:
        decision = evaluate_regression_gate(old_metrics, new_metrics, model_names=model_names)
        if not decision.approved:
            raise RegressionGateError(decision.reason)
    except RegressionGateError as exc:
        _deny(exc)
        raise

    transparency_log = transparency_log or get_default_transparency_log(session_factory)
    try:
        shas = sign_and_verify_candidate(
            candidate_dir,
            model_names,
            signing_key_path=signing_key_path,
            public_key=public_key,
            transparency_log=transparency_log,
        )
    except ArtifactTrustError as exc:
        _deny(exc)
        raise

    try:
        verify_candidate_compatibility(candidate_dir, model_names)
    except Exception as exc:  # ArtifactCompatibilityError
        _deny(exc)
        raise

    # --- All gates passed: publish to production ------------------------
    # Everything from here on is the critical section: two concurrent
    # promotions targeting the same model_dir must not interleave their
    # archive/publish/registry-update steps.
    with _promotion_lock(model_dir):
        archive_path = archive_production(model_dir)
        _publish_files_to_production(candidate_dir, model_dir)

        registry = registry or ModelArtifactRegistry(
            manifest_path=os.path.join(model_dir, "artifact_manifest.json"),
            trust_verifier=make_trust_verifier(
                model_dir, public_key=public_key, transparency_log=transparency_log
            ),
        )
        for name in model_names:
            artifact_path = os.path.join(model_dir, f"{name}.joblib")
            version = registry.register(
                name, artifact_path, metrics=new_metrics.get(name, {}) if new_metrics else {}
            )
            registry.validate(name, version)
            registry.promote(name, version)

        version_id = str(uuid.uuid4())
        with sf() as session:
            previous = (
                session.query(ModelVersionRecord)
                .filter(ModelVersionRecord.status == "production")
                .order_by(ModelVersionRecord.promoted_at.desc())
                .first()
            )
            if previous is not None:
                previous.status = "deprecated"

            record = ModelVersionRecord(
                version_id=version_id,
                model_artifact_path=model_dir,
                artifact_signature=",".join(f"{k}:{v[:16]}" for k, v in shas.items()),
                status="production",
                promoted_at=datetime.now(UTC),
                promoted_by=actor,
                parent_version_id=previous.version_id if previous is not None else None,
                promotion_blocked_reason=None,
                training_metadata=_json_or_none(
                    {"reason": reason, "archive_path": archive_path, "shas": shas}
                ),
            )
            session.add(record)
            session.commit()
            session.refresh(record)

    audit_log.record(
        actor=actor,
        action="promote",
        model_name=",".join(model_names),
        success=True,
        version_id=version_id,
        reason=reason,
        detail=f"archived_previous={archive_path}",
    )
    logger.info(
        "Promoted candidate %s -> %s as version %s (actor=%s)",
        candidate_dir,
        model_dir,
        version_id,
        actor,
    )
    return record


# ---------------------------------------------------------------------------
# G. Shadow-deployment bookkeeping (Task D) — makes ModelVersionRecord an
# actual queryable shadow -> production/rolled_back history instead of a
# defined-but-never-written table.
# ---------------------------------------------------------------------------


def record_shadow_start(
    version_id: str,
    candidate_dir: str,
    *,
    metrics: dict | None = None,
    session_factory: Any = None,
) -> None:
    """Insert a ``ModelVersionRecord`` row with ``status="shadow"`` when a
    candidate begins its shadow deployment period."""
    from detection.persistence import ModelVersionRecord, get_engine, get_session_factory

    sf = session_factory or get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
    with sf() as session:
        session.add(
            ModelVersionRecord(
                version_id=version_id,
                model_artifact_path=candidate_dir,
                status="shadow",
                shadow_start=datetime.now(UTC),
                training_metadata=_json_or_none({"metrics": metrics or {}}),
            )
        )
        session.commit()


def record_shadow_outcome(
    version_id: str,
    *,
    status: str,
    reason: str | None = None,
    drift_stats: dict | None = None,
    session_factory: Any = None,
) -> None:
    """Update the shadow ``ModelVersionRecord`` row once its shadow period
    ends, with the observed drift/FP statistics and the outcome
    (``"rolled_back"`` or ``"archived"`` — the latter when the candidate went
    on to a *new*, separately-recorded ``production`` row via
    :func:`promote_candidate`).
    """
    from detection.persistence import ModelVersionRecord, get_engine, get_session_factory

    sf = session_factory or get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
    with sf() as session:
        record = (
            session.query(ModelVersionRecord)
            .filter(ModelVersionRecord.version_id == version_id)
            .first()
        )
        if record is None:
            logger.warning("record_shadow_outcome: no ModelVersionRecord for %s", version_id)
            return
        record.status = status
        record.promotion_blocked_reason = reason
        if drift_stats:
            record.shadow_total_requests = drift_stats.get(
                "total_shadow_requests", record.shadow_total_requests
            )
            record.shadow_drift_events = drift_stats.get("drift_events", record.shadow_drift_events)
            record.shadow_drift_rate = drift_stats.get("drift_rate", record.shadow_drift_rate)
        if status == "rolled_back":
            record.rolled_back_at = datetime.now(UTC)
        session.commit()


# ---------------------------------------------------------------------------
# G2. Auto-quarantine (Issue #871) — a candidate flagged by the backdoor
# detector (`detection.adversarial.backdoor_detector.scan_and_quarantine`)
# is recorded here immediately, independent of whether anyone has attempted
# to promote it. `promote_candidate` below re-checks this for the *same*
# candidate_dir on every call, so quarantine cannot be bypassed by simply
# omitting `backdoor_report` on a later promotion attempt.
# ---------------------------------------------------------------------------


def quarantine_candidate(
    *,
    candidate_dir: str,
    reason: str,
    report: dict | None = None,
    actor: str = "system",
    session_factory: Any = None,
) -> ModelVersionRecordLike:
    """Record *candidate_dir* as quarantined pending human review.

    Inserts a new ``ModelVersionRecord`` with ``status="quarantined"`` — it
    never mutates an existing row, so the quarantine event itself is an
    immutable audit entry alongside the shadow/production history for the
    same artifact path. The full structured backdoor report is stored (as
    JSON) in ``training_metadata``; ``quarantine_reason`` holds the short
    human-readable summary.
    """
    from detection.persistence import ModelVersionRecord, get_engine, get_session_factory

    sf = session_factory or get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
    version_id = str(uuid.uuid4())
    with sf() as session:
        record = ModelVersionRecord(
            version_id=version_id,
            model_artifact_path=candidate_dir,
            status="quarantined",
            quarantined_at=datetime.now(UTC),
            quarantine_reason=reason,
            training_metadata=_json_or_none({"backdoor_report": report} if report else {}),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
    logger.warning(
        "Candidate %s quarantined pending review (actor=%s): %s", candidate_dir, actor, reason
    )
    return record


def _existing_quarantine_reason(candidate_dir: str, session_factory: Any) -> str | None:
    """Most recent quarantine reason recorded for *candidate_dir*, or `None`
    if it has never been quarantined."""
    from detection.persistence import ModelVersionRecord

    with session_factory() as session:
        record = (
            session.query(ModelVersionRecord)
            .filter(
                ModelVersionRecord.model_artifact_path == candidate_dir,
                ModelVersionRecord.status == "quarantined",
            )
            .order_by(ModelVersionRecord.quarantined_at.desc())
            .first()
        )
        return record.quarantine_reason if record is not None else None


# ---------------------------------------------------------------------------
# H. Rollback (Task D)
# ---------------------------------------------------------------------------


def rollback_production(
    *,
    model_dir: str,
    actor: str,
    credential: str,
    reason: str,
    target_version: str | None = None,
    signing_key_path: str | None = None,
    public_key: Any = None,
    transparency_log: Any = None,
    audit_log: Any = None,
    session_factory: Any = None,
) -> ModelVersionRecordLike:
    """Roll production back to the previous (or an explicitly named) known-good
    version — a single, tested, auditable operation.

    Restores from the archive directory recorded in the target version's
    ``training_metadata``, re-verifying its trust chain before it goes live
    (an archived directory could in principle have been tampered with at
    rest, so this is not weaker than a fresh promotion).
    """
    from detection.persistence import (
        ModelVersionRecord,
        PromotionAuditLog,
        get_default_transparency_log,
        get_engine,
        get_session_factory,
        load_trusted_public_key,
    )

    # See promote_candidate for why this is resolved once and reused rather
    # than letting `audit_log`'s default silently bind to a different DB
    # than the one `session_factory` points at.
    sf = session_factory or get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
    audit_log = audit_log or PromotionAuditLog(sf)

    def _deny(exc: Exception) -> None:
        _record_denied_metric("rollback")
        audit_log.record(
            actor=actor,
            action="rollback",
            model_name=os.path.basename(os.path.abspath(model_dir)),
            success=False,
            version_id=target_version,
            reason=str(exc),
        )

    try:
        authorize_actor(actor, credential)
    except UnauthorizedPromotionError as exc:
        _deny(exc)
        raise

    with _promotion_lock(model_dir), sf() as session:
        current = (
            session.query(ModelVersionRecord)
            .filter(ModelVersionRecord.status == "production")
            .order_by(ModelVersionRecord.promoted_at.desc())
            .first()
        )

        # A ModelVersionRecord's own `archive_path` is a snapshot of what was
        # live *before* it was promoted — i.e. a backup of its *parent*'s
        # content, not its own. So restoring version X's bytes means finding
        # X's *child* (the record that superseded it) and using that
        # child's archive_path — which is exactly `current` in the common
        # "roll back to the immediate previous version" case.
        if target_version:
            target = (
                session.query(ModelVersionRecord)
                .filter(ModelVersionRecord.version_id == target_version)
                .first()
            )
            archive_source = (
                session.query(ModelVersionRecord)
                .filter(ModelVersionRecord.parent_version_id == target_version)
                .order_by(ModelVersionRecord.promoted_at.asc())
                .first()
                if target is not None
                else None
            )
        elif current is not None and current.parent_version_id:
            target = (
                session.query(ModelVersionRecord)
                .filter(ModelVersionRecord.version_id == current.parent_version_id)
                .first()
            )
            archive_source = current
        else:
            target = None
            archive_source = None

        if target is None or archive_source is None:
            exc = NoRollbackTargetError(
                "No prior known-good ModelVersionRecord found to roll back to."
            )
            _deny(exc)
            raise exc

        archive_path = _archive_path_from_metadata(archive_source.training_metadata)
        if not archive_path or not os.path.isdir(archive_path):
            exc = NoRollbackTargetError(
                f"Rollback target {target.version_id} has no recoverable archive "
                f"directory (looked for {archive_path!r})."
            )
            _deny(exc)
            raise exc

        model_names = _candidate_model_names(archive_path, None)
        transparency_log = transparency_log or get_default_transparency_log(sf)
        public_key = public_key or load_trusted_public_key()
        try:
            from detection.persistence import ModelArtifactVerifier, ModelIntegrityError

            for name in model_names:
                ModelArtifactVerifier(transparency_log, archive_path).verify(
                    name, public_key=public_key
                )
        except ModelIntegrityError as exc:
            wrapped = ArtifactTrustError(
                f"Rollback target {target.version_id} failed trust-chain verification: {exc}"
            )
            _deny(wrapped)
            raise wrapped from exc

        # Trust chain verified — safe to make it live. Snapshot current
        # production first so a bad rollback is itself recoverable.
        safety_archive = archive_production(model_dir)
        _publish_files_to_production(archive_path, model_dir)

        if current is not None:
            current.status = "rolled_back"
            current.rolled_back_at = datetime.now(UTC)
            current.rolled_back_by = actor
            current.promotion_blocked_reason = reason

        new_version_id = str(uuid.uuid4())
        restored = ModelVersionRecord(
            version_id=new_version_id,
            model_artifact_path=model_dir,
            artifact_signature=target.artifact_signature,
            status="production",
            promoted_at=datetime.now(UTC),
            promoted_by=actor,
            parent_version_id=current.version_id if current is not None else None,
            training_metadata=_json_or_none(
                {
                    "rollback_of": current.version_id if current is not None else None,
                    "restored_from": target.version_id,
                    "reason": reason,
                    "pre_rollback_archive": safety_archive,
                    "archive_path": archive_path,
                }
            ),
        )
        session.add(restored)
        session.commit()
        session.refresh(restored)
        # Capture plain values before the session closes below — ORM
        # attribute access on `target`/`current` outside this block would
        # otherwise raise DetachedInstanceError (both were expired by the
        # commit above and never explicitly refreshed).
        target_version_id = target.version_id

    audit_log.record(
        actor=actor,
        action="rollback",
        model_name=",".join(model_names),
        success=True,
        version_id=new_version_id,
        reason=reason,
        detail=f"restored_from={target_version_id}",
    )
    logger.warning(
        "Rolled back production model in %s to version %s (new version_id=%s, actor=%s, reason=%s)",
        model_dir,
        target_version_id,
        new_version_id,
        actor,
        reason,
    )
    return restored


def _archive_path_from_metadata(training_metadata: str | None) -> str | None:
    if not training_metadata:
        return None
    try:
        data = json.loads(training_metadata)
    except (ValueError, TypeError):
        return None
    return data.get("archive_path")


def _json_or_none(data: dict) -> str:
    return json.dumps(data, default=str)


# Typing-only alias (ModelVersionRecord is a SQLAlchemy model; imported lazily
# above to avoid a hard import-time dependency from this module onto the DB
# layer for callers that only need the pure functions like
# evaluate_regression_gate/authorize_actor).
ModelVersionRecordLike = Any
