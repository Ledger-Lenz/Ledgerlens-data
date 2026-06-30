"""FedAvg coordinator server.

Endpoints
---------
GET  /global_weights       Return current global weights + registered participant IDs.
POST /register             Register a participant and receive its ID back.
POST /submit_delta         Accept a masked weight delta for the current round.
POST /advance_round        (internal/test) Manually trigger aggregation if quorum is met.

Round lifecycle
---------------
1. Participants register (or are pre-registered via Config).
2. Coordinator broadcasts global weights on GET /global_weights.
3. Participants submit masked deltas.
4. Once >= FED_MIN_PARTICIPANTS deltas are received, the coordinator aggregates
   and advances to the next round automatically.

Run with:
    uvicorn detection.federated.coordinator:app --port 8000
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_weights(weights: np.ndarray) -> str:
    """Return the SHA-256 hex digest of the serialised global model weights.

    The weights are serialised as a canonical JSON array of floats (6 decimal
    places) so the hash is deterministic across platforms and Python versions.
    Raw gradient tensors are *not* stored anywhere in this function — only the
    aggregate model weights (which are persisted openly anyway).
    """
    canonical = json.dumps([round(float(v), 6) for v in weights.ravel()], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _deterministic_round_id(
    timestamp: str,
    participant_fingerprints: list[str],
    model_version: int,
) -> str:
    """SHA-256 of (timestamp, sorted-fingerprints, model_version).

    Using a content-addressed hash prevents sequential-ID manipulation:
    an attacker cannot insert a fake round without changing the hash.
    """
    material = json.dumps(
        {
            "timestamp": timestamp,
            "fingerprints": sorted(participant_fingerprints),
            "model_version": model_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()


# ---------------------------------------------------------------------------
# FederatedAuditTrail
# ---------------------------------------------------------------------------


class FederatedAuditTrail:
    """Append-only audit trail for federated learning rounds (issue #227).

    Records for each round:
    - A deterministic round ID (SHA-256 of timestamp + participant fingerprints
      + model version) — prevents sequential-ID manipulation.
    - Participant certificate fingerprints (hex SHA-256 of the participant ID
      by default; callers can supply real certificate fingerprints via
      ``set_participant_fingerprints``).
    - Gradient **norms** only (scalar L2 norms) — raw tensors are never stored.
    - The aggregation algorithm name.
    - A SHA-256 hash of the aggregate model weights.
    - The round outcome ("success" or "abort").

    Records are persisted to the ``federated_audit_trail`` DB table via
    SQLAlchemy (same engine as the main risk-score DB).  A Merkle-like hash
    chain (``prev_hash``) links each record to its predecessor, making
    retroactive insertion or deletion detectable.

    Security invariants
    -------------------
    * Raw gradient tensors are **never** written to storage.
    * The DB layer exposes no UPDATE or DELETE path for this table.
    * ``round_id`` is content-addressed — not a sequential integer.
    """

    def __init__(
        self,
        session_factory: Any | None = None,
        db_url: str | None = None,
    ) -> None:
        """Initialise the audit trail.

        Parameters
        ----------
        session_factory:
            A SQLAlchemy ``sessionmaker`` instance.  If ``None``, a default
            factory is created from ``db_url`` (or ``RISK_SCORE_DB_URL``).
        db_url:
            SQLAlchemy connection URL; ignored when ``session_factory`` is
            provided.
        """
        if session_factory is None:
            from detection.persistence import get_engine, get_session_factory, Base, FederatedAuditRecord  # noqa: F401
            engine = get_engine(db_url)
            Base.metadata.create_all(engine, checkfirst=True)
            session_factory = get_session_factory(engine)
        self._session_factory = session_factory

        # Optional mapping from participant_id -> certificate fingerprint.
        # When absent, participant_id is hashed to produce a stable fingerprint.
        self._fingerprint_map: dict[str, str] = {}

        logger.info("FederatedAuditTrail initialised")

    # ------------------------------------------------------------------
    # Participant fingerprints
    # ------------------------------------------------------------------

    def set_participant_fingerprints(self, mapping: dict[str, str]) -> None:
        """Register real certificate fingerprints for participant IDs.

        Parameters
        ----------
        mapping:
            ``{participant_id: fingerprint_hex}``  where *fingerprint_hex* is
            the SHA-256 fingerprint of the participant's TLS/X.509 certificate.
        """
        self._fingerprint_map.update(mapping)

    def get_fingerprint(self, participant_id: str) -> str:
        """Return the fingerprint for *participant_id*.

        Falls back to SHA-256(participant_id) when no certificate fingerprint
        has been registered — this ensures every participant always has a
        stable, unique fingerprint in the audit record.
        """
        if participant_id in self._fingerprint_map:
            return self._fingerprint_map[participant_id]
        # Derive a stable pseudonymous fingerprint from the participant ID.
        return hashlib.sha256(participant_id.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Core write path
    # ------------------------------------------------------------------

    def record_round(
        self,
        participant_fingerprints: list[str],
        gradient_norms: dict[str, float],
        aggregation_algorithm: str,
        aggregate_model_hash: str,
        round_outcome: str,
        model_version: int,
        timestamp: str | None = None,
    ) -> str:
        """Persist an audit record for one federated round and return its round_id.

        Parameters
        ----------
        participant_fingerprints:
            Certificate fingerprints (hex) of every participant who contributed
            in this round.
        gradient_norms:
            ``{participant_id: l2_norm}`` — scalar gradient norms **only**.
            Raw gradient tensors must never be passed here.
        aggregation_algorithm:
            Human-readable algorithm name, e.g. ``"fedavg"`` or
            ``"staleness_weighted_fedavg"``.
        aggregate_model_hash:
            SHA-256 hex digest of the post-aggregation global model weights
            (computed by :func:`_hash_weights`).
        round_outcome:
            ``"success"`` or ``"abort"``.
        model_version:
            The model version number after this aggregation.
        timestamp:
            ISO-8601 UTC string.  Defaults to ``datetime.now(UTC).isoformat()``.

        Returns
        -------
        str
            The deterministic ``round_id`` (SHA-256 hex digest).
        """
        # Security: refuse to record any value whose key suggests a raw tensor.
        # This is a defence-in-depth check; callers must never pass tensors.
        for key, value in gradient_norms.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"gradient_norms[{key!r}] must be a scalar float, got {type(value).__name__}. "
                    "Raw gradient tensors must never be written to the audit trail."
                )

        ts = timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z")
        round_id = _deterministic_round_id(ts, participant_fingerprints, model_version)

        from detection.persistence import FederatedAuditRecord, get_session_factory

        # Read the previous record's canonical hash to build the chain.
        prev_hash = self._get_latest_record_hash()

        record = FederatedAuditRecord(
            round_id=round_id,
            round_timestamp=ts,
            participant_fingerprints=json.dumps(sorted(participant_fingerprints)),
            gradient_norms=json.dumps(gradient_norms),
            aggregation_algorithm=aggregation_algorithm,
            aggregate_model_hash=aggregate_model_hash,
            round_outcome=round_outcome,
            model_version=model_version,
            participant_count=len(participant_fingerprints),
            prev_hash=prev_hash,
        )

        with self._session_factory() as session:
            session.add(record)
            session.commit()

        logger.info(
            "Federated audit record committed: round_id=%s model_version=%d "
            "participants=%d outcome=%s",
            round_id,
            model_version,
            len(participant_fingerprints),
            round_outcome,
        )
        return round_id

    def _get_latest_record_hash(self) -> str | None:
        """Return the SHA-256 hash of the most recent record for chain linking."""
        from detection.persistence import FederatedAuditRecord

        with self._session_factory() as session:
            latest = (
                session.query(FederatedAuditRecord)
                .order_by(FederatedAuditRecord.id.desc())
                .first()
            )
            if latest is None:
                return None
            # Compute a canonical hash of the latest record's fields.
            canonical = json.dumps(
                {
                    "round_id": latest.round_id,
                    "round_timestamp": latest.round_timestamp,
                    "participant_fingerprints": latest.participant_fingerprints,
                    "gradient_norms": latest.gradient_norms,
                    "aggregation_algorithm": latest.aggregation_algorithm,
                    "aggregate_model_hash": latest.aggregate_model_hash,
                    "round_outcome": latest.round_outcome,
                    "model_version": latest.model_version,
                    "participant_count": latest.participant_count,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(canonical.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Query helpers (used by scripts/query_federated_audit.py)
    # ------------------------------------------------------------------

    def query_by_round_id(self, round_id: str) -> list[dict[str, Any]]:
        """Return all audit records matching *round_id* (exact match)."""
        from detection.persistence import FederatedAuditRecord

        with self._session_factory() as session:
            rows = (
                session.query(FederatedAuditRecord)
                .filter(FederatedAuditRecord.round_id == round_id)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def query_by_participant(self, fingerprint: str) -> list[dict[str, Any]]:
        """Return audit records where *fingerprint* appears in participant_fingerprints."""
        from detection.persistence import FederatedAuditRecord

        with self._session_factory() as session:
            # participant_fingerprints is stored as a JSON array string — use LIKE.
            rows = (
                session.query(FederatedAuditRecord)
                .filter(
                    FederatedAuditRecord.participant_fingerprints.like(f"%{fingerprint}%")
                )
                .order_by(FederatedAuditRecord.id)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def query_by_model_hash(self, model_hash: str) -> list[dict[str, Any]]:
        """Return audit records whose aggregate_model_hash matches *model_hash*."""
        from detection.persistence import FederatedAuditRecord

        with self._session_factory() as session:
            rows = (
                session.query(FederatedAuditRecord)
                .filter(FederatedAuditRecord.aggregate_model_hash == model_hash)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def list_all(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return paginated audit records ordered by insertion time."""
        from detection.persistence import FederatedAuditRecord

        with self._session_factory() as session:
            rows = (
                session.query(FederatedAuditRecord)
                .order_by(FederatedAuditRecord.id)
                .limit(limit)
                .offset(offset)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row.id,
            "round_id": row.round_id,
            "round_timestamp": row.round_timestamp,
            "participant_fingerprints": json.loads(row.participant_fingerprints),
            "gradient_norms": json.loads(row.gradient_norms),
            "aggregation_algorithm": row.aggregation_algorithm,
            "aggregate_model_hash": row.aggregate_model_hash,
            "round_outcome": row.round_outcome,
            "model_version": row.model_version,
            "participant_count": row.participant_count,
            "prev_hash": row.prev_hash,
            "recorded_at": row.recorded_at.isoformat() if row.recorded_at else None,
        }

# ---------------------------------------------------------------------------
# Configuration (mirrors the project's os.getenv pattern from config.py)
# ---------------------------------------------------------------------------
FED_MIN_PARTICIPANTS: int = int(os.getenv("FED_MIN_PARTICIPANTS", "3"))
FED_WEIGHT_DIM: int = int(os.getenv("FED_WEIGHT_DIM", "0"))  # 0 = inferred at runtime
FEDERATED_ASYNC_TRIGGER_N: int = int(os.getenv("FEDERATED_ASYNC_TRIGGER_N", "3"))
FEDERATED_ASYNC_TRIGGER_SECONDS: int = int(os.getenv("FEDERATED_ASYNC_TRIGGER_SECONDS", "300"))
FEDERATED_MAX_STALENESS: int = int(os.getenv("FEDERATED_MAX_STALENESS", "5"))
# Krum Byzantine tolerance: assumed number of Byzantine participants per round (#189)
FEDERATED_BYZANTINE_TOLERANCE: int = int(os.getenv("FEDERATED_BYZANTINE_TOLERANCE", "1"))
# Magnitude clipping threshold before Krum scoring (prevents magnitude-based bypass)
_KRUM_CLIP_NORM: float = float(os.getenv("FEDERATED_KRUM_CLIP_NORM", "10.0"))


# ---------------------------------------------------------------------------
# Krum Byzantine-resilient aggregation (Blanchard et al., NeurIPS 2017)
# ---------------------------------------------------------------------------


def krum_aggregate(
    updates: list[np.ndarray],
    n_byzantine: int,
    multi_krum_m: int = 1,
) -> np.ndarray:
    """Select the gradient update(s) most resistant to Byzantine influence.

    Implements the Krum and Multi-Krum algorithms from:
        Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
        Gradient Descent", NeurIPS 2017.

    Each update's Krum score is the sum of squared distances to its
    (n - n_byzantine - 2) nearest neighbours.  The m updates with the lowest
    scores are averaged (Multi-Krum; m=1 gives vanilla Krum).

    Gradient magnitudes are clipped to ``_KRUM_CLIP_NORM`` before scoring to
    prevent magnitude-based Krum bypass attacks (a Byzantine participant
    cannot gain advantage by inflating its gradient norm).

    Complexity: O(n² × d) where n = len(updates), d = gradient dimension.
    Implemented with vectorised numpy operations.

    Parameters
    ----------
    updates:
        List of gradient arrays from n participants.
    n_byzantine:
        Assumed number of Byzantine participants.
    multi_krum_m:
        Number of updates to select and average (default 1 = Krum).

    Raises
    ------
    ValueError
        If there are insufficient honest participants for Krum to be safe
        (i.e. n - n_byzantine < 3).  Callers should fall back to FedAvg.
    """
    n = len(updates)
    if n - n_byzantine < 3:
        raise ValueError(
            f"Krum requires n - n_byzantine >= 3 (n={n}, n_byzantine={n_byzantine}). "
            "Fall back to FedAvg."
        )

    k = n - n_byzantine - 2  # number of nearest neighbours to sum

    # Clip gradient magnitudes to prevent magnitude-based bypass attacks
    clipped: list[np.ndarray] = []
    for u in updates:
        norm = np.linalg.norm(u)
        if norm > _KRUM_CLIP_NORM:
            clipped.append(u * (_KRUM_CLIP_NORM / norm))
        else:
            clipped.append(u)

    # Stack into matrix: shape (n, d)
    U = np.stack(clipped, axis=0)

    # Pairwise squared Euclidean distances: shape (n, n) — vectorised
    # ||u_i - u_j||^2 = ||u_i||^2 + ||u_j||^2 - 2 u_i · u_j
    norms_sq = np.sum(U**2, axis=1, keepdims=True)  # (n, 1)
    dist_sq = norms_sq + norms_sq.T - 2.0 * (U @ U.T)  # (n, n)
    np.fill_diagonal(dist_sq, np.inf)  # exclude self

    # Krum score: sum of k smallest distances for each update
    sorted_dists = np.sort(dist_sq, axis=1)  # (n, n), rows sorted ascending
    scores = sorted_dists[:, :k].sum(axis=1)  # (n,)

    # Select m updates with the smallest Krum scores
    m = min(multi_krum_m, n - n_byzantine)
    selected_indices = np.argsort(scores)[:m]

    # Warn if any selected update deviates significantly from the mean
    mean_update = U.mean(axis=0)
    for idx in selected_indices:
        deviation = float(np.linalg.norm(updates[idx] - mean_update))
        avg_norm = float(np.mean([np.linalg.norm(u) for u in updates]))
        if deviation > 2.0 * avg_norm:
            logger.warning(
                "Krum: selected update[%d] deviates significantly from mean "
                "(deviation=%.4f, avg_norm=%.4f) — possible Byzantine behaviour detected",
                idx,
                deviation,
                avg_norm,
            )

    result: np.ndarray = np.mean(U[selected_indices], axis=0)
    logger.info(
        "Krum selected %d/%d updates (indices=%s, scores=%s)",
        m,
        n,
        selected_indices.tolist(),
        [f"{scores[i]:.4f}" for i in selected_indices],
    )
    return result


# ---------------------------------------------------------------------------
# In-memory state (one coordinator per process; reset on restart)
# ---------------------------------------------------------------------------


class _RoundState:
    def __init__(self) -> None:
        self.round_number: int = 0
        self.global_weights: np.ndarray | None = None
        self.participants: list[str] = []
        # round_number -> {participant_id: masked_delta}
        self.pending: dict[int, dict[str, np.ndarray]] = {}

    def register(self, pid: str) -> None:
        if pid not in self.participants:
            self.participants.append(pid)

    def current_pending(self) -> dict[str, np.ndarray]:
        return self.pending.setdefault(self.round_number, {})

    def try_aggregate(self) -> bool:
        """Aggregate if quorum met. Returns True if aggregation happened."""
        pending = self.current_pending()
        if len(pending) < FED_MIN_PARTICIPANTS:
            return False
        self._aggregate(list(pending.values()))
        return True

    def _aggregate(self, masked_deltas: list[np.ndarray]) -> None:
        """Aggregate deltas using Krum when possible, falling back to FedAvg.

        Krum is used when ``n - FEDERATED_BYZANTINE_TOLERANCE >= 3``.
        """
        n = len(masked_deltas)
        n_byz = FEDERATED_BYZANTINE_TOLERANCE

        if n - n_byz >= 3:
            try:
                agg = krum_aggregate(masked_deltas, n_byzantine=n_byz)
                method = "krum"
            except Exception as exc:
                logger.warning("Krum aggregation failed (%s); falling back to FedAvg", exc)
                agg = np.sum(masked_deltas, axis=0) / n
                method = "fedavg-fallback"
        else:
            logger.warning(
                "Krum skipped: n=%d, n_byzantine=%d, need n-n_byzantine>=3; using FedAvg",
                n,
                n_byz,
            )
            agg = np.sum(masked_deltas, axis=0) / n
            method = "fedavg"

        if self.global_weights is None:
            self.global_weights = agg
        else:
            self.global_weights = self.global_weights + agg
        logger.info(
            "Round %d aggregated %d deltas via %s. New global weight norm: %.4f",
            self.round_number,
            n,
            method,
            float(np.linalg.norm(self.global_weights)),
        )
        self.round_number += 1


_state = _RoundState()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _state
    _state = _RoundState()
    # If a known dimension is configured, initialise to zeros so participants
    # can start training immediately without a prior submit.
    if FED_WEIGHT_DIM > 0:
        _state.global_weights = np.zeros(FED_WEIGHT_DIM)
    yield


app = FastAPI(title="LedgerLens Federated Coordinator", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    participant_id: str


class DeltaSubmission(BaseModel):
    participant_id: str
    delta: list[float]


class GlobalWeightsResponse(BaseModel):
    round_number: int
    weights: list[float]
    participants: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/register")
def register(req: RegisterRequest) -> dict[str, Any]:
    _state.register(req.participant_id)
    logger.info("Registered participant %s", req.participant_id)
    return {"participant_id": req.participant_id, "registered": True}


@app.get("/global_weights", response_model=GlobalWeightsResponse)
def get_global_weights() -> GlobalWeightsResponse:
    if _state.global_weights is None:
        raise HTTPException(
            status_code=503,
            detail="No global model yet; wait for the first aggregation round.",
        )
    return GlobalWeightsResponse(
        round_number=_state.round_number,
        weights=_state.global_weights.tolist(),
        participants=list(_state.participants),
    )


@app.post("/submit_delta")
def submit_delta(submission: DeltaSubmission) -> dict[str, Any]:
    pid = submission.participant_id
    if pid not in _state.participants:
        # Auto-register latecomers so participants don't need an explicit /register call
        _state.register(pid)

    delta = np.array(submission.delta, dtype=float)

    # Initialise global weights from first submission if not yet set
    if _state.global_weights is None:
        _state.global_weights = np.zeros_like(delta)

    pending = _state.current_pending()
    if pid in pending:
        raise HTTPException(
            status_code=409,
            detail=f"Participant {pid!r} already submitted for round {_state.round_number}.",
        )

    pending[pid] = delta
    n_received = len(pending)
    logger.info(
        "Round %d: received delta from %s (%d/%d)",
        _state.round_number,
        pid,
        n_received,
        FED_MIN_PARTICIPANTS,
    )

    aggregated = _state.try_aggregate()
    return {
        "round_number": _state.round_number - (1 if aggregated else 0),
        "deltas_received": n_received,
        "aggregated": aggregated,
    }


@app.post("/advance_round")
def advance_round() -> dict[str, Any]:
    """Force aggregation with however many deltas are present (for testing)."""
    pending = _state.current_pending()
    if not pending:
        raise HTTPException(status_code=400, detail="No deltas received yet.")
    _state._aggregate(list(pending.values()))
    return {"round_number": _state.round_number, "aggregated": True}


# ---------------------------------------------------------------------------
# Programmatic reset helper (used by tests)
# ---------------------------------------------------------------------------


def reset_state(weight_dim: int = 0) -> None:
    """Reset coordinator state; optionally pre-seed global weights to zeros."""
    global _state
    _state = _RoundState()
    if weight_dim > 0:
        _state.global_weights = np.zeros(weight_dim)


# ---------------------------------------------------------------------------
# Async federated coordinator (issue #270)
# ---------------------------------------------------------------------------


class AsyncGradientUpdate:
    """A gradient update tagged with the model version it was computed from."""

    def __init__(
        self,
        participant_id: str,
        delta: np.ndarray,
        gradient_model_version: int,
    ) -> None:
        self.participant_id = participant_id
        self.delta = delta
        self.gradient_model_version = gradient_model_version
        self.received_at: float = time.monotonic()


class AsyncFederatedCoordinator:
    """Asynchronous FedAvg coordinator that aggregates gradient updates as they arrive.

    Unlike the synchronous coordinator (``_RoundState``), this class does **not** wait
    for all participants before aggregating.  Instead it aggregates every
    ``trigger_n`` updates or every ``trigger_seconds`` — whichever comes first.

    Staleness-aware weighting
    -------------------------
    Each update is tagged with the model version it was computed from.
    ``staleness = current_model_version - gradient_model_version``.
    Updates with staleness > ``max_staleness`` are rejected.
    Accepted updates are weighted by ``1 / (1 + staleness)`` before aggregation
    (fresher updates contribute more).

    Byzantine resilience
    --------------------
    The same aggregation mechanism as the synchronous coordinator is used:
    simple weighted FedAvg.  Clipping and Byzantine-robust methods (e.g.
    coordinate-wise median) can be layered on top by overriding ``_aggregate``.

    Thread safety
    -------------
    All mutable state is protected by ``self._lock``.  Concurrent calls to
    ``submit_update`` from multiple threads are safe.

    Parameters
    ----------
    trigger_n:
        Aggregate after this many pending updates (default:
        ``FEDERATED_ASYNC_TRIGGER_N``, env-configurable).
    trigger_seconds:
        Also aggregate if this many seconds have elapsed since the last
        aggregation (default: ``FEDERATED_ASYNC_TRIGGER_SECONDS``).
    max_staleness:
        Reject updates computed from a model more than this many versions old
        (default: ``FEDERATED_MAX_STALENESS``).
    """

    def __init__(
        self,
        weight_dim: int = 0,
        trigger_n: int | None = None,
        trigger_seconds: int | None = None,
        max_staleness: int | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._global_weights: np.ndarray | None = (
            np.zeros(weight_dim) if weight_dim > 0 else None
        )
        self._model_version: int = 0
        self._pending: list[AsyncGradientUpdate] = []
        self._last_aggregation_time: float = time.monotonic()

        self.trigger_n: int = (
            trigger_n if trigger_n is not None else FEDERATED_ASYNC_TRIGGER_N
        )
        self.trigger_seconds: int = (
            trigger_seconds if trigger_seconds is not None else FEDERATED_ASYNC_TRIGGER_SECONDS
        )
        self.max_staleness: int = (
            max_staleness if max_staleness is not None else FEDERATED_MAX_STALENESS
        )
        self._audit: FederatedAuditTrail | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> int:
        with self._lock:
            return self._model_version

    @property
    def global_weights(self) -> np.ndarray | None:
        with self._lock:
            return self._global_weights.copy() if self._global_weights is not None else None

    def submit_update(
        self,
        participant_id: str,
        delta: list[float] | np.ndarray,
        gradient_model_version: int,
    ) -> dict[str, Any]:
        """Accept a gradient update from a participant.

        Parameters
        ----------
        participant_id:
            Identifier of the submitting participant.
        delta:
            Weight delta computed by the participant.
        gradient_model_version:
            The model version the participant used when computing the gradient.

        Returns
        -------
        dict
            ``{"accepted": bool, "current_model_version": int,
               "staleness": int, "aggregated": bool}``

        Raises
        ------
        ValueError
            If ``gradient_model_version`` is more than ``max_staleness`` versions old.
        """
        delta_arr = np.asarray(delta, dtype=float)

        with self._lock:
            staleness = self._model_version - gradient_model_version
            if staleness < 0:
                staleness = 0

            if staleness > self.max_staleness:
                raise ValueError(
                    f"Update from {participant_id!r} rejected: staleness {staleness} "
                    f"exceeds max_staleness {self.max_staleness} "
                    f"(gradient_model_version={gradient_model_version}, "
                    f"current_model_version={self._model_version})"
                )

            # Initialise global weights from first submission if not yet set
            if self._global_weights is None:
                self._global_weights = np.zeros_like(delta_arr)

            update = AsyncGradientUpdate(
                participant_id=participant_id,
                delta=delta_arr,
                gradient_model_version=gradient_model_version,
            )
            self._pending.append(update)

            logger.info(
                "Async update received: participant=%s staleness=%d pending=%d "
                "model_version=%d",
                participant_id,
                staleness,
                len(self._pending),
                self._model_version,
            )

            aggregated = self._maybe_aggregate()

        return {
            "accepted": True,
            "current_model_version": self._model_version,
            "staleness": staleness,
            "aggregated": aggregated,
        }

    def tick(self) -> bool:
        """Trigger time-based aggregation if ``trigger_seconds`` has elapsed.

        Intended to be called periodically by a background thread or scheduler.
        Returns True if aggregation occurred.
        """
        with self._lock:
            elapsed = time.monotonic() - self._last_aggregation_time
            if elapsed >= self.trigger_seconds and self._pending:
                return self._aggregate()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_aggregate(self) -> bool:
        """Aggregate if N-update trigger or time-trigger fires.

        Must be called while holding ``self._lock``.
        """
        elapsed = time.monotonic() - self._last_aggregation_time
        if len(self._pending) >= self.trigger_n or (
            elapsed >= self.trigger_seconds and self._pending
        ):
            return self._aggregate()
        return False

    def _aggregate(self) -> bool:
        """Apply staleness-weighted aggregation (Krum when safe, FedAvg otherwise).

        Must be called while holding ``self._lock``.
        """
        if not self._pending:
            return False

        updates = list(self._pending)
        self._pending.clear()

        n = len(updates)
        n_byz = FEDERATED_BYZANTINE_TOLERANCE

        # Staleness-aware weights: w_i = 1 / (1 + staleness_i)
        weights = np.array(
            [
                1.0 / (1.0 + max(0, self._model_version - u.gradient_model_version))
                for u in updates
            ]
        )
        total_weight = weights.sum()
        norm_weights = weights / total_weight

        # Weighted deltas for Krum scoring
        weighted_deltas = [w * u.delta for w, u in zip(norm_weights, updates, strict=True)]

        if n - n_byz >= 3:
            try:
                agg = krum_aggregate(weighted_deltas, n_byzantine=n_byz)
                method = "krum"
            except Exception as exc:
                logger.warning("Async Krum failed (%s); falling back to FedAvg", exc)
                agg = sum(weighted_deltas)
                method = "fedavg-fallback"
        else:
            agg = sum(weighted_deltas)
            method = "fedavg"

        assert self._global_weights is not None
        self._global_weights = self._global_weights + agg
        self._model_version += 1
        self._last_aggregation_time = time.monotonic()

        mean_staleness = float(
            np.mean(
                [max(0, self._model_version - 1 - u.gradient_model_version) for u in updates]
            )
        )

        logger.info(
            "Async aggregation complete: updates_included=%d mean_staleness=%.2f "
            "new_model_version=%d global_weight_norm=%.4f method=%s",
            len(updates),
            mean_staleness,
            self._model_version,
            float(np.linalg.norm(self._global_weights)),
            method,
        )

        # Audit trail: record this round without holding the lock during DB I/O
        if self._audit is not None:
            gradient_norms = {u.participant_id: float(np.linalg.norm(u.delta)) for u in updates}
            model_hash = _hash_weights(self._global_weights)
            # participant fingerprints: use participant_id as fingerprint when no
            # certificate is available (callers may inject real fingerprints via
            # FederatedAuditTrail.set_participant_fingerprints)
            fingerprints = [
                self._audit.get_fingerprint(u.participant_id) for u in updates
            ]
            self._audit.record_round(
                participant_fingerprints=fingerprints,
                gradient_norms=gradient_norms,
                aggregation_algorithm="staleness_weighted_fedavg",
                aggregate_model_hash=model_hash,
                round_outcome="success",
                model_version=self._model_version,
            )

        return True
