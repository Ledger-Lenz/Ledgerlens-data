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

import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (mirrors the project's os.getenv pattern from config.py)
# ---------------------------------------------------------------------------
FED_MIN_PARTICIPANTS: int = int(os.getenv("FED_MIN_PARTICIPANTS", "3"))
FED_WEIGHT_DIM: int = int(os.getenv("FED_WEIGHT_DIM", "0"))  # 0 = inferred at runtime
FEDERATED_ASYNC_TRIGGER_N: int = int(os.getenv("FEDERATED_ASYNC_TRIGGER_N", "3"))
FEDERATED_ASYNC_TRIGGER_SECONDS: int = int(os.getenv("FEDERATED_ASYNC_TRIGGER_SECONDS", "300"))
FEDERATED_MAX_STALENESS: int = int(os.getenv("FEDERATED_MAX_STALENESS", "5"))


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
        """FedAvg: global += (1/N) * sum(masked_deltas)."""
        n = len(masked_deltas)
        agg = np.sum(masked_deltas, axis=0) / n
        if self.global_weights is None:
            self.global_weights = agg
        else:
            self.global_weights = self.global_weights + agg
        logger.info(
            "Round %d aggregated %d deltas. New global weight norm: %.4f",
            self.round_number,
            n,
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
        """Apply staleness-weighted FedAvg to all pending updates.

        Must be called while holding ``self._lock``.
        """
        if not self._pending:
            return False

        updates = list(self._pending)
        self._pending.clear()

        # Staleness-aware weights: w_i = 1 / (1 + staleness_i)
        weights = np.array(
            [
                1.0 / (1.0 + max(0, self._model_version - u.gradient_model_version))
                for u in updates
            ]
        )
        total_weight = weights.sum()
        norm_weights = weights / total_weight

        agg = sum(w * u.delta for w, u in zip(norm_weights, updates, strict=True))
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
            "new_model_version=%d global_weight_norm=%.4f",
            len(updates),
            mean_staleness,
            self._model_version,
            float(np.linalg.norm(self._global_weights)),
        )
        return True
