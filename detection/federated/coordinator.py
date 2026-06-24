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
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
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
