"""FastAPI REST API exposing LedgerLens wallet risk scores.

Endpoints:
    GET /v1/wallets/{address}/scores   — paginated risk score history
    GET /v1/wallets/{address}/latest   — latest score + top-3 features
    GET /v1/health                     — liveness / readiness check
"""

import re
from typing import Optional

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select

from config import config
from detection.persistence import RiskScoreRecord, get_session_factory
from detection.risk_score_store import RiskScoreStore
from detection.shap_explainer import ShapExplainer

# ---------------------------------------------------------------------------
# Stellar address validation
# ---------------------------------------------------------------------------
_STELLAR_ACCOUNT_RE = re.compile(r"^G[A-Z2-7]{55}$")


def _validate_stellar_address(address: str) -> str:
    if not _STELLAR_ACCOUNT_RE.match(address):
        raise HTTPException(status_code=400, detail="Invalid Stellar account address")
    return address


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check_api_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    for hashed in config.API_KEYS:
        if bcrypt.checkpw(api_key.encode(), hashed.encode()):
            return api_key
    raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="LedgerLens Risk Score API",
    version="1.0.0",
    description="Wallet risk scores for Stellar DEX wash-trade detection.",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class RiskScoreResponse(BaseModel):
    score_id: int
    wallet: str
    asset_pair: str
    score: int
    benford_flag: bool
    ml_flag: bool
    confidence: int
    propagated_risk: Optional[float] = None
    ring_id: Optional[str] = None
    updated_at: str


class PaginatedScoresResponse(BaseModel):
    items: list[RiskScoreResponse]
    next_cursor: Optional[int] = Field(None, description="score_id cursor for next page")
    total: int


class LatestScoreResponse(BaseModel):
    wallet: str
    asset_pair: str
    score: int
    benford_flag: bool
    ml_flag: bool
    confidence: int
    top_features: list[dict]


class HealthResponse(BaseModel):
    status: str
    db: str
    model: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_store = RiskScoreStore()
_session_factory = get_session_factory()


def _record_to_response(r: RiskScoreRecord) -> RiskScoreResponse:
    return RiskScoreResponse(
        score_id=r.id,
        wallet=r.wallet,
        asset_pair=r.asset_pair,
        score=r.score,
        benford_flag=r.benford_flag,
        ml_flag=r.ml_flag,
        confidence=r.confidence,
        propagated_risk=r.propagated_risk,
        ring_id=r.ring_id,
        updated_at=r.updated_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health", response_model=HealthResponse, tags=["ops"])
@limiter.limit(f"{config.API_RATE_LIMIT_RPM}/minute")
async def health(request: Request):
    """Liveness and readiness probe."""
    db_status = "ok"
    try:
        with _session_factory() as session:
            session.execute(select(RiskScoreRecord).limit(1))
    except Exception:
        db_status = "unavailable"

    import os

    model_status = "ok" if os.path.isdir(config.MODEL_DIR) else "unavailable"

    overall = "ok" if db_status == "ok" else "degraded"
    return HealthResponse(status=overall, db=db_status, model=model_status)


@app.get(
    "/v1/wallets/{address}/scores",
    response_model=PaginatedScoresResponse,
    tags=["scores"],
)
@limiter.limit(f"{config.API_RATE_LIMIT_RPM}/minute")
async def get_wallet_scores(
    request: Request,
    address: str,
    start_ts: Optional[int] = Query(None, description="Unix timestamp lower bound"),
    end_ts: Optional[int] = Query(None, description="Unix timestamp upper bound"),
    asset_pair: Optional[str] = Query(None),
    min_score: Optional[int] = Query(None, ge=0, le=100),
    cursor: Optional[int] = Query(None, description="Cursor from previous page (score_id)"),
    limit: int = Query(50, ge=1, le=200),
    _key: str = Depends(_check_api_key),
):
    """Paginated risk score history for a wallet (cursor-based on score_id)."""
    _validate_stellar_address(address)

    from datetime import UTC, datetime

    with _session_factory() as session:
        stmt = (
            select(RiskScoreRecord)
            .where(RiskScoreRecord.wallet == address)
            .order_by(RiskScoreRecord.id.desc())
        )
        if cursor is not None:
            stmt = stmt.where(RiskScoreRecord.id < cursor)
        if asset_pair is not None:
            stmt = stmt.where(RiskScoreRecord.asset_pair == asset_pair)
        if min_score is not None:
            stmt = stmt.where(RiskScoreRecord.score >= min_score)
        if start_ts is not None:
            stmt = stmt.where(
                RiskScoreRecord.updated_at >= datetime.fromtimestamp(start_ts, tz=UTC)
            )
        if end_ts is not None:
            stmt = stmt.where(
                RiskScoreRecord.updated_at <= datetime.fromtimestamp(end_ts, tz=UTC)
            )

        rows = list(session.scalars(stmt.limit(limit + 1)))

    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None

    return PaginatedScoresResponse(
        items=[_record_to_response(r) for r in items],
        next_cursor=next_cursor,
        total=len(items),
    )


@app.get(
    "/v1/wallets/{address}/latest",
    response_model=LatestScoreResponse,
    tags=["scores"],
)
@limiter.limit(f"{config.API_RATE_LIMIT_RPM}/minute")
async def get_latest_score(
    request: Request,
    address: str,
    asset_pair: Optional[str] = Query(None),
    _key: str = Depends(_check_api_key),
):
    """Latest risk score and top-3 contributing features for a wallet."""
    _validate_stellar_address(address)

    with _session_factory() as session:
        stmt = (
            select(RiskScoreRecord)
            .where(RiskScoreRecord.wallet == address)
            .order_by(RiskScoreRecord.updated_at.desc())
        )
        if asset_pair is not None:
            stmt = stmt.where(RiskScoreRecord.asset_pair == asset_pair)
        record = session.scalar(stmt.limit(1))

    if record is None:
        raise HTTPException(status_code=404, detail="No score found for wallet")

    # Try to fetch top-3 SHAP features; fall back gracefully.
    top_features: list[dict] = []
    try:
        explainer = ShapExplainer(model_dir=config.MODEL_DIR)
        shap_vals = explainer.explain({"wallet": address, "asset_pair": record.asset_pair})
        sorted_vals = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top_features = [{"feature": k, "shap_value": v} for k, v in sorted_vals]
    except Exception:
        pass

    return LatestScoreResponse(
        wallet=record.wallet,
        asset_pair=record.asset_pair,
        score=record.score,
        benford_flag=record.benford_flag,
        ml_flag=record.ml_flag,
        confidence=record.confidence,
        top_features=top_features,
    )
