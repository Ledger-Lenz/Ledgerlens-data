"""Repository for reading and writing `RiskScoreRecord`s.

Used by `run_pipeline.py` to persist `RiskScorer.score()` output for
`ledgerlens-api` to read, and to look up previously flagged wallets.
"""

import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from config import config
from detection.persistence import (
    ModelInversionQueryTracker,
    RiskScoreRecord,
    ShapQueryCount,
    get_session_factory,
)
from utils.logging import get_logger
from utils.tracing import get_tracer, hash_span_id

logger = get_logger(__name__)
_tracer = get_tracer(__name__)


class RiskScoreStore:
    """CRUD wrapper around `RiskScoreRecord` keyed by `(wallet, asset_pair)`."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or get_session_factory()

    def upsert(self, wallet: str, asset_pair: str, risk_score: dict) -> RiskScoreRecord:
        """Insert or update the `RiskScore` record for `(wallet, asset_pair)`."""
        with _tracer.start_as_current_span("score.stored") as span:
            span.set_attribute("wallet.id", hash_span_id(wallet))
            span.set_attribute("score.value", risk_score.get("score", -1))
            return self._upsert_impl(wallet, asset_pair, risk_score)

    def _upsert_impl(self, wallet: str, asset_pair: str, risk_score: dict) -> RiskScoreRecord:
        """Internal upsert logic called inside an OTel span."""
        for attempt in range(5):
            try:
                with self._session_factory() as session:
                    existing = session.scalar(
                        select(RiskScoreRecord).where(
                            RiskScoreRecord.wallet == wallet,
                            RiskScoreRecord.asset_pair == asset_pair,
                        )
                    )
                    if existing is None:
                        existing = RiskScoreRecord(wallet=wallet, asset_pair=asset_pair)
                        session.add(existing)

                    existing.score = int(risk_score["score"])
                    existing.benford_flag = bool(risk_score["benford_flag"])
                    existing.ml_flag = bool(risk_score["ml_flag"])
                    existing.confidence = int(risk_score["confidence"])
                    if "propagated_risk" in risk_score:
                        existing.propagated_risk = float(risk_score["propagated_risk"])
                    if "ring_id" in risk_score:
                        existing.ring_id = risk_score["ring_id"]

                    session.commit()
                    session.refresh(existing)
                    return existing
            except OperationalError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (2**attempt))

    def get(self, wallet: str, asset_pair: str) -> RiskScoreRecord | None:
        with self._session_factory() as session:
            return cast(
                RiskScoreRecord | None,
                session.scalar(
                    select(RiskScoreRecord).where(
                        RiskScoreRecord.wallet == wallet,
                        RiskScoreRecord.asset_pair == asset_pair,
                    )
                ),
            )

    def list_flagged(self, threshold: int) -> Iterable[RiskScoreRecord]:
        with self._session_factory() as session:
            return list(
                session.scalars(
                    select(RiskScoreRecord)
                    .where(RiskScoreRecord.score >= threshold)
                    .order_by(RiskScoreRecord.score.desc())
                )
            )

    # ------------------------------------------------------------------
    # SHAP query accounting (differential-privacy Rényi composition)
    # ------------------------------------------------------------------

    def increment_shap_query(self, wallet: str) -> int:
        """Atomically increment and return `wallet`'s SHAP query count."""
        with self._session_factory() as session:
            counter = session.get(ShapQueryCount, wallet)
            if counter is None:
                counter = ShapQueryCount(wallet=wallet, query_count=0)
                session.add(counter)
            counter.query_count += 1
            new_count = counter.query_count
            session.commit()
            return new_count

    def get_shap_query_count(self, wallet: str) -> int:
        """Return `wallet`'s current SHAP query count (0 if never queried)."""
        with self._session_factory() as session:
            counter = session.get(ShapQueryCount, wallet)
            return counter.query_count if counter is not None else 0

    # ------------------------------------------------------------------
    # Model inversion attack defence (Issue #264)
    # ------------------------------------------------------------------

    def check_query_limit(self, caller_id: str, wallet_id: str) -> tuple[bool, int]:
        """Check if caller has exceeded query limit for wallet (external API calls only).

        Returns:
            (exceeded, current_count): True if limit exceeded, plus current query count
        """
        with self._session_factory() as session:
            tracker = session.scalar(
                select(ModelInversionQueryTracker).where(
                    ModelInversionQueryTracker.caller_id == caller_id,
                    ModelInversionQueryTracker.wallet_id == wallet_id,
                )
            )
            if tracker is None:
                return False, 0
            exceeded = tracker.query_count >= config.MODEL_INVERSION_QUERY_LIMIT
            return exceeded, tracker.query_count

    def increment_query_count(self, caller_id: str, wallet_id: str) -> int:
        """Atomically increment and return query count for (caller_id, wallet_id).

        External API calls only. Returns the incremented count.
        """
        with self._session_factory() as session:
            tracker = session.scalar(
                select(ModelInversionQueryTracker).where(
                    ModelInversionQueryTracker.caller_id == caller_id,
                    ModelInversionQueryTracker.wallet_id == wallet_id,
                )
            )
            if tracker is None:
                tracker = ModelInversionQueryTracker(
                    caller_id=caller_id, wallet_id=wallet_id, query_count=0
                )
                session.add(tracker)
            tracker.query_count = (tracker.query_count or 0) + 1
            tracker.last_query_at = datetime.now(UTC)
            new_count = tracker.query_count
            session.commit()
            return new_count
