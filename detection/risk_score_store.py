"""Repository for reading and writing `RiskScoreRecord`s.

Used by `run_pipeline.py` to persist `RiskScorer.score()` output for
`ledgerlens-api` to read, and to look up previously flagged wallets.
"""

import time
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from detection.persistence import RiskScoreRecord, get_session_factory


class RiskScoreStore:
    """CRUD wrapper around `RiskScoreRecord` keyed by `(wallet, asset_pair)`."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None):
        self._session_factory = session_factory or get_session_factory()

    def upsert(self, wallet: str, asset_pair: str, risk_score: dict) -> RiskScoreRecord:
        """Insert or update the `RiskScore` record for `(wallet, asset_pair)`.

        `risk_score` is the dict returned by `RiskScorer.score()`:
        ``{"score", "benford_flag", "ml_flag", "confidence"}`` plus the
        optional ``"propagated_risk"`` float produced by
        :func:`detection.risk_propagation.propagate_risk_scores`.
        ``"timestamp"`` is ignored — ``updated_at`` is set server-side.
        """
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

                    session.commit()
                    session.refresh(existing)
                    return existing
            except OperationalError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (2**attempt))

    def get(self, wallet: str, asset_pair: str) -> RiskScoreRecord | None:
        with self._session_factory() as session:
            return session.scalar(
                select(RiskScoreRecord).where(
                    RiskScoreRecord.wallet == wallet,
                    RiskScoreRecord.asset_pair == asset_pair,
                )
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
