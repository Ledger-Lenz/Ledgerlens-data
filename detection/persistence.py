"""SQLAlchemy persistence model for `RiskScore` records.

This is the storage side of the `ledgerlens-data` -> `ledgerlens-api`
handoff described in the README: `RiskScorer.score()` output is written
here, keyed by `(wallet, asset_pair)`, for the API to read from
`RISK_SCORE_DB_URL`.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config import config


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
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    def to_risk_score(self) -> dict:
        """Return the on-chain/API `RiskScore` shape."""
        return {
            "score": self.score,
            "benford_flag": self.benford_flag,
            "ml_flag": self.ml_flag,
            "timestamp": int(self.updated_at.timestamp()),
            "confidence": self.confidence,
        }


def get_engine(db_url: str | None = None) -> Engine:
    return create_engine(db_url or config.RISK_SCORE_DB_URL, future=True)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)
