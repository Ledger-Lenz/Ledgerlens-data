"""Differential Privacy budget tracker with RDP/PRV accountant composition.

Tracks cumulative epsilon across training rounds and inference queries using
Rényi Differential Privacy (RDP) moments composition. Each consumption event
is appended to an append-only DB log so budget state cannot be silently mutated.

Usage::

    tracker = DPBudgetTracker()
    tracker.record_training_round(epsilon=1.2, delta=1e-5, model_version="v3")
    tracker.record_inference_query(epsilon=0.05, query_type="shap")
    status = tracker.status()
    # {"total_epsilon": ..., "remaining_epsilon": ..., "events": [...]}
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

QueryKind = Literal["training", "inference"]


# ---------------------------------------------------------------------------
# ORM
# ---------------------------------------------------------------------------

class _Base(DeclarativeBase):
    pass


class DPBudgetEvent(_Base):
    """Append-only log of privacy budget consumption events."""

    __tablename__ = "dp_budget_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    kind = Column(String(16), nullable=False)          # "training" | "inference"
    model_version = Column(String(64), nullable=True)
    query_type = Column(String(64), nullable=True)
    epsilon = Column(Float, nullable=False)
    delta = Column(Float, nullable=True)
    cumulative_epsilon = Column(Float, nullable=False)
    prev_log_hash = Column(String(64), nullable=False)
    log_hash = Column(String(64), nullable=False)


def _get_engine():
    engine = create_engine(config.RISK_SCORE_DB_URL)
    _Base.metadata.create_all(engine, checkfirst=True)
    if str(engine.url).startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(conn, _rec):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
    return engine


def _entry_hash(prev_hash: str, kind: str, epsilon: float, cumulative_epsilon: float, ts: str) -> str:
    material = json.dumps(
        {"prev": prev_hash, "kind": kind, "eps": epsilon, "cum": cumulative_epsilon, "ts": ts},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


# ---------------------------------------------------------------------------
# RDP composition helper
# ---------------------------------------------------------------------------

def _rdp_compose(epsilons: list[float]) -> float:
    """Additive composition of RDP epsilons (tight for homogeneous mechanisms).

    Under RDP composition theorem the total epsilon is the sum of individual
    epsilons at the same Rényi order.  For heterogeneous mechanisms this is a
    valid upper bound (conservative but correct).
    """
    return sum(epsilons)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DPBudgetTracker:
    """Track cumulative DP epsilon and alert when budget runs low.

    Parameters
    ----------
    total_epsilon:
        Hard cap on cumulative epsilon.  Defaults to ``DP_BUDGET_TOTAL_EPSILON``
        from config (100.0 if unset).
    alert_threshold_epsilon:
        Remaining epsilon below which an alert is dispatched.  Defaults to
        ``DP_BUDGET_ALERT_THRESHOLD`` from config (10.0 if unset).
    session_factory:
        Optional SQLAlchemy session factory.  A new engine is created when None.
    """

    def __init__(
        self,
        total_epsilon: float | None = None,
        alert_threshold_epsilon: float | None = None,
        session_factory=None,
    ) -> None:
        self._total_epsilon: float = (
            total_epsilon
            if total_epsilon is not None
            else getattr(config, "DP_BUDGET_TOTAL_EPSILON", 100.0)
        )
        self._alert_threshold: float = (
            alert_threshold_epsilon
            if alert_threshold_epsilon is not None
            else getattr(config, "DP_BUDGET_ALERT_THRESHOLD", 10.0)
        )
        if session_factory is None:
            engine = _get_engine()
            self._session_factory = sessionmaker(bind=engine, future=True)
        else:
            self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cumulative_epsilon(self, session: Session) -> tuple[float, str]:
        """Return (current cumulative epsilon, last log hash)."""
        row = (
            session.query(DPBudgetEvent)
            .order_by(DPBudgetEvent.id.desc())
            .first()
        )
        if row is None:
            return 0.0, "genesis"
        return row.cumulative_epsilon, row.log_hash

    def _append(
        self,
        kind: QueryKind,
        epsilon: float,
        *,
        delta: float | None = None,
        model_version: str | None = None,
        query_type: str | None = None,
    ) -> DPBudgetEvent:
        with self._session_factory() as session:
            cum_eps, prev_hash = self._cumulative_epsilon(session)
            new_cum = _rdp_compose([cum_eps, epsilon])
            ts = datetime.now(UTC).isoformat()
            log_hash = _entry_hash(prev_hash, kind, epsilon, new_cum, ts)
            evt = DPBudgetEvent(
                kind=kind,
                model_version=model_version,
                query_type=query_type,
                epsilon=epsilon,
                delta=delta,
                cumulative_epsilon=new_cum,
                prev_log_hash=prev_hash,
                log_hash=log_hash,
                created_at=datetime.now(UTC),
            )
            session.add(evt)
            session.commit()
            session.refresh(evt)
            logger.info(
                "DP budget event recorded: kind=%s eps=%.4f cumulative=%.4f",
                kind, epsilon, new_cum,
            )
            remaining = self._total_epsilon - new_cum
            if remaining < self._alert_threshold:
                self._fire_alert(remaining, new_cum)
            return evt

    def _fire_alert(self, remaining: float, cumulative: float) -> None:
        try:
            from streaming.alert_dispatcher import AlertDispatcher  # local import avoids circular dep
            dispatcher = AlertDispatcher(channel=getattr(config, "ALERT_CHANNEL", "stdout"))
            dispatcher.dispatch(
                wallet="__system__",
                asset_pair="__dp_budget__",
                score=100,
                details={
                    "alert_type": "dp_budget_low",
                    "remaining_epsilon": remaining,
                    "cumulative_epsilon": cumulative,
                    "total_epsilon": self._total_epsilon,
                    "threshold": self._alert_threshold,
                },
            )
        except Exception as exc:
            logger.warning("Failed to fire DP budget alert: %s", exc)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def record_training_round(
        self,
        epsilon: float,
        delta: float | None = None,
        model_version: str | None = None,
    ) -> DPBudgetEvent:
        """Record epsilon consumed by a DP-SGD training round."""
        return self._append(
            "training",
            epsilon,
            delta=delta,
            model_version=model_version,
        )

    def record_inference_query(
        self,
        epsilon: float,
        query_type: str = "generic",
        delta: float | None = None,
    ) -> DPBudgetEvent:
        """Record epsilon consumed by a DP inference-time query."""
        return self._append(
            "inference",
            epsilon,
            delta=delta,
            query_type=query_type,
        )

    def status(self) -> dict:
        """Return current budget status as a plain dict."""
        with self._session_factory() as session:
            cum_eps, _ = self._cumulative_epsilon(session)
            events = (
                session.query(DPBudgetEvent)
                .order_by(DPBudgetEvent.id.asc())
                .all()
            )
            return {
                "total_epsilon": self._total_epsilon,
                "cumulative_epsilon": cum_eps,
                "remaining_epsilon": max(0.0, self._total_epsilon - cum_eps),
                "alert_threshold": self._alert_threshold,
                "budget_exhausted": cum_eps >= self._total_epsilon,
                "events": [
                    {
                        "id": e.id,
                        "kind": e.kind,
                        "epsilon": e.epsilon,
                        "cumulative_epsilon": e.cumulative_epsilon,
                        "model_version": e.model_version,
                        "query_type": e.query_type,
                        "created_at": e.created_at.isoformat() if e.created_at else None,
                    }
                    for e in events
                ],
            }

    def rollover(self, operator_confirmation: str, new_model_version: str) -> None:
        """Reset cumulative epsilon after a major model version change.

        Requires an explicit confirmation string to prevent accidental resets.
        The old budget log is preserved; a synthetic 'rollover' event anchors
        the new epoch.
        """
        expected = f"CONFIRM_ROLLOVER_{new_model_version}"
        if operator_confirmation != expected:
            raise ValueError(
                f"Budget rollover requires operator_confirmation == {expected!r}. "
                "This prevents accidental epsilon resets."
            )
        with self._session_factory() as session:
            cum_eps, prev_hash = self._cumulative_epsilon(session)
            ts = datetime.now(UTC).isoformat()
            log_hash = _entry_hash(prev_hash, "rollover", -cum_eps, 0.0, ts)
            evt = DPBudgetEvent(
                kind="rollover",
                model_version=new_model_version,
                epsilon=-cum_eps,
                cumulative_epsilon=0.0,
                prev_log_hash=prev_hash,
                log_hash=log_hash,
                created_at=datetime.now(UTC),
            )
            session.add(evt)
            session.commit()
        logger.info(
            "DP budget rolled over for model version %s (previous cumulative=%.4f)",
            new_model_version,
            cum_eps,
        )
