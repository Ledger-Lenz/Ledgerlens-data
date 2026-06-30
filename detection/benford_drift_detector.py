"""Benford's Law Drift Detection for triggering automated model retraining.

The Benford engine computes digit-distribution metrics that serve as inputs to
the ensemble ML models. If the underlying trade distribution on the Stellar DEX
changes structurally (e.g., a new high-frequency market maker enters with a
distinctive lot-size pattern), the Benford signal baseline drifts, and the ML
models trained on the old distribution begin producing miscalibrated scores.

This detector monitors rolling chi-square and MAD statistics per asset pair
and emits a retraining trigger when the observed distribution deviates
significantly from the training-time baseline (using Welford's online variance
algorithm to avoid holding the full trade history in memory).

References
----------
Welford, B.P. (1962) "Note on a method for calculating corrected sums of
squares and products." Technometrics, 4(3), 419–420.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column, Float, Integer, String, DateTime, Boolean, JSON,
    create_engine, select
)
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.pool import NullPool

from config import config
from detection.benford_engine import (
    BenfordMetrics,
    chi_square_statistic,
    mad_score,
    compute_benford_metrics_for_windows,
)

logger = logging.getLogger(__name__)

Base = declarative_base()


class DriftStatus(Enum):
    """Drift detector status classification."""
    STABLE = "stable"
    DRIFTED = "drifted"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class BenfordBaseline:
    """Per-asset-pair baseline statistics for drift detection.
    
    Maintains running mean and variance (Welford's algorithm) of chi-square
    and MAD over the training set, allowing efficient computation of z-scores
    without holding the full history in memory.
    """
    pair_id: str
    chi_square_mean: float = 0.0
    chi_square_variance: float = 0.0
    chi_square_count: int = 0
    
    mad_mean: float = 0.0
    mad_variance: float = 0.0
    mad_count: int = 0
    
    # Per-window baselines (5 windows: 1h, 4h, 24h, 168h, 720h)
    per_window_baselines: dict[int, dict[str, float]] = field(default_factory=dict)
    
    last_drift_timestamp: str | None = None
    drift_count_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "pair_id": self.pair_id,
            "chi_square_mean": float(self.chi_square_mean),
            "chi_square_variance": float(self.chi_square_variance),
            "chi_square_count": int(self.chi_square_count),
            "mad_mean": float(self.mad_mean),
            "mad_variance": float(self.mad_variance),
            "mad_count": int(self.mad_count),
            "per_window_baselines": {
                str(k): v for k, v in self.per_window_baselines.items()
            },
            "last_drift_timestamp": self.last_drift_timestamp,
            "drift_count_total": int(self.drift_count_total),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "BenfordBaseline":
        """Deserialize from JSON dict."""
        baseline = BenfordBaseline(pair_id=d["pair_id"])
        baseline.chi_square_mean = float(d.get("chi_square_mean", 0.0))
        baseline.chi_square_variance = float(d.get("chi_square_variance", 0.0))
        baseline.chi_square_count = int(d.get("chi_square_count", 0))
        baseline.mad_mean = float(d.get("mad_mean", 0.0))
        baseline.mad_variance = float(d.get("mad_variance", 0.0))
        baseline.mad_count = int(d.get("mad_count", 0))
        
        # Parse per-window baselines (keys are strings from JSON)
        pwb = d.get("per_window_baselines", {})
        baseline.per_window_baselines = {int(k): v for k, v in pwb.items()}
        
        baseline.last_drift_timestamp = d.get("last_drift_timestamp")
        baseline.drift_count_total = int(d.get("drift_count_total", 0))
        
        return baseline


class BenfordDriftModel(Base):
    """SQLAlchemy model for persisting Benford drift baselines to the database."""
    __tablename__ = "benford_drift_baselines"
    
    id = Column(Integer, primary_key=True)
    pair_id = Column(String, unique=True, nullable=False)
    chi_square_mean = Column(Float, nullable=False)
    chi_square_variance = Column(Float, nullable=False)
    chi_square_count = Column(Integer, nullable=False)
    mad_mean = Column(Float, nullable=False)
    mad_variance = Column(Float, nullable=False)
    mad_count = Column(Integer, nullable=False)
    per_window_baselines = Column(JSON, nullable=False, default={})
    last_drift_timestamp = Column(String)
    drift_count_total = Column(Integer, default=0)
    updated_at = Column(DateTime, nullable=True)


class BenfordDriftDetector:
    """Monitors rolling Benford statistics per asset pair for distribution shift.
    
    Uses Welford's online variance algorithm to maintain running mean and
    variance without storing the full trade history. Detects drift when the
    current chi-square or MAD statistic deviates more than a z-score threshold
    (default 3.0 = 0.27% tail probability) from the training-time baseline.
    
    Parameters
    ----------
    db_url:
        SQLAlchemy database URL for persisting baselines (default: config.RISK_SCORE_DB_URL).
    z_threshold:
        Z-score threshold for flagging drift (default: config.BENFORD_DRIFT_Z_THRESHOLD, default 3.0).
    min_baseline_samples:
        Minimum baseline samples before drift detection is active (default 20).
    
    Attributes
    ----------
    baselines : dict[str, BenfordBaseline]
        Per-pair baselines, keyed by pair_id.
    _drifted_pairs : dict[str, int]
        Pairs that have drifted, with count of consecutive drifts (used for deduplication).
    """
    
    def __init__(
        self,
        db_url: str | None = None,
        z_threshold: float | None = None,
        min_baseline_samples: int = 20,
    ):
        self.db_url = db_url or config.RISK_SCORE_DB_URL
        self.z_threshold = z_threshold if z_threshold is not None else config.BENFORD_DRIFT_Z_THRESHOLD
        self.min_baseline_samples = min_baseline_samples
        
        # In-memory cache of baselines
        self.baselines: dict[str, BenfordBaseline] = {}
        
        # Track consecutively drifted pairs to avoid duplicate alerts
        self._drifted_pairs: dict[str, int] = {}
        
        # Initialize database engine
        try:
            # Use NullPool to avoid connection pooling issues in multi-threaded contexts
            self.engine = create_engine(self.db_url, poolclass=NullPool)
            Base.metadata.create_all(self.engine)
            self._load_baselines_from_db()
        except Exception as e:
            logger.warning(f"Failed to initialize Benford drift detector DB: {e}. Using in-memory only.")
            self.engine = None
    
    def _load_baselines_from_db(self) -> None:
        """Load persisted baselines from the database."""
        if not self.engine:
            return
        
        try:
            with Session(self.engine) as session:
                rows = session.query(BenfordDriftModel).all()
                for row in rows:
                    baseline = BenfordBaseline(pair_id=row.pair_id)
                    baseline.chi_square_mean = row.chi_square_mean
                    baseline.chi_square_variance = row.chi_square_variance
                    baseline.chi_square_count = row.chi_square_count
                    baseline.mad_mean = row.mad_mean
                    baseline.mad_variance = row.mad_variance
                    baseline.mad_count = row.mad_count
                    baseline.per_window_baselines = row.per_window_baselines or {}
                    baseline.last_drift_timestamp = row.last_drift_timestamp
                    baseline.drift_count_total = row.drift_count_total
                    
                    self.baselines[row.pair_id] = baseline
                    logger.debug(f"Loaded Benford baseline for pair {row.pair_id}")
        except Exception as e:
            logger.warning(f"Failed to load Benford baselines from DB: {e}")
    
    def _save_baseline_to_db(self, baseline: BenfordBaseline) -> None:
        """Persist a baseline to the database."""
        if not self.engine:
            return
        
        try:
            with Session(self.engine) as session:
                # Upsert: try to update, fall back to insert
                stmt = select(BenfordDriftModel).where(BenfordDriftModel.pair_id == baseline.pair_id)
                existing = session.execute(stmt).scalar_one_or_none()
                
                if existing:
                    existing.chi_square_mean = baseline.chi_square_mean
                    existing.chi_square_variance = baseline.chi_square_variance
                    existing.chi_square_count = baseline.chi_square_count
                    existing.mad_mean = baseline.mad_mean
                    existing.mad_variance = baseline.mad_variance
                    existing.mad_count = baseline.mad_count
                    existing.per_window_baselines = baseline.per_window_baselines
                    existing.last_drift_timestamp = baseline.last_drift_timestamp
                    existing.drift_count_total = baseline.drift_count_total
                else:
                    new_row = BenfordDriftModel(
                        pair_id=baseline.pair_id,
                        chi_square_mean=baseline.chi_square_mean,
                        chi_square_variance=baseline.chi_square_variance,
                        chi_square_count=baseline.chi_square_count,
                        mad_mean=baseline.mad_mean,
                        mad_variance=baseline.mad_variance,
                        mad_count=baseline.mad_count,
                        per_window_baselines=baseline.per_window_baselines,
                        last_drift_timestamp=baseline.last_drift_timestamp,
                        drift_count_total=baseline.drift_count_total,
                    )
                    session.add(new_row)
                
                session.commit()
        except Exception as e:
            logger.warning(f"Failed to save Benford baseline for {baseline.pair_id} to DB: {e}")
    
    def fit_baseline(self, pair_id: str, amounts: pd.Series) -> None:
        """Compute and store the baseline distribution from training data (Issue #180).
        
        Uses Welford's online variance algorithm to compute mean and variance
        incrementally without holding the full dataset in memory.
        
        Parameters
        ----------
        pair_id:
            Asset pair identifier (e.g. 'USDC:GA.../XLM:native').
        amounts:
            Series of trade amounts (positive floats).
        """
        amounts = amounts[amounts > 0]
        if amounts.empty:
            logger.warning(f"Pair {pair_id}: empty amounts for baseline fit. Skipping.")
            return
        
        # Compute first-digit metrics
        chi_sq = chi_square_statistic(amounts)
        mad = mad_score(amounts)
        
        # Validate that computed values are finite and positive
        if not np.isfinite(chi_sq) or chi_sq < 0:
            logger.warning(f"Pair {pair_id}: invalid chi-square {chi_sq}. Skipping baseline fit.")
            return
        if not np.isfinite(mad) or mad < 0:
            logger.warning(f"Pair {pair_id}: invalid MAD {mad}. Skipping baseline fit.")
            return
        
        # Initialize or update baseline using Welford's algorithm
        baseline = self.baselines.get(pair_id) or BenfordBaseline(pair_id=pair_id)
        
        # Update chi-square running mean and variance
        baseline.chi_square_count += 1
        n = baseline.chi_square_count
        delta = chi_sq - baseline.chi_square_mean
        baseline.chi_square_mean += delta / n
        delta2 = chi_sq - baseline.chi_square_mean
        baseline.chi_square_variance += delta * delta2
        
        # Update MAD running mean and variance
        baseline.mad_count += 1
        m = baseline.mad_count
        delta_mad = mad - baseline.mad_mean
        baseline.mad_mean += delta_mad / m
        delta2_mad = mad - baseline.mad_mean
        baseline.mad_variance += delta_mad * delta2_mad
        
        self.baselines[pair_id] = baseline
        self._save_baseline_to_db(baseline)
        
        logger.debug(
            f"Pair {pair_id}: fitted baseline (n={n}, chi_sq_mean={baseline.chi_square_mean:.3f}, "
            f"mad_mean={baseline.mad_mean:.5f})"
        )
    
    def check(
        self,
        pair_id: str,
        current_chi_square: float,
        current_mad: float,
    ) -> DriftStatus:
        """Check if the current metrics indicate distribution drift (Issue #180).
        
        Compares current chi-square and MAD against the baseline using z-scores.
        If either metric exceeds the z-threshold, returns DriftStatus.DRIFTED.
        
        Parameters
        ----------
        pair_id:
            Asset pair identifier.
        current_chi_square:
            Observed chi-square statistic from recent trades.
        current_mad:
            Observed MAD statistic from recent trades.
        
        Returns
        -------
        DriftStatus:
            STABLE if no drift detected.
            DRIFTED if chi-square or MAD z-score exceeds threshold.
            INSUFFICIENT_DATA if baseline has < min_baseline_samples observations.
        """
        # Check baseline existence
        baseline = self.baselines.get(pair_id)
        if not baseline:
            return DriftStatus.INSUFFICIENT_DATA
        
        # Check sample size
        if baseline.chi_square_count < self.min_baseline_samples:
            return DriftStatus.INSUFFICIENT_DATA
        
        # Validate input values
        if not (np.isfinite(current_chi_square) and np.isfinite(current_mad)):
            logger.warning(f"Pair {pair_id}: invalid current metrics (chi_sq={current_chi_square}, mad={current_mad})")
            return DriftStatus.STABLE
        
        # Compute z-scores (handle zero variance edge case)
        chi_sq_std = np.sqrt(baseline.chi_square_variance / baseline.chi_square_count)
        mad_std = np.sqrt(baseline.mad_variance / baseline.mad_count)
        
        chi_sq_z = abs(current_chi_square - baseline.chi_square_mean) / chi_sq_std if chi_sq_std > 0 else 0.0
        mad_z = abs(current_mad - baseline.mad_mean) / mad_std if mad_std > 0 else 0.0
        
        # Flag drift if either z-score exceeds threshold
        if chi_sq_z > self.z_threshold or mad_z > self.z_threshold:
            # Dedup: only log once per transition to drifted state
            if pair_id not in self._drifted_pairs:
                logger.warning(
                    f"Pair {pair_id}: Benford drift detected. "
                    f"chi_sq: {current_chi_square:.3f} (baseline {baseline.chi_square_mean:.3f}, z={chi_sq_z:.2f}), "
                    f"mad: {current_mad:.5f} (baseline {baseline.mad_mean:.5f}, z={mad_z:.2f})"
                )
                baseline.drift_count_total += 1
                self._save_baseline_to_db(baseline)
            
            self._drifted_pairs[pair_id] = self._drifted_pairs.get(pair_id, 0) + 1
            return DriftStatus.DRIFTED
        else:
            # Transition out of drift: reset dedup counter
            if pair_id in self._drifted_pairs:
                logger.info(f"Pair {pair_id}: Benford metrics returned to stable range.")
                del self._drifted_pairs[pair_id]
            
            return DriftStatus.STABLE
    
    def check_batch(
        self,
        per_pair_metrics: dict[str, dict[str, float]],
    ) -> dict[str, DriftStatus]:
        """Check drift status for multiple pairs at once.
        
        Parameters
        ----------
        per_pair_metrics:
            Dict mapping pair_id -> {chi_square: float, mad: float}.
        
        Returns
        -------
        dict[str, DriftStatus]:
            Map of pair_id -> drift status.
        """
        results = {}
        for pair_id, metrics in per_pair_metrics.items():
            chi_sq = metrics.get("chi_square", np.nan)
            mad = metrics.get("mad", np.nan)
            results[pair_id] = self.check(pair_id, chi_sq, mad)
        
        return results
    
    def get_drifted_pairs(self) -> list[str]:
        """Return list of pairs currently in drifted state."""
        return list(self._drifted_pairs.keys())
    
    def should_trigger_retrain(self, num_pairs_trigger: int | None = None) -> bool:
        """Check if enough pairs are drifted to warrant model retraining.
        
        Parameters
        ----------
        num_pairs_trigger:
            Minimum number of pairs that must be in drifted state (default: 
            config.BENFORD_DRIFT_NUM_PAIRS_TRIGGER, default 0 = any single pair triggers).
        
        Returns
        -------
        bool:
            True if the number of drifted pairs meets or exceeds the threshold.
        """
        if num_pairs_trigger is None:
            num_pairs_trigger = config.BENFORD_DRIFT_NUM_PAIRS_TRIGGER
        
        drifted_count = len(self._drifted_pairs)
        
        if num_pairs_trigger == 0:
            # Any single pair can trigger
            return drifted_count > 0
        else:
            return drifted_count >= num_pairs_trigger
