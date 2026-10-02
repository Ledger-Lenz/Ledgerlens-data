"""Data quality scoring and ledger import readiness evaluation (Issue #464).

Provides a reusable, contract-driven data quality scoring engine for evaluating
incoming ledger records (trades, payments, orderbooks, account activity) before ingestion,
plus stream-level anomaly detection on batch volume and field shape (Issue #913).
"""

from __future__ import annotations

import datetime
import re
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from alerts.router import Alert, AlertRouter, RouteDestination
from utils.logging import get_logger

logger = get_logger(__name__)


class QualityDimension(StrEnum):
    COMPLETENESS = "COMPLETENESS"
    VALIDITY = "VALIDITY"
    TIMELINESS = "TIMELINESS"
    UNIQUENESS = "UNIQUENESS"
    CONSISTENCY = "CONSISTENCY"
    ANOMALY = "ANOMALY"


class ReadinessStatus(StrEnum):
    READY = "READY"
    WARNING = "WARNING"
    QUARANTINE_REJECTED = "QUARANTINE_REJECTED"


# Stellar account public key format: 56 alphanumeric chars starting with G or C
STELLAR_ACCOUNT_REGEX = re.compile(r"^[GC][A-Z2-7]{55}$")


@dataclass
class QualityRuleResult:
    """Evaluation result for a single data quality rule."""

    rule_name: str
    dimension: QualityDimension
    passed: bool
    score: float  # 0.0 to 1.0
    weight: float = 1.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    failed_records_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dimension"] = (
            self.dimension.value if isinstance(self.dimension, QualityDimension) else self.dimension
        )
        return d


@dataclass
class QualityReport:
    """Comprehensive data quality report for ledger import readiness."""

    overall_score: float  # 0.0 to 100.0
    status: ReadinessStatus
    dimension_scores: dict[str, float]
    rule_results: list[QualityRuleResult]
    total_records: int
    evaluated_at: str = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC).isoformat()
    )
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "status": (
                self.status.value if isinstance(self.status, ReadinessStatus) else self.status
            ),
            "dimension_scores": {k: round(v, 2) for k, v in self.dimension_scores.items()},
            "rule_results": [r.to_dict() for r in self.rule_results],
            "total_records": self.total_records,
            "evaluated_at": self.evaluated_at,
            "diagnostics": self.diagnostics,
        }


class QualityRule:
    """Base abstract/contract class for data quality rules."""

    def __init__(
        self,
        name: str,
        dimension: QualityDimension,
        weight: float = 1.0,
    ) -> None:
        self.name = name
        self.dimension = dimension
        self.weight = weight

    def evaluate(self, df: pd.DataFrame) -> QualityRuleResult:
        raise NotImplementedError


class CompletenessRule(QualityRule):
    """Checks required columns and maximum permitted null ratio per column."""

    def __init__(
        self,
        required_columns: list[str],
        max_null_ratio: float = 0.05,
        weight: float = 1.0,
    ) -> None:
        super().__init__("completeness_check", QualityDimension.COMPLETENESS, weight)
        self.required_columns = required_columns
        self.max_null_ratio = max_null_ratio

    def evaluate(self, df: pd.DataFrame) -> QualityRuleResult:
        if df.empty:
            return QualityRuleResult(
                rule_name=self.name,
                dimension=self.dimension,
                passed=False,
                score=0.0,
                weight=self.weight,
                message="Dataset is empty.",
                failed_records_count=0,
            )

        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            return QualityRuleResult(
                rule_name=self.name,
                dimension=self.dimension,
                passed=False,
                score=0.0,
                weight=self.weight,
                message=f"Missing required columns: {missing_cols}",
                details={"missing_columns": missing_cols},
                failed_records_count=len(df),
            )

        total_rows = len(df)
        col_null_ratios = {}
        total_failed_rows = 0

        for col in self.required_columns:
            null_cnt = int(df[col].isna().sum())
            null_ratio = null_cnt / total_rows if total_rows > 0 else 0.0
            col_null_ratios[col] = null_ratio
            if null_cnt > 0:
                total_failed_rows = max(total_failed_rows, null_cnt)

        max_observed_ratio = max(col_null_ratios.values()) if col_null_ratios else 0.0
        passed = max_observed_ratio <= self.max_null_ratio

        # Linear score penalty based on null ratio vs threshold
        score = (
            max(0.0, 1.0 - (max_observed_ratio / max(self.max_null_ratio, 0.01)))
            if not passed
            else 1.0
        )

        msg = (
            "All required columns meet completeness thresholds."
            if passed
            else (
                f"Exceeded max null ratio threshold ({self.max_null_ratio:.1%}). Highest null ratio: {max_observed_ratio:.1%}"
            )
        )

        return QualityRuleResult(
            rule_name=self.name,
            dimension=self.dimension,
            passed=passed,
            score=score,
            weight=self.weight,
            message=msg,
            details={"col_null_ratios": col_null_ratios},
            failed_records_count=total_failed_rows,
        )


class StellarAddressValidityRule(QualityRule):
    """Validates format of Stellar account addresses."""

    def __init__(
        self,
        address_columns: list[str] | None = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__("stellar_address_validity", QualityDimension.VALIDITY, weight)
        self.address_columns = address_columns or [
            "account",
            "source_account",
            "destination_account",
            "seller",
            "buyer",
        ]

    def evaluate(self, df: pd.DataFrame) -> QualityRuleResult:
        target_cols = [c for c in self.address_columns if c in df.columns]
        if not target_cols or df.empty:
            return QualityRuleResult(
                rule_name=self.name,
                dimension=self.dimension,
                passed=True,
                score=1.0,
                weight=self.weight,
                message="No address columns present to validate.",
            )

        total_rows = len(df)
        invalid_counts = {}
        invalid_rows_mask = pd.Series(False, index=df.index)

        for col in target_cols:
            valid_mask = df[col].astype(str).str.match(STELLAR_ACCOUNT_REGEX, na=False)
            inv_cnt = int((~valid_mask).sum())
            invalid_counts[col] = inv_cnt
            invalid_rows_mask |= ~valid_mask

        failed_count = int(invalid_rows_mask.sum())
        score = max(0.0, 1.0 - (failed_count / total_rows))
        passed = failed_count == 0

        msg = (
            "All address fields contain valid Stellar account keys."
            if passed
            else (
                f"Found {failed_count} records ({failed_count/total_rows:.1%}) with invalid Stellar account keys."
            )
        )

        return QualityRuleResult(
            rule_name=self.name,
            dimension=self.dimension,
            passed=passed,
            score=score,
            weight=self.weight,
            message=msg,
            details={"invalid_counts_per_column": invalid_counts},
            failed_records_count=failed_count,
        )


class AmountValidityRule(QualityRule):
    """Validates that monetary trade/payment amounts are strictly positive and finite."""

    def __init__(
        self,
        amount_column: str = "amount",
        allow_zero: bool = False,
        weight: float = 1.0,
    ) -> None:
        super().__init__("amount_validity", QualityDimension.VALIDITY, weight)
        self.amount_column = amount_column
        self.allow_zero = allow_zero

    def evaluate(self, df: pd.DataFrame) -> QualityRuleResult:
        if self.amount_column not in df.columns or df.empty:
            return QualityRuleResult(
                rule_name=self.name,
                dimension=self.dimension,
                passed=True,
                score=1.0,
                weight=self.weight,
                message=f"Column '{self.amount_column}' not present in dataset.",
            )

        amounts = pd.to_numeric(df[self.amount_column], errors="coerce")
        if self.allow_zero:
            valid_mask = (amounts >= 0) & np.isfinite(amounts)
        else:
            valid_mask = (amounts > 0) & np.isfinite(amounts)

        failed_count = int((~valid_mask).sum())
        total_rows = len(df)
        score = max(0.0, 1.0 - (failed_count / total_rows))
        passed = failed_count == 0

        msg = (
            f"All '{self.amount_column}' values are valid non-negative numbers."
            if passed
            else (
                f"Found {failed_count} records with invalid/negative/non-numeric '{self.amount_column}'."
            )
        )

        return QualityRuleResult(
            rule_name=self.name,
            dimension=self.dimension,
            passed=passed,
            score=score,
            weight=self.weight,
            message=msg,
            failed_records_count=failed_count,
        )


class TimelinessRule(QualityRule):
    """Validates timestamp freshness and checks for future/corrupted dates."""

    def __init__(
        self,
        timestamp_column: str = "ledger_close_time",
        max_age_days: float = 365.0,
        max_future_seconds: float = 300.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__("timeliness_check", QualityDimension.TIMELINESS, weight)
        self.timestamp_column = timestamp_column
        self.max_age_days = max_age_days
        self.max_future_seconds = max_future_seconds

    def evaluate(self, df: pd.DataFrame) -> QualityRuleResult:
        if self.timestamp_column not in df.columns or df.empty:
            return QualityRuleResult(
                rule_name=self.name,
                dimension=self.dimension,
                passed=True,
                score=1.0,
                weight=self.weight,
                message=f"Timestamp column '{self.timestamp_column}' not found.",
            )

        parsed = pd.to_datetime(df[self.timestamp_column], errors="coerce", utc=True)
        unparseable_cnt = int(parsed.isna().sum())

        now = pd.Timestamp.now(tz="UTC")
        max_past_dt = now - pd.Timedelta(days=self.max_age_days)
        max_future_dt = now + pd.Timedelta(seconds=self.max_future_seconds)

        future_cnt = int((parsed > max_future_dt).sum())
        stale_cnt = int((parsed < max_past_dt).sum())

        failed_count = unparseable_cnt + future_cnt + stale_cnt
        total_rows = len(df)
        score = max(0.0, 1.0 - (failed_count / total_rows))
        passed = failed_count == 0

        msg = (
            "All timestamps are parseable, fresh, and within valid window."
            if passed
            else (
                f"Timeliness failures: {unparseable_cnt} unparseable, {future_cnt} in future, {stale_cnt} older than {self.max_age_days} days."
            )
        )

        return QualityRuleResult(
            rule_name=self.name,
            dimension=self.dimension,
            passed=passed,
            score=score,
            weight=self.weight,
            message=msg,
            details={
                "unparseable_count": unparseable_cnt,
                "future_count": future_cnt,
                "stale_count": stale_cnt,
            },
            failed_records_count=failed_count,
        )


class UniquenessRule(QualityRule):
    """Checks duplicate ratio on key identifier columns (trade_id, hash, op_id)."""

    def __init__(
        self,
        id_columns: list[str] | None = None,
        max_duplicate_ratio: float = 0.01,
        weight: float = 1.0,
    ) -> None:
        super().__init__("uniqueness_check", QualityDimension.UNIQUENESS, weight)
        self.id_columns = id_columns or ["trade_id", "transaction_hash", "operation_id", "id"]
        self.max_duplicate_ratio = max_duplicate_ratio

    def evaluate(self, df: pd.DataFrame) -> QualityRuleResult:
        present_cols = [c for c in self.id_columns if c in df.columns]
        if not present_cols or df.empty:
            return QualityRuleResult(
                rule_name=self.name,
                dimension=self.dimension,
                passed=True,
                score=1.0,
                weight=self.weight,
                message="No ID columns present to assess uniqueness.",
            )

        col = present_cols[0]
        total_rows = len(df)
        dup_mask = df.duplicated(subset=[col], keep="first")
        dup_count = int(dup_mask.sum())
        dup_ratio = dup_count / total_rows if total_rows > 0 else 0.0

        passed = dup_ratio <= self.max_duplicate_ratio
        score = (
            max(0.0, 1.0 - (dup_ratio / max(self.max_duplicate_ratio, 0.01))) if not passed else 1.0
        )

        msg = (
            f"Uniqueness check passed for column '{col}' (dup ratio {dup_ratio:.2%})."
            if passed
            else (
                f"Duplicate ratio on '{col}' ({dup_ratio:.2%}) exceeded maximum threshold ({self.max_duplicate_ratio:.2%})."
            )
        )

        return QualityRuleResult(
            rule_name=self.name,
            dimension=self.dimension,
            passed=passed,
            score=score,
            weight=self.weight,
            message=msg,
            details={"column": col, "duplicate_count": dup_count, "duplicate_ratio": dup_ratio},
            failed_records_count=dup_count,
        )


class OrderbookSpreadConsistencyRule(QualityRule):
    """Validates orderbook sanity (e.g., bid <= ask, positive volumes)."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__("orderbook_spread_consistency", QualityDimension.CONSISTENCY, weight)

    def evaluate(self, df: pd.DataFrame) -> QualityRuleResult:
        if "best_bid" not in df.columns or "best_ask" not in df.columns or df.empty:
            return QualityRuleResult(
                rule_name=self.name,
                dimension=self.dimension,
                passed=True,
                score=1.0,
                weight=self.weight,
                message="Orderbook spread columns not present.",
            )

        bids = pd.to_numeric(df["best_bid"], errors="coerce")
        asks = pd.to_numeric(df["best_ask"], errors="coerce")

        crossed = (bids > asks) & np.isfinite(bids) & np.isfinite(asks)
        failed_count = int(crossed.sum())
        total_rows = len(df)
        score = max(0.0, 1.0 - (failed_count / total_rows))
        passed = failed_count == 0

        msg = (
            "Orderbook spreads consistent (no crossed markets)."
            if passed
            else (f"Found {failed_count} crossed orderbook snapshots (best_bid > best_ask).")
        )

        return QualityRuleResult(
            rule_name=self.name,
            dimension=self.dimension,
            passed=passed,
            score=score,
            weight=self.weight,
            message=msg,
            failed_records_count=failed_count,
        )


class LedgerQualityScorer:
    """Data Quality Scorer and Import Readiness Engine."""

    def __init__(
        self,
        rules: list[QualityRule] | None = None,
        pass_threshold: float = 85.0,
        warning_threshold: float = 65.0,
        dimension_weights: dict[QualityDimension, float] | None = None,
    ) -> None:
        self.pass_threshold = pass_threshold
        self.warning_threshold = warning_threshold
        self.dimension_weights = dimension_weights or {
            QualityDimension.COMPLETENESS: 0.30,
            QualityDimension.VALIDITY: 0.30,
            QualityDimension.TIMELINESS: 0.15,
            QualityDimension.UNIQUENESS: 0.15,
            QualityDimension.CONSISTENCY: 0.10,
        }
        self.rules = rules if rules is not None else self._default_trade_rules()

    def _default_trade_rules(self) -> list[QualityRule]:
        return [
            CompletenessRule(required_columns=["amount"], max_null_ratio=0.02),
            StellarAddressValidityRule(),
            AmountValidityRule(amount_column="amount"),
            TimelinessRule(),
            UniquenessRule(),
            OrderbookSpreadConsistencyRule(),
        ]

    def add_rule(self, rule: QualityRule) -> None:
        """Add a custom quality rule to the scorer."""
        self.rules.append(rule)

    def evaluate_import_readiness(self, data: pd.DataFrame | list[dict[str, Any]]) -> QualityReport:
        """Evaluate dataset against all rules and compute readiness score & status."""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        total_records = len(df)
        if total_records == 0:
            report = QualityReport(
                overall_score=0.0,
                status=ReadinessStatus.QUARANTINE_REJECTED,
                dimension_scores={dim.value: 0.0 for dim in QualityDimension},
                rule_results=[],
                total_records=0,
                diagnostics=["CRITICAL: Input dataset is empty (0 records). Ingestion rejected."],
            )
            return report

        results: list[QualityRuleResult] = []
        dim_scores_sum: dict[QualityDimension, float] = {d: 0.0 for d in QualityDimension}
        dim_weights_sum: dict[QualityDimension, float] = {d: 0.0 for d in QualityDimension}

        diagnostics: list[str] = []

        for rule in self.rules:
            res = rule.evaluate(df)
            results.append(res)
            dim_scores_sum[res.dimension] += res.score * res.weight
            dim_weights_sum[res.dimension] += res.weight

            if not res.passed:
                diagnostics.append(
                    f"[{res.dimension.value}] Rule '{res.rule_name}' FAILED (score={res.score:.2f}, failed_records={res.failed_records_count}): {res.message}"
                )

        dim_final_scores: dict[str, float] = {}
        for dim, total_w in dim_weights_sum.items():
            if total_w > 0:
                dim_final_scores[dim.value] = (dim_scores_sum[dim] / total_w) * 100.0
            else:
                dim_final_scores[dim.value] = 100.0

        # Compute overall weighted score
        overall_weighted_sum = 0.0
        overall_weight_total = 0.0
        for dim, weight in self.dimension_weights.items():
            dim_score = dim_final_scores.get(dim.value, 100.0)
            overall_weighted_sum += dim_score * weight
            overall_weight_total += weight

        overall_score = (
            (overall_weighted_sum / overall_weight_total) if overall_weight_total > 0 else 0.0
        )

        if overall_score >= self.pass_threshold:
            status = ReadinessStatus.READY
        elif overall_score >= self.warning_threshold:
            status = ReadinessStatus.WARNING
        else:
            status = ReadinessStatus.QUARANTINE_REJECTED

        if status == ReadinessStatus.QUARANTINE_REJECTED:
            diagnostics.insert(
                0,
                f"REJECTED: Overall data quality score ({overall_score:.1f}/100) is below pass threshold ({self.pass_threshold}).",
            )
        elif status == ReadinessStatus.WARNING:
            diagnostics.insert(
                0,
                f"WARNING: Data quality score ({overall_score:.1f}/100) flagged for operational review.",
            )

        return QualityReport(
            overall_score=overall_score,
            status=status,
            dimension_scores=dim_final_scores,
            rule_results=results,
            total_records=total_records,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Stream-level anomaly detection (Issue #913)
#
# Per-record rules above cannot see a systemic upstream fault that produces
# individually valid records which are collectively wrong (a feed silently
# dropping 90% of trades, a field suddenly arriving null for every record).
# StreamQualityMonitor tracks rolling per-source baselines of batch volume and
# key-field null rates / means and routes deviations through alerts.router.
# ---------------------------------------------------------------------------

STREAM_QUALITY_DETECTOR = "stream_quality_monitor"

try:
    from prometheus_client import Counter, Gauge

    stream_metric_value = Gauge(
        "ledgerlens_ingestion_stream_metric_value",
        "Latest observed stream-level ingestion metric (batch volume, null rate, mean)",
        ["source", "metric"],
    )
    stream_metric_zscore = Gauge(
        "ledgerlens_ingestion_stream_metric_zscore",
        "Deviation of the latest stream-level metric from its rolling baseline",
        ["source", "metric"],
    )
    stream_anomalies_total = Counter(
        "ledgerlens_ingestion_stream_anomalies_total",
        "Stream-level ingestion anomalies detected, by suppression state",
        ["source", "metric", "direction", "suppressed"],
    )
except Exception:  # pragma: no cover - prometheus optional in tests
    stream_metric_value = None  # type: ignore[assignment]
    stream_metric_zscore = None  # type: ignore[assignment]
    stream_anomalies_total = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SuppressionWindow:
    """An acknowledged period during which anomalies for a source are expected.

    ``source="*"`` covers every source; ``metrics=None`` covers every metric.
    """

    source: str
    start: datetime.datetime
    end: datetime.datetime
    reason: str
    metrics: frozenset[str] | None = None

    def covers(self, source: str, metric: str, at: datetime.datetime) -> bool:
        if self.source not in ("*", source):
            return False
        if self.metrics is not None and metric not in self.metrics:
            return False
        return self.start <= at < self.end


@dataclass
class StreamAnomaly:
    """A stream-level metric that deviated from its rolling baseline."""

    source: str
    metric: str
    observed: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    detected_at: datetime.datetime
    suppressed_by: SuppressionWindow | None = None

    @property
    def direction(self) -> str:
        return "spike" if self.observed > self.baseline_mean else "drop"

    def to_alert(self) -> Alert:
        """Shape the anomaly as an ``alerts.router.Alert`` with triage context."""
        return {
            "detectors": [STREAM_QUALITY_DETECTOR],
            "source": self.source,
            "metric": self.metric,
            "direction": self.direction,
            "observed": self.observed,
            "baseline": self.baseline_mean,
            "magnitude": self.z_score,
            "detected_at": self.detected_at.isoformat(),
            "message": (
                f"{self.source}: {self.metric} {self.direction} "
                f"(observed={self.observed:.4g}, baseline={self.baseline_mean:.4g}, "
                f"z={self.z_score:+.1f})"
            ),
        }


class StreamQualityMonitor:
    """Rolling statistical checks on ingestion volume and key-field shape.

    Each call to :meth:`observe_batch` computes, per source, the batch
    ``volume``, ``null_rate:<field>`` for every key field and ``mean:<field>``
    for every distribution field, and compares each against a rolling window
    of that source's previous batches. A metric whose z-score exceeds
    ``z_threshold`` is an anomaly. The spread used for the z-score is floored
    (``relative_std_floor`` of the baseline mean, ``null_rate_std_floor`` for
    null rates) so a near-constant baseline does not turn ordinary jitter into
    alerts.

    Anomalous observations are kept out of the baseline so one bad batch does
    not normalise the next; after an intended level change, call
    :meth:`reset_baseline`. Anomalies inside an acknowledged
    :class:`SuppressionWindow` are counted but not routed.
    """

    def __init__(
        self,
        router: AlertRouter | None = None,
        dispatch: Callable[[RouteDestination, Alert], None] | None = None,
        key_fields: list[str] | None = None,
        distribution_fields: list[str] | None = None,
        window: int = 24,
        min_history: int = 6,
        z_threshold: float = 4.0,
        relative_std_floor: float = 0.05,
        null_rate_std_floor: float = 0.01,
    ) -> None:
        self.router = router
        self.dispatch = dispatch
        self.key_fields = key_fields if key_fields is not None else ["amount"]
        self.distribution_fields = distribution_fields or []
        self.window = window
        self.min_history = min_history
        self.z_threshold = z_threshold
        self.relative_std_floor = relative_std_floor
        self.null_rate_std_floor = null_rate_std_floor
        self._history: dict[tuple[str, str], deque[float]] = {}
        self._suppressions: list[SuppressionWindow] = []

    def acknowledge(
        self,
        source: str,
        start: datetime.datetime,
        end: datetime.datetime,
        reason: str,
        metrics: list[str] | None = None,
    ) -> SuppressionWindow:
        """Register an expected change (e.g. planned maintenance) for *source*."""
        if end <= start:
            raise ValueError("suppression window end must be after start")
        suppression = SuppressionWindow(
            source, start, end, reason, frozenset(metrics) if metrics is not None else None
        )
        self._suppressions.append(suppression)
        logger.info("Acknowledged stream-quality window for %s: %s", source, reason)
        return suppression

    def reset_baseline(self, source: str) -> None:
        """Forget *source*'s rolling history, e.g. after an intended level shift."""
        for key in [k for k in self._history if k[0] == source]:
            del self._history[key]

    def compute_metrics(self, df: pd.DataFrame) -> dict[str, float]:
        """Stream-level metrics for one ingestion batch."""
        metrics = {"volume": float(len(df))}
        for col in self.key_fields:
            if col in df.columns and len(df):
                metrics[f"null_rate:{col}"] = float(df[col].isna().mean())
        for col in self.distribution_fields:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(values):
                    metrics[f"mean:{col}"] = float(values.mean())
        return metrics

    def observe_batch(
        self,
        source: str,
        data: pd.DataFrame | list[dict[str, Any]],
        at: datetime.datetime | None = None,
    ) -> list[StreamAnomaly]:
        """Score one batch from *source*, route unsuppressed anomalies, return all."""
        df = pd.DataFrame(data) if isinstance(data, list) else data
        at = at or datetime.datetime.now(tz=datetime.UTC)
        self._suppressions = [s for s in self._suppressions if s.end > at]

        anomalies = []
        for metric, value in self.compute_metrics(df).items():
            anomaly = self._check(source, metric, value, at)
            if anomaly is not None:
                anomalies.append(anomaly)
                self._emit(anomaly)
        return anomalies

    def _check(
        self, source: str, metric: str, value: float, at: datetime.datetime
    ) -> StreamAnomaly | None:
        history = self._history.setdefault((source, metric), deque(maxlen=self.window))
        if stream_metric_value is not None:
            stream_metric_value.labels(source=source, metric=metric).set(value)

        if len(history) < self.min_history:
            history.append(value)
            return None

        mean = float(np.mean(history))
        floor = (
            self.null_rate_std_floor
            if metric.startswith("null_rate:")
            else self.relative_std_floor * abs(mean)
        )
        std = max(float(np.std(history)), floor, 1e-9)
        z_score = (value - mean) / std
        if stream_metric_zscore is not None:
            stream_metric_zscore.labels(source=source, metric=metric).set(z_score)

        if abs(z_score) <= self.z_threshold:
            history.append(value)
            return None

        suppressed_by = next((s for s in self._suppressions if s.covers(source, metric, at)), None)
        return StreamAnomaly(source, metric, value, mean, std, z_score, at, suppressed_by)

    def _emit(self, anomaly: StreamAnomaly) -> None:
        suppressed = anomaly.suppressed_by is not None
        if stream_anomalies_total is not None:
            stream_anomalies_total.labels(
                source=anomaly.source,
                metric=anomaly.metric,
                direction=anomaly.direction,
                suppressed=str(suppressed).lower(),
            ).inc()
        if suppressed:
            logger.info(
                "Suppressed stream anomaly for %s/%s (%s)",
                anomaly.source,
                anomaly.metric,
                anomaly.suppressed_by.reason,  # type: ignore[union-attr]
            )
            return

        alert = anomaly.to_alert()
        logger.warning("Stream-quality anomaly: %s", alert["message"])
        if self.router is None:
            return
        for destination in self.router.route(alert):
            if self.dispatch is not None:
                self.dispatch(destination, alert)
