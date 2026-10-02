"""CUSUM (Cumulative Sum) control chart for online change-point detection (issue #289).

Detects sustained upward or downward shifts in a streaming metric (e.g. the
LedgerLens risk score stream) in O(1) time and O(1) space per update.

Theory
------
The two-sided Page-CUSUM statistic maintains:

    S_high[n] = max(0, S_high[n-1] + x_n - (μ₀ + k))
    S_low[n]  = max(0, S_low[n-1]  - x_n + (μ₀ - k))

An alarm fires when either statistic exceeds h (decision threshold). After
acknowledgement both statistics are reset to zero.

Parameter guidance (in-control ARL ≈ 500, out-of-control ARL ≈ 10 for a
10-point shift with σ ≈ 15):
    k = 5.0   (half the minimum detectable shift in score units)
    h = 25.0

Public API
----------
CUSUMDetector
    .update(value)   -> bool   (True = alarm just triggered)
    .is_alarm        -> bool
    .acknowledge()

CUSUMFeedbackStore
    .record_feedback(alert_id, outcome)   # outcome: "true_positive" | "false_positive"
    .recalibrate(detector)               # adjust detector.h toward target FP rate

Recalibration methodology
--------------------------
When an operator marks a past alarm as a true positive (TP) or false positive
(FP), ``CUSUMFeedbackStore`` stores the outcome.  Once at least
``min_feedback_samples`` outcomes have accumulated, ``recalibrate()`` estimates
the empirical false-positive rate and nudges the decision threshold ``h``
upward (if FP rate is above target) or downward (if below target):

    h_new = h * (1 + α * (fp_rate - fp_rate_target))

where α is a configurable learning rate (default 0.1).  A safety bound
clamps h_new to the range [h_min, h_max] (defaults [1.0, 100.0]) so the
threshold can never drift to an operationally insane value.

After each recalibration the feedback store is cleared (the new threshold
becomes the new baseline) and the event is logged with the old/new threshold,
the sample size, and the empirical FP rate.

Documented cadence: recalibration is triggered explicitly by the operator
(or a scheduled job) by calling ``CUSUMFeedbackStore.recalibrate()``.  It
never fires automatically inside ``CUSUMDetector.update()`` — keeping the
detection hot-path free of side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from prometheus_client import Gauge

from config import config

logger = logging.getLogger(__name__)

_cusum_alarm_gauge = Gauge(
    "ledgerlens_cusum_alarm",
    "CUSUM change-point alarm (1=alarm, 0=in-control)",
    ["metric"],
)

try:
    from prometheus_client import Counter

    _cusum_recalibration_total = Counter(
        "ledgerlens_cusum_recalibrations_total",
        "Total number of CUSUM threshold recalibrations performed",
        ["metric"],
    )
    _cusum_threshold_gauge = Gauge(
        "ledgerlens_cusum_decision_threshold",
        "Current CUSUM decision threshold h (after recalibrations)",
        ["metric"],
    )
except Exception:  # pragma: no cover
    _cusum_recalibration_total = None  # type: ignore[assignment]
    _cusum_threshold_gauge = None  # type: ignore[assignment]

_REDIS_KEY_PREFIX = "ledgerlens:cusum:"

FeedbackOutcome = Literal["true_positive", "false_positive"]


@dataclass(frozen=True)
class PerPairAlarm:
    """Pair-level CUSUM alarm payload (issue #971).

    Carries the pair identifier, the metric that degraded, the direction of the
    shift and the magnitude of the CUSUM statistic that crossed the threshold so
    downstream alert routing can include pair-level context.
    """

    pair: str
    metric: str
    direction: str  # "high" or "low"
    magnitude: float
    threshold: float




class CUSUMDetector:
    """Two-sided CUSUM control chart with optional Redis state persistence.

    Args:
        metric_name: Logical name used for Prometheus labels and Redis key.
        target_mean: Expected in-control mean (μ₀).
        allowable_slack: Allowable slack k (typically half the minimum shift).
        decision_threshold: Alarm threshold h.
        redis_client: Optional ``redis.Redis`` instance for alarm persistence
            across worker restarts. When ``None`` alarm state is in-memory only.
    """

    def __init__(
        self,
        metric_name: str = "risk_score",
        target_mean: float | None = None,
        allowable_slack: float | None = None,
        decision_threshold: float | None = None,
        redis_client=None,
    ) -> None:
        self.metric_name = metric_name
        self.mu0: float = target_mean if target_mean is not None else config.CUSUM_TARGET_MEAN
        self.k: float = (
            allowable_slack if allowable_slack is not None else config.CUSUM_ALLOWABLE_SLACK
        )
        self.h: float = (
            decision_threshold
            if decision_threshold is not None
            else config.CUSUM_DECISION_THRESHOLD
        )

        if self.k < 0:
            raise ValueError("allowable_slack (k) must be >= 0")
        if self.h <= 0:
            raise ValueError("decision_threshold (h) must be > 0")

        self._redis = redis_client
        self._s_high: float = 0.0
        self._s_low: float = 0.0
        self._alarm: bool = False

        # Restore alarm state from Redis if available
        if self._redis is not None:
            try:
                val = self._redis.get(f"{_REDIS_KEY_PREFIX}{metric_name}:alarm")
                self._alarm = val == b"1"
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, value: float) -> bool:
        """Ingest one observation; return True if alarm newly triggered."""
        if self._alarm:
            return False  # already alarming — call acknowledge() first

        self._s_high = max(0.0, self._s_high + value - (self.mu0 + self.k))
        self._s_low = max(0.0, self._s_low - value + (self.mu0 - self.k))

        if self._s_high >= self.h or self._s_low >= self.h:
            self._alarm = True
            _cusum_alarm_gauge.labels(metric=self.metric_name).set(1)
            logger.warning(
                "CUSUM alarm: metric=%s s_high=%.2f s_low=%.2f h=%.2f",
                self.metric_name,
                self._s_high,
                self._s_low,
                self.h,
            )
            self._persist_alarm(True)
            self._reset_statistics()
            return True

        return False

    def _reset_statistics(self) -> None:
        """Reset accumulated CUSUM statistics to zero after an alarm."""
        self._s_high = 0.0
        self._s_low = 0.0

    def acknowledge(self) -> None:
        """Reset CUSUM statistics and clear the alarm."""
        self._s_high = 0.0
        self._s_low = 0.0
        self._alarm = False
        _cusum_alarm_gauge.labels(metric=self.metric_name).set(0)
        self._persist_alarm(False)
        logger.info("CUSUM alarm acknowledged and reset: metric=%s", self.metric_name)

    @property
    def is_alarm(self) -> bool:
        return self._alarm

    @property
    def s_high(self) -> float:
        return self._s_high

    @property
    def s_low(self) -> float:
        return self._s_low

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _persist_alarm(self, state: bool) -> None:
        if self._redis is None:
            return
        try:
            key = f"{_REDIS_KEY_PREFIX}{self.metric_name}:alarm"
            self._redis.set(key, "1" if state else "0")
        except Exception as exc:
            logger.warning("Failed to persist CUSUM alarm state to Redis: %s", exc)


# ---------------------------------------------------------------------------
# Feedback store and recalibration
# ---------------------------------------------------------------------------


@dataclass
class CUSUMFeedbackRecord:
    """A single operator feedback event tied to an alarm."""

    alert_id: str
    outcome: FeedbackOutcome  # "true_positive" or "false_positive"


@dataclass
class RecalibrationResult:
    """Summary of a recalibration event."""

    metric_name: str
    old_threshold: float
    new_threshold: float
    fp_rate: float
    fp_rate_target: float
    n_samples: int
    clamped: bool  # True if the safety bound was hit


class CUSUMFeedbackStore:
    """Accumulates operator feedback on CUSUM alarms and recalibrates thresholds.

    This class is deliberately separate from :class:`CUSUMDetector` to keep
    the real-time detection hot-path free of stateful side effects.  Operators
    (or an automated scheduler) call :meth:`record_feedback` for each resolved
    alarm and :meth:`recalibrate` periodically to apply accumulated feedback.

    Args:
        metric_name:
            Must match the ``metric_name`` of the :class:`CUSUMDetector` being
            managed.
        fp_rate_target:
            Desired false-positive rate (e.g. 0.05 = 5 %).
        min_feedback_samples:
            Minimum number of feedback outcomes required before recalibration
            is permitted.  Prevents premature adaptation on tiny samples.
        learning_rate:
            How aggressively to move the threshold.  Default 0.1 (10 %).
            At α=0.1 and an FP rate 10 pp above target, h grows by ~1 %.
        h_min:
            Safety lower bound for the decision threshold h.  Recalibration
            will never push h below this value.
        h_max:
            Safety upper bound for the decision threshold h.  Recalibration
            will never push h above this value.

    Recalibration methodology (documented for operators)
    -------------------------------------------------------
    1. Operator marks each resolved alarm as ``"true_positive"`` (genuine
       change-point) or ``"false_positive"`` (noise / benign variation) via
       :meth:`record_feedback`.
    2. When the feedback count reaches ``min_feedback_samples``, calling
       :meth:`recalibrate(detector)`` computes:

           fp_rate = n_fp / (n_tp + n_fp)
           delta   = α * (fp_rate - fp_rate_target)
           h_new   = h_current * (1 + delta)
           h_new   = clamp(h_new, h_min, h_max)

    3. The detector's ``h`` attribute is updated in-place.
    4. The feedback store is cleared so the next calibration epoch starts fresh.
    5. All recalibration events are logged (old h, new h, fp_rate, n_samples)
       and a Prometheus counter is incremented.

    Safety guarantee
    ----------------
    ``h_new`` is always clamped to [h_min, h_max].  Tests can verify this by
    passing extreme feedback streams and asserting the threshold stays within
    bounds.
    """

    def __init__(
        self,
        metric_name: str = "risk_score",
        fp_rate_target: float = 0.05,
        min_feedback_samples: int = 10,
        learning_rate: float = 0.1,
        h_min: float = 1.0,
        h_max: float = 100.0,
    ) -> None:
        if not (0.0 < fp_rate_target < 1.0):
            raise ValueError("fp_rate_target must be in (0, 1)")
        if min_feedback_samples < 1:
            raise ValueError("min_feedback_samples must be >= 1")
        if h_min <= 0:
            raise ValueError("h_min must be > 0")
        if h_max <= h_min:
            raise ValueError("h_max must be > h_min")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")

        self.metric_name = metric_name
        self.fp_rate_target = fp_rate_target
        self.min_feedback_samples = min_feedback_samples
        self.learning_rate = learning_rate
        self.h_min = h_min
        self.h_max = h_max

        self._records: list[CUSUMFeedbackRecord] = []

    # ------------------------------------------------------------------
    # Feedback recording
    # ------------------------------------------------------------------

    def record_feedback(self, alert_id: str, outcome: FeedbackOutcome) -> None:
        """Record operator feedback for a past alarm.

        Args:
            alert_id: Arbitrary string uniquely identifying the alarm (e.g.
                a timestamp, a Prometheus alert fingerprint).
            outcome: ``"true_positive"`` if the alarm was a genuine change-point,
                ``"false_positive"`` if it was noise.

        Duplicate ``alert_id`` values are allowed — the same alarm can be
        re-labelled if an operator changes their mind.  Only the last label
        counts (earlier records for the same ID are replaced).
        """
        if outcome not in ("true_positive", "false_positive"):
            raise ValueError(
                f"outcome must be 'true_positive' or 'false_positive', got {outcome!r}"
            )
        # Replace earlier entry for same alert_id, if any
        self._records = [r for r in self._records if r.alert_id != alert_id]
        self._records.append(CUSUMFeedbackRecord(alert_id=alert_id, outcome=outcome))

    def pending_count(self) -> int:
        """Return the number of pending feedback records."""
        return len(self._records)

    def fp_rate(self) -> float | None:
        """Compute current empirical false-positive rate.

        Returns ``None`` when there are no records yet.
        """
        if not self._records:
            return None
        n_fp = sum(1 for r in self._records if r.outcome == "false_positive")
        return n_fp / len(self._records)

    # ------------------------------------------------------------------
    # Recalibration
    # ------------------------------------------------------------------

    def recalibrate(self, detector: CUSUMDetector) -> RecalibrationResult | None:
        """Recalibrate *detector*'s decision threshold using accumulated feedback.

        Returns ``None`` (and logs a debug message) if there are fewer than
        ``min_feedback_samples`` records.  Returns a :class:`RecalibrationResult`
        describing the change otherwise.

        The feedback store is cleared after a successful recalibration so the
        next calibration epoch starts from zero.

        Safety bound: ``detector.h`` is clamped to [h_min, h_max].
        """
        n = len(self._records)
        if n < self.min_feedback_samples:
            logger.debug(
                "CUSUM recalibration deferred: only %d/%d feedback samples for metric=%s",
                n,
                self.min_feedback_samples,
                self.metric_name,
            )
            return None

        n_fp = sum(1 for r in self._records if r.outcome == "false_positive")
        fp_rate = n_fp / n

        old_h = detector.h
        delta = self.learning_rate * (fp_rate - self.fp_rate_target)
        h_new_raw = old_h * (1.0 + delta)

        # Safety clamp
        clamped = False
        if h_new_raw < self.h_min:
            h_new = self.h_min
            clamped = True
        elif h_new_raw > self.h_max:
            h_new = self.h_max
            clamped = True
        else:
            h_new = h_new_raw

        detector.h = h_new

        result = RecalibrationResult(
            metric_name=self.metric_name,
            old_threshold=old_h,
            new_threshold=h_new,
            fp_rate=fp_rate,
            fp_rate_target=self.fp_rate_target,
            n_samples=n,
            clamped=clamped,
        )

        logger.info(
            "CUSUM recalibration: metric=%s h %.3f→%.3f fp_rate=%.3f target=%.3f "
            "n=%d clamped=%s",
            self.metric_name,
            old_h,
            h_new,
            fp_rate,
            self.fp_rate_target,
            n,
            clamped,
        )

        # Prometheus instrumentation
        if _cusum_recalibration_total is not None:
            _cusum_recalibration_total.labels(metric=self.metric_name).inc()
        if _cusum_threshold_gauge is not None:
            _cusum_threshold_gauge.labels(metric=self.metric_name).set(h_new)

        # Clear the feedback store for the next epoch
        self._records.clear()

        return result

    def clear(self) -> None:
        """Discard all accumulated feedback without recalibrating."""
        self._records.clear()


class PerPairCUSUMTracker:
    """Per-pair rolling CUSUM tracking for model-degradation detection (#971).

    Maintains one :class:`CUSUMDetector` per trading pair, each labelled with a
    pair-scoped metric name so Prometheus series and Redis alarm state remain
    isolated. This lets a distribution shift confined to a single pair be
    detected even when the aggregate metric stays flat.

    Args:
        metric_name: Base metric name (e.g. ``"model_auc"``).
        target_mean: Expected in-control mean (μ₀) shared by all pairs.
        allowable_slack: Allowable slack k shared by all pairs.
        decision_threshold: Alarm threshold h shared by all pairs.
        redis_client: Optional ``redis.Redis`` instance for alarm persistence.
    """

    def __init__(
        self,
        metric_name: str = "model_metric",
        target_mean: float | None = None,
        allowable_slack: float | None = None,
        decision_threshold: float | None = None,
        redis_client=None,
    ) -> None:
        self.metric_name = metric_name
        self._target_mean = target_mean
        self._allowable_slack = allowable_slack
        self._decision_threshold = decision_threshold
        self._redis = redis_client
        self._detectors: dict[str, CUSUMDetector] = {}

    def _detector_for(self, pair: str) -> CUSUMDetector:
        detector = self._detectors.get(pair)
        if detector is None:
            detector = CUSUMDetector(
                metric_name=f"{self.metric_name}:{pair}",
                target_mean=self._target_mean,
                allowable_slack=self._allowable_slack,
                decision_threshold=self._decision_threshold,
                redis_client=self._redis,
            )
            self._detectors[pair] = detector
        return detector

    def update(self, pair: str, value: float) -> PerPairAlarm | None:
        """Ingest one observation for ``pair``.

        Returns a :class:`PerPairAlarm` when the pair's CUSUM statistic crosses
        the decision threshold, otherwise ``None``. Stable pairs return ``None``
        and never affect other pairs' statistics.
        """
        detector = self._detector_for(pair)
        # Capture the statistic that is about to cross before update() resets it.
        prev_high = detector.s_high
        prev_low = detector.s_low
        triggered = detector.update(value)
        if not triggered:
            return None

        # update() resets statistics on alarm, so recompute the crossing value
        # from the pre-update statistic plus the current observation.
        high_after = max(0.0, prev_high + value - (detector.mu0 + detector.k))
        low_after = max(0.0, prev_low - value + (detector.mu0 - detector.k))
        if high_after >= low_after:
            direction, magnitude = "high", high_after
        else:
            direction, magnitude = "low", low_after

        alarm = PerPairAlarm(
            pair=pair,
            metric=self.metric_name,
            direction=direction,
            magnitude=magnitude,
            threshold=detector.h,
        )
        logger.warning(
            "Per-pair CUSUM alarm: pair=%s metric=%s direction=%s magnitude=%.2f h=%.2f",
            pair,
            self.metric_name,
            direction,
            magnitude,
            detector.h,
        )
        return alarm

    def is_alarm(self, pair: str) -> bool:
        detector = self._detectors.get(pair)
        return detector.is_alarm if detector is not None else False

    def acknowledge(self, pair: str) -> None:
        detector = self._detectors.get(pair)
        if detector is not None:
            detector.acknowledge()

    @property
    def pairs(self) -> list[str]:
        return list(self._detectors.keys())
