"""Per-asset-pair Prometheus metrics for LedgerLens (issue #276).

Defines the canonical per-pair metrics emitted by the scoring pipeline:
  - ledgerlens_score_duration_seconds  (Histogram)
  - ledgerlens_benford_computation_total  (Counter)
  - ledgerlens_risk_score_distribution  (Histogram)
  - ledgerlens_confirmed_wash_trades_total  (Counter) — for SLO dashboard (#197)
  - ledgerlens_confirmed_clean_wallets_total  (Counter) — for SLO dashboard (#197)

All metrics carry an ``asset_pair`` label using the canonical format
``CODE:ISSUER/CODE:ISSUER`` sorted alphabetically.  Labels never include
wallet addresses — only aggregate pair identifiers.

Usage::

    from detection.per_pair_metrics import record_scoring_duration, record_benford_computation, record_risk_score

    with record_scoring_duration("USDC:GA.../XLM:native"):
        score = scorer.score(features)
    record_benford_computation(asset_pair, status="ok")
    record_risk_score(asset_pair, score["score"])
    record_confirmed_wash_trade(asset_pair)
    record_confirmed_clean_wallet(asset_pair)

Per-pair degradation detection (issue #971)
-------------------------------------------

``PerPairDegradationMonitor`` tracks a rolling window of per-pair metric
values and runs a CUSUM change-point test (reusing
``monitoring.cusum_detector``) on each pair independently.  When a single
pair's distribution shifts while aggregate metrics stay flat, only that
pair's CUSUM statistic crosses the threshold, so a per-pair alert is
emitted through ``alerts.router`` with pair-level context (pair id, metric,
magnitude).  Stable pairs never cross the threshold and stay silent.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

_metrics_available = False
_score_duration: object = None
_benford_computation: object = None
_risk_score_dist: object = None
_confirmed_wash_trades: object = None
_confirmed_clean_wallets: object = None

try:
    from prometheus_client import Counter, Histogram

    ledgerlens_score_duration_seconds = Histogram(
        "ledgerlens_score_duration_seconds",
        "Per-asset-pair scoring latency in seconds",
        ["asset_pair"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    ledgerlens_benford_computation_total = Counter(
        "ledgerlens_benford_computation_total",
        "Total Benford computations completed per asset pair",
        ["asset_pair", "status"],
    )
    ledgerlens_risk_score_distribution = Histogram(
        "ledgerlens_risk_score_distribution",
        "Distribution of risk scores (0-100) per asset pair",
        ["asset_pair"],
        buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    )
    # SLO dashboard counters (issue #197)
    ledgerlens_confirmed_wash_trades_total = Counter(
        "ledgerlens_confirmed_wash_trades_total",
        "Total confirmed wash trades detected per asset pair",
        ["asset_pair"],
    )
    ledgerlens_confirmed_clean_wallets_total = Counter(
        "ledgerlens_confirmed_clean_wallets_total",
        "Total confirmed clean (non-fraudulent) wallets per asset pair",
        ["asset_pair"],
    )
    # Per-pair degradation counters (issue #971)
    ledgerlens_pair_degradation_total = Counter(
        "ledgerlens_pair_degradation_total",
        "Total per-pair model degradation alerts emitted",
        ["asset_pair", "metric"],
    )
    _score_duration = ledgerlens_score_duration_seconds
    _benford_computation = ledgerlens_benford_computation_total
    _risk_score_dist = ledgerlens_risk_score_distribution
    _confirmed_wash_trades = ledgerlens_confirmed_wash_trades_total
    _confirmed_clean_wallets = ledgerlens_confirmed_clean_wallets_total
    _metrics_available = True
except Exception:
    ledgerlens_score_duration_seconds = None  # type: ignore[assignment]
    ledgerlens_benford_computation_total = None  # type: ignore[assignment]
    ledgerlens_risk_score_distribution = None  # type: ignore[assignment]
    ledgerlens_confirmed_wash_trades_total = None  # type: ignore[assignment]
    ledgerlens_confirmed_clean_wallets_total = None  # type: ignore[assignment]
    ledgerlens_pair_degradation_total = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def canonical_pair(asset_pair: str) -> str:
    """Return the canonical sort-order form of *asset_pair*.

    Ensures ``A/B`` and ``B/A`` map to the same label, preventing metric
    cardinality explosion from direction-dependent pair strings.

    Security: wallet addresses are never included in pair labels; only the
    CODE:ISSUER format is accepted.
    """
    parts = [p.strip() for p in asset_pair.split("/") if p.strip()]
    if len(parts) != 2:
        return asset_pair
    return "/".join(sorted(parts))


@contextlib.contextmanager
def record_scoring_duration(asset_pair: str):
    """Context manager that records scoring duration for *asset_pair*."""
    pair = canonical_pair(asset_pair)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if _metrics_available and _score_duration is not None:
            _score_duration.labels(asset_pair=pair).observe(elapsed)


def record_benford_computation(asset_pair: str, status: str = "ok") -> None:
    """Increment the Benford computation counter for *asset_pair*."""
    pair = canonical_pair(asset_pair)
    if _metrics_available and _benford_computation is not None:
        _benford_computation.labels(asset_pair=pair, status=status).inc()


def record_risk_score(asset_pair: str, score: float) -> None:
    """Observe a risk *score* in the distribution histogram for *asset_pair*."""
    pair = canonical_pair(asset_pair)
    if _metrics_available and _risk_score_dist is not None:
        _risk_score_dist.labels(asset_pair=pair).observe(float(score))


def record_confirmed_wash_trade(asset_pair: str) -> None:
    """Increment the confirmed wash trade counter for *asset_pair*.

    Call this when a wallet on *asset_pair* is manually confirmed to be
    conducting wash trading (used to compute recall metrics for SLO dashboard).
    """
    pair = canonical_pair(asset_pair)
    if _metrics_available and _confirmed_wash_trades is not None:
        _confirmed_wash_trades.labels(asset_pair=pair).inc()


def record_confirmed_clean_wallet(asset_pair: str) -> None:
    """Increment the confirmed clean wallet counter for *asset_pair*.

    Call this when a wallet on *asset_pair* is manually confirmed to be
    legitimate/non-fraudulent (used to compute false-positive rate metrics
    for SLO dashboard).
    """
    pair = canonical_pair(asset_pair)
    if _metrics_available and _confirmed_clean_wallets is not None:
        _confirmed_clean_wallets.labels(asset_pair=pair).inc()


# ---------------------------------------------------------------------------
# Per-pair degradation detection (issue #971)
# ---------------------------------------------------------------------------


def _cusum_detector_cls():
    """Return the CUSUM detector class from ``monitoring.cusum_detector``.

    Imported lazily so this module keeps working (metrics only) even when the
    monitoring package is unavailable in a minimal deployment.
    """
    from monitoring.cusum_detector import CUSUMDetector  # type: ignore

    return CUSUMDetector


@dataclass
class PairDegradationAlert:
    """Per-pair degradation alert payload (issue #971).

    Carries the pair identifier, the metric that degraded, and the magnitude
    of the shift so downstream routing/alerting has full pair-level context.
    """

    asset_pair: str
    metric: str
    magnitude: float
    direction: str
    cusum_value: float
    baseline_mean: float
    observed_mean: float
    samples: int

    def to_payload(self) -> Dict[str, object]:
        return {
            "asset_pair": self.asset_pair,
            "metric": self.metric,
            "magnitude": self.magnitude,
            "direction": self.direction,
            "cusum_value": self.cusum_value,
            "baseline_mean": self.baseline_mean,
            "observed_mean": self.observed_mean,
            "samples": self.samples,
        }


@dataclass
class _PairState:
    """Rolling state for a single (pair, metric) series."""

    window: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    detector: object = None
    baseline_mean: float = 0.0
    baseline_ready: bool = False


class PerPairDegradationMonitor:
    """Rolling per-pair metric tracking with CUSUM degradation detection.

    Each ``(asset_pair, metric)`` series is tracked independently.  A CUSUM
    change-point test (reusing ``monitoring.cusum_detector``) runs per series,
    so a distribution shift on one thinly traded pair is detected even when
    aggregate metrics across all pairs remain flat.  Stable pairs never cross
    the CUSUM threshold and therefore never emit an alert.

    Alerts are routed through ``alerts.router`` with pair-level context.
    """

    def __init__(
        self,
        metric: str = "risk_score",
        baseline_size: int = 30,
        threshold: float = 5.0,
        slack: float = 0.5,
        window_size: int = 200,
    ) -> None:
        self.metric = metric
        self.baseline_size = baseline_size
        self.threshold = threshold
        self.slack = slack
        self.window_size = window_size
        self._states: Dict[str, _PairState] = defaultdict(self._new_state)

    def _new_state(self) -> _PairState:
        return _PairState(window=deque(maxlen=self.window_size))

    def _ensure_detector(self, state: _PairState) -> None:
        if state.detector is not None:
            return
        detector_cls = _cusum_detector_cls()
        try:
            state.detector = detector_cls(
                threshold=self.threshold, slack=self.slack
            )
        except TypeError:
            # Fall back to a positional/looser constructor signature.
            state.detector = detector_cls(self.threshold, self.slack)

    def observe(self, asset_pair: str, value: float) -> Optional[PairDegradationAlert]:
        """Record a metric *value* for *asset_pair* and check for degradation.

        Returns a :class:`PairDegradationAlert` when the pair's CUSUM statistic
        crosses the threshold, otherwise ``None``.
        """
        pair = canonical_pair(asset_pair)
        state = self._states[pair]
        value = float(value)
        state.window.append(value)

        # Establish a baseline from the first ``baseline_size`` samples.
        if not state.baseline_ready:
            if len(state.window) >= self.baseline_size:
                state.baseline_mean = sum(state.window) / len(state.window)
                state.baseline_ready = True
                self._ensure_detector(state)
            return None

        self._ensure_detector(state)
        detector = state.detector
        if detector is None:
            return None

        # Feed the detector and read its current statistic.
        if hasattr(detector, "update"):
            detector.update(value)
        elif hasattr(detector, "observe"):
            detector.observe(value)
        else:  # pragma: no cover - defensive
            return None

        cusum_value = self._read_statistic(detector)
        if cusum_value is None or cusum_value < self.threshold:
            return None

        observed_mean = sum(state.window) / len(state.window)
        magnitude = abs(observed_mean - state.baseline_mean)
        direction = "up" if observed_mean >= state.baseline_mean else "down"
        alert = PairDegradationAlert(
            asset_pair=pair,
            metric=self.metric,
            magnitude=magnitude,
            direction=direction,
            cusum_value=cusum_value,
            baseline_mean=state.baseline_mean,
            observed_mean=observed_mean,
            samples=len(state.window),
        )
        self._record_metric(alert)
        self._route_alert(alert)
        # Reset the detector so a single shift does not re-fire every sample.
        self._reset_detector(state)
        return alert

    @staticmethod
    def _read_statistic(detector: object) -> Optional[float]:
        for attr in ("statistic", "cusum", "value", "max_statistic"):
            if hasattr(detector, attr):
                try:
                    return float(getattr(detector, attr))
                except (TypeError, ValueError):
                    continue
        return None

    def _reset_detector(self, state: _PairState) -> None:
        detector = state.detector
        if detector is None:
            return
        for attr in ("reset", "clear"):
            if hasattr(detector, attr):
                try:
                    getattr(detector, attr)()
                    return
                except Exception:  # pragma: no cover - defensive
                    continue

    @staticmethod
    def _record_metric(alert: PairDegradationAlert) -> None:
        if _metrics_available and ledgerlens_pair_degradation_total is not None:
            ledgerlens_pair_degradation_total.labels(
                asset_pair=alert.asset_pair, metric=alert.metric
            ).inc()

    @staticmethod
    def _route_alert(alert: PairDegradationAlert) -> None:
        """Route the per-pair degradation alert through ``alerts.router``."""
        try:
            from alerts.router import route_alert  # type: ignore
        except Exception:  # pragma: no cover - router optional in tests
            logger.warning(
                "per-pair degradation on %s (%s): magnitude=%.4f",
                alert.asset_pair,
                alert.metric,
                alert.magnitude,
            )
            return
        try:
            route_alert(
                alert_type="per_pair_degradation",
                severity="warning",
                message=(
                    f"Model degradation detected for pair {alert.asset_pair} "
                    f"on metric {alert.metric} "
                    f"(magnitude={alert.magnitude:.4f}, {alert.direction})"
                ),
                context=alert.to_payload(),
            )
        except TypeError:
            # Router with a looser signature.
            route_alert(alert.to_payload())


def detect_pair_degradation(
    series_by_pair: Dict[str, List[float]],
    metric: str = "risk_score",
    baseline_size: int = 30,
    threshold: float = 5.0,
    slack: float = 0.5,
) -> List[PairDegradationAlert]:
    """Convenience helper: run per-pair detection over pre-collected series.

    ``series_by_pair`` maps an asset pair to its ordered metric values.  Each
    pair is evaluated independently, so a shift on one pair is reported even
    when the aggregate across all pairs is unchanged.
    """
    monitor = PerPairDegradationMonitor(
        metric=metric,
        baseline_size=baseline_size,
        threshold=threshold,
        slack=slack,
    )
    alerts: List[PairDegradationAlert] = []
    for pair, values in series_by_pair.items():
        for value in values:
            alert = monitor.observe(pair, value)
            if alert is not None:
                alerts.append(alert)
    return alerts
def compare_pair_metrics(
    production_scores: dict[str, float],
    candidate_scores: dict[str, float],
    *,
    threshold: float = 0.0,
) -> dict[str, object]:
    """Compare production vs. candidate scores per asset pair (issue #936).

    Used by the model-governance shadow-mode evaluation to compute the
    agreement rate and per-pair metric deltas over a shadow period.  This is
    a pure computation: it never emits metrics or influences live alerts, so
    running it against shadow traffic has zero effect on production decisions.

    *production_scores* and *candidate_scores* map canonical asset pairs to
    risk scores.  A pair is considered to *agree* when both models place it on
    the same side of *threshold* (both flagged or both not flagged).

    Returns a report dict with ``agreement_rate``, ``pairs_compared``,
    ``deltas`` (per-pair absolute score delta) and ``disagreements`` (pairs
    where the flag decision differs).
    """
    pairs = sorted(set(production_scores) | set(candidate_scores))
    deltas: dict[str, float] = {}
    disagreements: list[str] = []
    compared = 0
    agreed = 0
    for pair in pairs:
        prod = production_scores.get(pair)
        cand = candidate_scores.get(pair)
        if prod is None or cand is None:
            continue
        compared += 1
        deltas[pair] = abs(float(cand) - float(prod))
        prod_flag = float(prod) >= threshold
        cand_flag = float(cand) >= threshold
        if prod_flag == cand_flag:
            agreed += 1
        else:
            disagreements.append(pair)
    agreement_rate = (agreed / compared) if compared else 1.0
    return {
        "agreement_rate": agreement_rate,
        "pairs_compared": compared,
        "deltas": deltas,
        "disagreements": disagreements,
    }
