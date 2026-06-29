"""Per-asset-pair Prometheus metrics for LedgerLens (issue #276).

Defines the three canonical per-pair metrics emitted by the scoring pipeline:
  - ledgerlens_score_duration_seconds  (Histogram)
  - ledgerlens_benford_computation_total  (Counter)
  - ledgerlens_risk_score_distribution  (Histogram)

All metrics carry an ``asset_pair`` label using the canonical format
``CODE:ISSUER/CODE:ISSUER`` sorted alphabetically.  Labels never include
wallet addresses — only aggregate pair identifiers.

Usage::

    from detection.per_pair_metrics import record_scoring_duration, record_benford_computation, record_risk_score

    with record_scoring_duration("USDC:GA.../XLM:native"):
        score = scorer.score(features)
    record_benford_computation(asset_pair, status="ok")
    record_risk_score(asset_pair, score["score"])
"""

from __future__ import annotations

import contextlib
import time

_metrics_available = False
_score_duration: object = None
_benford_computation: object = None
_risk_score_dist: object = None

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
    _score_duration = ledgerlens_score_duration_seconds
    _benford_computation = ledgerlens_benford_computation_total
    _risk_score_dist = ledgerlens_risk_score_distribution
    _metrics_available = True
except Exception:
    ledgerlens_score_duration_seconds = None  # type: ignore[assignment]
    ledgerlens_benford_computation_total = None  # type: ignore[assignment]
    ledgerlens_risk_score_distribution = None  # type: ignore[assignment]


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
