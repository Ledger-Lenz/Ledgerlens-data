"""Correlated-signal alert deduplication.

Groups alerts raised by independent detectors (Benford engine, GNN, Isolation
Forest, ...) for the same wallet/asset-pair within a short window into a
single enriched alert, so analysts see one notification per underlying event
instead of one per detector.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from prometheus_client import Counter

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

ledgerlens_alerts_deduplicated_total = Counter(
    "ledgerlens_alerts_deduplicated_total",
    "Total number of raw alerts folded into an existing group instead of "
    "being emitted as a standalone alert",
)


class _Group:
    __slots__ = ("wallet_address", "asset_pair", "detectors", "risk_score", "evidence",
                  "detected_at", "last_seen", "raw_count")

    def __init__(self, alert: dict[str, Any]):
        self.wallet_address = alert["wallet_address"]
        self.asset_pair = alert["asset_pair"]
        self.detectors: set[str] = {alert["detector"]}
        self.risk_score: float = alert["risk_score"]
        self.evidence: dict[str, Any] = dict(alert.get("evidence") or {})
        self.detected_at: float = alert["detected_at"]
        self.last_seen: float = alert["detected_at"]
        self.raw_count = 1

    def absorb(self, alert: dict[str, Any]) -> None:
        self.detectors.add(alert["detector"])
        self.risk_score = max(self.risk_score, alert["risk_score"])
        self.evidence.update(alert.get("evidence") or {})
        self.detected_at = min(self.detected_at, alert["detected_at"])
        self.last_seen = max(self.last_seen, alert["detected_at"])
        self.raw_count += 1

    def flush(self) -> dict[str, Any]:
        return {
            "wallet_address": self.wallet_address,
            "asset_pair": self.asset_pair,
            "detectors": sorted(self.detectors),
            "risk_score": self.risk_score,
            "evidence": self.evidence,
            "detected_at": self.detected_at,
        }


def deduplicate(
    alert_stream: Iterable[dict[str, Any]],
    window_seconds: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Group correlated alerts and yield one enriched alert per group.

    Alerts are buffered in-memory, keyed by ``(wallet_address, asset_pair)``.
    A group is flushed (yielded) once no new alert has landed in that group
    for ``window_seconds`` -- silence is measured using each alert's
    ``detected_at`` field (event time), not wall-clock time, so the function
    is deterministic and safe to drive from a finite, pre-recorded stream in
    tests as well as from a live feed.

    Ordering behaviour: alerts may arrive out of order within the window.
    Each incoming alert's ``detected_at`` advances that key's "last seen"
    high-water mark only if it is greater than what is already recorded, and
    every other open group is checked for staleness relative to the new
    alert's timestamp before it is buffered -- so a late-arriving alert that
    is still within the window correctly merges into the existing group
    rather than starting a new one.

    Evidence merging: the flushed alert carries the union of every
    contributing detector name, the maximum risk score across all signals in
    the group, the union of all ``evidence`` dict fields (later values win on
    key collision), and the *earliest* ``detected_at`` timestamp seen in the
    group. No input alert is ever dropped -- every alert ends up represented
    in exactly one flushed group (size 1 if it never correlates with
    anything else).

    Args:
        alert_stream: iterable of alert dicts, each requiring
            ``wallet_address``, ``asset_pair``, ``detector``, ``risk_score``,
            ``detected_at`` (epoch seconds), and optional ``evidence`` dict.
        window_seconds: silence window before a group is flushed. Defaults
            to ``config.ALERT_DEDUP_WINDOW_SECONDS``.

    Yields:
        Flushed grouped-alert dicts, in the order their groups went silent
        (remaining open groups are flushed, in key order, once the input
        stream is exhausted).
    """
    if window_seconds is None:
        window_seconds = config.ALERT_DEDUP_WINDOW_SECONDS

    groups: dict[tuple[str, str], _Group] = {}

    for alert in alert_stream:
        key = (alert["wallet_address"], alert["asset_pair"])
        now = alert["detected_at"]

        for stale_key in [k for k, g in groups.items() if now - g.last_seen >= window_seconds]:
            yield _flush(groups.pop(stale_key))

        if key in groups:
            groups[key].absorb(alert)
        else:
            groups[key] = _Group(alert)

    for key in sorted(groups):
        yield _flush(groups.pop(key))


def _flush(group: _Group) -> dict[str, Any]:
    if group.raw_count > 1:
        ledgerlens_alerts_deduplicated_total.inc(group.raw_count - 1)
    return group.flush()
