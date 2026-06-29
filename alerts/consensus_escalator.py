"""Consensus-driven alert escalation requiring multi-detector agreement."""

import time
from dataclasses import dataclass
from typing import Any

import redis
from prometheus_client import Counter

from config import config

MIN_DETECTOR_CONSENSUS = 2
CONSENSUS_WINDOW_SECONDS = 120

ledgerlens_consensus_alerts_total = Counter(
    "ledgerlens_consensus_alerts_total", "Total consensus alerts fired"
)
ledgerlens_single_detector_alerts_total = Counter(
    "ledgerlens_single_detector_alerts_total", "Total single-detector alerts fired"
)

DETECTOR_ALLOWLIST = {
    "benford",
    "ml",
    "graph",
    "liquidity",
    "cross_pair",
}


@dataclass
class EscalatedAlert:
    wallet: str
    pair: str
    detectors: list[str]
    time_span_seconds: float
    severity: str


class EscalationPolicy:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.consensus_threshold = MIN_DETECTOR_CONSENSUS
        self.window_seconds = CONSENSUS_WINDOW_SECONDS

    def _validate_detector(self, detector: str) -> None:
        if detector not in DETECTOR_ALLOWLIST:
            raise ValueError(f"Invalid detector: {detector}")

    def record_signal(self, wallet: str, pair: str, detector: str) -> None:
        self._validate_detector(detector)
        key = f"consensus:{wallet}:{pair}"
        now = time.time()
        pipe = self.redis.pipeline()
        pipe.hset(key, detector, now)
        pipe.expire(key, self.window_seconds + 10)
        pipe.execute()

    def check_consensus(self, wallet: str, pair: str) -> EscalatedAlert | None:
        key = f"consensus:{wallet}:{pair}"
        now = time.time()
        signals = self.redis.hgetall(key)
        if not signals:
            return None

        valid_signals = {
            k.decode(): float(v.decode())
            for k, v in signals.items()
            if now - float(v.decode()) <= self.window_seconds
        }

        if len(valid_signals) >= self.consensus_threshold:
            detectors = list(valid_signals.keys())
            time_span = max(valid_signals.values()) - min(valid_signals.values())
            ledgerlens_consensus_alerts_total.inc()
            return EscalatedAlert(
                wallet=wallet,
                pair=pair,
                detectors=detectors,
                time_span_seconds=time_span,
                severity="high",
            )
        return None

    def emit_single_detector_alert(self, wallet: str, pair: str, detector: str) -> None:
        self._validate_detector(detector)
        ledgerlens_single_detector_alerts_total.inc()
