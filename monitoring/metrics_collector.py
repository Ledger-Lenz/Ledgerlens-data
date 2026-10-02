"""Metrics collector that feeds scored wallet events into the CUSUM detector (issue #289).

Usage
-----
    collector = MetricsCollector()
    collector.record_score(wallet="G...", score=72.0)

Cardinality guardrails (issue #934)
-----------------------------------
Metric label values must never be sourced directly from user/transaction
identifiers (wallet addresses, transaction hashes, account IDs, ...). Doing so
blows up label cardinality and overloads the metrics backend. Use
:func:`emit_metric` (or :meth:`MetricsCollector.emit_metric`) so that offending
emissions are rejected at runtime. See ``docs/metrics_labeling.md`` for the
safe-labeling guidelines.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from monitoring.cusum_detector import CUSUMDetector

logger = logging.getLogger(__name__)

#: Maximum number of distinct values a single label may take before it is
#: considered high-cardinality and rejected.
MAX_LABEL_CARDINALITY = 100

#: Maximum length of a label value; longer values are almost always raw
#: identifiers rather than bounded enumerations.
MAX_LABEL_VALUE_LENGTH = 64

#: Known high-cardinality-risk label names. Values for these labels are
#: rejected outright because they are (almost) always sourced from
#: user/transaction identifiers.
HIGH_CARDINALITY_LABELS = frozenset(
    {
        "wallet",
        "wallet_address",
        "account",
        "account_id",
        "address",
        "tx",
        "tx_id",
        "tx_hash",
        "transaction",
        "transaction_id",
        "transaction_hash",
        "user",
        "user_id",
        "email",
        "session_id",
        "request_id",
        "trace_id",
        "span_id",
    }
)

#: Patterns that indicate a label value is a raw identifier rather than a
#: bounded enumeration (Stellar account IDs, hex hashes, UUIDs, emails).
_HIGH_CARDINALITY_VALUE_PATTERNS = (
    re.compile(r"^G[A-Z2-7]{55}$"),  # Stellar account ID
    re.compile(r"^[0-9a-fA-F]{32,}$"),  # hex hash / tx id
    re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    ),  # UUID
    re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),  # email
)


class HighCardinalityLabelError(ValueError):
    """Raised when a metric emission uses a high-cardinality label value."""


def _is_high_cardinality_value(value: Any) -> bool:
    """Return True if ``value`` looks like a raw user/transaction identifier."""
    if not isinstance(value, str):
        return False
    if len(value) > MAX_LABEL_VALUE_LENGTH:
        return True
    return any(pattern.match(value) for pattern in _HIGH_CARDINALITY_VALUE_PATTERNS)


def check_labels(labels: Mapping[str, Any] | None) -> list[str]:
    """Return a list of cardinality violations found in ``labels``.

    An empty list means the labels are safe to emit.
    """
    violations: list[str] = []
    if not labels:
        return violations
    for name, value in labels.items():
        if name in HIGH_CARDINALITY_LABELS:
            violations.append(
                f"label {name!r} is a known high-cardinality label; "
                "do not use user/transaction identifiers as label values"
            )
        elif _is_high_cardinality_value(value):
            violations.append(
                f"label {name!r} value looks like a raw identifier "
                "(high cardinality); use a bounded enumeration instead"
            )
    return violations


def emit_metric(
    name: str,
    value: float,
    labels: Mapping[str, Any] | None = None,
    *,
    strict: bool = True,
) -> bool:
    """Emit a metric after enforcing cardinality guardrails.

    Args:
        name: Metric name.
        value: Metric value.
        labels: Optional label mapping.
        strict: When True (default) a high-cardinality label raises
            :class:`HighCardinalityLabelError`. When False the emission is
            flagged (logged) and dropped instead.

    Returns:
        True if the metric was emitted, False if it was rejected in
        non-strict mode.

    Raises:
        HighCardinalityLabelError: If ``strict`` and a violation is found.
    """
    violations = check_labels(labels)
    if violations:
        message = f"rejected metric {name!r}: " + "; ".join(violations)
        if strict:
            raise HighCardinalityLabelError(message)
        logger.warning(message)
        return False
    logger.debug("metric %s=%s labels=%s", name, value, dict(labels or {}))
    return True


class MetricsCollector:
    """Collect per-wallet risk score events and forward them to CUSUM.

    Args:
        cusum: Optional pre-configured ``CUSUMDetector``. A default instance
            (using ``config`` defaults) is created when ``None``.
        redis_client: Forwarded to a default ``CUSUMDetector`` when one is
            created internally.
    """

    def __init__(
        self,
        cusum: CUSUMDetector | None = None,
        redis_client=None,
    ) -> None:
        self._cusum = cusum or CUSUMDetector(metric_name="risk_score", redis_client=redis_client)

    def record_score(self, wallet: str, score: float) -> bool:
        """Record a scored wallet event; return True if CUSUM alarm fires.

        Args:
            wallet: Stellar account ID (used only for logging).
            score: Risk score in [0, 100].

        Returns:
            True if the CUSUM alarm was just triggered by this observation.
        """
        alarmed = self._cusum.update(score)
        if alarmed:
            logger.warning("CUSUM alarm triggered by wallet %s with score %.1f", wallet, score)
        return alarmed

    def emit_metric(
        self,
        name: str,
        value: float,
        labels: Mapping[str, Any] | None = None,
        *,
        strict: bool = True,
    ) -> bool:
        """Emit a metric through the cardinality guardrails.

        See :func:`emit_metric` for argument semantics.
        """
        return emit_metric(name, value, labels, strict=strict)

    @property
    def cusum(self) -> CUSUMDetector:
        return self._cusum
