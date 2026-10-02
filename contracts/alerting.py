"""Alert-channel contracts for the streaming / alerts boundary.

Defines the ``AlertChannel`` protocol so that alternative alert
backends (webhook, websocket, Slack, PagerDuty, …) can be
plugged in without changing ``AlertDispatcher``.
"""

from __future__ import annotations

import typing
from typing import Protocol, runtime_checkable


class AlertEvent(typing.TypedDict, total=False):
    """Shape of an alert event flowing through the alerting subsystem.

    Required
    --------
    - ``wallet`` — the flagged wallet
    - ``asset_pair`` — traded pair
    - ``score`` — risk score 0–100
    - ``detectors`` — list of detector names that fired
    - ``timestamp`` — unix seconds

    Freshness attribution
    ---------------------
    - ``source`` — originating data source (e.g. ``"ethereum"``)
    - ``source_event_ts`` — unix seconds of the on-chain event that
      triggered this alert; propagated through ingestion -> feature ->
      scoring -> alert so end-to-end freshness can be measured.
    - ``stage_timestamps`` — per-stage unix-second timestamps keyed by
      stage name (``ingestion``, ``feature``, ``scoring``, ``alert``),
      enabling per-stage freshness breakdowns for root-causing.
    """

    wallet: str
    asset_pair: str
    score: int
    detectors: list[str]
    timestamp: int
    severity: str
    message: str
    source: str
    source_event_ts: int
    stage_timestamps: dict[str, int]


@runtime_checkable
class AlertChannel(Protocol):
    """Interface for an alert delivery channel.

    Usage::

        channel: AlertChannel = WebhookChannel(...)
        channel.dispatch(event)
    """

    def dispatch(self, event: AlertEvent) -> None:
        """Deliver *event* through this channel."""
