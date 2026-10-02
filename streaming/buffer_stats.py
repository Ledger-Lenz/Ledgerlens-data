"""Periodic buffer-size logger for the streaming pipeline.

Logs the trade count per wallet held in a ``FeatureBuffer`` at a configurable
interval.  Runs in a background daemon thread so it never blocks the main
pipeline.

Usage::

    from streaming.buffer_stats import start_buffer_stats_logger

    stats = start_buffer_stats_logger(buffer, interval_seconds=60)
    # returns the Thread; call stats.join() if you need to wait for it.
"""

from __future__ import annotations

import threading

from streaming.feature_buffer import FeatureBuffer
from utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_INTERVAL_SECONDS = 60


def start_buffer_stats_logger(
    buffer: FeatureBuffer,
    interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    """Start a daemon thread that logs buffer sizes every *interval_seconds*.

    Args:
        buffer: The ``FeatureBuffer`` instance to monitor.
        interval_seconds: How often to emit a log line (default 60 s).
        stop_event: Optional ``threading.Event``; the thread exits when it is set.

    Returns:
        The started ``threading.Thread`` (daemon=True).
    """
    _stop = stop_event or threading.Event()

    def _run() -> None:
        last_evictions = buffer.evictions_total
        while not _stop.wait(timeout=interval_seconds):
            # stats() also publishes occupancy gauges to Prometheus (Issue #901).
            s = buffer.stats()
            evicted = s["evictions_total"] - last_evictions
            last_evictions = s["evictions_total"]
            logger.info(
                "FeatureBuffer stats: %d wallet(s), %d total buffered trade(s), "
                "occupancy=%.2f, eviction_rate=%.3f/s",
                s["wallets"],
                s["trades"],
                s["occupancy"],
                evicted / interval_seconds,
            )

    t = threading.Thread(target=_run, daemon=True, name="buffer-stats-logger")
    t.start()
    return t
