"""In-memory TTL+LRU cache for per-wallet feature matrices.

In the WebSocket feed scenario (see ``streaming/streaming_scorer.py``), a
wallet may be re-scored many times per minute as new trade events arrive.
Rebuilding the feature matrix from scratch on every event (Benford windows,
wallet graph metrics, cross-asset coordination, hardening features, ...) is
the dominant cost of a re-score. Caching the last computed matrix for a
short TTL eliminates the redundant recomputation during these high-activity
bursts.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import pandas as pd

from config import config

try:
    from prometheus_client import Counter

    feature_cache_hits_total = Counter(
        "feature_cache_hits_total",
        "Number of FeatureCache lookups served from cache",
    )
    feature_cache_misses_total = Counter(
        "feature_cache_misses_total",
        "Number of FeatureCache lookups that were not cached or had expired",
    )
except Exception:  # pragma: no cover
    feature_cache_hits_total = None  # type: ignore[assignment]
    feature_cache_misses_total = None  # type: ignore[assignment]


class FeatureCache:
    """Thread-safe TTL cache mapping wallet -> feature matrix (``pd.Series``).

    Entries older than ``ttl_seconds`` are treated as a miss and evicted on
    next access. When the cache is at ``maxsize``, the least-recently-used
    entry is evicted to make room for a new one (entries refreshed via
    :meth:`get` or :meth:`put` are moved to the most-recently-used position).
    """

    def __init__(self, ttl_seconds: int | None = None, maxsize: int | None = None) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else config.FEATURE_CACHE_TTL_SECONDS
        self._maxsize = maxsize if maxsize is not None else config.FEATURE_CACHE_MAXSIZE
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[pd.Series, float]] = OrderedDict()

    def get(self, wallet: str) -> pd.Series | None:
        """Return the cached feature matrix for *wallet*, or ``None`` on a miss."""
        with self._lock:
            entry = self._cache.get(wallet)
            if entry is None:
                self._record_miss()
                return None

            series, cached_at = entry
            if time.monotonic() - cached_at >= self._ttl:
                del self._cache[wallet]
                self._record_miss()
                return None

            self._cache.move_to_end(wallet)
            self._record_hit()
            return series

    def put(self, wallet: str, features: pd.Series) -> None:
        """Cache *features* for *wallet*, evicting the LRU entry if at capacity."""
        with self._lock:
            self._cache.pop(wallet, None)
            self._cache[wallet] = (features, time.monotonic())
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, wallet: str) -> None:
        """Remove *wallet* from the cache, if present."""
        with self._lock:
            self._cache.pop(wallet, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @staticmethod
    def _record_hit() -> None:
        if feature_cache_hits_total is not None:
            feature_cache_hits_total.inc()

    @staticmethod
    def _record_miss() -> None:
        if feature_cache_misses_total is not None:
            feature_cache_misses_total.inc()
