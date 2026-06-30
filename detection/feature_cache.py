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
from typing import TYPE_CHECKING

import pandas as pd

from config import config

if TYPE_CHECKING:
    from detection.wallet_graph import IncrementalWalletGraph

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


class WalletGraphCache:
    """Thread-safe singleton cache holding the live :class:`IncrementalWalletGraph`.

    Wraps :class:`~detection.wallet_graph.IncrementalWalletGraph` so that the
    streaming scorer and feature engineering pipeline share a single in-memory
    graph that is updated incrementally as new trade edges arrive.

    Usage::

        cache = WalletGraphCache.instance()
        cache.add_trade_edge(src, dst, trade)
        subgraph = cache.get_ego_subgraph(wallet_id)
        removed = cache.remove_stale_edges()
    """

    _singleton: "WalletGraphCache | None" = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        from detection.wallet_graph import IncrementalWalletGraph

        self._graph = IncrementalWalletGraph()
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "WalletGraphCache":
        """Return the process-wide singleton (created on first call)."""
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    def add_trade_edge(self, src_wallet: str, dst_wallet: str, trade: dict) -> None:
        """Delegate to the underlying :class:`IncrementalWalletGraph`."""
        self._graph.add_trade_edge(src_wallet, dst_wallet, trade)

    def remove_stale_edges(self, max_age_hours: float | None = None) -> int:
        """Remove stale edges and return the count removed."""
        return self._graph.remove_stale_edges(max_age_hours)

    def get_ego_subgraph(self, wallet_id: str, hops: int = 2):
        """Extract a k-hop ego subgraph from the live graph."""
        return self._graph.get_ego_subgraph(wallet_id, hops)

    def graph_metrics(self) -> dict:
        """Return ``{nodes, edges, stale_edges}`` for the live graph."""
        return self._graph.graph_metrics()

    def snapshot(self):
        """Return a full copy of the current graph state."""
        return self._graph.snapshot()
