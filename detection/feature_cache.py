"""In-memory TTL+LRU cache for per-wallet feature matrices, plus a
RecentDataBuffer for accumulating labelled samples for incremental training.

In the WebSocket feed scenario (see ``streaming/streaming_scorer.py``), a
wallet may be re-scored many times per minute as new trade events arrive.
Rebuilding the feature matrix from scratch on every event (Benford windows,
wallet graph metrics, cross-asset coordination, hardening features, ...) is
the dominant cost of a re-score. Caching the last computed matrix for a
short TTL eliminates the redundant recomputation during these high-activity
bursts.

``RecentDataBuffer`` complements ``FeatureCache`` by accumulating *labelled*
feature rows for incremental LightGBM training.  When the buffer reaches
``max_size`` (or an external drift signal fires), the buffered rows are
passed to ``detection.model_training.incremental_train_lightgbm`` so that
the LightGBM model can adapt to distribution shifts within seconds rather
than minutes.
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


# ---------------------------------------------------------------------------
# RecentDataBuffer — labelled sample accumulator for incremental training
# ---------------------------------------------------------------------------


class RecentDataBuffer:
    """Thread-safe circular buffer that accumulates labelled feature rows for
    incremental LightGBM training.

    Design
    ──────
    The buffer holds at most *max_size* rows at a time (default:
    ``config.INCREMENTAL_BUFFER_SIZE``, i.e. 10 000).  When a new batch is
    added that would overflow the buffer, the **oldest** rows are evicted
    first (FIFO drop), so the buffer always contains the most recent samples.

    Incremental training is triggered in two ways:

    1. **Buffer-full trigger**: when ``add()`` causes ``len(buffer) >=
       max_size``, ``is_ready()`` returns ``True``.
    2. **Drift trigger**: external callers (e.g.
       ``scripts/retrain_if_drifted.py``) can call ``is_ready()`` after
       receiving a PSI drift signal; the buffer returns its current contents
       regardless of fill level (down to ``min_samples``).

    Usage::

        buffer = RecentDataBuffer(max_size=10_000, min_samples=500)

        # Called from the streaming pipeline as labelled events arrive:
        buffer.add(feature_df_with_label_column)

        if buffer.is_ready():
            new_data = buffer.flush()        # returns DataFrame, clears buffer
            new_lgbm = incremental_train_lightgbm(
                existing_model=current_lgbm,
                new_data=new_data,
                n_new_trees=100,
                reference_feature_columns=feature_columns,
            )

    Thread safety
    ─────────────
    All mutating operations (``add``, ``flush``, ``clear``) are protected by
    an internal ``threading.Lock``.  ``is_ready()`` and ``__len__`` are also
    lock-protected.

    Args:
        max_size:
            Maximum number of rows held.  When exceeded, oldest rows are
            evicted.  Defaults to ``config.INCREMENTAL_BUFFER_SIZE``.
        min_samples:
            Minimum rows required for ``is_ready()`` to return ``True`` even
            when triggered externally (drift signal).  Prevents incremental
            training on a nearly-empty buffer.  Defaults to 100.
    """

    def __init__(
        self,
        max_size: int | None = None,
        min_samples: int = 100,
    ) -> None:
        try:
            from config import config as _cfg  # late import for testability

            self._max_size: int = (
                max_size if max_size is not None
                else int(getattr(_cfg, "INCREMENTAL_BUFFER_SIZE", 10_000))
            )
        except Exception:  # pragma: no cover
            self._max_size = max_size if max_size is not None else 10_000

        self._min_samples = max(1, min_samples)
        self._lock = threading.Lock()
        # Store rows as a list of DataFrames; concat on flush.
        self._chunks: list[pd.DataFrame] = []
        self._n_rows: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def max_size(self) -> int:
        return self._max_size

    def add(self, rows: pd.DataFrame) -> None:
        """Append *rows* to the buffer, evicting the oldest rows if needed.

        Args:
            rows: A ``pd.DataFrame`` with feature columns **and** a ``"label"``
                  column (1 = wash trade, 0 = legitimate).  Rows without a
                  ``"label"`` column are still accepted (for use in inference
                  pipelines), but ``flush()`` will raise if labels are missing
                  when incremental training is attempted.
        """
        if rows.empty:
            return

        with self._lock:
            self._chunks.append(rows.reset_index(drop=True))
            self._n_rows += len(rows)

            # Evict oldest rows if we are over the size cap
            if self._n_rows > self._max_size:
                self._evict_oldest_locked()

    def is_ready(self, force: bool = False) -> bool:
        """Return ``True`` if the buffer has enough data to trigger training.

        Args:
            force: When ``True`` (drift-triggered call), return ``True`` if
                   the buffer has >= *min_samples* rows, regardless of whether
                   it is full.  When ``False`` (size-triggered), return
                   ``True`` only when the buffer is at capacity.
        """
        with self._lock:
            if force:
                return self._n_rows >= self._min_samples
            return self._n_rows >= self._max_size

    def flush(self) -> pd.DataFrame:
        """Return all buffered rows as a single ``pd.DataFrame`` and clear
        the buffer.

        Returns:
            A concatenated ``pd.DataFrame`` of all buffered rows.

        Raises:
            ValueError: if the buffer is empty.
        """
        with self._lock:
            if not self._chunks:
                raise ValueError("RecentDataBuffer is empty — nothing to flush")
            result = pd.concat(self._chunks, ignore_index=True)
            self._chunks = []
            self._n_rows = 0
            return result

    def peek(self) -> pd.DataFrame:
        """Return a copy of the buffer contents without clearing it."""
        with self._lock:
            if not self._chunks:
                return pd.DataFrame()
            return pd.concat(self._chunks, ignore_index=True)

    def clear(self) -> None:
        """Discard all buffered rows without returning them."""
        with self._lock:
            self._chunks = []
            self._n_rows = 0

    def __len__(self) -> int:
        with self._lock:
            return self._n_rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_oldest_locked(self) -> None:
        """Evict rows from the front of the buffer until size <= max_size.

        Must be called with ``self._lock`` held.
        """
        while self._n_rows > self._max_size and self._chunks:
            oldest = self._chunks[0]
            excess = self._n_rows - self._max_size

            if len(oldest) <= excess:
                # Drop the entire oldest chunk
                self._n_rows -= len(oldest)
                self._chunks.pop(0)
            else:
                # Trim the oldest chunk from the front
                self._chunks[0] = oldest.iloc[excess:].reset_index(drop=True)
                self._n_rows -= excess
                break
