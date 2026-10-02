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
from detection.model_compatibility import FEATURE_CONTRACT_VERSION

if TYPE_CHECKING:
    pass

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


#: See ``streaming.feature_store.UNSCOPED_TENANT_NAMESPACE``.
UNSCOPED_TENANT_NAMESPACE = "_unscoped"


class FeatureCache:
    """Thread-safe TTL cache mapping wallet -> feature matrix (``pd.Series``).

    Entries older than ``ttl_seconds`` are treated as a miss and evicted on
    next access. When the cache is at ``maxsize``, the least-recently-used
    entry is evicted to make room for a new one (entries refreshed via
    :meth:`get` or :meth:`put` are moved to the most-recently-used position).

    Feature schema invalidation: values are tracked against the active
    ``feature_contract_version``. If the schema version changes, the full cache
    is cleared so stale rows are never served against a newer feature contract.
    """

    def __init__(
        self,
        ttl_seconds: int | None = None,
        maxsize: int | None = None,
        tenant_id: str | None = None,
        schema_version: int | str | None = None,
    ) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else config.FEATURE_CACHE_TTL_SECONDS
        self._maxsize = maxsize if maxsize is not None else config.FEATURE_CACHE_MAXSIZE
        self._schema_version = (
            schema_version if schema_version is not None else FEATURE_CONTRACT_VERSION
        )
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[pd.Series, float, int | str]] = OrderedDict()
        self.tenant_id = tenant_id

    @property
    def schema_version(self) -> int | str:
        """The active feature contract version this cache enforces."""
        return self._schema_version

    @schema_version.setter
    def schema_version(self, value: int | str) -> None:
        """Update the active schema version, clearing any stale cached rows."""
        with self._lock:
            self._schema_version = value
            # Purge all entries: their schema version may no longer match.
            self._cache.clear()

    def _key(self, wallet: str) -> str:
        """Namespace the cache key by tenant.

        This cache is per-process, but a process serving more than one tenant
        would otherwise let the first tenant to compute a wallet's features
        serve them to the second. Keying by tenant removes that path.
        """
        return f"{self.tenant_id or UNSCOPED_TENANT_NAMESPACE}\x1f{wallet}"

    def get(self, wallet: str) -> pd.Series | None:
        """Return the cached feature matrix for *wallet*, or ``None`` on a miss."""
        key = self._key(wallet)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._record_miss()
                return None

            series, cached_at, cached_schema = entry
            if cached_schema != self._schema_version:
                del self._cache[wallet]
                self._record_miss()
                return None
            if time.monotonic() - cached_at >= self._ttl:
                del self._cache[key]
                self._record_miss()
                return None

            self._cache.move_to_end(key)
            self._record_hit()
            return series

    def put(self, wallet: str, features: pd.Series, schema_version: int | str | None = None) -> None:
        """Cache *features* for *wallet*, evicting the LRU entry if at capacity."""
        key = self._key(wallet)
        sv = schema_version if schema_version is not None else self._schema_version
        with self._lock:
            self._cache.pop(key, None)
            self._cache[key] = (features, time.monotonic(), sv)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, wallet: str) -> None:
        """Remove *wallet* from the cache, if present."""
        key = self._key(wallet)
        with self._lock:
            self._cache.pop(key, None)

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
                max_size
                if max_size is not None
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


# ---------------------------------------------------------------------------
# CacheWarmer — cold-start warm-up strategy (Issue #969)
# ---------------------------------------------------------------------------


class CacheWarmer:
    """Pre-populate a :class:`FeatureCache` with the most-active wallets before
    an instance is marked ready to receive production traffic.

    Cold-start problem
    ------------------
    After every deployment the feature cache is empty.  The first wave of
    requests re-builds every feature matrix from scratch, causing a latency
    spike that typically lasts until the TTL window is filled (~1–2 minutes at
    production trade volumes).  ``CacheWarmer`` eliminates this spike by
    computing and caching features for the *hot set* — the wallets most likely
    to be scored in the next scoring cycle — **before** the readiness probe
    marks the instance live.

    Hot-set definition
    ------------------
    The hot set is derived from recent ``RiskScore`` records persisted in the
    database.  Wallets with the highest trade activity or score frequency in
    the last ``lookback_hours`` hours are the most likely to be re-scored
    immediately after deployment.  The default hot-set size is
    ``config.CACHE_WARM_HOT_SET_SIZE`` (100 wallets).

    Usage
    -----
    ::

        cache = FeatureCache()
        warmer = CacheWarmer(cache, feature_builder=my_feature_fn)
        warmer.warm(hot_set=[(wallet_id, features_series), ...])

        # Or use the convenience method that queries the risk-score store:
        warmer.warm_from_store(risk_store, trades_df, ...)

    Deployment integration
    ----------------------
    Call :meth:`warm` (or :meth:`warm_from_store`) in the instance startup
    sequence **before** signalling readiness to the load-balancer or
    container orchestrator.  In Kubernetes this means calling it before the
    readiness probe HTTP endpoint starts returning 200.  The
    :attr:`is_warm` flag turns ``True`` when warm-up completes and can be
    checked by your readiness probe handler.

    Metrics
    -------
    ``CacheWarmer`` exposes a Prometheus gauge (``cache_warm_entries_total``)
    that tracks how many entries were inserted during the last warm-up run.
    A Prometheus counter (``cache_warm_duration_seconds``) records elapsed
    time.  These are optional: if ``prometheus_client`` is not installed the
    attributes are ``None`` and no metrics are emitted.

    Args:
        cache: The :class:`FeatureCache` instance to warm.
        hot_set_size: Maximum number of wallets to pre-warm. Defaults to
            ``config.CACHE_WARM_HOT_SET_SIZE`` (100).
        timeout_seconds: Maximum wall-clock time allowed for the warm-up phase.
            If the warm-up takes longer than this, it is stopped early and
            :attr:`is_warm` is still set to ``True`` (partial warm-up is better
            than delaying deployment). Defaults to
            ``config.CACHE_WARM_TIMEOUT_SECONDS`` (30).
    """

    def __init__(
        self,
        cache: FeatureCache,
        hot_set_size: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._cache = cache
        try:
            from config import config as _cfg

            self._hot_set_size = int(
                hot_set_size
                if hot_set_size is not None
                else getattr(_cfg, "CACHE_WARM_HOT_SET_SIZE", 100)
            )
            self._timeout = float(
                timeout_seconds
                if timeout_seconds is not None
                else getattr(_cfg, "CACHE_WARM_TIMEOUT_SECONDS", 30.0)
            )
        except Exception:  # pragma: no cover
            self._hot_set_size = hot_set_size if hot_set_size is not None else 100
            self._timeout = timeout_seconds if timeout_seconds is not None else 30.0

        self._is_warm: bool = False
        self._entries_warmed: int = 0
        self._last_warm_duration_seconds: float | None = None

        # Optional Prometheus metrics
        try:
            from prometheus_client import Counter, Gauge

            self._warm_entries_gauge: object | None = Gauge(
                "cache_warm_entries_total",
                "Number of entries inserted during the last cache warm-up run",
            )
            self._warm_duration_counter: object | None = Counter(
                "cache_warm_duration_seconds_total",
                "Total seconds spent in cache warm-up runs",
            )
        except Exception:  # pragma: no cover
            self._warm_entries_gauge = None
            self._warm_duration_counter = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_warm(self) -> bool:
        """``True`` once :meth:`warm` has completed (even if partial)."""
        return self._is_warm

    @property
    def entries_warmed(self) -> int:
        """Number of entries inserted during the last warm-up run."""
        return self._entries_warmed

    @property
    def last_warm_duration_seconds(self) -> float | None:
        """Elapsed seconds for the most recent warm-up run, or ``None`` if
        no warm-up has been performed yet."""
        return self._last_warm_duration_seconds

    def warm(
        self,
        hot_set: list[tuple[str, "pd.Series"]],
    ) -> int:
        """Pre-populate the cache from a pre-computed hot set.

        Args:
            hot_set: List of ``(wallet_id, feature_series)`` tuples to insert.
                The list is processed in order; up to :attr:`hot_set_size`
                entries are inserted. Processing stops early if
                :attr:`timeout_seconds` elapses.

        Returns:
            Number of cache entries actually inserted.
        """
        start = time.monotonic()
        inserted = 0
        limit = min(self._hot_set_size, len(hot_set))

        for wallet_id, features in hot_set[:limit]:
            elapsed = time.monotonic() - start
            if elapsed >= self._timeout:
                break
            try:
                self._cache.put(wallet_id, features)
                inserted += 1
            except Exception:  # pragma: no cover
                # Never let a warm-up error block startup
                pass

        elapsed_total = time.monotonic() - start
        self._entries_warmed = inserted
        self._last_warm_duration_seconds = elapsed_total
        self._is_warm = True

        # Update Prometheus metrics
        if self._warm_entries_gauge is not None:
            self._warm_entries_gauge.set(inserted)  # type: ignore[union-attr]
        if self._warm_duration_counter is not None:
            self._warm_duration_counter.inc(elapsed_total)  # type: ignore[union-attr]

        return inserted

    def warm_from_store(
        self,
        risk_store: object,
        build_features_fn,
        trades_df: "pd.DataFrame",
        lookback_hours: int = 24,
        **feature_kwargs,
    ) -> int:
        """Query the risk-score store for the hot set and warm the cache.

        Identifies the most-active wallets from *risk_store* (those scored
        most recently within *lookback_hours* hours), builds their feature
        matrices using *build_features_fn*, and inserts them into the cache.

        Args:
            risk_store: A repository object with a ``get_recent_wallets``
                method (or equivalent) that returns
                ``list[str]`` — wallet IDs ordered by recency.
            build_features_fn: Callable ``(wallet_id, wallet_trades_df, **kwargs)
                -> pd.Series`` that computes the feature vector for a wallet.
            trades_df: Full trade DataFrame used to build per-wallet features.
                Only rows matching each hot-set wallet are passed to
                *build_features_fn*.
            lookback_hours: How far back to look when selecting the hot set.
                Default 24 hours.
            **feature_kwargs: Extra keyword arguments forwarded to
                *build_features_fn* (e.g. ``funding_graph``, ``all_pairs_df``).

        Returns:
            Number of cache entries inserted.
        """
        try:
            wallets: list[str] = list(
                risk_store.get_recent_wallets(  # type: ignore[union-attr]
                    limit=self._hot_set_size,
                    lookback_hours=lookback_hours,
                )
            )
        except Exception:  # pragma: no cover
            # Store unavailable at startup — proceed without warming
            self._is_warm = True
            return 0

        start = time.monotonic()
        hot_set: list[tuple[str, "pd.Series"]] = []

        for wallet_id in wallets:
            elapsed = time.monotonic() - start
            if elapsed >= self._timeout:
                break
            try:
                wallet_trades = (
                    trades_df[trades_df["wallet_id"] == wallet_id]
                    if not trades_df.empty
                    else trades_df
                )
                features = build_features_fn(wallet_id, wallet_trades, **feature_kwargs)
                hot_set.append((wallet_id, pd.Series(features) if isinstance(features, dict) else features))
            except Exception:  # pragma: no cover
                # Feature build errors must not block warm-up
                continue

        return self.warm(hot_set)

    def describe(self) -> dict:
        """Return a summary of the last warm-up run for operator diagnostics.

        Useful for logging at startup and for the readiness-probe endpoint::

            if not warmer.is_warm:
                return {"ready": False}
            info = warmer.describe()
            logger.info("Cache warmed: %s", info)

        Returns:
            Dictionary with ``is_warm``, ``entries_warmed``,
            ``last_warm_duration_seconds``, and ``hot_set_size``.
        """
        return {
            "is_warm": self._is_warm,
            "entries_warmed": self._entries_warmed,
            "last_warm_duration_seconds": self._last_warm_duration_seconds,
            "hot_set_size": self._hot_set_size,
            "timeout_seconds": self._timeout,
            "cache_size": len(self._cache),
        }
