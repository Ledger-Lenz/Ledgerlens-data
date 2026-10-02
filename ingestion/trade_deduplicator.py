"""Idempotent trade ingestion using the unified exactly-once dedup store.

Handles duplicate trades from the Stellar Horizon SSE stream by staging/
committing trade hashes in Redis via ``pipeline.exactly_once.ExactlyOnceStore``.

Unlike the previous implementation, this module is **fail-closed** (Issue
#670, invariant 8): when Redis is unreachable, ``is_duplicate`` raises
``DedupBackendUnavailableError`` instead of silently allowing every event
through. Callers that cannot tolerate raising should catch that error
explicitly and decide whether to queue, halt, or (with an explicit, logged
decision) skip deduplication — there is no built-in silent fallback.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping

from config import config
from pipeline.exactly_once import (
    DedupBackendUnavailableError,
    DedupKey,
    DedupState,
    ExactlyOnceStore,
    RedisExactlyOnceBackend,
)
from utils.logging import get_logger

logger = get_logger(__name__)

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter

    ledgerlens_duplicate_events_total = Counter(
        "ledgerlens_duplicate_events_total",
        "Total number of duplicate trade events discarded",
        ["asset_pair"],
    )
    ledgerlens_dedup_cache_hits_total = Counter(
        "ledgerlens_dedup_cache_hits_total",
        "Total cache hits in trade deduplication",
    )
except ImportError:
    ledgerlens_duplicate_events_total = None
    ledgerlens_dedup_cache_hits_total = None


class SeenEventCache:
    """Fail-closed dedup cache for Horizon trade events.

    Keyed by ``sha256(paging_token or trade_id)`` scoped to ``asset_pair``,
    via ``pipeline.exactly_once.DedupKey(source=f"horizon_trade:{asset_pair}",
    external_id=trade_hash)``. A trade hash is committed (durably marked seen)
    the first time it is checked — this class has no separate side-effect
    boundary of its own, so "checked" and "committed" happen atomically from
    the caller's point of view.

    Raises ``DedupBackendUnavailableError`` from every method when Redis is
    unreachable. There is intentionally no "allow through" fallback.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
        key_prefix: str | None = None,
    ):
        self.redis_url = redis_url or config.REDIS_URL
        self.ttl_seconds = ttl_seconds or config.TRADE_DEDUP_TTL_SECONDS
        self.key_prefix = key_prefix or config.TRADE_DEDUP_CACHE_KEY_PREFIX

        self._backend = RedisExactlyOnceBackend(self.redis_url, key_prefix=self.key_prefix)
        self._store = ExactlyOnceStore(self._backend, ttl_seconds=float(self.ttl_seconds))

    def _key(self, trade_id: str, paging_token: str | None, asset_pair: str) -> DedupKey:
        import hashlib

        hash_input = paging_token or trade_id
        trade_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        return DedupKey(source=f"horizon_trade:{asset_pair}", external_id=trade_hash)

    def is_duplicate(
        self,
        trade_id: str,
        paging_token: str | None = None,
        asset_pair: str = "unknown",
    ) -> bool:
        """Check if trade has been seen before, and mark it seen if not.

        Raises:
            DedupBackendUnavailableError: Redis is unreachable — the caller
                must not treat this as "not a duplicate".
        """
        key = self._key(trade_id, paging_token, asset_pair)
        decision = self._store.check_and_stage(key)

        if decision.state is DedupState.COMMITTED:
            logger.debug("Trade duplicate detected: %s (%s…)", trade_id, key.external_id[:8])
            if ledgerlens_duplicate_events_total:
                ledgerlens_duplicate_events_total.labels(asset_pair=asset_pair).inc()
            if ledgerlens_dedup_cache_hits_total:
                ledgerlens_dedup_cache_hits_total.inc()
            return True

        # NEW, STAGED (redo), or TTL_EXPIRED_REVERIFY: not yet committed. This
        # cache has no side-effect boundary of its own (see class docstring),
        # so "checked" and "committed" happen atomically here rather than via
        # a separate downstream commit() call.
        self._store.commit(key)
        logger.debug("Trade cached: %s (%s…)", trade_id, key.external_id[:8])
        return False

    def cache_trade(
        self,
        trade_id: str,
        paging_token: str | None = None,
        asset_pair: str = "unknown",
    ) -> None:
        """Explicitly add a trade to the cache (used when not checking duplicates).

        Args:
            trade_id: Horizon trade ID.
            paging_token: Horizon paging token.
            asset_pair: Asset pair for organization.

        Raises:
            DedupBackendUnavailableError: Redis is unreachable.
        """
        key = self._key(trade_id, paging_token, asset_pair)
        self._store.commit(key)

    def get_cache_size(self, asset_pair: str = "unknown") -> int:
        """Return the number of cached trade hashes for an asset pair.

        Returns -1 if Redis is unavailable (ops introspection only — this is
        not on the correctness-critical dedup path).
        """
        try:
            return self._backend.scan_count(f"horizon_trade:{asset_pair}")
        except DedupBackendUnavailableError as e:
            logger.warning(f"Failed to get cache size: {e}")
            return -1

    def clear_cache(self, asset_pair: str | None = None) -> bool:
        """Clear cached trades (for testing/ops). Returns False if Redis unavailable."""
        try:
            if asset_pair:
                self._backend.scan_delete(f"horizon_trade:{asset_pair}")
            else:
                self._backend.scan_delete("horizon_trade")
            return True
        except Exception as e:  # noqa: BLE001
            # Broad catch justified: Redis operations can fail on network/timeout.
            # Return False (failure status) rather than crashing test cleanup.
            logger.warning(f"Failed to clear cache: {e}")
            return False

    def health_check(self) -> bool:
        """Check whether the dedup backend is available.

        Returns:
            True if the backend is available and responsive, False otherwise.
        """
        return self._store.is_available()


# ---------------------------------------------------------------------------
# Global singleton + convenience wrapper
# ---------------------------------------------------------------------------

_cache_instance: SeenEventCache | None = None


def get_trade_dedup_cache() -> SeenEventCache:
    """Get or create the global trade deduplication cache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SeenEventCache()
    return _cache_instance


def is_duplicate_trade(
    trade_id: str,
    paging_token: str | None = None,
    asset_pair: str = "unknown",
) -> bool:
    """Convenience wrapper around the global cache.

    Raises:
        DedupBackendUnavailableError: propagated from ``SeenEventCache.is_duplicate``.
    """
    cache = get_trade_dedup_cache()
    return cache.is_duplicate(trade_id, paging_token, asset_pair)


__all__ = [
    "DedupBackendUnavailableError",
    "SeenEventCache",
    "get_trade_dedup_cache",
    "is_duplicate_trade",
]


# --- Cross-source deduplication (Issue #905) ---------------------------------
#
# Redundant source connectors (see ``source_connectors.py``) observe the same
# on-ledger trade under different source-local message ids and at different
# times. ``CrossSourceDeduplicator`` collapses them using a *canonical*,
# source-agnostic identity: ``(ledger_sequence, operation_index, trade_index)``.
#
# Dedup window: ``window_seconds`` (default ``config.CROSS_SOURCE_DEDUP_WINDOW_SECONDS``
# if set, else 300s) bounds how long a canonical key is remembered after it
# was first seen. It must exceed the worst-case lag between the fastest and
# slowest redundant source; ``max_entries`` caps memory regardless of window.

CanonicalTradeKey = tuple[int, int, int]


def canonical_trade_key(
    ledger_sequence: int, operation_index: int, trade_index: int = 0
) -> CanonicalTradeKey:
    """Builds the source-agnostic identity of an on-ledger trade."""
    return (int(ledger_sequence), int(operation_index), int(trade_index))


def canonical_key_from_record(record: Mapping | object) -> CanonicalTradeKey:
    """Derives the canonical key from a raw record or ``Trade``.

    Uses explicit ``ledger_sequence``/``operation_index`` fields when present;
    otherwise decodes a Horizon-style ``"<operation_id>-<trade_index>"`` id,
    where the 64-bit operation id (SEP TOID) packs ledger seq (high 32 bits)
    and the ledger-local operation position (low 32 bits: tx order + op
    index), so trades from different transactions never share a key.
    """
    get = record.get if isinstance(record, Mapping) else lambda k: getattr(record, k, None)
    ledger, op_index = get("ledger_sequence"), get("operation_index")
    if ledger is not None and op_index is not None:
        return canonical_trade_key(ledger, op_index, get("trade_index") or 0)
    trade_id = get("trade_id") or get("id") or get("paging_token")
    if not trade_id:
        raise ValueError("record has no ledger/operation fields or Horizon trade id")
    op_id_str, _, trade_idx = str(trade_id).partition("-")
    op_id = int(op_id_str)
    return canonical_trade_key(op_id >> 32, op_id & 0xFFFFFFFF, int(trade_idx or 0))


class CrossSourceDeduplicator:
    """In-process, windowed dedup of trades observed through multiple sources."""

    def __init__(
        self,
        window_seconds: float | None = None,
        max_entries: int = 100_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        if window_seconds is None:
            window_seconds = float(getattr(config, "CROSS_SOURCE_DEDUP_WINDOW_SECONDS", 300.0))
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.window_seconds = window_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._seen: OrderedDict[CanonicalTradeKey, tuple[float, str]] = OrderedDict()

    def _evict(self, now: float) -> None:
        while self._seen:
            key, (seen_at, _) = next(iter(self._seen.items()))
            if now - seen_at < self.window_seconds and len(self._seen) <= self.max_entries:
                break
            self._seen.popitem(last=False)

    def accept(self, record: Mapping | object, source: str = "unknown") -> bool:
        """Returns True if the record is new (emit it), False if a duplicate."""
        now = self._clock()
        self._evict(now)
        key = canonical_key_from_record(record)
        if key in self._seen:
            first_source = self._seen[key][1]
            logger.debug("cross-source duplicate %s from %s (first: %s)", key, source, first_source)
            if ledgerlens_duplicate_events_total is not None:
                ledgerlens_duplicate_events_total.labels(asset_pair="cross_source").inc()
            return False
        self._seen[key] = (now, source)
        return True

    def __len__(self) -> int:
        return len(self._seen)
