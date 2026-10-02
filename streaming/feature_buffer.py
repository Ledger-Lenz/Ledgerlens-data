"""Thread-safe per-wallet rolling trade buffer.

Phase 1 of the real-time detection pipeline (Issue #12).

Thread-safety model
-------------------
- A top-level ``threading.RLock`` (``_registry_lock``) guards mutations to the
  dict of wallets and their per-wallet locks.
- Each wallet gets its own ``threading.Lock`` that is held only while
  reading/writing that wallet's deque.  Unrelated wallets can therefore be
  updated concurrently with no contention between them.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from typing import TYPE_CHECKING

import pandas as pd

from config import config
from detection.feature_engineering import build_feature_vector
from detection.streaming_benford import StreamingBenfordSketch
from ingestion.data_models import Trade
from utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

try:  # pragma: no cover - metrics are optional at import time
    from prometheus_client import Counter, Gauge

    _EVICTIONS_TOTAL = Counter(
        "ledgerlens_feature_buffer_evictions_total",
        "Wallets evicted from the FeatureBuffer by the LRU bound (Issue #901)",
    )
    _BUFFER_WALLETS = Gauge(
        "ledgerlens_feature_buffer_wallets", "Wallets currently held in the FeatureBuffer"
    )
    _BUFFER_TRADES = Gauge(
        "ledgerlens_feature_buffer_trades", "Trades currently held in the FeatureBuffer"
    )
    _BUFFER_OCCUPANCY = Gauge(
        "ledgerlens_feature_buffer_occupancy_ratio",
        "Wallet occupancy of the FeatureBuffer relative to max_wallets (0-1)",
    )
except ImportError:  # pragma: no cover
    _EVICTIONS_TOTAL = _BUFFER_WALLETS = _BUFFER_TRADES = _BUFFER_OCCUPANCY = None


class FeatureBuffer:
    """Per-wallet rolling deque of recent trades, safe for concurrent access.

    Memory bound (Issue #901)
    -------------------------
    Memory is hard-bounded at ``max_wallets * max_trades`` trade records:

    - ``max_trades`` caps each wallet's deque (oldest trade dropped first).
    - ``max_wallets`` caps the number of tracked wallets.  When a *new*
      wallet arrives and the buffer is full, the **least-recently-used**
      wallet (by last ``update``/``get_feature_row`` access) is evicted with
      all its state (trades, dedup index, Benford sketches).

    ``max_wallets=None`` (or ``0``) disables the wallet bound.  See
    ``docs/streaming_architecture.md`` for operator tradeoffs.
    """

    def __init__(self, max_trades: int = 1000, max_wallets: int | None = None) -> None:
        self.max_trades = max_trades
        if max_wallets is None:
            max_wallets = int(getattr(config, "FEATURE_BUFFER_MAX_WALLETS", 0) or 0)
        self.max_wallets = max_wallets or None
        self.evictions_total = 0
        # Guards creation of new wallet entries in _buffers/_locks.
        self._registry_lock = threading.RLock()
        # OrderedDict doubles as the LRU index: most-recently-used at the end.
        self._buffers: OrderedDict[str, deque] = OrderedDict()
        self._wallet_locks: dict[str, threading.Lock] = {}
        # trade_id -> None per wallet, kept in the same append/evict order as
        # ``_buffers[wallet]`` so membership checks for the "have I already
        # applied this trade_id to this wallet's state" dedup guard are O(1)
        # instead of an O(max_trades) scan (Issue #670, invariant 1: at
        # least-once Kafka/replay redelivery of the same trade_id must never
        # produce more than one feature-state update).
        self._seen_trade_ids: dict[str, dict[str, None]] = {}

        # Benford sketches: wallet -> window_hours -> StreamingBenfordSketch
        self._benford_sketches: dict[str, dict[int, StreamingBenfordSketch]] = {}
        # Per-pair Benford sketches: wallet -> pair_id -> window_hours -> StreamingBenfordSketch
        self._pair_benford_sketches: dict[str, dict[str, dict[int, StreamingBenfordSketch]]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_wallet(self, wallet: str) -> threading.Lock:
        """Return the lock for *wallet*, creating both lock and deque if absent."""
        with self._registry_lock:
            if wallet in self._buffers:
                self._buffers.move_to_end(wallet)
            else:
                if self.max_wallets is not None:
                    while len(self._buffers) >= self.max_wallets:
                        self._evict_lru()
                self._wallet_locks[wallet] = threading.Lock()
                self._buffers[wallet] = deque(maxlen=self.max_trades)
                self._benford_sketches[wallet] = {
                    h: StreamingBenfordSketch(h * 3600) for h in config.BENFORD_WINDOWS_HOURS
                }
                self._pair_benford_sketches[wallet] = {}
                self._seen_trade_ids[wallet] = {}
            return self._wallet_locks[wallet]

    def _evict_lru(self) -> None:
        """Drop the least-recently-used wallet. Caller holds ``_registry_lock``."""
        victim, _ = self._buffers.popitem(last=False)
        self._wallet_locks.pop(victim, None)
        self._benford_sketches.pop(victim, None)
        self._pair_benford_sketches.pop(victim, None)
        self._seen_trade_ids.pop(victim, None)
        self.evictions_total += 1
        if _EVICTIONS_TOTAL is not None:
            _EVICTIONS_TOTAL.inc()
        logger.debug("FeatureBuffer: evicted LRU wallet=%s", victim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, trade: Trade) -> None:
        """Add *trade* to both ``base_account`` and ``counter_account`` buffers.

        When a wallet's deque is at capacity, ``deque(maxlen=…)`` automatically
        evicts the oldest entry on ``append()``.

        Idempotent per ``(wallet, trade_id)``: a redelivered trade (at-least-
        once Kafka redelivery when an offset was left uncommitted, or a stream
        replay reprocessing an uncommitted tail) is a no-op for any wallet
        that already applied it — this is what makes it safe for callers to
        redo a partially-processed message without double-counting feature
        state (Issue #670, invariant 1).
        """
        amount = float(trade.amount)
        record = {
            "trade_id": trade.trade_id,
            "ledger_close_time": trade.ledger_close_time,
            "base_account": trade.base_account,
            "counter_account": trade.counter_account,
            "base_asset": str(trade.base_asset.code),
            "counter_asset": str(trade.counter_asset.code),
            "amount": amount,
        }
        pair_id = trade.base_asset.pair_id(trade.counter_asset)

        for wallet in (trade.base_account, trade.counter_account):
            lock = self._ensure_wallet(wallet)
            with lock:
                seen = self._seen_trade_ids.get(wallet)
                if seen is None:  # evicted concurrently by the LRU bound
                    continue
                if trade.trade_id in seen:
                    logger.debug(
                        "FeatureBuffer.update: trade_id=%s already applied to wallet=%s — "
                        "skipping duplicate (redelivery)",
                        trade.trade_id,
                        wallet,
                    )
                    continue

                buf = self._buffers[wallet]
                if len(buf) >= self.max_trades and buf:
                    # ``deque(maxlen=…)`` is about to silently drop buf[0] on
                    # append() below — evict its trade_id from the companion
                    # index in lock-step so membership checks stay accurate.
                    seen.pop(buf[0].get("trade_id"), None)

                buf.append(record)
                seen[trade.trade_id] = None

                # Update wallet-level Benford sketches
                for sketch in self._benford_sketches[wallet].values():
                    sketch.update(amount, trade.ledger_close_time)

                # Update pair-level Benford sketches
                if pair_id not in self._pair_benford_sketches[wallet]:
                    self._pair_benford_sketches[wallet][pair_id] = {
                        h: StreamingBenfordSketch(h * 3600) for h in config.BENFORD_WINDOWS_HOURS
                    }
                for sketch in self._pair_benford_sketches[wallet][pair_id].values():
                    sketch.update(amount, trade.ledger_close_time)

    def get_feature_row(self, wallet: str) -> pd.Series | None:
        """Build and return the feature row for *wallet*.

        Returns ``None`` if the wallet has no trades in the buffer.
        """
        lock = self._ensure_wallet(wallet)
        with lock:
            if wallet not in self._buffers:  # evicted concurrently
                return None
            records = list(self._buffers[wallet])
            # Prepare pre-computed Benford metrics
            benford_metrics = {h: s.to_metrics() for h, s in self._benford_sketches[wallet].items()}
            # Prepare per-pair sketches for cross-asset features.
            # Shallow copy to avoid RuntimeError if new pairs are added during iteration.
            pair_benford_sketches = dict(self._pair_benford_sketches[wallet])

        if not records:
            return None

        wallet_df = pd.DataFrame(records)
        features = build_feature_vector(
            wallet,
            wallet_df,
            all_pairs_df=wallet_df,
            benford_metrics=benford_metrics,
            pair_benford_sketches=pair_benford_sketches,
        )
        return pd.Series(features).fillna(0.0)

    def wallet_trade_count(self, wallet: str) -> int:
        """Return the number of trades currently buffered for *wallet*."""
        with self._registry_lock:
            buf = self._buffers.get(wallet)
        if buf is None:
            return 0
        # The deque's own lock isn't needed for len() — CPython's GIL makes
        # len() of a deque atomic, and a brief race here is acceptable for a
        # count-only read.
        return len(buf)

    def all_wallets(self) -> list[str]:
        """Return all wallets currently tracked in the buffer."""
        with self._registry_lock:
            return list(self._buffers.keys())

    def stats(self) -> dict[str, float | int | None]:
        """Return occupancy/eviction stats and publish them as Prometheus gauges."""
        with self._registry_lock:
            wallets = len(self._buffers)
            trades = sum(len(b) for b in self._buffers.values())
        occupancy = wallets / self.max_wallets if self.max_wallets else 0.0
        if _BUFFER_WALLETS is not None:
            _BUFFER_WALLETS.set(wallets)
            _BUFFER_TRADES.set(trades)
            _BUFFER_OCCUPANCY.set(occupancy)
        return {
            "wallets": wallets,
            "trades": trades,
            "max_wallets": self.max_wallets,
            "max_trades": self.max_trades,
            "occupancy": occupancy,
            "evictions_total": self.evictions_total,
        }
