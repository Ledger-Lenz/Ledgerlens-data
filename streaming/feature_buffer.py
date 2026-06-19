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
from collections import deque
from typing import TYPE_CHECKING

import pandas as pd

from detection.feature_engineering import build_feature_vector
from ingestion.data_models import Trade

if TYPE_CHECKING:
    pass


class FeatureBuffer:
    """Per-wallet rolling deque of recent trades, safe for concurrent access."""

    def __init__(self, max_trades: int = 1000) -> None:
        self.max_trades = max_trades
        # Guards creation of new wallet entries in _buffers/_locks.
        self._registry_lock = threading.RLock()
        self._buffers: dict[str, deque] = {}
        self._wallet_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_wallet(self, wallet: str) -> threading.Lock:
        """Return the lock for *wallet*, creating both lock and deque if absent."""
        with self._registry_lock:
            if wallet not in self._wallet_locks:
                self._wallet_locks[wallet] = threading.Lock()
                self._buffers[wallet] = deque(maxlen=self.max_trades)
            return self._wallet_locks[wallet]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, trade: Trade) -> None:
        """Add *trade* to both ``base_account`` and ``counter_account`` buffers.

        When a wallet's deque is at capacity, ``deque(maxlen=…)`` automatically
        evicts the oldest entry on ``append()``.
        """
        record = {
            "trade_id": trade.trade_id,
            "ledger_close_time": trade.ledger_close_time,
            "base_account": trade.base_account,
            "counter_account": trade.counter_account,
            "base_asset": str(trade.base_asset.code),
            "counter_asset": str(trade.counter_asset.code),
            "amount": trade.amount,
        }
        for wallet in (trade.base_account, trade.counter_account):
            lock = self._ensure_wallet(wallet)
            with lock:
                self._buffers[wallet].append(record)

    def get_feature_row(self, wallet: str) -> pd.Series | None:
        """Build and return the feature row for *wallet*.

        Returns ``None`` if the wallet has no trades in the buffer.
        """
        lock = self._ensure_wallet(wallet)
        with lock:
            records = list(self._buffers[wallet])

        if not records:
            return None

        wallet_df = pd.DataFrame(records)
        features = build_feature_vector(wallet, wallet_df, all_pairs_df=wallet_df)
        return pd.Series(features)

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
