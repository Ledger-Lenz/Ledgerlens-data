"""Probabilistic data structures for high-throughput wallet feature extraction.

Implements HyperLogLog (counterparty cardinality) and Count-Min Sketch (trade
amount frequency) per wallet, enabling fixed-memory aggregations for 100k+
concurrent wallets at Stellar DEX throughput.

References
----------
- Cormode & Muthukrishnan, 'An Improved Data Stream Summary: The Count-Min Sketch' (2005)
- Flajolet et al., 'HyperLogLog: The analysis of a near-optimal cardinality estimation
  algorithm' (2007)
"""

from __future__ import annotations

import hashlib
import math
import struct
import threading
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# HyperLogLog — counterparty cardinality
# ---------------------------------------------------------------------------

# p=13 → m=8 192 registers (uint8, 8 KB), standard error ≈ 1.15 %
_HLL_P: Final[int] = 13
_HLL_M: Final[int] = 1 << _HLL_P  # 8 192

_ALPHA_M: Final[float] = 0.7213 / (1.0 + 1.079 / _HLL_M)

# ---------------------------------------------------------------------------
# Count-Min Sketch — trade-amount frequency
# ---------------------------------------------------------------------------

# 6 rows × 256 columns of uint16 → 3 072 bytes (≈ 3 KB)
_CMS_D: Final[int] = 6
_CMS_W: Final[int] = 256

# Pairwise hash parameters (a, b): h_i(x) = (a * x + b) & 0xFFFFFFFF % W
_CMS_PARAMS: Final[tuple[tuple[int, int], ...]] = (
    (2654435761, 1234567891),
    (2246822519, 2345678901),
    (3266489917, 3456789012),
    (668265263, 4567890123 % (1 << 32)),
    (374761393, 5678901234 % (1 << 32)),
    (1452429799, 6789012345 % (1 << 32)),
)

# ---------------------------------------------------------------------------
# Amount discretisation
# ---------------------------------------------------------------------------

_AMOUNT_MIN: Final[float] = 0.01
_AMOUNT_MAX: Final[float] = 1e9
_AMOUNT_BUCKETS: Final[int] = 32

_LOG_AMOUNT_MIN: Final[float] = math.log(_AMOUNT_MIN)
_LOG_AMOUNT_MAX: Final[float] = math.log(_AMOUNT_MAX)
_LOG_AMOUNT_RANGE: Final[float] = _LOG_AMOUNT_MAX - _LOG_AMOUNT_MIN


def _amount_bucket(amount: float) -> int:
    """Return log-scale bucket index in ``[0, _AMOUNT_BUCKETS)`` for *amount*."""
    if amount <= _AMOUNT_MIN:
        return 0
    if amount >= _AMOUNT_MAX:
        return _AMOUNT_BUCKETS - 1
    ratio = (math.log(amount) - _LOG_AMOUNT_MIN) / _LOG_AMOUNT_RANGE
    return min(int(ratio * _AMOUNT_BUCKETS), _AMOUNT_BUCKETS - 1)


def _bucket_to_amount(bucket: int) -> float:
    """Return the representative XLM amount (log-scale midpoint) for *bucket*."""
    mid = (bucket + 0.5) / _AMOUNT_BUCKETS
    return math.exp(_LOG_AMOUNT_MIN + mid * _LOG_AMOUNT_RANGE)


# ---------------------------------------------------------------------------
# Internal: 64-bit hash + rho
# ---------------------------------------------------------------------------


def _hash64(value: bytes) -> int:
    """Deterministic 64-bit integer hash of *value* via MD5."""
    digest = hashlib.md5(value, usedforsecurity=False).digest()
    return struct.unpack_from("<Q", digest)[0]


def _rho(w: int, max_bits: int) -> int:
    """Position of the leftmost 1-bit among *max_bits* bits (1-indexed from left).

    Returns ``max_bits + 1`` when *w* is zero (all bits are leading zeros).
    """
    if w == 0:
        return max_bits + 1
    return max_bits - w.bit_length() + 1


# ---------------------------------------------------------------------------
# Internal: HyperLogLog
# ---------------------------------------------------------------------------


class _HyperLogLog:
    """Minimal HyperLogLog for distinct-counterparty cardinality estimation.

    Uses p=13 (m=8 192 registers, uint8) for ≈ 8 KB and ≈ 1.15 % standard error.
    """

    __slots__ = ("_registers",)

    def __init__(self) -> None:
        self._registers = np.zeros(_HLL_M, dtype=np.uint8)

    def update(self, value: bytes) -> None:
        """Ingest *value* into the sketch."""
        h = _hash64(value)
        j = h >> (64 - _HLL_P)  # top p bits → register index
        w = h & ((1 << (64 - _HLL_P)) - 1)  # remaining 51 bits
        # Inline _rho to eliminate function-call overhead in the hot path.
        rho_val = (64 - _HLL_P + 1) if w == 0 else (64 - _HLL_P - w.bit_length() + 1)
        if rho_val > self._registers[j]:
            self._registers[j] = rho_val

    def count(self) -> float:
        """Return the estimated distinct-element cardinality."""
        regs = self._registers.astype(np.float64)
        Z = 1.0 / float(np.sum(np.power(2.0, -regs)))
        E = _ALPHA_M * float(_HLL_M) ** 2 * Z

        # Small-range correction: use linear counting when many registers are empty.
        if E <= 2.5 * _HLL_M:
            V = int(np.sum(regs == 0))
            if V > 0:
                E = _HLL_M * math.log(float(_HLL_M) / V)

        # Large-range correction (only relevant for n > 2^32 / 30 ≈ 143 M).
        elif E > (1 << 32) / 30.0:
            E = -(1 << 32) * math.log(1.0 - E / (1 << 32))

        return E

    def nbytes(self) -> int:
        """Memory used by the register array in bytes."""
        return int(self._registers.nbytes)


# ---------------------------------------------------------------------------
# Internal: Count-Min Sketch
# ---------------------------------------------------------------------------


class _CountMinSketch:
    """Count-Min Sketch for trade-amount-bucket frequency estimation.

    Uses 6 rows × 256 columns of uint16 counters → 3 072 bytes (≈ 3 KB).
    """

    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table = np.zeros((_CMS_D, _CMS_W), dtype=np.uint16)

    @staticmethod
    def _col(item: int, row: int) -> int:
        a, b = _CMS_PARAMS[row]
        return int((a * item + b) & 0xFFFFFFFF) % _CMS_W

    def add(self, item: int) -> None:
        """Increment the frequency counter for *item* in every hash row.

        (a, b) parameters are unpacked inline — avoids 6 staticmethod call
        overheads per trade on the hot path.
        """
        table = self._table
        for row, (a, b) in enumerate(_CMS_PARAMS):
            col = int((a * item + b) & 0xFFFFFFFF) % _CMS_W
            val = int(table[row, col]) + 1
            table[row, col] = val if val < 65535 else 65535  # saturating add

    def estimate(self, item: int) -> int:
        """Return the minimum-count estimate (upper bound) for *item*."""
        return int(min(self._table[row, self._col(item, row)] for row in range(_CMS_D)))

    def mode_bucket(self) -> int:
        """Return the amount-bucket index with the highest estimated frequency."""
        totals = [self.estimate(b) for b in range(_AMOUNT_BUCKETS)]
        return int(np.argmax(totals))

    def nbytes(self) -> int:
        """Memory used by the CMS table in bytes."""
        return int(self._table.nbytes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WalletSketchBook:
    """Per-wallet probabilistic feature store for high-throughput ingestion.

    Each wallet is tracked with two fixed-size sketches:

    - **HyperLogLog** (p=13, ≈ 8 KB, ≈ 1.15 % standard error) for
      counterparty-cardinality estimation.
    - **Count-Min Sketch** (6 × 256 uint16, ≈ 3 KB) for trade-amount
      frequency, enabling ``amount_mode`` queries.

    Total memory per wallet: ≈ 11 KB — well within the 15 KB budget, and
    constant regardless of trade volume.

    Thread safety
    -------------
    A dedicated ``threading.Lock`` is held per wallet during reads/writes so
    that unrelated wallets can be updated concurrently with minimal contention.
    A registry lock guards creation of new wallet entries only.

    Parameters
    ----------
    max_wallets:
        Expected upper bound on concurrent wallets (informational; no hard cap
        is enforced internally).
    """

    def __init__(self, max_wallets: int = 100_000) -> None:
        self._max_wallets = max_wallets
        self._registry_lock = threading.Lock()
        self._hll: dict[str, _HyperLogLog] = {}
        self._cms: dict[str, _CountMinSketch] = {}
        self._wallet_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_wallet(self, wallet: str) -> threading.Lock:
        with self._registry_lock:
            if wallet not in self._wallet_locks:
                self._wallet_locks[wallet] = threading.Lock()
                self._hll[wallet] = _HyperLogLog()
                self._cms[wallet] = _CountMinSketch()
            return self._wallet_locks[wallet]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_trade(self, wallet: str, counterparty: str, amount: float) -> None:
        """Ingest one trade event for *wallet*.

        Updates the HyperLogLog with *counterparty* and the Count-Min Sketch
        with the log-scale bucket of *amount*.

        The registry lock is bypassed on the hot path via a try/except dict
        lookup; the slow path through ``_ensure_wallet`` is taken only on the
        first call for a new wallet.
        """
        try:
            lock = self._wallet_locks[wallet]
        except KeyError:
            lock = self._ensure_wallet(wallet)
        bucket = _amount_bucket(amount)
        encoded = counterparty.encode()
        with lock:
            self._hll[wallet].update(encoded)
            self._cms[wallet].add(bucket)

    def counterparty_count(self, wallet: str) -> int:
        """Return the HyperLogLog estimate of distinct counterparty count.

        Expected standard error ≈ 1.15 % (p = 13).
        """
        lock = self._ensure_wallet(wallet)
        with lock:
            return int(self._hll[wallet].count())

    def amount_mode(self, wallet: str) -> float:
        """Return the representative XLM amount for the most frequent trade-size bucket.

        Uses the Count-Min Sketch mode estimate and maps it back to the log-scale
        bucket midpoint.
        """
        lock = self._ensure_wallet(wallet)
        with lock:
            bucket = self._cms[wallet].mode_bucket()
        return _bucket_to_amount(bucket)

    def wallet_memory_bytes(self, wallet: str) -> int:
        """Return combined sketch memory (bytes) for *wallet*.

        Returns 0 for wallets that have never received a trade.
        """
        with self._registry_lock:
            if wallet not in self._hll:
                return 0
        lock = self._wallet_locks[wallet]
        with lock:
            return self._hll[wallet].nbytes() + self._cms[wallet].nbytes()

    def all_wallets(self) -> list[str]:
        """Return a snapshot list of all currently tracked wallet IDs."""
        with self._registry_lock:
            return list(self._hll.keys())
