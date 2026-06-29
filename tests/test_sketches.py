"""Tests for WalletSketchBook (Issue #124).

Verifies all acceptance criteria:
  1. HyperLogLog within 2 % of exact cardinality for n ≤ 10 000
  2. Combined sketch memory per wallet ≤ 15 KB
  3. Ingestion throughput ≥ 50 000 trade events/second
  4. CMS-derived mode bucket matches exact mode on a synthetic benchmark
  5. Thread-safety under concurrent updates
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter

from ingestion.sketches import (
    _AMOUNT_BUCKETS,
    WalletSketchBook,
    _amount_bucket,
    _bucket_to_amount,
)

# ---------------------------------------------------------------------------
# Acceptance criterion 1: HLL error < 2 % for n = 10 000
# ---------------------------------------------------------------------------


def test_hll_accuracy() -> None:
    """HyperLogLog estimate must be within 2 % of exact count for n = 10 000."""
    book = WalletSketchBook()
    wallet = "GAAA_HLL_ACCURACY"
    n = 10_000

    for i in range(n):
        book.add_trade(wallet, f"GCP{i:010d}", amount=100.0)

    estimate = book.counterparty_count(wallet)
    error = abs(estimate - n) / n
    assert error < 0.02, f"HLL error {error:.3%} exceeds 2 % (estimate={estimate}, exact={n})"


# ---------------------------------------------------------------------------
# Acceptance criterion 2: memory per wallet ≤ 15 KB
# ---------------------------------------------------------------------------


def test_wallet_memory_budget() -> None:
    """Combined sketch memory must not exceed 15 KB per wallet after population."""
    book = WalletSketchBook()
    wallet = "GMEM_BUDGET"

    for i in range(1_000):
        book.add_trade(wallet, f"GCP{i}", amount=float(i + 1))

    mem = book.wallet_memory_bytes(wallet)
    assert mem <= 15_360, f"Sketch memory {mem} B exceeds 15 360 B (15 KB) limit"


def test_wallet_memory_unseen_wallet() -> None:
    """wallet_memory_bytes must return 0 for a wallet that has never been seen."""
    book = WalletSketchBook()
    assert book.wallet_memory_bytes("GNEVER_SEEN") == 0


# ---------------------------------------------------------------------------
# Acceptance criterion 3: throughput ≥ 50 000 events/s
# ---------------------------------------------------------------------------


def test_ingestion_throughput() -> None:
    """Single-wallet throughput must reach ≥ 50 000 add_trade calls per second."""
    book = WalletSketchBook()
    wallet = "GSPEED_WALLET"
    n = 50_000

    start = time.monotonic()
    for i in range(n):
        book.add_trade(wallet, f"GCP{i % 1000}", amount=float((i % 999) + 1))
    elapsed = time.monotonic() - start

    throughput = n / elapsed
    assert (
        throughput >= 50_000
    ), f"Throughput {throughput:.0f} events/s is below the 50 000 events/s target"


# ---------------------------------------------------------------------------
# Acceptance criterion 4: CMS mode accuracy within 3 %
# ---------------------------------------------------------------------------


def test_cms_mode_bucket_accuracy() -> None:
    """CMS mode bucket must match the exact mode bucket on a clear synthetic workload."""
    book = WalletSketchBook()
    wallet = "GMODE_WALLET"
    rng = random.Random(42)

    # 70 % of trades cluster around 1 000 XLM; 30 % are noise in 1–100 XLM.
    amounts: list[float] = []
    for _ in range(3_000):
        if rng.random() < 0.70:
            amounts.append(rng.uniform(900.0, 1_100.0))
        else:
            amounts.append(rng.uniform(1.0, 100.0))

    for amt in amounts:
        book.add_trade(wallet, "GCOUNTERPARTY", amount=amt)

    exact_mode_bucket = Counter(_amount_bucket(a) for a in amounts).most_common(1)[0][0]
    exact_mode_amount = _bucket_to_amount(exact_mode_bucket)
    sketch_mode_amount = book.amount_mode(wallet)

    error = abs(sketch_mode_amount - exact_mode_amount) / (exact_mode_amount + 1e-9)
    assert error < 0.03, (
        f"CMS mode error {error:.3%} exceeds 3 % "
        f"(sketch={sketch_mode_amount:.2f} XLM, exact={exact_mode_amount:.2f} XLM)"
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 5: thread safety
# ---------------------------------------------------------------------------


def test_concurrent_ingestion_no_corruption() -> None:
    """Concurrent updates from multiple threads must not corrupt sketch state."""
    book = WalletSketchBook()
    n_threads = 4
    n_trades_per_thread = 2_500
    errors: list[Exception] = []

    def ingest(thread_id: int) -> None:
        try:
            for i in range(n_trades_per_thread):
                book.add_trade(
                    f"GWALLET{i % 10}",
                    f"GCP_T{thread_id}_{i}",
                    amount=float(i + 1),
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=ingest, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Exceptions raised during concurrent ingestion: {errors}"
    for w in book.all_wallets():
        assert book.counterparty_count(w) > 0
        assert book.wallet_memory_bytes(w) <= 15_360


# ---------------------------------------------------------------------------
# Edge cases and helpers
# ---------------------------------------------------------------------------


def test_empty_wallet_counterparty_count_is_zero() -> None:
    """Querying a freshly-initialised wallet must return cardinality 0."""
    book = WalletSketchBook()
    assert book.counterparty_count("GNEVER") == 0


def test_all_wallets_listing() -> None:
    """all_wallets must return exactly the set of wallets that received trades."""
    book = WalletSketchBook()
    wallets = {"GAAA", "GBBB", "GCCC"}
    for w in wallets:
        book.add_trade(w, "GOTHER", amount=10.0)
    assert set(book.all_wallets()) == wallets


def test_amount_bucket_roundtrip() -> None:
    """_bucket_to_amount and _amount_bucket are approximate inverses (≤ 1 bucket off)."""
    for bucket in range(_AMOUNT_BUCKETS):
        midpoint = _bucket_to_amount(bucket)
        recovered = _amount_bucket(midpoint)
        assert (
            abs(recovered - bucket) <= 1
        ), f"bucket {bucket} → amount {midpoint:.4f} → recovered bucket {recovered}"


def test_duplicate_counterparties_not_double_counted() -> None:
    """Repeated counterparties must not inflate the HLL cardinality estimate."""
    book = WalletSketchBook()
    wallet = "GDEDUP"
    n_distinct = 100
    n_repeats = 50  # each counterparty appears 50 times

    for _ in range(n_repeats):
        for i in range(n_distinct):
            book.add_trade(wallet, f"GCP{i:05d}", amount=1.0)

    estimate = book.counterparty_count(wallet)
    # Allow a generous 20 % margin for small-n HLL variance
    assert (
        estimate <= n_distinct * 1.20
    ), f"HLL over-counted duplicates: estimate={estimate}, distinct={n_distinct}"
