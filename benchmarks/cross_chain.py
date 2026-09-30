"""
benchmarks/cross_chain.py — Labelled benchmarks for cross-chain detection.

Two reproducible labelled sets:

  - Identity pairs (Issue #879): known same-entity and coincidental
    cross-chain address pairs with the evidence each would produce
    (bridge memo, amount fingerprint, timing correlation).  Used to report
    precision/recall of ``combine_confidence`` across thresholds and to
    justify ``config.CROSS_CHAIN_MIN_CONFIDENCE``.
  - Bridge transactions (Issue #880): lock-and-mint and liquidity-pool
    bridge transfers plus look-alike non-bridge transactions (same-chain
    wrapping, same-chain DEX swaps, mismatched mints, spoofed pool events).
    Used to report mechanism classification accuracy and per-mechanism
    precision/recall versus the unified (mechanism-agnostic) baseline.

Run ``python -m benchmarks.cross_chain`` to print the report.  Results are
recorded in docs/cross_chain_identity.md.
"""

from __future__ import annotations

import json
import random
from typing import Any

from detection.cross_chain.bridge_mechanisms import (
    LIQUIDITY_POOL,
    LOCK_AND_MINT,
    analyze_bridge_transaction,
    detect_bridge_unified,
)
from detection.cross_chain.confidence import Evidence, combine_confidence, precision_recall

DEFAULT_SEED = 879
THRESHOLDS = (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------
# Identity pairs (Issue #879)
# ---------------------------------------------------------------------------


def build_identity_pairs(
    n_true: int = 200, n_false: int = 400, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Labelled cross-chain identity pairs with the evidence each produces."""
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    for i in range(n_true):
        evidence: list[Evidence] = []
        if rng.random() < 0.35:
            evidence.append(Evidence("bridge", 1.0))
        if rng.random() < 0.6:
            evidence.append(Evidence("amount_fingerprint", rng.uniform(0.999, 1.0)))
        if rng.random() < 0.6:
            evidence.append(Evidence("timing_correlation", rng.uniform(0.8, 0.98)))
        if not evidence:
            evidence.append(Evidence("timing_correlation", rng.uniform(0.8, 0.95)))
        pairs.append({"pair": (f"GTRUE{i}", f"0xtrue{i}"), "label": True, "evidence": evidence})

    for i in range(n_false):
        evidence = []
        # Coincidental matches: round-number amounts and busy-hour activity overlap.
        # Coincidences rarely stack, so only ~10% carry both signals.
        roll = rng.random()
        if roll < 0.45 or roll >= 0.9:
            evidence.append(Evidence("amount_fingerprint", rng.uniform(0.999, 1.0)))
        if roll >= 0.45:
            evidence.append(Evidence("timing_correlation", rng.uniform(0.8, 0.88)))
        pairs.append({"pair": (f"GFALSE{i}", f"0xfalse{i}"), "label": False, "evidence": evidence})

    return pairs


def evaluate_identity_confidence(
    pairs: list[dict[str, Any]] | None = None,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> list[dict[str, float]]:
    """Precision/recall of combined confidence at each threshold."""
    pairs = pairs if pairs is not None else build_identity_pairs()
    scored = {p["pair"]: combine_confidence(p["evidence"]) for p in pairs}
    truth = {p["pair"] for p in pairs if p["label"]}
    return [precision_recall(scored, truth, t) for t in thresholds]


# ---------------------------------------------------------------------------
# Bridge transactions (Issue #880)
# ---------------------------------------------------------------------------


def _lock_and_mint(rng: random.Random, i: int, drift: float, dest_chain: str) -> dict[str, Any]:
    amount = round(rng.uniform(10, 50_000), 2)
    events = [
        {"type": "lock", "amount": amount, "chain": "stellar", "asset": "USDC"},
        {"type": "mint", "amount": amount * (1 - drift), "chain": dest_chain, "asset": "wUSDC"},
    ]
    if rng.random() < 0.3:
        events.insert(1, {"type": "transfer", "amount": 0.01, "chain": "stellar"})
    return {"id": f"lm-{i}", "events": events}


def _liquidity_pool(
    rng: random.Random, i: int, fee: float, dest_chain: str, delta_error: float = 0.0
) -> dict[str, Any]:
    amount = round(rng.uniform(10, 50_000), 2)
    out = amount * (1 - fee)
    src_before = rng.uniform(1e5, 1e6)
    dst_before = rng.uniform(1e5, 1e6)
    balances = {"stellar": {"before": src_before, "after": src_before + amount * (1 + delta_error)}}
    if dest_chain != "stellar":
        balances[dest_chain] = {"before": dst_before, "after": dst_before - out}
    return {
        "id": f"lp-{i}",
        "events": [
            {"type": "pool_deposit", "amount": amount, "chain": "stellar", "asset": "USDC"},
            {"type": "pool_withdraw", "amount": out, "chain": dest_chain, "asset": "USDC"},
        ],
        "pool_balances": balances,
    }


def build_bridge_transactions(
    n_per_class: int = 150, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Labelled bridge / look-alike transactions, tagged with their true mechanism."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    half = n_per_class // 2

    for i in range(n_per_class):
        # Genuine lock-and-mint; a minority carry decimal-rounding drift.
        drift = rng.uniform(0.0, 0.0015) if rng.random() < 0.2 else 0.0
        tx = _lock_and_mint(rng, i, drift, rng.choice(["ethereum", "solana"]))
        rows.append({"tx": tx, "mechanism": LOCK_AND_MINT, "is_bridge": True})
    for i in range(half):
        # Same-chain wrapping: identical lock/mint amounts, no chain hop.
        tx = _lock_and_mint(rng, n_per_class + i, 0.0, "stellar")
        rows.append({"tx": tx, "mechanism": LOCK_AND_MINT, "is_bridge": False})
    for i in range(half):
        # Unrelated mint that coincides with a lock (amounts 0.2-0.9% apart).
        tx = _lock_and_mint(rng, 2 * n_per_class + i, rng.uniform(0.002, 0.009), "ethereum")
        rows.append({"tx": tx, "mechanism": LOCK_AND_MINT, "is_bridge": False})

    for i in range(n_per_class):
        tx = _liquidity_pool(rng, i, rng.uniform(0.0005, 0.008), rng.choice(["ethereum", "solana"]))
        rows.append({"tx": tx, "mechanism": LIQUIDITY_POOL, "is_bridge": True})
    for i in range(half):
        # Same-chain DEX swap: pool in/out with a 0.3% fee, never leaves Stellar.
        tx = _liquidity_pool(rng, n_per_class + i, 0.003, "stellar")
        rows.append({"tx": tx, "mechanism": LIQUIDITY_POOL, "is_bridge": False})
    for i in range(half):
        # Spoofed pool events: amounts look right but pool balances disagree.
        tx = _liquidity_pool(
            rng,
            2 * n_per_class + i,
            rng.uniform(0.001, 0.008),
            "ethereum",
            delta_error=rng.uniform(0.02, 0.1),
        )
        rows.append({"tx": tx, "mechanism": LIQUIDITY_POOL, "is_bridge": False})

    return rows


def _pr(preds: list[bool], labels: list[bool]) -> dict[str, float]:
    tp = sum(p and y for p, y in zip(preds, labels, strict=True))
    fp = sum(p and not y for p, y in zip(preds, labels, strict=True))
    fn = sum(y and not p for p, y in zip(preds, labels, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "n": float(len(labels))}


def evaluate_bridge_mechanisms(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Classification accuracy and per-mechanism precision/recall vs the unified baseline."""
    rows = rows if rows is not None else build_bridge_transactions()
    analyses = [analyze_bridge_transaction(r["tx"]) for r in rows]
    accuracy = sum(
        a["mechanism"] == r["mechanism"] for a, r in zip(analyses, rows, strict=True)
    ) / len(rows)

    per_mechanism: dict[str, dict[str, Any]] = {}
    for mechanism in (LOCK_AND_MINT, LIQUIDITY_POOL):
        idx = [i for i, r in enumerate(rows) if r["mechanism"] == mechanism]
        labels = [rows[i]["is_bridge"] for i in idx]
        per_mechanism[mechanism] = {
            "mechanism_specific": _pr([analyses[i]["is_bridge"] for i in idx], labels),
            "unified": _pr([detect_bridge_unified(rows[i]["tx"]) for i in idx], labels),
        }
    return {"classification_accuracy": accuracy, "per_mechanism": per_mechanism}


def main() -> None:
    report = {
        "identity_confidence": evaluate_identity_confidence(),
        "bridge_mechanisms": evaluate_bridge_mechanisms(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
