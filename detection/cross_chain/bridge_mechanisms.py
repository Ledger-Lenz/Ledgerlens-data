"""Mechanism-aware bridge transaction detection (Issue #880).

Lock-and-mint bridges and liquidity-pool (swap-based) bridges leave
structurally different on-chain footprints:

* **lock-and-mint** - the source asset is locked (or burned) and the same
  amount of a wrapped asset is minted (or unlocked) on the destination chain.
  The signature is a 1:1 lock/burn <-> mint/unlock amount match.
* **liquidity-pool** - the user deposits into a pool on the source chain and
  a sister pool on the destination chain pays out ``amount - fee``.  The
  signature is a pair of opposite pool balance deltas on *different* chains
  with a small, bounded fee.

Transactions are first classified by mechanism, then scored by that
mechanism's own heuristics.  Mechanisms are pluggable: subclass
``BridgeMechanismHandler`` and decorate it with ``@register_bridge_mechanism``.

A bridge transaction record looks like::

    {
        "id": "...",
        "events": [
            {"type": "lock", "amount": 100.0, "chain": "stellar", "asset": "USDC"},
            {"type": "mint", "amount": 100.0, "chain": "ethereum", "asset": "wUSDC"},
        ],
        # liquidity-pool bridges may also carry per-chain pool balances:
        "pool_balances": {
            "stellar": {"before": 1000.0, "after": 1100.0},
            "ethereum": {"before": 5000.0, "after": 4900.3},
        },
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

LOCK_AND_MINT = "lock_and_mint"
LIQUIDITY_POOL = "liquidity_pool"
UNKNOWN_MECHANISM = "unknown"

# Minimum classification score for a handler to claim a transaction.
MIN_CLASSIFICATION_SCORE = 0.5
# Minimum mechanism-specific confidence for a transaction to count as a bridge.
MIN_DETECTION_CONFIDENCE = 0.5

_REGISTRY: dict[str, BridgeMechanismHandler] = {}


def _events(tx: dict[str, Any], types: set[str]) -> list[dict[str, Any]]:
    return [e for e in tx.get("events") or [] if str(e.get("type", "")).lower() in types]


def _total(events: list[dict[str, Any]]) -> float:
    return sum(float(e.get("amount") or 0.0) for e in events)


def _chains(events: list[dict[str, Any]]) -> set[str]:
    return {str(e.get("chain", "")).lower() for e in events if e.get("chain")}


class BridgeMechanismHandler(ABC):
    """Classification + feature extraction + detection for one bridge mechanism."""

    name: str = UNKNOWN_MECHANISM

    @abstractmethod
    def classification_score(self, tx: dict[str, Any]) -> float:
        """How strongly ``tx`` looks like this mechanism, in [0, 1]."""

    @abstractmethod
    def extract_features(self, tx: dict[str, Any]) -> dict[str, float]:
        """Mechanism-specific features used by ``detect``."""

    @abstractmethod
    def detect(self, features: dict[str, float]) -> float:
        """Confidence in [0, 1] that the transaction is a genuine bridge transfer."""


def register_bridge_mechanism(
    cls: type[BridgeMechanismHandler],
) -> type[BridgeMechanismHandler]:
    """Class decorator registering a handler under its ``name``."""
    _REGISTRY[cls.name] = cls()
    return cls


def get_bridge_mechanisms() -> dict[str, BridgeMechanismHandler]:
    return dict(_REGISTRY)


def unregister_bridge_mechanism(name: str) -> None:
    _REGISTRY.pop(name, None)


@register_bridge_mechanism
class LockAndMintHandler(BridgeMechanismHandler):
    """Lock/burn on the source chain matched by mint/unlock on the destination."""

    name = LOCK_AND_MINT
    _SOURCE = {"lock", "burn"}
    _DEST = {"mint", "unlock"}
    # Lock-and-mint is 1:1; allow only rounding-level drift.
    amount_tolerance = 0.001

    def classification_score(self, tx: dict[str, Any]) -> float:
        events = tx.get("events") or []
        if not events:
            return 0.0
        relevant = _events(tx, self._SOURCE | self._DEST)
        return len(relevant) / len(events)

    def extract_features(self, tx: dict[str, Any]) -> dict[str, float]:
        source = _events(tx, self._SOURCE)
        dest = _events(tx, self._DEST)
        locked = _total(source)
        minted = _total(dest)
        ratio = min(locked, minted) / max(locked, minted) if locked > 0 and minted > 0 else 0.0
        cross_chain = bool(_chains(source) and _chains(dest) and _chains(source) != _chains(dest))
        return {
            "locked_amount": locked,
            "minted_amount": minted,
            "mint_burn_ratio": ratio,
            "has_source_leg": float(bool(source)),
            "has_destination_leg": float(bool(dest)),
            "cross_chain": float(cross_chain),
        }

    def detect(self, features: dict[str, float]) -> float:
        if not (features["has_source_leg"] and features["has_destination_leg"]):
            return 0.0
        if not features["cross_chain"]:
            return 0.0
        drift = 1.0 - features["mint_burn_ratio"]
        if drift <= self.amount_tolerance:
            return 1.0
        # Confidence decays linearly to zero at 3x the tolerance.
        return max(0.0, 1.0 - drift / (3 * self.amount_tolerance))


@register_bridge_mechanism
class LiquidityPoolHandler(BridgeMechanismHandler):
    """Pool deposit on the source chain matched by a pool payout on the destination."""

    name = LIQUIDITY_POOL
    _DEPOSIT = {"pool_deposit", "swap_in", "deposit"}
    _PAYOUT = {"pool_withdraw", "swap_out", "withdraw"}
    # Pool bridges charge a fee; above this it is not a plausible bridge fee.
    max_fee_ratio = 0.01
    # How closely pool balance deltas must track the event amounts.
    delta_tolerance = 0.001

    def classification_score(self, tx: dict[str, Any]) -> float:
        events = tx.get("events") or []
        relevant = _events(tx, self._DEPOSIT | self._PAYOUT | {"swap"})
        score = len(relevant) / len(events) if events else 0.0
        if tx.get("pool_balances"):
            score = min(1.0, score + 0.5)
        return score

    def extract_features(self, tx: dict[str, Any]) -> dict[str, float]:
        deposits = _events(tx, self._DEPOSIT)
        payouts = _events(tx, self._PAYOUT)
        amount_in = _total(deposits)
        amount_out = _total(payouts)
        fee_ratio = 1.0 - amount_out / amount_in if amount_in > 0 and amount_out > 0 else 1.0
        dep_chains = _chains(deposits)
        pay_chains = _chains(payouts)
        cross_chain = bool(dep_chains and pay_chains and dep_chains.isdisjoint(pay_chains))

        balances = tx.get("pool_balances") or {}
        source_delta = sum(
            float(balances[c]["after"]) - float(balances[c]["before"])
            for c in dep_chains
            if c in balances
        )
        dest_delta = sum(
            float(balances[c]["before"]) - float(balances[c]["after"])
            for c in pay_chains
            if c in balances
        )
        has_balances = float(any(c in balances for c in dep_chains | pay_chains))
        delta_error = 0.0
        if has_balances and amount_in > 0:
            delta_error = max(
                abs(source_delta - amount_in) / amount_in,
                abs(dest_delta - amount_out) / amount_in,
            )
        return {
            "amount_in": amount_in,
            "amount_out": amount_out,
            "fee_ratio": fee_ratio,
            "source_pool_delta": source_delta,
            "destination_pool_delta": dest_delta,
            "pool_delta_error": delta_error,
            "has_pool_balances": has_balances,
            "cross_chain": float(cross_chain),
        }

    def detect(self, features: dict[str, float]) -> float:
        if not features["cross_chain"]:
            return 0.0
        fee = features["fee_ratio"]
        if fee < 0.0 or fee > self.max_fee_ratio:
            return 0.0
        confidence = 1.0
        if features["has_pool_balances"] and features["pool_delta_error"] > self.delta_tolerance:
            confidence = max(0.0, 1.0 - features["pool_delta_error"] / (10 * self.delta_tolerance))
        return confidence


def classify_bridge_mechanism(tx: dict[str, Any]) -> tuple[str, float]:
    """Return ``(mechanism, score)`` for the best-matching registered handler."""
    best_name, best_score = UNKNOWN_MECHANISM, 0.0
    for name, handler in _REGISTRY.items():
        score = handler.classification_score(tx)
        if score > best_score:
            best_name, best_score = name, score
    if best_score < MIN_CLASSIFICATION_SCORE:
        return UNKNOWN_MECHANISM, best_score
    return best_name, best_score


def analyze_bridge_transaction(tx: dict[str, Any]) -> dict[str, Any]:
    """Classify ``tx`` by mechanism and apply that mechanism's detection heuristics."""
    mechanism, class_score = classify_bridge_mechanism(tx)
    handler = _REGISTRY.get(mechanism)
    if handler is None:
        return {
            "tx_id": tx.get("id") or tx.get("hash", ""),
            "mechanism": UNKNOWN_MECHANISM,
            "classification_score": class_score,
            "features": {},
            "confidence": 0.0,
            "is_bridge": False,
        }
    features = handler.extract_features(tx)
    confidence = handler.detect(features)
    return {
        "tx_id": tx.get("id") or tx.get("hash", ""),
        "mechanism": mechanism,
        "classification_score": class_score,
        "features": features,
        "confidence": confidence,
        "is_bridge": confidence >= MIN_DETECTION_CONFIDENCE,
    }


def detect_bridge_unified(tx: dict[str, Any], amount_tolerance: float = 0.01) -> bool:
    """Mechanism-agnostic baseline: first and last event amounts match.

    Kept as the benchmark reference the mechanism-specific path is compared
    against (see benchmarks/cross_chain.py).
    """
    events = tx.get("events") or []
    if len(events) < 2:
        return False
    first = float(events[0].get("amount") or 0.0)
    last = float(events[-1].get("amount") or 0.0)
    if first <= 0:
        return False
    return abs(first - last) / first <= amount_tolerance
