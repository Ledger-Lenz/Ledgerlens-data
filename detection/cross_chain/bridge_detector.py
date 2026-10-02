"""Bridge transaction detector.

Identifies Stellar <-> Ethereum/Solana bridge transfers from transaction lists
by detecting Ethereum/Solana addresses inside SEP-0006 memos.

Also implements detect_bridge_wash_trade() for round-trip anchor analysis
(Issue #278): identifies wallets that send and receive bridge payments to/from
the same anchor within BRIDGE_ROUNDTRIP_WINDOW_HOURS, computing a ratio that
flags wash traders exploiting cross-chain opacity.

Bridge transactions carrying on-chain ``events`` are classified by bridge
mechanism (lock-and-mint vs liquidity-pool) and scored with
mechanism-specific heuristics from ``bridge_mechanisms`` (Issue #880).

Where the bridge publishes a verifiable proof of the transfer (Issue #884),
detect_bridge_links() verifies it via ``integrations.bridge_attestation`` and
upgrades the link to a high-confidence ``attestation`` level; otherwise the
link stays heuristic (memo correlation). See docs/bridge_attestation.md.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

from detection.cross_chain.bridge_mechanisms import analyze_bridge_transaction
from integrations.bridge_attestation import (
    ATTESTATION_CONFIDENCE,
    ATTESTATION_HEURISTIC,
    BridgeAttestationVerifier,
    attested_recipient_matches,
)

logger = logging.getLogger(__name__)

# Stellar G-address: G + 55 base32 chars = 56 chars total
_STELLAR_GADDR_RE = re.compile(r"^G[A-Z2-7]{55}$")

# EVM address regex: 0x followed by 40 hex chars
_EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_EVM_NO_PREFIX_RE = re.compile(r"^[a-fA-F0-9]{40}$")

# Solana address regex: base58 string of length 32 to 44
_SOL_ADDR_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def bytes_to_base58(b: bytes) -> str:
    """Encode bytes to base58 string."""
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = int.from_bytes(b, "big")
    res = []
    while n > 0:
        n, r = divmod(n, 58)
        res.append(alphabet[r])
    for x in b:
        if x == 0:
            res.append(alphabet[0])
        else:
            break
    return "".join(reversed(res))


class BridgeDetector:
    """Detects cross-chain identities via Stellar bridge transaction memos."""

    def __init__(
        self,
        anchor_addresses: list[str] | None = None,
        attestation_verifier: BridgeAttestationVerifier | None = None,
    ):
        self.anchor_addresses = set(anchor_addresses or [])
        self._attestation_verifier = attestation_verifier

    @property
    def attestation_verifier(self) -> BridgeAttestationVerifier:
        if self._attestation_verifier is None:
            self._attestation_verifier = BridgeAttestationVerifier()
        return self._attestation_verifier

    def _attest(
        self, tx: dict[str, Any], linked_addr: str, chain: str
    ) -> tuple[str, float, dict[str, Any]]:
        """Return (attestation level, confidence, details) for a memo-derived link.

        A proof only upgrades the link when it verifies *and* attests a transfer
        to ``linked_addr``; a valid proof for some other transfer is ignored.
        """
        proof = tx.get("bridge_proof")
        heuristic = ATTESTATION_CONFIDENCE[ATTESTATION_HEURISTIC]
        if not proof:
            return ATTESTATION_HEURISTIC, heuristic, {}
        result = self.attestation_verifier.verify(proof)
        if not result.verified:
            return ATTESTATION_HEURISTIC, heuristic, {"proof_rejected": result.reason}
        if not attested_recipient_matches(result, linked_addr, chain):
            return (
                ATTESTATION_HEURISTIC,
                heuristic,
                {"proof_rejected": "proof does not attest a transfer to the linked address"},
            )
        return result.level, result.confidence, {"bridge": result.bridge, **result.details}

    def parse_memo_address(self, memo_type: str, memo_val: Any) -> tuple[str, str] | None:
        """Parse an Ethereum or Solana address from a memo.

        Returns (address, chain) if parsed, otherwise None.
        """
        if not memo_val:
            return None

        memo_type = memo_type.lower()

        if memo_type == "text":
            # Memo is a text string
            memo_str = str(memo_val).strip()
            # Check EVM address
            if _EVM_ADDR_RE.match(memo_str):
                return memo_str.lower(), "ethereum"
            if _EVM_NO_PREFIX_RE.match(memo_str):
                return f"0x{memo_str}".lower(), "ethereum"
            # Check Solana address
            if _SOL_ADDR_RE.match(memo_str):
                return memo_str, "solana"

        elif memo_type in ("hash", "return"):
            # Memo is 32-byte hash. Might be represented as a base64 string or hex string.
            raw_bytes = None
            if isinstance(memo_val, bytes):
                raw_bytes = memo_val
            elif isinstance(memo_val, str):
                memo_str = memo_val.strip()
                # Try base64
                try:
                    b = base64.b64decode(memo_str)
                    if len(b) == 32:
                        raw_bytes = b
                except Exception:
                    pass
                # Try hex if base64 failed or didn't produce 32 bytes
                if not raw_bytes:
                    try:
                        b = bytes.fromhex(memo_str)
                        if len(b) == 32:
                            raw_bytes = b
                    except Exception:
                        pass

            if not raw_bytes or len(raw_bytes) != 32:
                return None

            # 1. Check for padded EVM address (20 bytes)
            # Left padded (12 bytes of zeros followed by 20 bytes EVM)
            if raw_bytes[:12] == b"\x00" * 12:
                evm_addr = "0x" + raw_bytes[12:].hex()
                return evm_addr.lower(), "ethereum"
            # Right padded (20 bytes EVM followed by 12 bytes of zeros)
            if raw_bytes[20:] == b"\x00" * 12:
                evm_addr = "0x" + raw_bytes[:20].hex()
                return evm_addr.lower(), "ethereum"

            # 2. Treat the raw 32 bytes as a Solana public key
            sol_addr = bytes_to_base58(raw_bytes)
            # Verify it is a valid base58 Solana address format
            if _SOL_ADDR_RE.match(sol_addr):
                return sol_addr, "solana"

        return None

    def detect_bridge_links(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify bridge links from a list of transaction records.

        Each transaction record should look like:
        {
            "id": "...",
            "source_account": "...",
            "memo_type": "...",
            "memo": "...",
            "from": "...",  # optional
            "to": "...",    # optional
            "bridge_proof": {"bridge": "wormhole", "vaa": "<base64>"},  # optional
        }

        Each link carries ``attestation`` (``zk_proof`` / ``guardian_signatures``
        / ``heuristic``) and a ``confidence`` weighted by that level.
        """
        links = []
        for tx in transactions:
            memo_type = tx.get("memo_type")
            memo = tx.get("memo")
            if not memo_type or not memo:
                continue

            parsed = self.parse_memo_address(memo_type, memo)
            if not parsed:
                continue

            linked_addr, chain = parsed
            tx_id = tx.get("id") or tx.get("hash", "")

            # Determine the Stellar user address
            # Default to transaction fee payer/source_account
            stellar_addr = tx.get("source_account")

            # If there's an explicit payment source or destination, we can refine:
            tx_from = tx.get("from")
            tx_to = tx.get("to")

            if tx_from and tx_to:
                # If we know the anchor address, the user is the opposite party
                if tx_from in self.anchor_addresses:
                    stellar_addr = tx_to
                elif tx_to in self.anchor_addresses:
                    stellar_addr = tx_from
                else:
                    # Otherwise, use the source_account, or fall back to 'from'
                    stellar_addr = stellar_addr or tx_from

            if stellar_addr:
                attestation, confidence, details = self._attest(tx, linked_addr, chain)
                link = {
                    "stellar_address": stellar_addr,
                    "linked_address": linked_addr,
                    "chain": chain,
                    "tx_id": tx_id,
                    "memo": str(memo),
                    "confidence": confidence,
                    "attestation": attestation,
                    "attestation_details": details,
                }
                if tx.get("events"):
                    analysis = analyze_bridge_transaction(tx)
                    link["mechanism"] = analysis["mechanism"]
                    link["mechanism_confidence"] = analysis["confidence"]
                    link["mechanism_features"] = analysis["features"]
                links.append(link)

        return links

    def classify_transactions(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify each transaction by bridge mechanism and score it.

        Returns one ``analyze_bridge_transaction`` result per transaction:
        ``mechanism``, ``classification_score``, ``features``, ``confidence``
        and ``is_bridge``.
        """
        return [analyze_bridge_transaction(tx) for tx in transactions]


# ---------------------------------------------------------------------------
# Bridge anchor address management
# ---------------------------------------------------------------------------

_BRIDGE_ANCHORS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "bridge_anchors.json"
)
_cached_anchors: frozenset[str] | None = None


def _validate_stellar_address(addr: str) -> bool:
    return bool(_STELLAR_GADDR_RE.match(addr))


def load_bridge_anchors(path: str | None = None) -> frozenset[str]:
    """Load and validate bridge anchor addresses from data/bridge_anchors.json.

    Validates all entries are well-formed Stellar G-addresses (56-char base32).
    Raises ValueError for any malformed address so misconfiguration is caught at
    startup rather than silently skipped at scoring time.
    """
    global _cached_anchors
    if _cached_anchors is not None and path is None:
        return _cached_anchors

    resolved = path or _BRIDGE_ANCHORS_PATH
    with open(resolved) as fh:
        data = json.load(fh)

    addrs: list[str] = []
    for entry in data.get("anchors", []):
        addr = entry if isinstance(entry, str) else entry["address"]
        if not _validate_stellar_address(addr):
            raise ValueError(f"Invalid Stellar G-address in bridge_anchors.json: {addr!r}")
        addrs.append(addr)

    result = frozenset(addrs)
    if path is None:
        _cached_anchors = result
    return result


# ---------------------------------------------------------------------------
# Wash-trade round-trip detection
# ---------------------------------------------------------------------------


def detect_bridge_wash_trade(
    wallet_id: str,
    transactions: list[dict[str, Any]],
    bridge_contracts: list[str] | None = None,
    window_hours: int | None = None,
) -> dict[str, Any]:
    """Detect bridge wash-trading via round-trip anchor analysis.

    A bridge round-trip occurs when the same wallet sends assets *to* a bridge
    anchor and later receives assets *from* the same anchor within
    `window_hours` (default: config.BRIDGE_ROUNDTRIP_WINDOW_HOURS = 72 h).

    Args:
        wallet_id:        Stellar G-address of the wallet being scored.
        transactions:     List of payment records (dicts with keys: ``from``,
                          ``to``, ``amount``, ``timestamp``, and optionally
                          ``anchor``).
        bridge_contracts: Override anchor address set. Defaults to
                          ``load_bridge_anchors()``.
        window_hours:     Round-trip matching window in hours.

    Returns:
        Dict with:
          - ``bridge_round_trip_ratio``: round-trips / total bridge txns (0–1)
          - ``bridge_round_trips``: matched round-trip count
          - ``total_bridge_txns``: total payments to/from any anchor
    """
    from config import config

    if window_hours is None:
        window_hours = config.BRIDGE_ROUNDTRIP_WINDOW_HOURS

    anchors: frozenset[str]
    if bridge_contracts is not None:
        for a in bridge_contracts:
            if not _validate_stellar_address(a):
                raise ValueError(f"Invalid bridge anchor address: {a!r}")
        anchors = frozenset(bridge_contracts)
    else:
        anchors = load_bridge_anchors()

    # Partition transactions into outbound (wallet→anchor) and inbound (anchor→wallet)
    outbound: list[dict[str, Any]] = []
    inbound: list[dict[str, Any]] = []

    for tx in transactions:
        tx_from = tx.get("from", "")
        tx_to = tx.get("to", "")
        ts_raw = tx.get("timestamp")
        if ts_raw is None:
            continue
        ts = datetime.fromisoformat(str(ts_raw)) if isinstance(ts_raw, str) else ts_raw
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        anchor_addr = tx_from if tx_from in anchors else (tx_to if tx_to in anchors else None)
        if anchor_addr is None:
            continue

        record = {**tx, "_ts": ts, "_anchor": anchor_addr}
        if tx_from == wallet_id and tx_to in anchors:
            outbound.append(record)
        elif tx_to == wallet_id and tx_from in anchors:
            inbound.append(record)

    total_bridge = len(outbound) + len(inbound)
    if total_bridge == 0:
        return {"bridge_round_trip_ratio": 0.0, "bridge_round_trips": 0, "total_bridge_txns": 0}

    from datetime import timedelta

    window = timedelta(hours=window_hours)
    matched_out: set[int] = set()
    matched_in: set[int] = set()
    round_trips = 0

    for i, out_tx in enumerate(outbound):
        if i in matched_out:
            continue
        for j, in_tx in enumerate(inbound):
            if j in matched_in:
                continue
            if (
                abs(in_tx["_ts"] - out_tx["_ts"]) <= window
                and in_tx["_anchor"] == out_tx["_anchor"]
            ):
                matched_out.add(i)
                matched_in.add(j)
                round_trips += 1
                break

    possible_round_trips = min(len(outbound), len(inbound))
    ratio = round_trips / possible_round_trips if possible_round_trips else 0.0
    return {
        "bridge_round_trip_ratio": ratio,
        "bridge_round_trips": round_trips,
        "total_bridge_txns": total_bridge,
    }
