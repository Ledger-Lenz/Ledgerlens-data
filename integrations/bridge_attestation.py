"""Cryptographic attestation verification for cross-chain bridge proofs (#884).

Bridge detection in ``detection/cross_chain`` is otherwise heuristic: a memo
that happens to contain an EVM/Solana address, or a pair of transfers whose
amount and timing line up. Some bridges publish a *verifiable* proof of each
lock/mint event, and when one is available we can replace "these two legs
look related" with "the bridge itself attested that this transfer happened".

Attestation levels (strongest first):

``zk_proof``
    A succinct validity proof (e.g. Groth16 over BN254) that the source-chain
    event was finalised. Verification is trust-minimised: it depends only on
    the verifying key, not on any committee. Verifiers register through
    :func:`register_zk_verifier` (see ``docs/bridge_attestation.md`` for the
    bridges that publish such proofs and why none is wired in yet).

``guardian_signatures``
    A committee attestation that can be checked offline, e.g. a Wormhole VAA
    signed by a supermajority (> 2/3) of the pinned guardian set. Verification
    is cryptographic (secp256k1 ``ecrecover`` against pinned guardian keys) but
    trusts the committee's honest majority rather than a proof of execution.

``heuristic``
    No proof available or verification failed; the link rests on
    memo/amount/timing correlation only.

The Ethereum-flavoured Keccak-256 used by Wormhole (original Keccak padding,
not NIST SHA3) is implemented here in pure Python so the verifier has no
dependency beyond ``ecdsa``, which is already installed.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ATTESTATION_ZK = "zk_proof"
ATTESTATION_GUARDIAN = "guardian_signatures"
ATTESTATION_HEURISTIC = "heuristic"

# Confidence assigned to a link by attestation level. Downstream scoring
# (identity-graph edges, risk propagation) consumes ``confidence`` directly,
# so these weights are what make verified links count for more.
ATTESTATION_CONFIDENCE: dict[str, float] = {
    ATTESTATION_ZK: 1.0,
    ATTESTATION_GUARDIAN: 0.98,
    # Heuristic links keep their own confidence but are capped below any
    # verified link so a verified edge always outranks an unverified one.
    ATTESTATION_HEURISTIC: 0.9,
}

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "bridge_attestation"


# ---------------------------------------------------------------------------
# Keccak-256 (Ethereum variant)
# ---------------------------------------------------------------------------

_KECCAK_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]  # fmt: skip
_KECCAK_ROT = [
    [0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56], [27, 20, 39, 8, 14],
]  # fmt: skip
_MASK64 = (1 << 64) - 1


def _rotl(x: int, n: int) -> int:
    return ((x << n) | (x >> (64 - n))) & _MASK64 if n else x


def _keccak_f(a: list[list[int]]) -> None:
    for rc in _KECCAK_RC:
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                a[x][y] ^= d[x]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rotl(a[x][y], _KECCAK_ROT[x][y])
        for x in range(5):
            for y in range(5):
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
        a[0][0] ^= rc


def keccak256(data: bytes) -> bytes:
    """Ethereum Keccak-256 (0x01 padding, unlike NIST SHA3-256's 0x06)."""
    rate = 136
    padded = bytearray(data) + b"\x01" + b"\x00" * ((-len(data) - 1) % rate)
    padded[-1] |= 0x80
    state = [[0] * 5 for _ in range(5)]
    for off in range(0, len(padded), rate):
        block = padded[off : off + rate]
        for i in range(rate // 8):
            state[i % 5][i // 5] ^= int.from_bytes(block[8 * i : 8 * i + 8], "little")
        _keccak_f(state)
    return b"".join(state[i % 5][i // 5].to_bytes(8, "little") for i in range(4))


# ---------------------------------------------------------------------------
# Wormhole VAA parsing + guardian-signature verification
# ---------------------------------------------------------------------------

WORMHOLE_CHAIN_IDS: dict[int, str] = {
    1: "solana", 2: "ethereum", 4: "bsc", 5: "polygon", 6: "avalanche",
    23: "arbitrum", 24: "optimism", 30: "base",
}  # fmt: skip
_EVM_CHAINS = frozenset(WORMHOLE_CHAIN_IDS.values()) - {"solana"}


class AttestationError(ValueError):
    """Raised for malformed attestation payloads."""


@dataclass(frozen=True)
class WormholeVAA:
    """A parsed (not yet verified) Wormhole v1 VAA."""

    guardian_set_index: int
    signatures: list[tuple[int, bytes]]  # (guardian_index, 65-byte r||s||v)
    body: bytes
    timestamp: int
    nonce: int
    emitter_chain: int
    emitter_address: bytes
    sequence: int
    consistency_level: int
    payload: bytes

    @property
    def digest(self) -> bytes:
        """keccak256(keccak256(body)) — the value each guardian signs."""
        return keccak256(keccak256(self.body))

    @property
    def id(self) -> str:
        return f"{self.emitter_chain}/{self.emitter_address.hex()}/{self.sequence}"


def parse_vaa(raw: bytes | str) -> WormholeVAA:
    """Parse a Wormhole v1 VAA from raw bytes or its base64 encoding."""
    data = base64.b64decode(raw) if isinstance(raw, str) else raw
    try:
        version, gsi, n_sigs = struct.unpack_from(">BIB", data, 0)
        if version != 1:
            raise AttestationError(f"Unsupported VAA version {version}")
        off = 6
        sigs = []
        for _ in range(n_sigs):
            sigs.append((data[off], data[off + 1 : off + 66]))
            off += 66
        body = data[off:]
        ts, nonce, chain = struct.unpack_from(">IIH", body, 0)
        emitter = body[10:42]
        (sequence,) = struct.unpack_from(">Q", body, 42)
        consistency = body[50]
        payload = body[51:]
    except (struct.error, IndexError) as exc:
        raise AttestationError(f"Truncated VAA: {exc}") from exc
    if len(emitter) != 32 or any(len(s) != 65 for _, s in sigs):
        raise AttestationError("Truncated VAA")
    return WormholeVAA(
        guardian_set_index=gsi,
        signatures=sigs,
        body=body,
        timestamp=ts,
        nonce=nonce,
        emitter_chain=chain,
        emitter_address=emitter,
        sequence=sequence,
        consistency_level=consistency,
        payload=payload,
    )


def ecrecover_address(digest: bytes, signature: bytes) -> bytes | None:
    """Recover the 20-byte Ethereum address that produced ``signature`` over ``digest``."""
    from ecdsa import BadSignatureError, SECP256k1, VerifyingKey
    from ecdsa.util import sigdecode_string

    if len(signature) != 65 or signature[64] > 3:
        return None
    rs, recid = signature[:64], signature[64]
    try:
        keys = VerifyingKey.from_public_key_recovery_with_digest(
            rs, digest, SECP256k1, sigdecode=sigdecode_string, allow_truncate=False
        )
    except (BadSignatureError, ValueError, AssertionError):
        return None
    # ecdsa returns candidates in recovery-id order for the given r.
    if recid >= len(keys):
        return None
    return keccak256(keys[recid].to_string("raw"))[12:]


def load_guardian_sets(path: str | Path | None = None) -> dict[int, list[bytes]]:
    """Load pinned Wormhole guardian sets: ``{index: [20-byte address, ...]}``."""
    resolved = Path(path) if path else _FIXTURE_DIR / "wormhole_guardian_sets.json"
    raw = json.loads(resolved.read_text())
    return {
        int(idx): [bytes.fromhex(a.removeprefix("0x")) for a in addrs]
        for idx, addrs in raw["guardian_sets"].items()
    }


@dataclass(frozen=True)
class VAAVerification:
    valid: bool
    reason: str
    valid_signatures: int = 0
    quorum: int = 0


def verify_vaa(vaa: WormholeVAA, guardian_sets: dict[int, list[bytes]]) -> VAAVerification:
    """Check a VAA carries a guardian quorum (floor(2n/3) + 1) of valid signatures.

    Mirrors ``Messages.verifyVM`` in the Wormhole core contract: signatures
    must be in strictly ascending guardian-index order (no duplicates) and
    each must ecrecover to the pinned guardian at that index.
    """
    guardians = guardian_sets.get(vaa.guardian_set_index)
    if guardians is None:
        return VAAVerification(False, f"unknown guardian set {vaa.guardian_set_index}")
    quorum = len(guardians) * 2 // 3 + 1
    if len(vaa.signatures) < quorum:
        return VAAVerification(False, "insufficient signatures", 0, quorum)
    digest = vaa.digest
    last = -1
    for g_idx, sig in vaa.signatures:
        if g_idx <= last:
            return VAAVerification(False, "signature indices not ascending", 0, quorum)
        last = g_idx
        if g_idx >= len(guardians):
            return VAAVerification(False, f"guardian index {g_idx} out of range", 0, quorum)
        if ecrecover_address(digest, sig) != guardians[g_idx]:
            return VAAVerification(False, f"invalid signature from guardian {g_idx}", 0, quorum)
    return VAAVerification(True, "ok", len(vaa.signatures), quorum)


# ---------------------------------------------------------------------------
# Wormhole token-bridge transfer payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenBridgeTransfer:
    payload_id: int  # 1 = Transfer, 3 = TransferWithPayload
    amount: int  # normalised to 8 decimals
    token_address: bytes
    token_chain: int
    to_address: bytes
    to_chain: int


def parse_token_bridge_transfer(payload: bytes) -> TokenBridgeTransfer | None:
    """Parse a Wormhole Token Bridge Transfer(1)/TransferWithPayload(3) payload."""
    if len(payload) < 133 or payload[0] not in (1, 3):
        return None
    return TokenBridgeTransfer(
        payload_id=payload[0],
        amount=int.from_bytes(payload[1:33], "big"),
        token_address=payload[33:65],
        token_chain=int.from_bytes(payload[65:67], "big"),
        to_address=payload[67:99],
        to_chain=int.from_bytes(payload[99:101], "big"),
    )


# ---------------------------------------------------------------------------
# Supported-bridge registry + dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeInfo:
    name: str
    attestation: str  # one of the ATTESTATION_* levels
    proof_format: str
    notes: str = ""


# Keep in sync with the table in docs/bridge_attestation.md.
SUPPORTED_BRIDGES: dict[str, BridgeInfo] = {
    "wormhole": BridgeInfo(
        "wormhole",
        ATTESTATION_GUARDIAN,
        "VAA v1 (base64), 13-of-19 secp256k1 guardian signatures",
        "Verified offline against pinned guardian sets.",
    ),
    "allbridge": BridgeInfo(
        "allbridge",
        ATTESTATION_HEURISTIC,
        "none published",
        "Stellar<->EVM/Solana; validator-signed off-chain, no public proof.",
    ),
    "stellar_anchor_sep6": BridgeInfo(
        "stellar_anchor_sep6",
        ATTESTATION_HEURISTIC,
        "none",
        "Custodial anchors; memo correlation only.",
    ),
}

ZKVerifier = Callable[[dict[str, Any]], bool]
_ZK_VERIFIERS: dict[str, ZKVerifier] = {}


def register_zk_verifier(bridge: str, verifier: ZKVerifier, proof_format: str) -> None:
    """Register a zk-proof verifier for ``bridge`` and mark it ``zk_proof``-attested.

    ``verifier`` receives the proof dict from the transaction record and must
    return True only if the proof verifies against a pinned verifying key.
    """
    _ZK_VERIFIERS[bridge] = verifier
    SUPPORTED_BRIDGES[bridge] = BridgeInfo(bridge, ATTESTATION_ZK, proof_format)


@dataclass(frozen=True)
class AttestationResult:
    level: str
    bridge: str | None = None
    verified: bool = False
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return ATTESTATION_CONFIDENCE[self.level]


_HEURISTIC = AttestationResult(ATTESTATION_HEURISTIC, reason="no proof")


def attested_recipient_matches(result: AttestationResult, address: str, chain: str) -> bool:
    """True if a verified attestation names ``address`` on ``chain`` as its recipient.

    Wormhole encodes recipients as 32 bytes: EVM addresses left-padded with
    zeros, Solana pubkeys verbatim. Note that for Solana the recipient is
    usually the destination *token account* (an ATA PDA), not the wallet.
    """
    to_hex = result.details.get("to_address")
    to_chain = result.details.get("to_chain")
    if not result.verified or not to_hex:
        return False
    # Memo parsing labels every 0x address "ethereum"; the same key is valid on
    # any EVM chain, so compare address families rather than chain names.
    same_family = to_chain == chain or (chain in _EVM_CHAINS and to_chain in _EVM_CHAINS)
    if not same_family:
        return False
    to_bytes = bytes.fromhex(to_hex)
    if address.lower().startswith("0x"):
        return to_bytes == b"\x00" * 12 + bytes.fromhex(address[2:])
    from detection.cross_chain.solana_resolver import SolanaValidationError, base58_decode

    try:
        return base58_decode(address) == to_bytes
    except SolanaValidationError:
        return False


class BridgeAttestationVerifier:
    """Verify the bridge proof attached to a transaction record, if any.

    A record opts in by carrying ``bridge_proof = {"bridge": <name>, ...}``.
    Wormhole proofs carry ``{"bridge": "wormhole", "vaa": <base64>}``; zk
    proofs carry whatever their registered verifier expects.
    """

    def __init__(self, guardian_sets: dict[int, list[bytes]] | None = None):
        self._guardian_sets = guardian_sets

    @property
    def guardian_sets(self) -> dict[int, list[bytes]]:
        if self._guardian_sets is None:
            self._guardian_sets = load_guardian_sets()
        return self._guardian_sets

    def verify(self, proof: dict[str, Any] | None) -> AttestationResult:
        if not proof:
            return _HEURISTIC
        bridge = str(proof.get("bridge", "")).lower()
        if bridge in _ZK_VERIFIERS:
            try:
                ok = bool(_ZK_VERIFIERS[bridge](proof))
            except Exception as exc:  # a broken proof must never raise into detection
                logger.warning("zk verifier for %s raised: %s", bridge, exc)
                ok = False
            if ok:
                return AttestationResult(ATTESTATION_ZK, bridge, True, "ok")
            return AttestationResult(ATTESTATION_HEURISTIC, bridge, False, "zk proof invalid")
        if bridge == "wormhole" and proof.get("vaa"):
            return self._verify_wormhole(proof["vaa"])
        return AttestationResult(ATTESTATION_HEURISTIC, bridge or None, False, "unsupported bridge")

    def _verify_wormhole(self, raw_vaa: str | bytes) -> AttestationResult:
        try:
            vaa = parse_vaa(raw_vaa)
        except (AttestationError, ValueError) as exc:
            return AttestationResult(ATTESTATION_HEURISTIC, "wormhole", False, str(exc))
        check = verify_vaa(vaa, self.guardian_sets)
        if not check.valid:
            logger.info("Wormhole VAA %s rejected: %s", vaa.id, check.reason)
            return AttestationResult(ATTESTATION_HEURISTIC, "wormhole", False, check.reason)
        details: dict[str, Any] = {
            "vaa_id": vaa.id,
            "digest": vaa.digest.hex(),
            "guardian_set_index": vaa.guardian_set_index,
            "signatures": check.valid_signatures,
            "quorum": check.quorum,
            "emitter_chain": WORMHOLE_CHAIN_IDS.get(vaa.emitter_chain, str(vaa.emitter_chain)),
        }
        transfer = parse_token_bridge_transfer(vaa.payload)
        if transfer is not None:
            details.update(
                amount=transfer.amount,
                to_chain=WORMHOLE_CHAIN_IDS.get(transfer.to_chain, str(transfer.to_chain)),
                to_address=transfer.to_address.hex(),
            )
        return AttestationResult(ATTESTATION_GUARDIAN, "wormhole", True, "ok", details)
