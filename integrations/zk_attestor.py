"""ZK proof batching system for amortised on-chain verification.

Each wallet attestation produces a ``SNARKProof``.  Verifying N proofs on-chain
individually costs O(N) Soroban compute budget.  ``aggregate_proofs()`` combines
N proofs into a single ``AggregateProof`` that can be verified on-chain in O(1)
regardless of N, using a SnarkPack-style Fiat-Shamir inner-product argument.

``BatchingController`` accumulates individual proofs and submits an aggregate
once the batch reaches ``ZK_BATCH_SIZE`` (default 10) or
``ZK_BATCH_TIMEOUT_SECONDS`` (default 300) elapses, whichever comes first.

Simulation note
---------------
This module implements the *protocol* of ZK proof aggregation using
HMAC-SHA256 as a stand-in for the SNARK proving/verification operations.
In a production deployment the proof primitives would be replaced with a
real pairing-based library (e.g. py_ecc for BLS12-381, or a Groth16 binding).
The aggregate size, batching, and on-chain verification logic are identical.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

ZK_BATCH_SIZE: int = int(os.getenv("ZK_BATCH_SIZE", "10"))
ZK_BATCH_TIMEOUT_SECONDS: int = int(os.getenv("ZK_BATCH_TIMEOUT_SECONDS", "300"))

# Simulated Groth16 proof size: pi_A (32 B) + pi_B (64 B) + pi_C (32 B) = 128 bytes.
_INDIVIDUAL_PROOF_SIZE = 128


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SNARKProof:
    """A SNARK proof attesting to the risk score of a single wallet.

    Fields
    ------
    wallet_id:
        Stellar account ID (G…) of the attested wallet.
    proof_data:
        Serialised proof bytes (simulates Groth16 pi_A + pi_B + pi_C).
        Exactly ``_INDIVIDUAL_PROOF_SIZE`` bytes for a valid proof.
    public_inputs:
        Public witness vector: [risk_score, unix_timestamp, pair_hash].
    verification_key:
        Per-circuit verification-key fingerprint (32 bytes).
    """

    wallet_id: str
    proof_data: bytes
    public_inputs: list[int]
    verification_key: bytes

    def is_valid(self) -> bool:
        """Return True if the proof is internally consistent.

        Validity check: the first 32 bytes of ``proof_data`` must equal
        HMAC-SHA256(verification_key, encode(public_inputs)).  This simulates
        the pairing check in a real Groth16 verifier.
        """
        if len(self.proof_data) < 32:
            return False
        msg = b"".join(struct.pack(">q", v) for v in self.public_inputs)
        expected = hmac.new(self.verification_key, msg, hashlib.sha256).digest()
        return hmac.compare_digest(self.proof_data[:32], expected)


@dataclass
class AggregateProof:
    """An aggregate proof covering N wallet attestations in O(1) verification cost.

    ``aggregate_data`` (64 bytes) encodes the Fiat-Shamir challenge and the
    linear combination of all individual proofs.  ``public_input_commitment``
    (32 bytes) is the Merkle root of all public input vectors.

    Total size: 96 bytes — always smaller than N × 128 bytes (N ≥ 1).
    """

    wallet_ids: list[str]
    aggregate_data: bytes           # 64 bytes: challenge || linear_combination
    public_input_commitment: bytes  # 32 bytes: Merkle root of public inputs
    n_proofs: int

    @property
    def byte_size(self) -> int:
        return len(self.aggregate_data) + len(self.public_input_commitment)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidProofError(ValueError):
    """Raised when a proof fails individual verification before aggregation."""

    def __init__(self, wallet_id: str, reason: str) -> None:
        super().__init__(f"Invalid proof for wallet {wallet_id!r}: {reason}")
        self.wallet_id = wallet_id


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def aggregate_proofs(proofs: list[SNARKProof]) -> AggregateProof:
    """Aggregate N individually-valid SNARK proofs into a single AggregateProof.

    Algorithm (SnarkPack-style):
    1. Verify each proof individually; raise ``InvalidProofError`` on first failure.
    2. Derive a Fiat-Shamir challenge by hashing all commitments in order.
    3. Build a linear combination of all ``proof_data`` vectors under the challenge.
    4. Commit to all public inputs via a binary Merkle tree.
    5. Return the 96-byte aggregate (always smaller than N × 128 bytes).

    Raises
    ------
    ValueError
        If ``proofs`` is empty.
    InvalidProofError
        If any proof fails individual verification; ``exc.wallet_id`` names it.
    """
    if not proofs:
        raise ValueError("Cannot aggregate an empty proof list")

    # Step 1 — individual verification
    for proof in proofs:
        if not proof.is_valid():
            raise InvalidProofError(proof.wallet_id, "proof verification failed")

    # Step 2 — Fiat-Shamir challenge: H(wallet_id || proof_data || public_inputs …)
    h = hashlib.sha256()
    for proof in proofs:
        h.update(proof.wallet_id.encode())
        h.update(proof.proof_data)
        h.update(b"".join(struct.pack(">q", v) for v in proof.public_inputs))
    challenge: bytes = h.digest()  # 32 bytes

    # Step 3 — linear combination under challenge
    # In production: inner-product argument over BLS12-381 pairings.
    combined = bytearray(32)
    for i, proof in enumerate(proofs):
        weight = hashlib.sha256(challenge + struct.pack(">I", i)).digest()
        for j in range(32):
            combined[j] ^= proof.proof_data[j % len(proof.proof_data)] ^ weight[j]

    # Step 4 — Merkle root of public inputs
    leaves = [
        hashlib.sha256(b"".join(struct.pack(">q", v) for v in p.public_inputs)).digest()
        for p in proofs
    ]
    root = _merkle_root(leaves)

    return AggregateProof(
        wallet_ids=[p.wallet_id for p in proofs],
        aggregate_data=challenge + bytes(combined),  # 64 bytes
        public_input_commitment=root,                 # 32 bytes
        n_proofs=len(proofs),
    )


def verify_aggregate_proof(agg: AggregateProof, proofs: list[SNARKProof]) -> bool:
    """Verify an aggregate proof against its constituent proofs.

    In production this is a single O(1) pairing check on-chain (Soroban contract).
    Returns True if the aggregate is consistent with all proofs.
    """
    if agg.n_proofs != len(proofs):
        return False

    # Re-derive Fiat-Shamir challenge
    h = hashlib.sha256()
    for proof in proofs:
        h.update(proof.wallet_id.encode())
        h.update(proof.proof_data)
        h.update(b"".join(struct.pack(">q", v) for v in proof.public_inputs))
    expected_challenge = h.digest()

    if not hmac.compare_digest(agg.aggregate_data[:32], expected_challenge):
        return False

    leaves = [
        hashlib.sha256(b"".join(struct.pack(">q", v) for v in p.public_inputs)).digest()
        for p in proofs
    ]
    return hmac.compare_digest(agg.public_input_commitment, _merkle_root(leaves))


def make_valid_proof(
    wallet_id: str,
    risk_score: int,
    timestamp: int,
    verification_key: bytes,
) -> SNARKProof:
    """Construct a valid ``SNARKProof`` for testing and integration use.

    The proof is deterministic given the same inputs — useful for idempotency tests.
    """
    public_inputs = [risk_score, timestamp, hash(wallet_id) & 0xFFFF_FFFF]
    msg = b"".join(struct.pack(">q", v) for v in public_inputs)
    mac = hmac.new(verification_key, msg, hashlib.sha256).digest()  # 32 bytes
    # Pad to _INDIVIDUAL_PROOF_SIZE bytes simulating pi_A + pi_B + pi_C
    padding = hashlib.sha256(mac).digest() * 3  # 96 bytes
    proof_data = (mac + padding)[:_INDIVIDUAL_PROOF_SIZE]
    return SNARKProof(
        wallet_id=wallet_id,
        proof_data=proof_data,
        public_inputs=public_inputs,
        verification_key=verification_key,
    )


def _merkle_root(leaves: list[bytes]) -> bytes:
    """Binary Merkle root of a list of 32-byte leaf hashes."""
    if not leaves:
        return hashlib.sha256(b"").digest()
    layer = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last leaf on odd count
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0]


# ---------------------------------------------------------------------------
# Batching controller
# ---------------------------------------------------------------------------


class BatchingController:
    """Accumulates proofs and submits aggregate batches to the on-chain contract.

    Triggers a flush when either:
    - the batch reaches ``batch_size`` proofs, or
    - ``timeout_seconds`` elapses since the first proof was added.

    Idempotency
    -----------
    Submitting the same ``wallet_id`` twice within an unflushed batch is silently
    deduplicated.  Cross-batch uniqueness is enforced by the on-chain contract's
    unique constraint on (wallet_id, asset_pair).
    """

    def __init__(
        self,
        submit_fn: Callable[[AggregateProof], None],
        batch_size: int = ZK_BATCH_SIZE,
        timeout_seconds: int = ZK_BATCH_TIMEOUT_SECONDS,
    ) -> None:
        self._submit_fn = submit_fn
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

        self._pending: list[SNARKProof] = []
        self._seen_wallets: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_proof(self, proof: SNARKProof) -> None:
        """Validate and enqueue a proof.  Flushes immediately if batch is full.

        Raises ``InvalidProofError`` if the proof fails individual verification.
        """
        if not proof.is_valid():
            raise InvalidProofError(proof.wallet_id, "proof verification failed")

        with self._lock:
            if proof.wallet_id in self._seen_wallets:
                logger.debug("Duplicate proof for %r — skipped", proof.wallet_id)
                return
            self._pending.append(proof)
            self._seen_wallets.add(proof.wallet_id)
            if len(self._pending) == 1:
                self._arm_timer()
            if len(self._pending) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> None:
        """Manually flush any pending proofs regardless of batch size."""
        with self._lock:
            self._flush_locked()

    def stop(self) -> None:
        """Cancel the pending timer and flush any remaining proofs."""
        with self._lock:
            self._cancel_timer()
            self._flush_locked()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _arm_timer(self) -> None:
        """Schedule a timeout flush (must be called under lock)."""
        self._cancel_timer()
        timer = threading.Timer(self.timeout_seconds, self._timeout_flush)
        timer.daemon = True
        timer.start()
        self._timer = timer

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _timeout_flush(self) -> None:
        with self._lock:
            if self._pending:
                logger.info(
                    "ZK batch timeout — flushing %d proof(s)", len(self._pending)
                )
                self._flush_locked()

    def _flush_locked(self) -> None:
        """Aggregate and submit current batch (must be called under lock)."""
        if not self._pending:
            return
        batch = list(self._pending)
        self._pending.clear()
        self._seen_wallets.clear()
        self._cancel_timer()
        try:
            agg = aggregate_proofs(batch)
            self._submit_fn(agg)
            logger.info(
                "Submitted aggregate ZK proof covering %d wallet(s)", agg.n_proofs
            )
        except InvalidProofError as exc:
            logger.error("Batch rejected — invalid proof: %s", exc)
        except Exception as exc:
            logger.error("Failed to submit aggregate proof: %s", exc)
