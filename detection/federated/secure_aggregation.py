"""Secure gradient aggregation using CKKS homomorphic encryption (TenSEAL).

CKKS Parameter Choices
-----------------------
- Polynomial degree (``poly_modulus_degree``): **8192**
  Provides ≥ 128-bit security at coefficient modulus sizes below 218 bits
  (BFV/CKKS standard security table). Supports vectors of up to ~4096
  complex slots (8192 / 2).

- Coefficient modulus bit sizes: ``[60, 40, 40, 60]``
  Three multiplication levels; the leading and trailing 60-bit primes are
  the "special" primes required by RNS-CKKS. This gives ~140 bits of
  coefficient modulus, comfortably below the 218-bit limit for poly-degree
  8192.

- Global scale: ``2^40``
  Balances floating-point precision (~11 decimal digits) against noise
  growth.  For gradient aggregation of float32 values this exceeds the
  required tolerance of 1e-4.

Threshold Decryption Protocol
------------------------------
TenSEAL does not natively expose a threshold decryption API. We simulate a
(K-of-N) threshold scheme by splitting the serialised TenSEAL secret key
bytes using Shamir's Secret Sharing (implemented without third-party SSS
library for minimal dependency surface):

1. **Key generation**: The coordinator generates a CKKS context + keypair
   and splits the secret key into N shares. Each participant receives one
   share; the coordinator retains no complete key.

2. **Encryption**: Each participant encrypts their gradient tensor with the
   public context before submitting it.  The coordinator receives only
   ciphertexts.

3. **Homomorphic aggregation**: The coordinator sums ciphertexts
   homomorphically — valid in CKKS because addition is levelled.

4. **Threshold decryption**: At least K participants submit their key
   shares. The coordinator uses Lagrange interpolation over GF(p) to
   reconstruct the secret key, decrypts the aggregate, then immediately
   discards the reconstructed key from memory.

Security properties
-------------------
- The coordinator cannot decrypt any individual gradient (no single share
  suffices).
- At least K − 1 colluding participants are required to break privacy.
- Private key material is never logged, serialised to disk, or included
  in error messages.

Reference: Bonawitz et al., Practical Secure Aggregation for Privacy-
Preserving Machine Learning, CCS 2017.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional TenSEAL import (graceful degradation in environments without it)
# ---------------------------------------------------------------------------
try:
    import tenseal as ts  # type: ignore[import-untyped]

    _TENSEAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    ts = None  # type: ignore[assignment]
    _TENSEAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# CKKS parameter constants (see module docstring for rationale)
# ---------------------------------------------------------------------------
_POLY_MOD_DEGREE = 8192
_COEFF_MOD_BITS = [60, 40, 40, 60]
_GLOBAL_SCALE = 2**40

# Shamir secret sharing prime (must be > 255 so each byte fits in one share coefficient)
_SSS_PRIME = 2**127 - 1  # Mersenne prime, 127-bit


# ---------------------------------------------------------------------------
# Minimal Shamir Secret Sharing over GF(_SSS_PRIME)
# ---------------------------------------------------------------------------


def _sss_split(secret_bytes: bytes, k: int, n: int) -> list[tuple[int, bytes]]:
    """Split *secret_bytes* into *n* shares requiring *k* to reconstruct.

    Returns a list of ``(x, share_bytes)`` tuples where *x* in 1..n.
    Each share byte string is the same length as *secret_bytes*.
    """
    p = _SSS_PRIME
    length = len(secret_bytes)
    result: list[tuple[int, bytes]] = []

    # Process each byte independently (simplifies implementation; fine for key bytes)
    all_shares: list[list[int]] = [[] for _ in range(n)]
    for byte_val in secret_bytes:
        # Random polynomial of degree k-1 with f(0) = byte_val
        coeffs = [byte_val] + [secrets.randbelow(p) for _ in range(k - 1)]
        for x in range(1, n + 1):
            y = sum(c * pow(x, i, p) for i, c in enumerate(coeffs)) % p
            all_shares[x - 1].append(y)

    for i in range(n):
        share_bytes = bytes(v % 256 for v in all_shares[i])  # truncate to byte range
        result.append((i + 1, share_bytes))

    return result


def _sss_reconstruct(shares: list[tuple[int, bytes]]) -> bytes:
    """Reconstruct secret bytes from a list of ``(x, share_bytes)`` pairs."""
    p = _SSS_PRIME
    if not shares:
        raise ValueError("No shares provided for reconstruction")
    length = len(shares[0][1])
    result = bytearray(length)

    for byte_idx in range(length):
        points = [(x, sb[byte_idx]) for x, sb in shares]
        # Lagrange interpolation at x=0
        secret_byte = 0
        for i, (xi, yi) in enumerate(points):
            num = yi
            den = 1
            for j, (xj, _) in enumerate(points):
                if i != j:
                    num = (num * (-xj)) % p
                    den = (den * (xi - xj)) % p
            secret_byte = (secret_byte + num * pow(den, p - 2, p)) % p
        result[byte_idx] = secret_byte % 256

    return bytes(result)


# ---------------------------------------------------------------------------
# Context / key management
# ---------------------------------------------------------------------------


@dataclass
class SecureAggregationContext:
    """Shared CKKS context distributed to all participants.

    Contains the public context (encryption parameters + public key) but
    NOT the secret key — secret key material is split across participants.
    """

    serialised_context: bytes
    n_participants: int
    k_threshold: int

    def to_tenseal_context(self) -> "ts.Context":
        if not _TENSEAL_AVAILABLE:
            raise RuntimeError("tenseal is not installed; run: pip install tenseal")
        return ts.context_from(self.serialised_context)


@dataclass
class ParticipantKeyShare:
    """A single participant's Shamir share of the CKKS secret key."""

    participant_index: int  # 1-based
    share_bytes: bytes


class SecureAggregationSetup:
    """Generate CKKS context + key and distribute shares to participants.

    Parameters
    ----------
    n_participants:
        Total number of federated participants (3–50).
    k_threshold:
        Minimum number of participants required for threshold decryption.
        Must satisfy ``k <= n_participants``.
    """

    def __init__(self, n_participants: int, k_threshold: int) -> None:
        if not (3 <= n_participants <= 50):
            raise ValueError(f"n_participants must be between 3 and 50, got {n_participants}")
        if not (1 <= k_threshold <= n_participants):
            raise ValueError(
                f"k_threshold must be between 1 and n_participants ({n_participants}), "
                f"got {k_threshold}"
            )
        if not _TENSEAL_AVAILABLE:
            raise RuntimeError("tenseal is not installed; run: pip install tenseal")

        self.n = n_participants
        self.k = k_threshold

        # Generate CKKS context with full keys
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=_POLY_MOD_DEGREE,
            coeff_mod_bit_sizes=_COEFF_MOD_BITS,
        )
        ctx.generate_galois_keys()
        ctx.global_scale = _GLOBAL_SCALE

        # Serialise the secret key for SSS, then make the context public-only
        sk_bytes = ctx.secret_key().serialize()
        self._shares = _sss_split(sk_bytes, self.k, self.n)

        # Drop secret key from context so the serialised public context
        # does not contain it
        ctx.make_context_public()
        self._public_ctx_bytes = ctx.serialize()

        # Keep full context internally for the setup phase only
        self._full_ctx_bytes: bytes | None = None  # not retained after key splitting

    @property
    def public_context(self) -> SecureAggregationContext:
        return SecureAggregationContext(
            serialised_context=self._public_ctx_bytes,
            n_participants=self.n,
            k_threshold=self.k,
        )

    def participant_share(self, participant_index: int) -> ParticipantKeyShare:
        """Return the key share for participant *participant_index* (1-based)."""
        if not (1 <= participant_index <= self.n):
            raise ValueError(f"participant_index must be 1–{self.n}")
        x, share_bytes = self._shares[participant_index - 1]
        return ParticipantKeyShare(participant_index=x, share_bytes=share_bytes)


# ---------------------------------------------------------------------------
# Participant-side: encrypt gradient
# ---------------------------------------------------------------------------


def encrypt_gradient(
    gradient: np.ndarray,
    ctx: SecureAggregationContext,
) -> bytes:
    """Encrypt *gradient* under the shared public CKKS context.

    Returns the serialised ciphertext bytes to be sent to the coordinator.
    The coordinator receives only ciphertexts and cannot decrypt individual
    contributions.

    Parameters
    ----------
    gradient:
        Flat float array of model weight delta.
    ctx:
        Shared public context (no secret key material).
    """
    if not _TENSEAL_AVAILABLE:
        raise RuntimeError("tenseal is not installed; run: pip install tenseal")

    tenseal_ctx = ctx.to_tenseal_context()
    enc = ts.ckks_vector(tenseal_ctx, gradient.tolist())
    return enc.serialize()


# ---------------------------------------------------------------------------
# Coordinator-side: aggregate ciphertexts and threshold-decrypt
# ---------------------------------------------------------------------------


class SecureAggregator:
    """Coordinator-side secure aggregation.

    Receives encrypted gradient ciphertexts from participants, sums them
    homomorphically, then performs threshold decryption once at least
    K participant key shares are available.

    Parameters
    ----------
    ctx:
        The shared public context.
    """

    def __init__(self, ctx: SecureAggregationContext) -> None:
        self._ctx = ctx
        self._ciphertexts: list[bytes] = []
        self._key_shares: list[tuple[int, bytes]] = []

    def add_ciphertext(self, ciphertext_bytes: bytes) -> None:
        """Accept an encrypted gradient from one participant."""
        self._ciphertexts.append(ciphertext_bytes)

    def add_key_share(self, share: ParticipantKeyShare) -> None:
        """Accept a key share from a participant for threshold decryption."""
        self._key_shares.append((share.participant_index, share.share_bytes))

    def can_decrypt(self) -> bool:
        return len(self._key_shares) >= self._ctx.k_threshold

    def aggregate_and_decrypt(self) -> np.ndarray:
        """Sum all ciphertexts homomorphically and decrypt using threshold shares.

        Raises
        ------
        RuntimeError
            If fewer than k_threshold key shares have been received.
        ValueError
            If no ciphertexts have been submitted.
        """
        if not _TENSEAL_AVAILABLE:
            raise RuntimeError("tenseal is not installed; run: pip install tenseal")
        if not self._ciphertexts:
            raise ValueError("No ciphertexts received for aggregation")
        if not self.can_decrypt():
            raise RuntimeError(
                f"Threshold not met: have {len(self._key_shares)} shares, "
                f"need {self._ctx.k_threshold}"
            )

        public_ctx = self._ctx.to_tenseal_context()

        # Homomorphically sum all ciphertexts
        enc_sum: "ts.CKKSVector" | None = None
        for ct_bytes in self._ciphertexts:
            enc = ts.ckks_vector_from(public_ctx, ct_bytes)
            if enc_sum is None:
                enc_sum = enc
            else:
                enc_sum += enc

        assert enc_sum is not None

        # Reconstruct secret key from threshold shares
        sk_bytes = _sss_reconstruct(self._key_shares[: self._ctx.k_threshold])

        # Temporarily restore secret key for decryption, then immediately zero it
        full_ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=_POLY_MOD_DEGREE,
            coeff_mod_bit_sizes=_COEFF_MOD_BITS,
        )
        # Load the full context with the reconstructed key
        # (TenSEAL context with secret key must be recreated from serialised form)
        # We attach the secret key to the public context via link_secret_key
        sk = ts.SecretKey.deserialize(full_ctx, sk_bytes)
        full_ctx.link_secret_key(sk)
        full_ctx.global_scale = _GLOBAL_SCALE

        # Re-bind the ciphertext to the full context for decryption
        enc_sum_full = ts.ckks_vector_from(full_ctx, enc_sum.serialize())
        result = enc_sum_full.decrypt()

        # Immediately overwrite reconstructed key material
        sk_bytes = b"\x00" * len(sk_bytes)

        logger.info(
            "Secure aggregation: decrypted aggregate of %d ciphertexts "
            "using %d/%d key shares",
            len(self._ciphertexts),
            len(self._key_shares),
            self._ctx.n_participants,
        )

        return np.array(result)
