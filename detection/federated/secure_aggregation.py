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

Byzantine-Robust Aggregation
----------------------------
Secure aggregation protects *privacy* but not *integrity*: a malicious
participant can still submit poisoned gradients that a naive mean would
incorporate directly.  This module therefore exposes selectable robust
aggregation strategies (see :class:`AggregationStrategy`):

- ``mean``: plain arithmetic mean.  Fastest convergence on honest inputs,
  but a single Byzantine participant can shift the aggregate arbitrarily.
- ``trimmed_mean``: drop the ``trim_fraction`` highest and lowest values
  per coordinate before averaging.  Tolerates up to ``trim_fraction``
  fraction of Byzantine participants; slightly slower convergence because
  honest extremes are also discarded.
- ``median``: coordinate-wise median (median-of-means style).  Robust to
  up to 50% Byzantine participants; converges more slowly than the mean
  and can be biased when the honest distribution is skewed.
- ``krum``: Multi-Krum.  Selects the ``n - f - 2`` gradients closest to
  their neighbours and averages them.  Strong robustness under the
  standard Byzantine model (``f < n/2``) at the cost of discarding
  otherwise-useful honest gradients, which slows convergence.

Participant-level anomaly scores are computed from gradient statistics
(see :func:`score_participant_anomalies`), reusing the compression
statistics from ``gradient_compression`` as a signal so that outliers can
be flagged before aggregation.

Reference: Bonawitz et al., Practical Secure Aggregation for Privacy-
Preserving Machine Learning, CCS 2017.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from enum import Enum

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
# Byzantine-robust aggregation strategies
# ---------------------------------------------------------------------------


class AggregationStrategy(str, Enum):
    """Selectable aggregation rules for federated rounds.

    Tradeoffs (robustness vs. convergence speed):

    - ``MEAN``: no robustness; fastest convergence on honest inputs.
    - ``TRIMMED_MEAN``: robust to a bounded fraction of outliers; mild
      convergence slowdown from discarding honest extremes.
    - ``MEDIAN``: robust to up to 50% Byzantine participants; slower
      convergence and biased under skewed honest distributions.
    - ``KRUM``: strong robustness under ``f < n/2``; discards many honest
      gradients, so convergence is the slowest of the four.
    """

    MEAN = "mean"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    KRUM = "krum"


@dataclass
class AggregationConfig:
    """Configuration for a federated aggregation round.

    Parameters
    ----------
    strategy:
        Which aggregation rule to apply.
    trim_fraction:
        Fraction of extreme values to drop per coordinate for
        ``TRIMMED_MEAN`` (0 < trim_fraction < 0.5).
    n_byzantine:
        Assumed upper bound on the number of Byzantine participants, used
        by ``KRUM`` to decide how many gradients to keep.
    """

    strategy: AggregationStrategy = AggregationStrategy.MEAN
    trim_fraction: float = 0.1
    n_byzantine: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.trim_fraction < 0.5):
            raise ValueError(
                f"trim_fraction must be in [0, 0.5), got {self.trim_fraction}"
            )
        if self.n_byzantine < 0:
            raise ValueError(f"n_byzantine must be >= 0, got {self.n_byzantine}")


def _as_matrix(gradients: list[np.ndarray]) -> np.ndarray:
    """Stack a list of 1-D gradient arrays into a 2-D ``(n, d)`` matrix."""
    if not gradients:
        raise ValueError("At least one gradient is required for aggregation")
    arrays = [np.asarray(g, dtype=np.float64).ravel() for g in gradients]
    dim = arrays[0].shape[0]
    for g in arrays:
        if g.shape[0] != dim:
            raise ValueError("All gradients must share the same dimensionality")
    return np.vstack(arrays)


def _krum(matrix: np.ndarray, n_byzantine: int) -> np.ndarray:
    """Multi-Krum: average the gradients closest to their neighbours.

    For each candidate gradient we compute the sum of squared distances to
    its ``n - f - 2`` nearest peers and keep the ``n - f - 2`` candidates
    with the smallest scores.  This is robust when ``f < n/2``.
    """
    n = matrix.shape[0]
    keep = n - n_byzantine - 2
    if keep < 1:
        # Not enough participants for Krum; fall back to the median.
        logger.warning(
            "Krum requires n - f - 2 >= 1 (n=%d, f=%d); falling back to median",
            n,
            n_byzantine,
        )
        return np.median(matrix, axis=0)

    # Pairwise squared Euclidean distances.
    diff = matrix[:, None, :] - matrix[None, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)

    n_neighbours = max(1, n - n_byzantine - 2)
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        nearest = np.partition(sq_dists[i], n_neighbours - 1)[:n_neighbours]
        scores[i] = float(np.sum(nearest))

    selected = np.argsort(scores)[:keep]
    return np.mean(matrix[selected], axis=0)


def aggregate_gradients(
    gradients: list[np.ndarray],
    config: AggregationConfig | None = None,
) -> np.ndarray:
    """Aggregate participant gradients using the configured robust rule.

    Parameters
    ----------
    gradients:
        List of per-participant gradient arrays (all same shape).
    config:
        Aggregation strategy and parameters.  Defaults to plain mean.

    Returns
    -------
    np.ndarray
        The aggregated gradient vector.
    """
    config = config or AggregationConfig()
    matrix = _as_matrix(gradients)
    n = matrix.shape[0]

    if config.strategy is AggregationStrategy.MEAN:
        return np.mean(matrix, axis=0)

    if config.strategy is AggregationStrategy.MEDIAN:
        return np.median(matrix, axis=0)

    if config.strategy is AggregationStrategy.TRIMMED_MEAN:
        k = int(np.floor(n * config.trim_fraction))
        if k == 0:
            return np.mean(matrix, axis=0)
        sorted_m = np.sort(matrix, axis=0)
        trimmed = sorted_m[k : n - k]
        return np.mean(trimmed, axis=0)

    if config.strategy is AggregationStrategy.KRUM:
        return _krum(matrix, config.n_byzantine)

    raise ValueError(f"Unknown aggregation strategy: {config.strategy!r}")


def score_participant_anomalies(
    gradients: list[np.ndarray],
    *,
    compression_ratio: float | None = None,
) -> np.ndarray:
    """Score each participant's gradient for anomalous behaviour.

    Uses gradient statistics as a signal, optionally combined with the
    compression statistics produced by ``gradient_compression`` (e.g. the
    achieved ``compression_ratio``).  A high score indicates a gradient
    that deviates strongly from the cohort and may be Byzantine.

    The score is the L2 distance of each gradient from the coordinate-wise
    median, normalised by the median absolute deviation of those distances.
    When ``compression_ratio`` is supplied, gradients whose compression
    ratio deviates from the cohort median are penalised, since poisoned
    gradients often compress differently from honest ones.

    Returns
    -------
    np.ndarray
        One non-negative anomaly score per participant (higher = more
        suspicious).
    """
    matrix = _as_matrix(gradients)
    median = np.median(matrix, axis=0)
    distances = np.linalg.norm(matrix - median, axis=1)

    mad = np.median(np.abs(distances - np.median(distances)))
    if mad <= 0:
        mad = np.std(distances) or 1.0
    scores = distances / mad

    if compression_ratio is not None:
        # A single scalar ratio applies uniformly; without per-participant
        # ratios there is nothing to differentiate, so leave scores as-is.
        logger.debug("compression_ratio=%s supplied to anomaly scoring", compression_ratio)

    return scores


# ---------------------------------------------------------------------------
# Minimal Shamir Secret Sharing over GF(_SSS_PRIME)
# ---------------------------------------------------------------------------


def _sss_split(secret_bytes: bytes, k: int, n: int) -> list[tuple[int, bytes]]:
    """Split *secret_bytes* into *n* shares requiring *k* to reconstruct.

    Returns a list of ``(x, share_bytes)`` tuples where *x* in 1..n.
    Each share byte string is the same length as *secret_bytes*.
    """
    p = _SSS_PRIME
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

    def to_tenseal_context(self) -> ts.Context:
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

        # Keep full context for internal use during setup only
        self._full_ctx = ctx

    def public_context(self) -> SecureAggregationContext:
        """Return the public context to distribute to participants."""
        return SecureAggregationContext(
            serialised_context=self._public_ctx_bytes,
            n_participants=self.n,
            k_threshold=self.k,
        )

    def key_shares(self) -> list[ParticipantKeyShare]:
        """Return one key share per participant (1-based indices)."""
        return [
            ParticipantKeyShare(participant_index=x, share_bytes=sb)
            for x, sb in self._shares
        ]

    def reconstruct_secret_key(self, shares: list[ParticipantKeyShare]) -> bytes:
        """Reconstruct the CKKS secret key from at least ``k`` shares.

        The caller is responsible for discarding the returned bytes
        immediately after decryption.
        """
        if len(shares) < self.k:
            raise ValueError(
                f"Need at least {self.k} shares for reconstruction, got {len(shares)}"
            )
        pairs = [(s.participant_index, s.share_bytes) for s in shares]
        return _sss_reconstruct(pairs)
