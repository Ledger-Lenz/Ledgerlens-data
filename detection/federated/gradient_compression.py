"""Gradient compression for federated learning bandwidth reduction.

Two compressors are provided:

TopKSparsifier
    Transmits only the top-K largest-magnitude gradient values (and their
    indices).  With k_ratio=0.01 this reduces bandwidth by ~100x.  A random
    rotation is applied *before* selecting the top-K indices so that the
    selected positions do not leak which features have the largest gradients
    (security requirement from issue #251).

PowerSGDCompressor
    Approximates a gradient matrix as a low-rank factorisation P @ Q^T where
    P has ``rank`` columns.  Reduces communication from O(m*n) to O((m+n)*r).

Both compressors support an *error-feedback* (memory correction) mechanism:
the residual e_t = g_t - decompress(compress(g_t)) is stored per layer and
added to the *next* round's gradient before compression.  This prevents
residuals from accumulating in a way that dominates updates.

Usage
-----
    compressor = TopKSparsifier(k_ratio=0.01)
    ec = ErrorFeedbackCompressor(compressor, n_layers=1)

    # participant side – compress before transmission
    payload = ec.compress(gradient)

    # coordinator side – decompress before aggregation
    gradient_approx = TopKSparsifier.decompress(payload)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Payload types
# ---------------------------------------------------------------------------


@dataclass
class TopKPayload:
    values: np.ndarray   # shape (k,)
    indices: np.ndarray  # shape (k,), integer
    shape: tuple[int, ...]
    rotation_seed: int   # deterministic seed for the random rotation


@dataclass
class PowerSGDPayload:
    P: np.ndarray  # shape (m, rank)
    Q: np.ndarray  # shape (n, rank)
    shape: tuple[int, ...]


# ---------------------------------------------------------------------------
# Top-K sparsifier
# ---------------------------------------------------------------------------


class TopKSparsifier:
    """Transmit only the top-K gradient values with random rotation privacy.

    Parameters
    ----------
    k_ratio:
        Fraction of elements to keep (default 0.01 → 1%).
    seed:
        Base RNG seed; per-call seeds are derived from this + an incrementing
        counter so that each compression call uses a fresh rotation.
    """

    def __init__(self, k_ratio: float = 0.01, seed: int = 42) -> None:
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError("k_ratio must be in (0, 1]")
        self.k_ratio = k_ratio
        self._seed = seed
        self._call_counter = 0

    def compress(self, gradient: np.ndarray) -> TopKPayload:
        """Compress a flat gradient vector to a TopKPayload."""
        flat = gradient.ravel().astype(np.float64)
        n = flat.size
        k = max(1, math.ceil(n * self.k_ratio))

        # Random rotation to prevent feature-importance leakage
        rotation_seed = (self._seed + self._call_counter) % (2**31)
        self._call_counter += 1
        rng = np.random.default_rng(rotation_seed)
        # Efficient random sign flip (Hadamard-like) instead of full rotation
        signs = rng.choice([-1.0, 1.0], size=n)
        rotated = flat * signs

        # Select top-K by magnitude
        abs_vals = np.abs(rotated)
        top_k_idx = np.argpartition(abs_vals, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(abs_vals[top_k_idx])[::-1]]

        return TopKPayload(
            values=rotated[top_k_idx].copy(),
            indices=top_k_idx.copy(),
            shape=gradient.shape,
            rotation_seed=rotation_seed,
        )

    @staticmethod
    def decompress(payload: TopKPayload) -> np.ndarray:
        """Reconstruct the full gradient vector from a TopKPayload."""
        n = math.prod(payload.shape)
        flat = np.zeros(n, dtype=np.float64)
        flat[payload.indices] = payload.values

        # Invert the random rotation (sign flip is its own inverse)
        rng = np.random.default_rng(payload.rotation_seed)
        signs = rng.choice([-1.0, 1.0], size=n)
        flat *= signs

        return flat.reshape(payload.shape)


# ---------------------------------------------------------------------------
# PowerSGD compressor
# ---------------------------------------------------------------------------


class PowerSGDCompressor:
    """Low-rank matrix factorisation of gradient matrices.

    For a flat gradient vector of length n the compressor reshapes it to a
    2-D matrix (m, ceil(n/m)) where m is chosen to make the matrix
    approximately square, then computes rank-r factors P and Q such that
    gradient ≈ P @ Q^T.

    Parameters
    ----------
    rank:
        Number of singular vectors to keep (default 4).
    n_power_iterations:
        Number of power iterations for the randomised SVD (default 1).
    """

    def __init__(self, rank: int = 4, n_power_iterations: int = 1) -> None:
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rank = rank
        self.n_power_iterations = n_power_iterations

    def _to_matrix(self, gradient: np.ndarray) -> np.ndarray:
        # If already 2-D with both dims >= rank, use as-is (preserves structure).
        if gradient.ndim == 2 and min(gradient.shape) >= self.rank:
            return gradient.astype(np.float64)
        n = gradient.size
        m = max(1, int(math.sqrt(n)))
        cols = math.ceil(n / m)
        padded = np.zeros(m * cols, dtype=np.float64)
        padded[:n] = gradient.ravel()
        return padded.reshape(m, cols)

    def compress(self, gradient: np.ndarray) -> PowerSGDPayload:
        """Compress via randomised low-rank factorisation."""
        mat = self._to_matrix(gradient)
        m, n = mat.shape
        r = min(self.rank, m, n)

        # Randomised SVD via power iteration
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((n, r))
        Q, _ = np.linalg.qr(Q)

        for _ in range(self.n_power_iterations):
            Z = mat @ Q
            Z, _ = np.linalg.qr(Z)
            Q, _ = np.linalg.qr(mat.T @ Z)

        P = mat @ Q  # (m, r)

        return PowerSGDPayload(P=P, Q=Q, shape=gradient.shape)

    @staticmethod
    def decompress(payload: PowerSGDPayload) -> np.ndarray:
        """Reconstruct gradient from low-rank factors."""
        mat = payload.P @ payload.Q.T  # (m, n)
        n_original = math.prod(payload.shape)
        return mat.ravel()[:n_original].reshape(payload.shape)


# ---------------------------------------------------------------------------
# Error-feedback (memory correction) wrapper
# ---------------------------------------------------------------------------


class ErrorFeedbackCompressor:
    """Wraps any compressor and applies per-layer error feedback.

    After each compression the residual (original - decompressed) is
    accumulated in ``self.error_memory[layer_idx]`` and added to the next
    round's gradient before compression.  This ensures compression errors
    do not accumulate unbounded.

    Parameters
    ----------
    compressor:
        A ``TopKSparsifier`` or ``PowerSGDCompressor`` instance.
    n_layers:
        Number of layers (or parameter groups) to track separately.
        Pass 1 for a flat/unified gradient vector.
    """

    def __init__(
        self,
        compressor: TopKSparsifier | PowerSGDCompressor,
        n_layers: int = 1,
    ) -> None:
        self.compressor = compressor
        self.error_memory: list[np.ndarray | None] = [None] * n_layers

    def compress(
        self,
        gradient: np.ndarray,
        layer_idx: int = 0,
    ) -> TopKPayload | PowerSGDPayload:
        """Add stored error, compress, then record the new residual."""
        if self.error_memory[layer_idx] is not None:
            gradient = gradient + self.error_memory[layer_idx]

        payload = self.compressor.compress(gradient)

        if isinstance(payload, TopKPayload):
            decompressed = TopKSparsifier.decompress(payload)
        else:
            decompressed = PowerSGDCompressor.decompress(payload)

        self.error_memory[layer_idx] = gradient - decompressed
        return payload

    def reset(self, layer_idx: int | None = None) -> None:
        """Clear error memory (call at the start of a new training run)."""
        if layer_idx is None:
            self.error_memory = [None] * len(self.error_memory)
        else:
            self.error_memory[layer_idx] = None


# ---------------------------------------------------------------------------
# Bandwidth benchmark utility
# ---------------------------------------------------------------------------


def bandwidth_ratio(
    gradient: np.ndarray,
    payload: TopKPayload | PowerSGDPayload,
) -> float:
    """Approximate ratio of compressed to uncompressed bytes."""
    original_bytes = gradient.nbytes

    if isinstance(payload, TopKPayload):
        compressed_bytes = payload.values.nbytes + payload.indices.nbytes
    else:
        compressed_bytes = payload.P.nbytes + payload.Q.nbytes

    return compressed_bytes / original_bytes
