"""Pairwise additive secret sharing for secure gradient aggregation.

Each participant i generates random masks r_{i,j} = -r_{j,i} for every peer j.
Its net mask is sum_j r_{i,j}, which ensures sum_i mask_i == 0, so the
coordinator recovers the true aggregate without learning individual deltas.

Reference: Bonawitz et al., Practical Secure Aggregation (CCS 2017).
"""

from __future__ import annotations

import numpy as np


def generate_masks(
    participant_ids: list[str],
    delta_shape: tuple[int, ...],
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Return per-participant additive masks that sum to zero.

    Parameters
    ----------
    participant_ids:
        Ordered list of participant identifiers.
    delta_shape:
        Shape of each weight-delta array.
    rng:
        Optional seeded RNG for reproducibility in tests.

    Returns
    -------
    masks : dict mapping participant_id -> mask ndarray, same shape as delta.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(participant_ids)
    masks: dict[str, np.ndarray] = {pid: np.zeros(delta_shape) for pid in participant_ids}

    # For each ordered pair (i, j) with i < j, draw a symmetric cancelling mask.
    for i in range(n):
        for j in range(i + 1, n):
            r = rng.standard_normal(delta_shape)
            masks[participant_ids[i]] += r
            masks[participant_ids[j]] -= r

    return masks


def mask_delta(delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return delta + mask (masked weight update sent to coordinator)."""
    result: np.ndarray = delta + mask
    return result


def secure_sum(masked_deltas: list[np.ndarray]) -> np.ndarray:
    """Sum masked deltas; masks cancel so result == sum of true deltas."""
    result: np.ndarray = np.sum(masked_deltas, axis=0)
    return result
