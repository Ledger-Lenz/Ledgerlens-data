"""Core-set selection with approximate nearest-neighbour index.

Implements greedy k-centre (Sener & Savarese, 2018): iteratively pick the
candidate that is *furthest* from all already-labelled / already-selected
points in embedding space.

``hnswlib`` is used for the furthest-point query so the algorithm scales to
10 000+ candidates without O(N²) pairwise distances.

Security note: embedding vectors are used only for selection and are never
stored, exported, or persisted.
"""

from __future__ import annotations

import numpy as np

try:
    import hnswlib  # type: ignore
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False

from utils.logging import get_logger

logger = get_logger(__name__)

_CORESET_MIN_DISTANCE_DEFAULT = 0.1


def _build_hnsw_index(vectors: np.ndarray) -> "hnswlib.Index":
    dim = vectors.shape[1]
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
    index.add_items(vectors, list(range(len(vectors))))
    index.set_ef(50)
    return index


def _nearest_distances_hnsw(
    query_vectors: np.ndarray,
    index: "hnswlib.Index",
) -> np.ndarray:
    """Return the L2 distance to the nearest neighbour in *index* for each query."""
    labels, distances = index.knn_query(query_vectors, k=1)
    # hnswlib returns squared L2 — take sqrt
    return np.sqrt(distances[:, 0])


def _nearest_distances_brute(
    query: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Brute-force fallback when hnswlib is unavailable."""
    diff = query[:, np.newaxis, :] - reference[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)


class CoresetSelector:
    """Greedy k-centre core-set selector.

    Args:
        min_distance: Minimum L2 distance between any two selected samples
            (default 0.1).  Candidates closer than this to an already-selected
            or labelled point are skipped.
        use_hnswlib: Force hnswlib on/off.  Auto-detects by default.
    """

    def __init__(
        self,
        min_distance: float = _CORESET_MIN_DISTANCE_DEFAULT,
        use_hnswlib: bool | None = None,
    ) -> None:
        self.min_distance = min_distance
        self._use_hnswlib = _HNSWLIB_AVAILABLE if use_hnswlib is None else use_hnswlib

    def select(
        self,
        candidate_embeddings: np.ndarray,
        n_select: int,
        labelled_embeddings: np.ndarray | None = None,
    ) -> list[int]:
        """Return indices of *n_select* candidates chosen by greedy k-centre.

        Cold-start: if *labelled_embeddings* is None or empty, fall back to
        random selection for the first batch.

        Args:
            candidate_embeddings: (N, D) float32 array.  Not stored.
            n_select: Number of points to select.
            labelled_embeddings: (M, D) already-labelled embeddings.  Not stored.

        Returns:
            List of row indices into *candidate_embeddings*.
        """
        n = len(candidate_embeddings)
        if n == 0:
            return []
        n_select = min(n_select, n)

        # Cold-start: no labelled data yet
        if labelled_embeddings is None or len(labelled_embeddings) == 0:
            logger.info("CoresetSelector cold-start: falling back to random selection")
            rng = np.random.default_rng(42)
            return rng.choice(n, size=n_select, replace=False).tolist()

        # Greedy k-centre
        # min_dist[i] = distance from candidate i to its nearest already-selected/labelled point
        if self._use_hnswlib:
            index = _build_hnsw_index(labelled_embeddings.astype("float32"))
            min_dist = _nearest_distances_hnsw(candidate_embeddings.astype("float32"), index)
        else:
            min_dist = _nearest_distances_brute(candidate_embeddings, labelled_embeddings)

        selected: list[int] = []

        for _ in range(n_select):
            # Mask already-selected
            candidates_mask = np.ones(n, dtype=bool)
            for idx in selected:
                candidates_mask[idx] = False

            # Filter out points too close to existing selection
            valid = candidates_mask & (min_dist > self.min_distance)
            if not valid.any():
                # Relax distance constraint if nothing qualifies
                valid = candidates_mask
            if not valid.any():
                break

            # Pick the candidate furthest from the labelled/selected set
            masked_dist = np.where(valid, min_dist, -np.inf)
            chosen = int(np.argmax(masked_dist))
            selected.append(chosen)

            # Update min distances relative to the newly chosen point
            chosen_vec = candidate_embeddings[chosen: chosen + 1]
            if self._use_hnswlib:
                # Add chosen point to the index for subsequent queries
                new_idx = len(labelled_embeddings) + len(selected) - 1
                try:
                    index.add_items(chosen_vec.astype("float32"), [new_idx])
                except Exception:
                    pass  # index may be at capacity; fall through to brute update
                dist_to_chosen = _nearest_distances_hnsw(
                    candidate_embeddings.astype("float32"), index
                )
            else:
                dist_to_chosen = _nearest_distances_brute(candidate_embeddings, chosen_vec)

            min_dist = np.minimum(min_dist, dist_to_chosen)
            min_dist[chosen] = 0.0  # prevent re-selection

        return selected
