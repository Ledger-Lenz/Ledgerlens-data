"""Domain-aware hard negative miner for contrastive pre-training.

Two negative sources are combined:
  - **Semi-hard negatives**: confirmed-clean wallets whose GNN embedding is
    *closest* to a given wash-trade anchor (hard in embedding space but from
    the opposite class).  Selected via an approximate nearest-neighbour index
    (FAISS HNSW) in O(k log n).
  - **Positive pairs from rings**: wash-trade wallet pairs that belong to the
    *same* detected ring are pulled together as positives.

All wallet IDs used during mining are HMAC-SHA256 hashed before they enter
any data structure or log line — raw G-addresses never appear in training
state.

Curriculum schedule
-------------------
Over the first ``CONTRASTIVE_CURRICULUM_EPOCHS`` epochs (default 5) the
sampler linearly interpolates the fraction of hard negatives from 0 → 1,
starting with purely random (easy) negatives and ramping up to fully hard by
the end of the warm-up period.

    hard_fraction(epoch) = min(epoch / curriculum_epochs, 1.0)

References
----------
- Robinson et al. (2021) "Contrastive Learning with Hard Negative Samples"
  https://arxiv.org/abs/2010.04592
- You et al. (2020) "Graph Contrastive Learning with Augmentations" (GraphCL)
  https://arxiv.org/abs/2010.13902
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("ledgerlens.negative_miner")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTRASTIVE_CURRICULUM_EPOCHS: int = int(
    os.getenv("CONTRASTIVE_CURRICULUM_EPOCHS", "5")
)
EVENT_HMAC_SECRET: str = os.getenv("EVENT_HMAC_SECRET", "ledgerlens-event-hmac-default")

# FAISS HNSW build parameters
_HNSW_M: int = 32          # number of bi-directional links per node
_HNSW_EF_CONSTRUCTION: int = 200
_HNSW_EF_SEARCH: int = 64

# ---------------------------------------------------------------------------
# Privacy helper
# ---------------------------------------------------------------------------


def _hash_wallet(wallet_id: str) -> str:
    """Return HMAC-SHA256 hex digest keyed by EVENT_HMAC_SECRET."""
    return hmac.new(
        EVENT_HMAC_SECRET.encode(),
        wallet_id.encode(),
        hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# FAISS wrapper with graceful fallback to brute-force
# ---------------------------------------------------------------------------


class _ANN:
    """Approximate nearest-neighbour index backed by FAISS HNSW.

    Falls back to exact brute-force cosine search when FAISS is not
    installed, so the miner works in any environment.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._embeddings: Optional[np.ndarray] = None  # (n, dim) float32
        self._index = None

        try:
            import faiss  # noqa: F401
            self._faiss_available = True
        except ImportError:
            logger.warning(
                "faiss not installed — falling back to brute-force cosine search. "
                "Install faiss-cpu for O(k log n) mining."
            )
            self._faiss_available = False

    def build(self, embeddings: np.ndarray) -> None:
        """Build the index from an (n, dim) float32 array of L2-normalised vectors."""
        emb = np.asarray(embeddings, dtype=np.float32)
        # L2-normalise so inner product == cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms
        self._embeddings = emb

        if self._faiss_available:
            import faiss

            index = faiss.IndexHNSWFlat(self._dim, _HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = _HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = _HNSW_EF_SEARCH
            index.add(emb)
            self._index = index
        # else: brute-force uses self._embeddings directly

    def query(self, query_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) of the k approximate nearest neighbours.

        Both arrays have shape (len(query_vecs), k).
        Distances are cosine similarities (higher = more similar).
        """
        q = np.asarray(query_vecs, dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        q = q / norms

        if self._faiss_available and self._index is not None:
            import faiss  # noqa: F401

            distances, indices = self._index.search(q, k)
            return distances, indices

        # Brute-force fallback
        assert self._embeddings is not None
        sims = q @ self._embeddings.T  # (q, n)
        top_k = np.argsort(-sims, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top_k, axis=1)
        return top_sims, top_k

    @property
    def is_built(self) -> bool:
        return self._embeddings is not None


# ---------------------------------------------------------------------------
# Ring-positive registry
# ---------------------------------------------------------------------------


class RingRegistry:
    """Stores wash-trade ring membership for positive-pair construction.

    All wallet IDs are stored as HMAC hashes — raw addresses never appear.

    Parameters
    ----------
    rings:
        List of sets of raw wallet G-addresses belonging to the same ring.
    """

    def __init__(self) -> None:
        self._hash_to_ring: dict[str, int] = {}
        self._rings: list[list[str]] = []

    @classmethod
    def from_rings(cls, rings: list[list[str]]) -> "RingRegistry":
        """Build from a list of raw-address ring groups."""
        registry = cls()
        for ring_idx, members in enumerate(rings):
            hashed_members = [_hash_wallet(w) for w in members]
            registry._rings.append(hashed_members)
            for h in hashed_members:
                registry._hash_to_ring[h] = ring_idx
        return registry

    def get_ring_partners(self, wallet_hash: str) -> list[str]:
        """Return hashed wallet IDs that are in the same ring as *wallet_hash*."""
        ring_idx = self._hash_to_ring.get(wallet_hash)
        if ring_idx is None:
            return []
        return [h for h in self._rings[ring_idx] if h != wallet_hash]

    def __len__(self) -> int:
        return sum(len(r) for r in self._rings)


# ---------------------------------------------------------------------------
# Main miner
# ---------------------------------------------------------------------------


class HardNegativeMiner:
    """Domain-aware hard negative miner for SimCLR contrastive pre-training.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the encoder output (h, before projector).
    curriculum_epochs:
        Number of warm-up epochs over which hard-negative fraction ramps
        from 0 → 1.  After this many epochs, 100 % hard negatives are used.
    rng_seed:
        NumPy random seed for reproducibility.

    Usage
    -----
    1. Call ``build_clean_index(clean_embeddings)`` once per epoch (or less
       frequently) to refresh the approximate nearest-neighbour index.
    2. Call ``mine_negatives(anchor_embeddings, wash_labels, n_negatives,
       epoch)`` to get a set of negative indices for each anchor in the batch.
    3. Call ``get_positive_pairs(hashed_wallet_ids)`` to retrieve ring-based
       positive pairs.
    """

    def __init__(
        self,
        embedding_dim: int,
        curriculum_epochs: int = CONTRASTIVE_CURRICULUM_EPOCHS,
        rng_seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.curriculum_epochs = curriculum_epochs
        self._rng = np.random.default_rng(rng_seed)
        self._ann = _ANN(embedding_dim)
        self._ring_registry: Optional[RingRegistry] = None
        # Total number of clean wallets in the index
        self._n_clean: int = 0

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_clean_index(self, clean_embeddings: np.ndarray) -> None:
        """Build (or rebuild) the ANN index over confirmed-clean wallet embeddings.

        Parameters
        ----------
        clean_embeddings:
            (n_clean, embedding_dim) float32 array of encoder outputs for
            wallets whose label == 0 (confirmed clean).
        """
        if len(clean_embeddings) == 0:
            logger.warning("build_clean_index called with empty embeddings — skipping")
            return
        self._ann.build(np.asarray(clean_embeddings, dtype=np.float32))
        self._n_clean = len(clean_embeddings)
        logger.debug("Built clean-wallet ANN index: %d vectors, dim=%d", self._n_clean, self.embedding_dim)

    def set_ring_registry(self, registry: RingRegistry) -> None:
        """Attach a RingRegistry for positive-pair construction."""
        self._ring_registry = registry

    # ------------------------------------------------------------------
    # Curriculum schedule
    # ------------------------------------------------------------------

    def hard_fraction(self, epoch: int) -> float:
        """Return the fraction of hard negatives to use at *epoch* (0-indexed).

        Linearly ramps from 0.0 at epoch 0 to 1.0 at epoch
        ``curriculum_epochs``.
        """
        if self.curriculum_epochs <= 0:
            return 1.0
        return min(float(epoch) / self.curriculum_epochs, 1.0)

    # ------------------------------------------------------------------
    # Negative mining
    # ------------------------------------------------------------------

    def mine_negatives(
        self,
        anchor_embeddings: np.ndarray,
        n_negatives: int,
        epoch: int,
    ) -> np.ndarray:
        """Mine negative indices (into the clean-wallet index) for each anchor.

        Parameters
        ----------
        anchor_embeddings:
            (batch, embedding_dim) float32 — embeddings of the wash-trade
            anchor wallets in the current batch.
        n_negatives:
            Number of negatives to return per anchor.
        epoch:
            Current training epoch (0-indexed).  Controls curriculum mix.

        Returns
        -------
        np.ndarray of shape (batch, n_negatives) with integer indices into
        the clean-wallet embedding array passed to ``build_clean_index``.
        """
        batch = len(anchor_embeddings)
        if self._n_clean == 0 or not self._ann.is_built:
            # No index — fall back to random negatives
            return self._random_negatives(batch, n_negatives)

        frac = self.hard_fraction(epoch)
        n_hard = int(round(frac * n_negatives))
        n_easy = n_negatives - n_hard

        result = np.empty((batch, n_negatives), dtype=np.int64)

        # --- hard negatives via ANN ---
        if n_hard > 0:
            k = min(n_hard + 1, self._n_clean)  # +1 to allow dedup
            _, nn_indices = self._ann.query(anchor_embeddings, k=k)
            # nn_indices: (batch, k) — take the top-n_hard
            hard_part = nn_indices[:, :n_hard]
            # Pad/trim to exactly n_hard columns
            if hard_part.shape[1] < n_hard:
                pad = self._random_negatives(batch, n_hard - hard_part.shape[1])
                hard_part = np.concatenate([hard_part, pad], axis=1)
            result[:, :n_hard] = hard_part

        # --- easy negatives (random) ---
        if n_easy > 0:
            result[:, n_hard:] = self._random_negatives(batch, n_easy)

        return result

    def _random_negatives(self, batch: int, k: int) -> np.ndarray:
        """Return random indices into the clean-wallet pool of shape (batch, k)."""
        n = max(self._n_clean, 1)
        return self._rng.integers(0, n, size=(batch, k))

    # ------------------------------------------------------------------
    # Positive pair construction from rings
    # ------------------------------------------------------------------

    def get_ring_positives(
        self,
        hashed_wallet_ids: list[str],
    ) -> list[tuple[int, int]]:
        """Return (i, j) index pairs where wallets i and j are in the same ring.

        Parameters
        ----------
        hashed_wallet_ids:
            Ordered list of HMAC-hashed wallet IDs for the current batch.

        Returns
        -------
        List of ``(anchor_idx, positive_idx)`` tuples where both indices
        point into *hashed_wallet_ids*.  May be empty if no ring partners
        are present in the batch.
        """
        if self._ring_registry is None:
            return []

        hash_to_idx = {h: i for i, h in enumerate(hashed_wallet_ids)}
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        for i, wh in enumerate(hashed_wallet_ids):
            for partner_hash in self._ring_registry.get_ring_partners(wh):
                j = hash_to_idx.get(partner_hash)
                if j is not None and i != j:
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)

        return pairs

    # ------------------------------------------------------------------
    # Convenience: build a mixed batch for NT-Xent
    # ------------------------------------------------------------------

    def build_contrastive_batch(
        self,
        wash_embeddings: np.ndarray,
        clean_embeddings_pool: np.ndarray,
        hashed_wash_ids: list[str],
        epoch: int,
        n_negatives_per_anchor: int = 4,
    ) -> dict:
        """Assemble a full contrastive batch dict ready for NT-Xent loss.

        Returns
        -------
        dict with keys:
          ``anchors``       — (n_wash, dim) wash-trade anchor embeddings
          ``positives``     — (n_pos, dim) ring-partner embeddings (may be empty)
          ``negatives``     — (n_wash, n_neg, dim) mined negative embeddings
          ``ring_pairs``    — list of (i, j) anchor index pairs for ring positives
          ``hard_fraction`` — float, hard negative fraction used this epoch
        """
        neg_indices = self.mine_negatives(wash_embeddings, n_negatives_per_anchor, epoch)
        # Gather negative embedding vectors
        neg_vecs = clean_embeddings_pool[neg_indices]  # (batch, n_neg, dim)

        ring_pairs = self.get_ring_positives(hashed_wash_ids)

        pos_embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
        if ring_pairs:
            pos_idx = list({j for _, j in ring_pairs} | {i for i, _ in ring_pairs})
            pos_embeddings = wash_embeddings[pos_idx]

        return {
            "anchors": wash_embeddings,
            "positives": pos_embeddings,
            "negatives": neg_vecs,
            "ring_pairs": ring_pairs,
            "hard_fraction": self.hard_fraction(epoch),
        }
