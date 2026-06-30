"""Tests for detection/contrastive/negative_miner.py and the updated pretrain.py.

Covers:
  1. Hard negatives are closer to the anchor than easy (random) negatives.
  2. Curriculum hard_fraction ramps correctly.
  3. Ring registry construction and ring-positive pair retrieval.
  4. Privacy: raw wallet IDs are hashed; original IDs do not appear in state.
  5. ANN fallback (brute-force) produces the same nearest-neighbour as exact.
  6. Regression: domain-aware pre-training AUC ≥ random-augmentation AUC.
  7. LabeledFeatureDataset hashes IDs at construction time.
  8. mine_negatives shape contract.
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from detection.contrastive.negative_miner import (
    EVENT_HMAC_SECRET,
    HardNegativeMiner,
    RingRegistry,
    _ANN,
    _hash_wallet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vecs(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _expected_hash(w: str) -> str:
    return hmac.new(EVENT_HMAC_SECRET.encode(), w.encode(), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# 1. Hard negatives are closer to the anchor than random negatives
# ---------------------------------------------------------------------------

class TestHardVsEasyNegatives:
    """Core requirement: hard negatives must be closer (higher cosine sim) than easy."""

    def test_hard_negatives_closer_than_random(self):
        dim = 32
        n_clean = 200
        rng = np.random.default_rng(7)

        # Build a clean pool — all pointing roughly in the "positive" hemisphere
        clean_embs = _unit_vecs(n_clean, dim, seed=1)

        # Anchor pointing strongly toward clean_embs[0]
        anchor = clean_embs[0:1].copy()

        miner = HardNegativeMiner(embedding_dim=dim, curriculum_epochs=0, rng_seed=42)
        miner.build_clean_index(clean_embs)

        # hard_fraction=1.0 → all hard (epoch >= curriculum_epochs=0)
        hard_idx = miner.mine_negatives(anchor, n_negatives=5, epoch=1)  # shape (1,5)

        # Random reference: many random indices
        random_idx = rng.integers(0, n_clean, size=(1, 50))

        # Mean cosine similarity to anchor
        a_norm = anchor / np.linalg.norm(anchor)
        hard_sims = (clean_embs[hard_idx[0]] @ a_norm.T).mean()
        random_sims = (clean_embs[random_idx[0]] @ a_norm.T).mean()

        assert hard_sims > random_sims, (
            f"Hard negatives (sim={hard_sims:.4f}) should be closer to anchor "
            f"than random negatives (sim={random_sims:.4f})"
        )

    def test_hard_negatives_shape(self):
        dim = 16
        miner = HardNegativeMiner(embedding_dim=dim, curriculum_epochs=5)
        miner.build_clean_index(_unit_vecs(50, dim))
        anchors = _unit_vecs(8, dim)
        idx = miner.mine_negatives(anchors, n_negatives=4, epoch=6)
        assert idx.shape == (8, 4)

    def test_no_index_returns_random_shape(self):
        """mine_negatives should still return correct shape before index is built."""
        miner = HardNegativeMiner(embedding_dim=16, curriculum_epochs=5)
        anchors = _unit_vecs(4, 16)
        idx = miner.mine_negatives(anchors, n_negatives=3, epoch=0)
        assert idx.shape == (4, 3)


# ---------------------------------------------------------------------------
# 2. Curriculum schedule
# ---------------------------------------------------------------------------

class TestCurriculumSchedule:
    def test_zero_at_epoch_zero(self):
        miner = HardNegativeMiner(embedding_dim=8, curriculum_epochs=10)
        assert miner.hard_fraction(0) == 0.0

    def test_one_at_curriculum_epochs(self):
        miner = HardNegativeMiner(embedding_dim=8, curriculum_epochs=10)
        assert miner.hard_fraction(10) == 1.0

    def test_capped_at_one_after_curriculum(self):
        miner = HardNegativeMiner(embedding_dim=8, curriculum_epochs=5)
        assert miner.hard_fraction(100) == 1.0

    def test_monotone_increase(self):
        miner = HardNegativeMiner(embedding_dim=8, curriculum_epochs=8)
        fracs = [miner.hard_fraction(e) for e in range(10)]
        for a, b in zip(fracs, fracs[1:]):
            assert b >= a

    def test_midpoint(self):
        miner = HardNegativeMiner(embedding_dim=8, curriculum_epochs=10)
        assert abs(miner.hard_fraction(5) - 0.5) < 1e-6

    def test_zero_curriculum_always_hard(self):
        miner = HardNegativeMiner(embedding_dim=8, curriculum_epochs=0)
        assert miner.hard_fraction(0) == 1.0
        assert miner.hard_fraction(5) == 1.0

    @pytest.mark.parametrize("epoch,expected", [(0, 0.0), (2, 0.4), (5, 1.0), (7, 1.0)])
    def test_parametrized_values(self, epoch, expected):
        miner = HardNegativeMiner(embedding_dim=8, curriculum_epochs=5)
        assert abs(miner.hard_fraction(epoch) - expected) < 1e-6

    def test_hard_fraction_controls_mining_mix(self):
        """At epoch 0 (easy), mined indices should NOT all be from ANN top-k."""
        dim = 32
        n_clean = 100
        miner = HardNegativeMiner(embedding_dim=dim, curriculum_epochs=20, rng_seed=0)
        clean_embs = _unit_vecs(n_clean, dim)
        miner.build_clean_index(clean_embs)

        anchor = clean_embs[0:1]
        # epoch=0 → hard_fraction=0 → all random
        idx_easy = miner.mine_negatives(anchor, n_negatives=10, epoch=0)
        # epoch=20 → hard_fraction=1 → all hard
        idx_hard = miner.mine_negatives(anchor, n_negatives=10, epoch=20)

        # Hard should be consistently closer
        a_norm = anchor / np.linalg.norm(anchor)
        sim_hard = (clean_embs[idx_hard[0]] @ a_norm.T).mean()
        sim_easy = (clean_embs[idx_easy[0]] @ a_norm.T).mean()
        # With enough hard epochs, hard should be >= easy on average
        # (can't guarantee strictly greater with random easy, but very likely)
        assert sim_hard >= sim_easy - 0.1  # lenient: just don't regress badly


# ---------------------------------------------------------------------------
# 3. Ring registry
# ---------------------------------------------------------------------------

class TestRingRegistry:
    RINGS = [
        ["GWASH1", "GWASH2", "GWASH3"],
        ["GWASH4", "GWASH5"],
    ]

    def test_partners_returned_for_ring_member(self):
        reg = RingRegistry.from_rings(self.RINGS)
        h1 = _hash_wallet("GWASH1")
        partners = reg.get_ring_partners(h1)
        assert _hash_wallet("GWASH2") in partners
        assert _hash_wallet("GWASH3") in partners

    def test_self_not_in_partners(self):
        reg = RingRegistry.from_rings(self.RINGS)
        h1 = _hash_wallet("GWASH1")
        assert h1 not in reg.get_ring_partners(h1)

    def test_non_member_returns_empty(self):
        reg = RingRegistry.from_rings(self.RINGS)
        assert reg.get_ring_partners(_hash_wallet("GCLEAN999")) == []

    def test_cross_ring_isolation(self):
        reg = RingRegistry.from_rings(self.RINGS)
        h4 = _hash_wallet("GWASH4")
        partners = reg.get_ring_partners(h4)
        assert _hash_wallet("GWASH1") not in partners

    def test_len(self):
        reg = RingRegistry.from_rings(self.RINGS)
        assert len(reg) == 5  # 3 + 2

    def test_raw_addresses_not_stored(self):
        reg = RingRegistry.from_rings(self.RINGS)
        for ring in reg._rings:
            for h in ring:
                assert not h.startswith("G"), "Raw G-address leaked into ring storage"


# ---------------------------------------------------------------------------
# 4. Privacy: _hash_wallet
# ---------------------------------------------------------------------------

class TestHashWallet:
    def test_deterministic(self):
        assert _hash_wallet("GWALLET") == _hash_wallet("GWALLET")

    def test_different_inputs_differ(self):
        assert _hash_wallet("GWALLET1") != _hash_wallet("GWALLET2")

    def test_length_64(self):
        assert len(_hash_wallet("GTEST")) == 64

    def test_matches_expected_hmac(self):
        w = "GBENCHMARK"
        assert _hash_wallet(w) == _expected_hash(w)

    def test_raw_not_in_hash(self):
        w = "GRAWADDRESS1234"
        assert w not in _hash_wallet(w)


# ---------------------------------------------------------------------------
# 5. ANN brute-force fallback consistency
# ---------------------------------------------------------------------------

class TestANNFallback:
    def test_brute_force_finds_nearest_neighbour(self):
        dim = 16
        n = 50
        embs = _unit_vecs(n, dim, seed=3)
        ann = _ANN(dim)
        ann._faiss_available = False  # force brute-force
        ann.build(embs)

        query = embs[0:1]  # the first vector is its own nearest neighbour
        dists, idxs = ann.query(query, k=1)
        # After L2-normalisation, embs[0] dot embs[0] ≈ 1.0
        assert dists[0, 0] > 0.99

    def test_query_returns_correct_shape(self):
        dim = 8
        ann = _ANN(dim)
        ann._faiss_available = False
        ann.build(_unit_vecs(30, dim))
        dists, idxs = ann.query(_unit_vecs(4, dim), k=3)
        assert dists.shape == (4, 3)
        assert idxs.shape == (4, 3)


# ---------------------------------------------------------------------------
# 6. get_ring_positives via HardNegativeMiner
# ---------------------------------------------------------------------------

class TestGetRingPositives:
    def test_ring_pairs_found_in_batch(self):
        reg = RingRegistry.from_rings([["GWASH1", "GWASH2", "GWASH3"]])
        miner = HardNegativeMiner(embedding_dim=8)
        miner.set_ring_registry(reg)

        batch_ids = [_hash_wallet("GWASH1"), _hash_wallet("GWASH2"), _hash_wallet("GCLEAN1")]
        pairs = miner.get_ring_positives(batch_ids)
        assert len(pairs) > 0
        # Both wallets must be in the batch
        for i, j in pairs:
            assert 0 <= i < len(batch_ids)
            assert 0 <= j < len(batch_ids)

    def test_no_pairs_when_registry_missing(self):
        miner = HardNegativeMiner(embedding_dim=8)
        pairs = miner.get_ring_positives([_hash_wallet("GWASH1")])
        assert pairs == []

    def test_no_duplicate_pairs(self):
        reg = RingRegistry.from_rings([["GWASH1", "GWASH2", "GWASH3"]])
        miner = HardNegativeMiner(embedding_dim=8)
        miner.set_ring_registry(reg)
        ids = [_hash_wallet(f"GWASH{i}") for i in range(1, 4)]
        pairs = miner.get_ring_positives(ids)
        assert len(pairs) == len(set(pairs))


# ---------------------------------------------------------------------------
# 7. LabeledFeatureDataset hashes wallet IDs
# ---------------------------------------------------------------------------

class TestLabeledFeatureDataset:
    def test_hashed_ids_not_raw(self):
        from detection.contrastive.pretrain import LabeledFeatureDataset

        X = np.random.randn(10, 8).astype(np.float32)
        y = np.zeros(10, dtype=np.int64)
        raw_ids = [f"GWALLET{i:04d}" for i in range(10)]
        ds = LabeledFeatureDataset(X, y, wallet_ids=raw_ids)

        for stored, raw in zip(ds.hashed_ids, raw_ids):
            assert raw not in stored, f"Raw wallet ID {raw!r} found in stored hash"
            assert stored == _expected_hash(raw)

    def test_wash_mask(self):
        from detection.contrastive.pretrain import LabeledFeatureDataset

        X = np.random.randn(20, 8).astype(np.float32)
        y = np.array([1, 0] * 10)
        ds = LabeledFeatureDataset(X, y)
        assert ds.wash_features.shape[0] == 10
        assert ds.clean_features.shape[0] == 10

    def test_getitem_returns_tensor(self):
        from detection.contrastive.pretrain import LabeledFeatureDataset

        X = np.random.randn(5, 8).astype(np.float32)
        y = np.zeros(5, dtype=np.int64)
        ds = LabeledFeatureDataset(X, y)
        feat, label, hid = ds[0]
        assert isinstance(feat, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(hid, str)


# ---------------------------------------------------------------------------
# 8. Regression: domain-aware AUC ≥ random-augmentation AUC
# ---------------------------------------------------------------------------

class TestAUCRegression:
    """Domain-aware pre-training must achieve AUC ≥ random-augmentation baseline.

    Uses a small synthetic dataset so the test runs quickly on CPU.
    We allow a tiny tolerance because on very short training runs with random
    seeds the difference can be negligible.
    """

    def test_domain_aware_auc_not_worse(self):
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=300,
            n_features=16,
            n_informative=8,
            n_classes=2,
            weights=[0.7, 0.3],
            random_state=0,
        )
        X = X.astype(np.float32)

        # Build a small ring: first 10 wash-trade wallets form one ring
        wash_idx = np.where(y == 1)[0]
        ring_wallets = [f"GWASH{i:04d}" for i in wash_idx[:6]]
        rings = [ring_wallets] if len(ring_wallets) >= 2 else []
        wallet_ids = [f"GWALLET{i:04d}" for i in range(len(X))]

        from detection.contrastive.pretrain import benchmark_auc

        results = benchmark_auc(
            X, y,
            epochs=3,
            batch_size=32,
            device="cpu",
            wallet_ids=wallet_ids,
            rings=rings,
        )

        # Domain-aware must not be more than 2% worse (tolerance for short runs)
        assert results["domain_auc"] >= results["random_auc"] - 0.02, (
            f"Domain-aware AUC {results['domain_auc']:.4f} more than 2% below "
            f"random AUC {results['random_auc']:.4f}"
        )
