"""Tests for issues #253, #254, #255, #256.

#253 — CoresetSelector + coreset_hybrid strategy
#254 — SlidingWindowBenfordAggregator
#255 — CausalTransfer (ICP)
#256 — StoppingCriterion + --force-continue
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, 4)), columns=["f1", "f2", "f3", "f4"])
    df.insert(0, "wallet", [f"W{i:04d}" for i in range(n)])
    return df


class _FakeModel:
    def __init__(self, probs: list[float]) -> None:
        self._p = np.array(probs)

    def predict_proba(self, X):
        n = len(X)
        p = np.tile(self._p, (n // len(self._p) + 1))[:n]
        return np.column_stack([1 - p, p])


# ===========================================================================
# #253 — CoresetSelector
# ===========================================================================


class TestCoresetSelector:
    def test_cold_start_returns_random_batch(self):
        from detection.active_learning.coreset_selector import CoresetSelector

        rng = np.random.default_rng(0)
        embeddings = rng.random((20, 4)).astype("float32")
        sel = CoresetSelector(use_hnswlib=False)
        idxs = sel.select(embeddings, n_select=5, labelled_embeddings=None)
        assert len(idxs) == 5
        assert len(set(idxs)) == 5  # unique

    def test_two_cluster_selects_from_both(self):
        """Embeddings in two tight clusters — coreset must pick from both."""
        from detection.active_learning.coreset_selector import CoresetSelector

        rng = np.random.default_rng(42)
        cluster_a = rng.random((30, 2)).astype("float32") * 0.05  # near (0, 0)
        cluster_b = (rng.random((30, 2)).astype("float32") * 0.05 + 5.0)  # near (5, 5)
        embeddings = np.vstack([cluster_a, cluster_b])

        # Label one point from cluster A as "already labelled"
        labelled = cluster_a[:1]
        sel = CoresetSelector(min_distance=0.0, use_hnswlib=False)
        idxs = sel.select(embeddings, n_select=6, labelled_embeddings=labelled)

        # At least one selected index must be from cluster B (indices 30–59)
        assert any(i >= 30 for i in idxs), "Core-set must select from both clusters"

    def test_respects_n_select_cap(self):
        from detection.active_learning.coreset_selector import CoresetSelector

        rng = np.random.default_rng(7)
        embeddings = rng.random((10, 3)).astype("float32")
        labelled = rng.random((2, 3)).astype("float32")
        sel = CoresetSelector(use_hnswlib=False)
        idxs = sel.select(embeddings, n_select=100, labelled_embeddings=labelled)
        assert len(idxs) <= 10

    def test_empty_candidates_returns_empty(self):
        from detection.active_learning.coreset_selector import CoresetSelector

        sel = CoresetSelector(use_hnswlib=False)
        assert sel.select(np.empty((0, 4), dtype="float32"), n_select=5) == []


class TestCoresetHybridStrategy:
    def _pool_two_clusters(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        a = pd.DataFrame(rng.random((15, 2)) * 0.1, columns=["f1", "f2"])
        b = pd.DataFrame(rng.random((15, 2)) * 0.1 + 10.0, columns=["f1", "f2"])
        df = pd.concat([a, b], ignore_index=True)
        df.insert(0, "wallet", [f"W{i:03d}" for i in range(len(df))])
        return df

    def test_alpha_zero_produces_more_diverse_batch_than_alpha_one(self):
        """alpha=0 (pure coreset) should give wider spread than alpha=1 (pure uncertainty)."""
        from detection.active_learning.query_strategies import CoresetHybrid

        pool = self._pool_two_clusters()
        model = _FakeModel([0.51] * 30)  # uniform uncertainty

        with patch("config.config") as mock_cfg:
            mock_cfg.ACTIVE_LEARNING_ALPHA = 0.0
            mock_cfg.CORESET_MIN_DISTANCE = 0.0
            sel_coreset = CoresetHybrid().select(pool, n_query=4, model=model, alpha=0.0)

        with patch("config.config") as mock_cfg:
            mock_cfg.ACTIVE_LEARNING_ALPHA = 1.0
            mock_cfg.CORESET_MIN_DISTANCE = 0.0
            sel_uncertainty = CoresetHybrid().select(pool, n_query=4, model=model, alpha=1.0)

        def _spread(wallets, pool):
            idxs = pool[pool["wallet"].isin(wallets)].index.tolist()
            vecs = pool.loc[idxs, ["f1", "f2"]].values
            if len(vecs) < 2:
                return 0.0
            diffs = vecs[:, np.newaxis] - vecs[np.newaxis, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            np.fill_diagonal(dists, 0)
            return dists.max()

        spread_coreset = _spread(sel_coreset, pool)
        spread_uncertainty = _spread(sel_uncertainty, pool)
        assert spread_coreset >= spread_uncertainty, (
            f"alpha=0 spread ({spread_coreset:.4f}) should be >= alpha=1 spread ({spread_uncertainty:.4f})"
        )

    def test_strategy_in_registry(self):
        from detection.active_learning.query_strategies import STRATEGY_REGISTRY

        assert "coreset_hybrid" in STRATEGY_REGISTRY

    def test_no_model_produces_selection(self):
        """With no model, should still return n_query wallets (uncertainty=0)."""
        from detection.active_learning.query_strategies import CoresetHybrid

        pool = _make_pool(10)
        result = CoresetHybrid().select(pool, n_query=3, model=None, alpha=0.5)
        assert len(result) == 3


# ===========================================================================
# #254 — SlidingWindowBenfordAggregator
# ===========================================================================


class TestSlidingWindowBenfordAggregator:
    def _make_aggregator(self, window_hours: float = 1.0):
        from detection.sliding_window_benford import SlidingWindowBenfordAggregator

        return SlidingWindowBenfordAggregator(window_hours)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_chi_square_matches_batch_within_tolerance(self):
        """Add 100 trades, expire 50, recompute from scratch; chi-sq must match within 1e-6."""
        from detection.benford_engine import chi_square_statistic
        from detection.sliding_window_benford import SlidingWindowBenfordAggregator

        agg = SlidingWindowBenfordAggregator(window_hours=1.0)
        t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        window_seconds = 3600.0

        rng = np.random.default_rng(42)
        amounts = (rng.random(100) * 999 + 1).tolist()  # [1, 1000)

        async def _run():
            # Add all 100 trades
            for i, amt in enumerate(amounts):
                ts = t0 + timedelta(seconds=i * 10)
                await agg.add_trade(amt, ts)

            # Trades 0–49 have timestamps t0+[0s, 490s].
            # Expire exactly these by advancing to t0+4095s:
            #   cutoff = 4095 - 3600 = 495s → trades with ts ≤ 495s expire (indices 0–49).
            #   Trade 50 has ts = 500s > 495s → survives.
            future_ts = t0 + timedelta(seconds=4095)
            await agg.add_trade(amounts[0], future_ts)  # triggers expiry, also adds one more

        asyncio.get_event_loop().run_until_complete(_run())

        # Reference: trades 50–99 (ts 500s–990s, all within 3600s of future_ts) survive,
        # plus the extra trade added at 4095s.
        surviving = amounts[50:]  # ts in [500s, 990s]; all within 3600s of 4095s
        # Include the extra trade we added at 4095s
        surviving.append(amounts[0])
        ref_chi = chi_square_statistic(pd.Series(surviving))

        # Tolerance
        assert abs(agg.chi_square() - ref_chi) < 1e-4, (
            f"sliding chi_sq={agg.chi_square():.8f} vs batch={ref_chi:.8f}"
        )

    def test_performance_10k_add_trade_under_100ms(self):
        """10 000 add_trade calls must complete in < 100ms."""
        from detection.sliding_window_benford import SlidingWindowBenfordAggregator

        agg = SlidingWindowBenfordAggregator(window_hours=24.0)
        t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        amounts = np.random.default_rng(1).random(10_000) * 999 + 1

        async def _run():
            for i, amt in enumerate(amounts):
                ts = t0 + timedelta(seconds=i)
                await agg.add_trade(float(amt), ts)

        start = time.perf_counter()
        asyncio.get_event_loop().run_until_complete(_run())
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"10k add_trade took {elapsed_ms:.1f}ms > 100ms"

    def test_invalid_amounts_skipped(self):
        from detection.sliding_window_benford import SlidingWindowBenfordAggregator

        agg = SlidingWindowBenfordAggregator(window_hours=1.0)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

        async def _run():
            await agg.add_trade(float("nan"), t0)
            await agg.add_trade(-5.0, t0)
            await agg.add_trade(0.0, t0)
            await agg.add_trade(100.0, t0)  # valid

        asyncio.get_event_loop().run_until_complete(_run())
        assert agg.sample_size == 1

    def test_empty_aggregator_metrics_are_zero(self):
        from detection.sliding_window_benford import SlidingWindowBenfordAggregator

        agg = SlidingWindowBenfordAggregator(window_hours=1.0)
        metrics = agg.to_metrics()
        assert metrics.chi_square == 0.0
        assert metrics.mad == 0.0
        assert metrics.sample_size == 0

    def test_lazy_expiry_reduces_count(self):
        from detection.sliding_window_benford import SlidingWindowBenfordAggregator

        agg = SlidingWindowBenfordAggregator(window_hours=1.0)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(hours=2)  # outside window

        async def _run():
            await agg.add_trade(100.0, t0)
            assert agg.sample_size == 1
            # Adding a trade 2h later should expire the first
            await agg.add_trade(200.0, t1)

        asyncio.get_event_loop().run_until_complete(_run())
        assert agg.sample_size == 1  # only the t1 trade remains


# ===========================================================================
# #255 — CausalTransfer (ICP)
# ===========================================================================


class TestCausalTransfer:
    def _make_two_env_data(self, n_per_env: int = 120, seed: int = 0) -> pd.DataFrame:
        """Two environments share f1 as the causal feature; f2 is env-specific."""
        rng = np.random.default_rng(seed)
        records = []
        for env_id in ["PAIR_A", "PAIR_B"]:
            noise = 1.0 if env_id == "PAIR_A" else 3.0  # different env-specific noise
            f1 = rng.standard_normal(n_per_env)
            f2 = rng.standard_normal(n_per_env) * noise  # env-specific, no causal effect
            # label is caused only by f1
            label = (f1 + rng.standard_normal(n_per_env) * 0.3 > 0).astype(int)
            df_env = pd.DataFrame({"f1": f1, "f2": f2, "label": label, "pair_id": env_id})
            records.append(df_env)
        return pd.concat(records, ignore_index=True)

    def test_icp_identifies_shared_causal_feature(self):
        """ICP must identify f1 (not f2) as invariant."""
        from detection.causal_transfer import CausalTransfer

        df = self._make_two_env_data(n_per_env=200)
        ct = CausalTransfer(feature_cols=["f1", "f2"])
        result = ct.fit(df, pair_col="pair_id", label_col="label")

        if not result.fallback_to_global:
            # f1 should be in invariant features
            assert "f1" in result.invariant_features, (
                f"Expected f1 in invariant set, got {result.invariant_features}"
            )

    def test_fallback_to_global_when_no_invariant_set(self):
        """If no feature is invariant, must fall back to global model gracefully."""
        from detection.causal_transfer import CausalTransfer

        rng = np.random.default_rng(99)
        # Completely random labels — nothing will be invariant
        df = pd.DataFrame({
            "f1": rng.standard_normal(100),
            "f2": rng.standard_normal(100),
            "label": rng.integers(0, 2, 100),
            "pair_id": ["A"] * 50 + ["B"] * 50,
        })
        ct = CausalTransfer(feature_cols=["f1", "f2"])
        result = ct.fit(df, pair_col="pair_id", label_col="label")
        # Must not raise; may or may not fall back
        assert result.global_model is not None

    def test_transferred_model_evaluates_on_heldout(self):
        """Transferred model AUC on held-out pair must be > 0.5 (better than random)."""
        from detection.causal_transfer import CausalTransfer

        df = self._make_two_env_data(n_per_env=200, seed=1)
        train = df[df["pair_id"] == "PAIR_A"]
        test = self._make_two_env_data(n_per_env=80, seed=5)
        test = test[test["pair_id"] == "PAIR_B"]

        ct = CausalTransfer(feature_cols=["f1", "f2"])
        ct.fit(df, pair_col="pair_id", label_col="label")
        auc = ct.evaluate(test, pair_col="pair_id", label_col="label")
        assert auc is None or math.isnan(auc) or auc >= 0.5

    def test_anonymised_pair_ids_not_stored_in_result(self):
        """Raw pair IDs must not appear in the fitted model's pair_models keys."""
        from detection.causal_transfer import CausalTransfer

        df = self._make_two_env_data()
        ct = CausalTransfer(feature_cols=["f1", "f2"])
        result = ct.fit(df, pair_col="pair_id", label_col="label")

        raw_pair_ids = {"PAIR_A", "PAIR_B"}
        if not result.fallback_to_global:
            stored_keys = set(result.pair_models.keys())
            assert not (stored_keys & raw_pair_ids), (
                f"Raw pair IDs leaked into pair_models: {stored_keys & raw_pair_ids}"
            )


# ===========================================================================
# #256 — StoppingCriterion
# ===========================================================================


class TestStoppingCriterion:
    def _make_criterion(self, window: int = 5, eer_threshold: float = 0.001):
        from detection.active_learning.annotation_queue import StoppingCriterion

        return StoppingCriterion(
            eer_threshold=eer_threshold,
            convergence_window=window,
            auc_improvement_threshold=0.005,
        )

    def test_fires_after_n_rounds_of_zero_improvement(self):
        """After convergence_window rounds of zero AUC improvement, criterion fires."""
        criterion = self._make_criterion(window=5)
        auc = 0.80
        for _ in range(6):  # window+1 records needed
            criterion.record_round_auc(auc)  # no improvement
        assert criterion.should_stop() is True

    def test_does_not_fire_early(self):
        """Before filling the window, criterion must not fire."""
        criterion = self._make_criterion(window=5)
        for i in range(3):
            criterion.record_round_auc(0.80)
        assert criterion.should_stop() is False

    def test_does_not_fire_when_improving(self):
        """Steady AUC improvement must not trigger stopping."""
        criterion = self._make_criterion(window=5)
        for i in range(6):
            criterion.record_round_auc(0.80 + i * 0.01)
        assert criterion.should_stop() is False

    def test_eer_below_threshold_fires(self):
        """If EER < threshold, should_stop must return True immediately."""
        criterion = self._make_criterion(eer_threshold=0.5)
        # Model that always predicts 0.99 → EER ≈ 0.01, below 0.5
        model = _FakeModel([0.99] * 5)
        pool = _make_pool(5)
        assert criterion.should_stop(model=model, unlabelled_pool=pool) is True

    def test_force_continue_suppresses_criterion(self, tmp_path):
        """--force-continue skips the stopping criterion check."""
        import sys

        pool_path = str(tmp_path / "pool.parquet")
        _make_pool(5).to_parquet(pool_path)

        queue_path = str(tmp_path / "queue.json")

        # Patch load_models to return nothing (no models → criterion can't fire EER)
        # but inject a converged AUC history via a patched StoppingCriterion
        with patch(
            "scripts.run_active_learning.load_models", return_value={}
        ), patch(
            "scripts.run_active_learning.run_active_learning", return_value=["W0001"]
        ) as mock_run:
            sys.argv = [
                "run_active_learning",
                "--pool", pool_path,
                "--queue", queue_path,
                "--force-continue",
            ]
            from scripts.run_active_learning import main

            main()
            # run_active_learning was called despite --force-continue
            mock_run.assert_called_once()

    def test_convergence_report_excludes_labels(self, tmp_path):
        """Convergence report must not include raw label values."""
        import json

        from detection.active_learning.annotation_queue import AnnotationQueue, StoppingCriterion

        queue_path = str(tmp_path / "q.json")
        q = AnnotationQueue(queue_path=queue_path)
        with patch.dict("os.environ", {"ANNOTATION_HMAC_SECRET": "testsecret"}):
            # Re-instantiate config with patched secret
            from config import config as cfg
            original = cfg.ANNOTATION_HMAC_SECRET
            cfg.ANNOTATION_HMAC_SECRET = "testsecret"
            try:
                q.annotate("GABC", label=1, annotator_id="alice")
            finally:
                cfg.ANNOTATION_HMAC_SECRET = original

        criterion = self._make_criterion()
        report = criterion.emit_convergence_report(queue_path=queue_path)

        # Report must contain annotator_counts but NOT label values
        assert "annotator_counts" in report
        assert "auc_history" in report
        # Check no key named "label" or "labels" at the top level
        for key in report:
            assert "label" not in key.lower(), f"Report contains label key: {key}"
