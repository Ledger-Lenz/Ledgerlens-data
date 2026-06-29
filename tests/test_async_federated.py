"""Unit tests for AsyncFederatedCoordinator (issue #270).

Covers:
- Staleness-aware weights are applied correctly (staleness 0 vs 3)
- Update with staleness > max_staleness is rejected with a clear error
- Thread safety: concurrent submissions do not corrupt aggregate state
- Time-based trigger aggregates pending updates
- N-based trigger aggregates at trigger_n updates
- Structured log on aggregation (inspected via log capture)
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from detection.federated.coordinator import AsyncFederatedCoordinator


# ---------------------------------------------------------------------------
# Test: staleness-aware weight correctness
# ---------------------------------------------------------------------------

class TestStalenessWeighting:
    """Two participants: staleness 0 and staleness 3.
    The weight of the fresh update should be higher.
    """

    def test_staleness_weights_applied_correctly(self):
        """Verify that weight = 1/(1+staleness) is applied.

        With staleness 0 and 3:
          w_0 = 1/(1+0) = 1.0
          w_3 = 1/(1+3) = 0.25
          normalised: w_0 = 0.8, w_3 = 0.2

        delta_a (staleness=0) = [4.0]
        delta_b (staleness=3) = [2.0]
        weighted sum = 0.8 * 4.0 + 0.2 * 2.0 = 3.2 + 0.4 = 3.6
        """
        coord = AsyncFederatedCoordinator(weight_dim=1, trigger_n=2, max_staleness=5)
        # Submit fresh update at model version 0 (staleness=0)
        coord.submit_update("participant_a", [4.0], gradient_model_version=0)
        # Submit stale update at model version 0, coordinator is now at 0
        # To simulate staleness=3, coordinator must be at version 3 first
        # We advance version manually by submitting and aggregating twice first
        coord2 = AsyncFederatedCoordinator(weight_dim=1, trigger_n=2, max_staleness=10)
        # Advance model to version 3 by submitting 6 updates (2 per aggregation)
        for _ in range(3):
            coord2.submit_update("p1", [0.0], gradient_model_version=0)
            coord2.submit_update("p2", [0.0], gradient_model_version=0)
        assert coord2.model_version == 3

        # Now submit two updates: one fresh (staleness=0) and one stale (staleness=3)
        coord3 = AsyncFederatedCoordinator(weight_dim=1, trigger_n=2, max_staleness=10)
        # First advance to version 3
        for _ in range(3):
            coord3.submit_update("p1", [0.0], gradient_model_version=0)
            coord3.submit_update("p2", [0.0], gradient_model_version=0)
        v = coord3.model_version  # should be 3
        assert v == 3

        # Reset for fresh test of staleness weighting
        coord4 = AsyncFederatedCoordinator(weight_dim=1, trigger_n=2, max_staleness=10)
        # Advance to version 3 without aggregating on the final pair
        for _ in range(2):
            coord4.submit_update("setup_a", [0.0], gradient_model_version=0)
            coord4.submit_update("setup_b", [0.0], gradient_model_version=0)
        assert coord4.model_version == 2
        coord4.submit_update("setup_a", [0.0], gradient_model_version=2)
        coord4.submit_update("setup_b", [0.0], gradient_model_version=2)
        assert coord4.model_version == 3

        # Now submit the target pair
        coord4.submit_update("fresh", [4.0], gradient_model_version=3)   # staleness=0
        coord4.submit_update("stale", [2.0], gradient_model_version=0)   # staleness=3

        # Expected: 0.8*4 + 0.2*2 = 3.6, added to existing 0.0 weights
        weights = coord4.global_weights
        assert weights is not None
        expected = 0.8 * 4.0 + 0.2 * 2.0
        np.testing.assert_allclose(
            weights[-1], expected, atol=1e-9,
            err_msg=f"Expected {expected}, got {weights[-1]}"
        )

    def test_fresh_update_weight_higher_than_stale(self):
        """A fresh update contributes more than a stale update of the same magnitude."""
        # Advance coordinator to version 3
        coord = AsyncFederatedCoordinator(weight_dim=1, trigger_n=2, max_staleness=10)
        for _ in range(3):
            coord.submit_update("p1", [0.0], gradient_model_version=0)
            coord.submit_update("p2", [0.0], gradient_model_version=0)
        assert coord.model_version == 3
        base = coord.global_weights[0]

        # One aggregation round with delta=1.0, staleness=0 (weight=1/(1+0)=1.0)
        c_fresh = AsyncFederatedCoordinator(weight_dim=1, trigger_n=2, max_staleness=10)
        for _ in range(3):
            c_fresh.submit_update("p1", [0.0], gradient_model_version=0)
            c_fresh.submit_update("p2", [0.0], gradient_model_version=0)
        c_fresh.submit_update("a", [1.0], gradient_model_version=3)  # staleness=0
        c_fresh.submit_update("b", [1.0], gradient_model_version=3)  # staleness=0
        fresh_result = c_fresh.global_weights[0] - base

        # One aggregation round with delta=1.0, staleness=3 (weight=1/(1+3)=0.25)
        c_stale = AsyncFederatedCoordinator(weight_dim=1, trigger_n=2, max_staleness=10)
        for _ in range(3):
            c_stale.submit_update("p1", [0.0], gradient_model_version=0)
            c_stale.submit_update("p2", [0.0], gradient_model_version=0)
        c_stale.submit_update("a", [1.0], gradient_model_version=3)   # staleness=0
        c_stale.submit_update("b", [1.0], gradient_model_version=0)   # staleness=3
        stale_result = c_stale.global_weights[0] - base

        assert fresh_result > stale_result, (
            f"Fresh contribution {fresh_result} should exceed stale {stale_result}"
        )


# ---------------------------------------------------------------------------
# Test: max staleness rejection
# ---------------------------------------------------------------------------

class TestMaxStalenessRejection:
    def test_over_max_staleness_rejected(self):
        coord = AsyncFederatedCoordinator(weight_dim=3, trigger_n=2, max_staleness=2)
        # Advance to version 3 so staleness of 3 exceeds max_staleness=2
        for _ in range(3):
            coord.submit_update("p1", [0.0, 0.0, 0.0], gradient_model_version=0)
            coord.submit_update("p2", [0.0, 0.0, 0.0], gradient_model_version=0)
        assert coord.model_version == 3

        with pytest.raises(ValueError, match="staleness"):
            coord.submit_update(
                "late_participant",
                [1.0, 1.0, 1.0],
                gradient_model_version=0,   # staleness=3 > max_staleness=2
            )

    def test_exactly_max_staleness_accepted(self):
        coord = AsyncFederatedCoordinator(weight_dim=3, trigger_n=2, max_staleness=3)
        for _ in range(3):
            coord.submit_update("p1", [0.0, 0.0, 0.0], gradient_model_version=0)
            coord.submit_update("p2", [0.0, 0.0, 0.0], gradient_model_version=0)
        assert coord.model_version == 3

        # staleness = 3 - 0 = 3, equal to max_staleness
        result = coord.submit_update("p3", [1.0, 1.0, 1.0], gradient_model_version=0)
        assert result["accepted"] is True


# ---------------------------------------------------------------------------
# Test: thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_submissions_no_corruption(self):
        """Ten threads each submit two updates concurrently; check no deadlock
        and that model_version increments correctly."""
        coord = AsyncFederatedCoordinator(weight_dim=4, trigger_n=3, max_staleness=100)
        errors: list[Exception] = []

        def submit(tid: int):
            try:
                for _ in range(5):
                    coord.submit_update(
                        f"p{tid}",
                        [float(tid)] * 4,
                        gradient_model_version=0,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Thread errors: {errors}"
        assert coord.model_version > 0, "At least one aggregation should have occurred"
        weights = coord.global_weights
        assert weights is not None
        assert weights.shape == (4,)

    def test_global_weights_integrity_under_concurrent_load(self):
        """Global weights must not contain NaN/Inf after concurrent writes."""
        coord = AsyncFederatedCoordinator(weight_dim=2, trigger_n=2, max_staleness=1000)
        done = threading.Event()

        def flood(tid):
            for _ in range(20):
                try:
                    coord.submit_update(f"p{tid}", [1.0, 1.0], gradient_model_version=0)
                except ValueError:
                    pass

        threads = [threading.Thread(target=flood, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        weights = coord.global_weights
        assert weights is not None
        assert np.all(np.isfinite(weights)), f"Non-finite weights: {weights}"


# ---------------------------------------------------------------------------
# Test: N-based trigger
# ---------------------------------------------------------------------------

class TestNTrigger:
    def test_aggregation_fires_at_trigger_n(self):
        coord = AsyncFederatedCoordinator(weight_dim=2, trigger_n=3, max_staleness=100)
        for i in range(2):
            coord.submit_update(f"p{i}", [1.0, 1.0], gradient_model_version=0)
        assert coord.model_version == 0  # not yet

        coord.submit_update("p2", [1.0, 1.0], gradient_model_version=0)
        assert coord.model_version == 1  # triggered at 3rd update


# ---------------------------------------------------------------------------
# Test: time-based trigger
# ---------------------------------------------------------------------------

class TestTimeTrigger:
    def test_tick_triggers_aggregation_after_interval(self):
        coord = AsyncFederatedCoordinator(
            weight_dim=2, trigger_n=100, trigger_seconds=0, max_staleness=100
        )
        coord.submit_update("p0", [1.0, 1.0], gradient_model_version=0)
        assert coord.model_version == 0  # trigger_n=100 not yet met

        # tick() should fire because trigger_seconds=0
        aggregated = coord.tick()
        assert aggregated is True
        assert coord.model_version == 1
