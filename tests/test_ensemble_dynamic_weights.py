"""Unit tests for EnsembleDynamicWeightController (issue #268).

Covers:
- High-FP model weight decreases vs low-FP models
- Weight bounds [0.05, 0.80] respected even at 100% FP rate
- Systemic reset fires when all models exceed FP threshold
- Minimum feedback count gating
- Authentication requirement (annotator_id + audit_trail_id)
"""

from __future__ import annotations

import pytest

from detection.ensemble_calibrator import (
    EnsembleDynamicWeightController,
    _DYNAMIC_WEIGHT_MAX,
    _DYNAMIC_WEIGHT_MIN,
    _MIN_FP_FEEDBACK,
)


def _make_controller(training_weights=None, **kwargs):
    if training_weights is None:
        training_weights = {"rf": 1 / 3, "xgb": 1 / 3, "lgbm": 1 / 3}
    return EnsembleDynamicWeightController(training_weights, **kwargs)


def _send_fp(controller, model_predictions, n=1, annotator="analyst1", audit="aud-001"):
    """Feed n identical FP feedback events."""
    for i in range(n):
        controller.observe_false_positive(
            wallet=f"GTEST{i:04d}",
            model_predictions=model_predictions,
            annotator_id=annotator,
            audit_trail_id=audit,
        )


# ---------------------------------------------------------------------------
# Test: high-FP model weight decreases
# ---------------------------------------------------------------------------

class TestWeightDecreasesForHighFPModel:
    """One model at 50% FP rate; two models at 5% FP rate.
    After enough feedback the high-FP model's weight must decrease.
    """

    def test_high_fp_weight_decreases(self):
        ctrl = _make_controller()
        initial_weights = ctrl.current_weights()
        initial_rf = initial_weights["rf"]

        # rf has 50% FP rate: it predicts >= 0.5 every other time
        # xgb and lgbm have 5% FP rate: they predict >= 0.5 only 1 in 20 times
        n = 20  # enough to exceed _MIN_FP_FEEDBACK
        for i in range(n):
            rf_pred = 0.9 if i % 2 == 0 else 0.1   # 50% FP rate
            xgb_pred = 0.9 if i == 0 else 0.1       # 5% FP rate
            lgbm_pred = 0.9 if i == 1 else 0.1      # 5% FP rate
            ctrl.observe_false_positive(
                wallet=f"GTEST{i:04d}",
                model_predictions={"rf": rf_pred, "xgb": xgb_pred, "lgbm": lgbm_pred},
                annotator_id="analyst1",
                audit_trail_id="aud-001",
            )

        updated = ctrl.current_weights()
        # The high-FP model (rf) should weigh less after adjustment
        assert updated["rf"] < initial_rf, (
            f"rf weight {updated['rf']:.4f} should be less than initial {initial_rf:.4f}"
        )
        # Low-FP models (xgb, lgbm) should weigh more
        assert updated["xgb"] > initial_weights["xgb"] or updated["lgbm"] > initial_weights["lgbm"]

    def test_weights_sum_to_one(self):
        ctrl = _make_controller()
        _send_fp(ctrl, {"rf": 0.9, "xgb": 0.1, "lgbm": 0.1}, n=15)
        total = sum(ctrl.current_weights().values())
        assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1, got {total}"


# ---------------------------------------------------------------------------
# Test: weight bounds [0.05, 0.80]
# ---------------------------------------------------------------------------

class TestWeightBounds:
    """Even when one model has 100% FP rate, weights must stay in [0.05, 0.80]."""

    def test_lower_bound_respected(self):
        ctrl = _make_controller(smoothing_alpha=1.0)  # instant full update
        # rf always FP (100%), xgb and lgbm never FP
        n = 30
        for i in range(n):
            ctrl.observe_false_positive(
                wallet=f"GTEST{i:04d}",
                model_predictions={"rf": 0.99, "xgb": 0.01, "lgbm": 0.01},
                annotator_id="analyst1",
                audit_trail_id="aud-001",
            )
        weights = ctrl.current_weights()
        for name, w in weights.items():
            assert w >= _DYNAMIC_WEIGHT_MIN - 1e-9, (
                f"Model {name} weight {w:.4f} below minimum {_DYNAMIC_WEIGHT_MIN}"
            )

    def test_upper_bound_respected(self):
        ctrl = _make_controller(smoothing_alpha=1.0)
        n = 30
        for i in range(n):
            ctrl.observe_false_positive(
                wallet=f"GTEST{i:04d}",
                model_predictions={"rf": 0.99, "xgb": 0.01, "lgbm": 0.01},
                annotator_id="analyst1",
                audit_trail_id="aud-001",
            )
        weights = ctrl.current_weights()
        for name, w in weights.items():
            assert w <= _DYNAMIC_WEIGHT_MAX + 1e-9, (
                f"Model {name} weight {w:.4f} above maximum {_DYNAMIC_WEIGHT_MAX}"
            )


# ---------------------------------------------------------------------------
# Test: systemic reset
# ---------------------------------------------------------------------------

class TestSystemicReset:
    """Systemic reset fires when ALL three models' FP rates exceed threshold."""

    def test_systemic_reset_fires(self):
        ctrl = _make_controller(systemic_fp_threshold=0.3)
        training_weights = ctrl.training_weights.copy()

        # All three models always predict >= 0.5 → 100% FP rate for each
        n = 20
        for i in range(n):
            ctrl.observe_false_positive(
                wallet=f"GTEST{i:04d}",
                model_predictions={"rf": 0.9, "xgb": 0.9, "lgbm": 0.9},
                annotator_id="analyst1",
                audit_trail_id="aud-001",
            )

        # After systemic reset, weights should equal training-time weights
        weights = ctrl.current_weights()
        for name, w in weights.items():
            assert abs(w - training_weights[name]) < 1e-9, (
                f"After systemic reset, {name} weight {w} != training weight "
                f"{training_weights[name]}"
            )

    def test_no_reset_when_not_all_high(self):
        ctrl = _make_controller(systemic_fp_threshold=0.3, smoothing_alpha=1.0)
        training_weights = ctrl.training_weights.copy()

        # Only rf has 100% FP; xgb and lgbm have 0%
        n = 20
        for i in range(n):
            ctrl.observe_false_positive(
                wallet=f"GTEST{i:04d}",
                model_predictions={"rf": 0.9, "xgb": 0.1, "lgbm": 0.1},
                annotator_id="analyst1",
                audit_trail_id="aud-001",
            )

        # Should NOT have reset — rf weight should be lower than initial
        weights = ctrl.current_weights()
        assert weights["rf"] < training_weights["rf"], (
            "rf should be penalised, not reset to training weight"
        )


# ---------------------------------------------------------------------------
# Test: minimum feedback count gating
# ---------------------------------------------------------------------------

class TestMinimumFeedbackGating:
    def test_no_adjustment_below_minimum(self):
        ctrl = _make_controller()
        initial = ctrl.current_weights()

        # Send fewer than _MIN_FP_FEEDBACK events
        for i in range(_MIN_FP_FEEDBACK - 1):
            ctrl.observe_false_positive(
                wallet=f"GTEST{i:04d}",
                model_predictions={"rf": 0.99, "xgb": 0.01, "lgbm": 0.01},
                annotator_id="analyst1",
                audit_trail_id="aud-001",
            )

        # Weights should still be at training-time values
        current = ctrl.current_weights()
        for name in initial:
            assert abs(current[name] - initial[name]) < 1e-9, (
                f"Weight for {name} changed before minimum feedback count reached"
            )


# ---------------------------------------------------------------------------
# Test: authentication enforcement
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_empty_annotator_id_rejected(self):
        ctrl = _make_controller()
        with pytest.raises(ValueError, match="annotator_id"):
            ctrl.observe_false_positive(
                wallet="GTEST0001",
                model_predictions={"rf": 0.9, "xgb": 0.9, "lgbm": 0.9},
                annotator_id="",
                audit_trail_id="aud-001",
            )

    def test_empty_audit_trail_id_rejected(self):
        ctrl = _make_controller()
        with pytest.raises(ValueError, match="audit_trail_id"):
            ctrl.observe_false_positive(
                wallet="GTEST0001",
                model_predictions={"rf": 0.9, "xgb": 0.9, "lgbm": 0.9},
                annotator_id="analyst1",
                audit_trail_id="",
            )
