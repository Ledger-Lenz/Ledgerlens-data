"""Unit tests for AdaptiveBatchController PID logic (Issue #243)."""

import pytest

from streaming.streaming_scorer import AdaptiveBatchController


def test_initial_batch_size_within_bounds():
    ctrl = AdaptiveBatchController(
        target_p95_latency=2.0, min_batch=1, max_batch=500
    )
    assert ctrl.min_batch <= ctrl.batch_size <= ctrl.max_batch


def test_latency_spike_decreases_batch_size():
    """When observed latency exceeds target, batch size must decrease."""
    ctrl = AdaptiveBatchController(
        target_p95_latency=2.0, min_batch=1, max_batch=500, kp=1.0, ki=0.0, kd=0.0
    )
    initial = ctrl.batch_size

    # Simulate repeated latency spike (4× the target)
    for _ in range(10):
        ctrl.update(8.0)

    assert ctrl.batch_size < initial, "Batch size should decrease under sustained high latency"


def test_latency_below_target_increases_batch_size():
    """After latency returns below target, the controller should increase batch size."""
    ctrl = AdaptiveBatchController(
        target_p95_latency=2.0, min_batch=1, max_batch=500, kp=1.0, ki=0.0, kd=0.0
    )

    # First drive batch size down with a spike
    for _ in range(10):
        ctrl.update(8.0)
    low_batch = ctrl.batch_size

    # Then recover with sub-target latency
    for _ in range(20):
        ctrl.update(0.5)

    assert ctrl.batch_size > low_batch, (
        "Batch size should increase when latency is comfortably below target"
    )


def test_batch_size_never_exceeds_max():
    ctrl = AdaptiveBatchController(
        target_p95_latency=2.0, min_batch=1, max_batch=100, kp=5.0, ki=1.0, kd=0.5
    )
    for _ in range(100):
        ctrl.update(0.001)  # extremely low latency → aggressive growth
    assert ctrl.batch_size <= 100


def test_batch_size_never_below_min():
    ctrl = AdaptiveBatchController(
        target_p95_latency=2.0, min_batch=5, max_batch=500, kp=5.0, ki=1.0, kd=0.5
    )
    for _ in range(100):
        ctrl.update(999.0)  # massive latency → aggressive shrink
    assert ctrl.batch_size >= 5


def test_anti_windup_prevents_unbounded_integral():
    """Integral term must be clamped; repeated overload should not keep growing."""
    ctrl = AdaptiveBatchController(
        target_p95_latency=2.0, min_batch=1, max_batch=500, kp=0.0, ki=1.0, kd=0.0
    )
    for _ in range(1000):
        ctrl.update(100.0)  # huge sustained overload
    # Integral is clamped to _INTEGRAL_CLAMP, so batch_size settles at min
    assert ctrl.batch_size == ctrl.min_batch
    # Internal integral must not grow unboundedly
    assert abs(ctrl._integral) <= AdaptiveBatchController._INTEGRAL_CLAMP + 1e-6


def test_update_returns_int():
    ctrl = AdaptiveBatchController()
    result = ctrl.update(1.5)
    assert isinstance(result, int)
