"""Tests for monitoring/cusum_detector.py (CUSUMDetector + CUSUMFeedbackStore).

Covers:
- Alarm triggers when statistic exceeds threshold
- Statistics reset after alarm so subsequent normal observations do not re-trigger
- acknowledge() clears alarm and statistics
- is_alarm reflects current state
- Feedback recording and recalibration (new)
- Safety bounds on recalibration (new)
- Minimum sample requirement for recalibration (new)
"""

import pytest

from monitoring.cusum_detector import (
    CUSUMDetector,
    CUSUMFeedbackStore,
    RecalibrationResult,
)


def _make_detector(**kwargs):
    defaults = dict(
        metric_name="test_metric",
        target_mean=0.0,
        allowable_slack=0.5,
        decision_threshold=5.0,
    )
    defaults.update(kwargs)
    return CUSUMDetector(**defaults)


def _make_store(**kwargs):
    defaults = dict(
        metric_name="test_metric",
        fp_rate_target=0.10,
        min_feedback_samples=10,
        learning_rate=0.1,
        h_min=1.0,
        h_max=100.0,
    )
    defaults.update(kwargs)
    return CUSUMFeedbackStore(**defaults)


# ---------------------------------------------------------------------------
# Alarm triggers correctly
# ---------------------------------------------------------------------------


def test_alarm_triggers_on_sustained_shift():
    detector = _make_detector()
    assert not detector.is_alarm

    for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
        detector.update(value)

    assert detector.is_alarm is True


# ---------------------------------------------------------------------------
# Statistics reset after alarm — no continuous re-alarm (issue #785)
# ---------------------------------------------------------------------------


def test_no_continuous_alarm_after_reset():
    detector = _make_detector()

    trigger_values = [10.0] * 10
    for value in trigger_values:
        detector.update(value)

    assert detector.is_alarm is True

    baseline_values = [0.0] * 10
    alarm_count = 0
    for value in baseline_values:
        if detector.update(value):
            alarm_count += 1

    assert alarm_count == 0, (
        f"Expected 0 alarms after reset, got {alarm_count}. "
        "CUSUM statistics were not reset correctly after the initial alarm."
    )
    assert detector.is_alarm is True


# ---------------------------------------------------------------------------
# acknowledge() clears alarm and statistics
# ---------------------------------------------------------------------------


def test_acknowledge_clears_alarm():
    detector = _make_detector()
    for value in [10.0] * 10:
        detector.update(value)
    assert detector.is_alarm is True

    detector.acknowledge()
    assert detector.is_alarm is False
    assert detector.s_high == 0.0
    assert detector.s_low == 0.0


def test_acknowledge_allows_future_alarms():
    detector = _make_detector()
    for value in [10.0] * 10:
        detector.update(value)
    detector.acknowledge()

    for value in [10.0] * 10:
        detector.update(value)

    assert detector.is_alarm is True


# ---------------------------------------------------------------------------
# is_alarm reflects state transitions
# ---------------------------------------------------------------------------


def test_is_alarm_false_before_alarm():
    detector = _make_detector()
    assert detector.is_alarm is False


def test_update_returns_false_when_already_alarming():
    detector = _make_detector()
    for value in [10.0] * 10:
        detector.update(value)
    assert detector.is_alarm is True

    result = detector.update(20.0)
    assert result is False


# ---------------------------------------------------------------------------
# CUSUMFeedbackStore — record_feedback
# ---------------------------------------------------------------------------


def test_record_feedback_stores_outcome():
    store = _make_store(min_feedback_samples=1)
    store.record_feedback("alert-001", "false_positive")
    assert store.pending_count() == 1
    assert store.fp_rate() == 1.0


def test_record_feedback_deduplicates_by_alert_id():
    store = _make_store()
    store.record_feedback("alert-001", "false_positive")
    store.record_feedback("alert-001", "true_positive")  # override
    assert store.pending_count() == 1
    assert store.fp_rate() == 0.0  # only TP remains


def test_record_feedback_rejects_invalid_outcome():
    store = _make_store()
    with pytest.raises(ValueError, match="outcome must be"):
        store.record_feedback("x", "ambiguous")  # type: ignore[arg-type]


def test_fp_rate_none_with_no_records():
    store = _make_store()
    assert store.fp_rate() is None


def test_fp_rate_mixed():
    store = _make_store()
    for j in range(3):
        store.record_feedback(f"fp{j}", "false_positive")
    for j in range(7):
        store.record_feedback(f"tp{j}", "true_positive")
    assert abs(store.fp_rate() - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# CUSUMFeedbackStore — recalibrate() gating
# ---------------------------------------------------------------------------


def test_recalibrate_returns_none_below_min_samples():
    detector = _make_detector()
    store = _make_store(min_feedback_samples=10)
    for j in range(5):
        store.record_feedback(f"a{j}", "false_positive")
    result = store.recalibrate(detector)
    assert result is None


def test_recalibrate_preserves_records_when_insufficient():
    detector = _make_detector()
    store = _make_store(min_feedback_samples=10)
    for j in range(5):
        store.record_feedback(f"a{j}", "false_positive")
    store.recalibrate(detector)
    assert store.pending_count() == 5


def test_recalibrate_clears_records_on_success():
    detector = _make_detector()
    store = _make_store(min_feedback_samples=5)
    for j in range(5):
        store.record_feedback(f"a{j}", "false_positive")
    result = store.recalibrate(detector)
    assert result is not None
    assert store.pending_count() == 0


# ---------------------------------------------------------------------------
# CUSUMFeedbackStore — threshold adjustment direction
# ---------------------------------------------------------------------------


def test_h_increases_when_fp_rate_exceeds_target():
    """More FPs than target → raise h (be less sensitive)."""
    detector = _make_detector(decision_threshold=10.0)
    store = _make_store(fp_rate_target=0.05, min_feedback_samples=10, learning_rate=0.1)
    for j in range(10):
        store.record_feedback(f"fp{j}", "false_positive")  # fp_rate = 1.0 >> 0.05
    old_h = detector.h
    store.recalibrate(detector)
    assert detector.h > old_h


def test_h_decreases_when_fp_rate_below_target():
    """Fewer FPs than target → lower h (be more sensitive)."""
    detector = _make_detector(decision_threshold=10.0)
    store = _make_store(fp_rate_target=0.5, min_feedback_samples=10, learning_rate=0.1)
    for j in range(10):
        store.record_feedback(f"tp{j}", "true_positive")  # fp_rate = 0.0 < 0.5
    old_h = detector.h
    store.recalibrate(detector)
    assert detector.h < old_h


# ---------------------------------------------------------------------------
# CUSUMFeedbackStore — safety bounds (AC-5)
# ---------------------------------------------------------------------------


def test_safety_bound_clamps_h_to_h_max():
    """h must not exceed h_max even with extreme learning."""
    detector = _make_detector(decision_threshold=99.0)
    store = _make_store(
        fp_rate_target=0.01, min_feedback_samples=5, learning_rate=1.0, h_max=100.0
    )
    for j in range(5):
        store.record_feedback(f"fp{j}", "false_positive")
    result = store.recalibrate(detector)
    assert result is not None
    assert detector.h <= 100.0
    assert result.clamped is True


def test_safety_bound_clamps_h_to_h_min():
    """h must not fall below h_min even with extreme learning."""
    detector = _make_detector(decision_threshold=2.0)
    store = _make_store(
        fp_rate_target=0.99, min_feedback_samples=5, learning_rate=1.0, h_min=1.0
    )
    for j in range(5):
        store.record_feedback(f"tp{j}", "true_positive")  # fp=0.0 << 0.99
    result = store.recalibrate(detector)
    assert result is not None
    assert detector.h >= 1.0
    assert result.clamped is True


def test_no_clamp_when_within_bounds():
    """When adjustment keeps h within bounds, clamped=False."""
    detector = _make_detector(decision_threshold=10.0)
    store = _make_store(fp_rate_target=0.10, min_feedback_samples=10, learning_rate=0.1)
    # Balanced feedback: 10% FP = target → delta ≈ 0
    for j in range(1):
        store.record_feedback(f"fp{j}", "false_positive")
    for j in range(9):
        store.record_feedback(f"tp{j}", "true_positive")
    result = store.recalibrate(detector)
    assert result is not None
    assert result.clamped is False


# ---------------------------------------------------------------------------
# CUSUMFeedbackStore — RecalibrationResult fields
# ---------------------------------------------------------------------------


def test_recalibration_result_contains_all_fields():
    detector = _make_detector(decision_threshold=10.0)
    store = _make_store(fp_rate_target=0.10, min_feedback_samples=5)
    for j in range(5):
        store.record_feedback(f"fp{j}", "false_positive")
    result = store.recalibrate(detector)
    assert isinstance(result, RecalibrationResult)
    assert result.metric_name == "test_metric"
    assert result.old_threshold == 10.0
    assert result.new_threshold == detector.h
    assert 0.0 <= result.fp_rate <= 1.0
    assert result.fp_rate_target == 0.10
    assert result.n_samples == 5
    assert isinstance(result.clamped, bool)


# ---------------------------------------------------------------------------
# CUSUMFeedbackStore — input validation
# ---------------------------------------------------------------------------


def test_fp_rate_target_must_be_in_01():
    with pytest.raises(ValueError):
        CUSUMFeedbackStore(fp_rate_target=0.0)
    with pytest.raises(ValueError):
        CUSUMFeedbackStore(fp_rate_target=1.0)


def test_min_feedback_samples_must_be_positive():
    with pytest.raises(ValueError):
        CUSUMFeedbackStore(min_feedback_samples=0)


def test_h_min_must_be_positive():
    with pytest.raises(ValueError):
        CUSUMFeedbackStore(h_min=0.0)


def test_h_max_must_exceed_h_min():
    with pytest.raises(ValueError):
        CUSUMFeedbackStore(h_min=10.0, h_max=5.0)
