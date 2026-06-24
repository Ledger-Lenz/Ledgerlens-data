"""Unit tests for StreamingBenfordSketch (Issue #55)."""

import math
import tracemalloc
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from detection.benford_engine import compute_benford_metrics
from detection.streaming_benford import StreamingBenfordSketch


def test_sketch_accuracy():
    """Verify sketch produces metrics within 5% of exact computation on 10k trades."""
    window_hours = 24
    window_seconds = window_hours * 3600
    sketch = StreamingBenfordSketch(window_seconds)

    # Generate 10k synthetic trades following Benford's Law
    rng = np.random.default_rng(42)
    # P(d) = log10(1 + 1/d)
    digits = rng.choice(range(1, 10), size=10000, p=[math.log10(1 + 1 / d) for d in range(1, 10)])
    # Convert digits to actual amounts
    amounts = [float(d * (10 ** rng.uniform(0, 3))) for d in digits]

    start_time = datetime(2024, 1, 1, 12, 0, 0)

    # Ingest all trades at the same time (no decay for now to compare with exact)
    for amount in amounts:
        sketch.update(amount, start_time)

    sketch_metrics = sketch.to_metrics()
    exact_metrics = compute_benford_metrics(pd.Series(amounts))

    assert sketch_metrics.sample_size == 10000
    assert exact_metrics.sample_size == 10000

    # MAD should be very close
    assert abs(sketch_metrics.mad - exact_metrics.mad) < 0.001
    # Chi-square should be very close
    assert (
        abs(sketch_metrics.chi_square - exact_metrics.chi_square)
        / (exact_metrics.chi_square + 1e-9)
        < 0.05
    )
    # Z-scores
    for d in range(1, 10):
        assert abs(sketch_metrics.z_scores[d] - exact_metrics.z_scores[d]) < 0.1


def test_memory_boundedness():
    """Verify O(1) memory footprint regardless of trade volume."""
    window_seconds = 3600
    sketch = StreamingBenfordSketch(window_seconds)
    now = datetime.now()

    tracemalloc.start()
    # Baseline
    for i in range(100):
        sketch.update(float(i + 1), now)
    _, peak_100 = tracemalloc.get_traced_memory()

    # Large volume
    for i in range(10000):
        sketch.update(float(i + 1), now)
    _, peak_10000 = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak memory should not have grown significantly
    # (Allowing some buffer for Python's internal management, but definitely not O(N))
    assert peak_10000 < peak_100 * 1.5


def test_decay_correspondence():
    """Verify EWMA decay correctly reflects the sliding window."""
    window_seconds = 3600  # 1 hour
    sketch = StreamingBenfordSketch(window_seconds)

    t0 = datetime(2024, 1, 1, 12, 0, 0)
    sketch.update(100.0, t0)
    assert sketch.n == 1.0

    # Advance time by exactly one window length
    t1 = t0 + timedelta(seconds=window_seconds)
    # We need to trigger decay by calling update or _apply_decay (which is internal)
    # We'll use a dummy update
    sketch.update(0.0, t1)  # update with 0 doesn't add count but applies decay

    # After one window length, count should be 1/e ≈ 0.3678
    assert sketch.n == pytest.approx(math.exp(-1), rel=1e-5)

    # Advance another window length
    t2 = t1 + timedelta(seconds=window_seconds)
    sketch.update(0.0, t2)
    assert sketch.n == pytest.approx(math.exp(-2), rel=1e-5)


def test_empty_sketch():
    """Metrics for an empty sketch should be 0.0."""
    sketch = StreamingBenfordSketch(3600)
    metrics = sketch.to_metrics()
    assert metrics.chi_square == 0.0
    assert metrics.mad == 0.0
    assert metrics.sample_size == 0
    assert all(z == 0.0 for z in metrics.z_scores.values())
