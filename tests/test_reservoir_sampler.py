"""Tests for data/reservoir_sampler.py.

Validates reservoir size invariant, drift-biased replacement, sample(n) behavior,
and atomic write to disk.
"""

import os

import numpy as np
import pandas as pd
import pytest

from data.reservoir_sampler import DriftAwareReservoirSampler, RESERVOIR_SIZE


class MockCUSUMDetector:
    """Mock CUSUM detector for testing drift mode behavior."""

    def __init__(self, alarm_state: bool = False):
        self._alarm = alarm_state

    @property
    def is_alarm(self) -> bool:
        return self._alarm

    def acknowledge(self) -> None:
        self._alarm = False

    def update(self, value: float) -> bool:
        """Trigger alarm if value > 100 (for testing)."""
        if value > 100:
            self._alarm = True
            return True
        return False


# ---------------------------------------------------------------------------
# Test 1: Stable mode - reservoir size remains exactly RESERVOIR_SIZE
# ---------------------------------------------------------------------------


def test_stable_mode_reservoir_size_invariant(tmp_path):
    """After 100,000 updates in stable mode, reservoir size remains exactly RESERVOIR_SIZE."""
    sampler = DriftAwareReservoirSampler(
        reservoir_size=10000,
        flush_interval=100000,  # Don't auto-flush during test
        drift_detector=MockCUSUMDetector(alarm_state=False),
        buffer_path=str(tmp_path / "reservoir.parquet"),
    )

    # Feed 100,000 examples in stable mode
    for i in range(100000):
        sampler.update({"value": float(i)}, timestamp=float(i))

    assert sampler.size == 10000, f"Expected size 10000, got {sampler.size}"
    assert sampler.size == sampler.reservoir_size


# ---------------------------------------------------------------------------
# Test 2: Drift mode - recent examples appear at higher rate
# ---------------------------------------------------------------------------


def test_drift_mode_recency_bias(tmp_path):
    """In drift mode, recent examples should appear more frequently than old ones."""
    reservoir_size = 1000
    sampler = DriftAwareReservoirSampler(
        reservoir_size=reservoir_size,
        flush_interval=100000,
        drift_detector=MockCUSUMDetector(alarm_state=True),  # Start in drift mode
        buffer_path=str(tmp_path / "reservoir.parquet"),
    )

    # Fill reservoir first
    for i in range(reservoir_size):
        sampler.update({"value": float(i)}, timestamp=float(i))

    # Now do many more updates in drift mode
    n_updates = 50000
    for i in range(reservoir_size, reservoir_size + n_updates):
        sampler.update({"value": float(i)}, timestamp=float(i))

    # Count how many "recent" (high-value) examples are in buffer
    recent_threshold = reservoir_size - 100
    buffer_values = set()
    for ex in sampler._buffer:
        buffer_values.add(int(ex["value"]))

    recent_count = sum(1 for v in buffer_values if v >= recent_threshold)

    # With recency bias, recent examples should dominate
    # At least 50% of reservoir should be recent (very conservative threshold)
    assert recent_count > reservoir_size * 0.5, (
        f"Expected >50% recent examples in drift mode, got {recent_count}/{reservoir_size}"
    )


def test_drift_mode_vs_stable_comparison(tmp_path):
    """Compare drift mode vs stable mode: drift mode should have more recent examples."""
    reservoir_size = 1000
    n_updates = 20000

    # Stable mode sampler
    stable_sampler = DriftAwareReservoirSampler(
        reservoir_size=reservoir_size,
        flush_interval=100000,
        drift_detector=MockCUSUMDetector(alarm_state=False),
        buffer_path=str(tmp_path / "stable.parquet"),
    )

    # Drift mode sampler
    drift_sampler = DriftAwareReservoirSampler(
        reservoir_size=reservoir_size,
        flush_interval=100000,
        drift_detector=MockCUSUMDetector(alarm_state=True),
        buffer_path=str(tmp_path / "drift.parquet"),
    )

    # Fill both reservoirs
    for s in [stable_sampler, drift_sampler]:
        for i in range(reservoir_size):
            s.update({"value": float(i)}, timestamp=float(i))

    # Apply same updates to both
    for i in range(reservoir_size, reservoir_size + n_updates):
        stable_sampler.update({"value": float(i)}, timestamp=float(i))
        drift_sampler.update({"value": float(i)}, timestamp=float(i))

    # Count recent examples in each
    recent_threshold = reservoir_size - 100

    def count_recent(buffer):
        values = set()
        for ex in buffer:
            values.add(int(ex["value"]))
        return sum(1 for v in values if v >= recent_threshold)

    stable_recent = count_recent(stable_sampler._buffer)
    drift_recent = count_recent(drift_sampler._buffer)

    # Drift mode should have more recent examples than stable mode
    assert drift_recent > stable_recent, (
        f"Drift mode ({drift_recent} recent) should have more recent examples "
        f"than stable mode ({stable_recent} recent)"
    )


# ---------------------------------------------------------------------------
# Test 3: sample(n) raises ValueError when n > reservoir size
# ---------------------------------------------------------------------------


def test_sample_raises_when_n_exceeds_size(tmp_path):
    """sample(n) raises ValueError when n > reservoir size."""
    sampler = DriftAwareReservoirSampler(
        reservoir_size=100,
        flush_interval=100000,
        drift_detector=MockCUSUMDetector(alarm_state=False),
        buffer_path=str(tmp_path / "reservoir.parquet"),
    )

    # Fill partially
    for i in range(50):
        sampler.update({"value": float(i)}, timestamp=float(i))

    # Should raise when n > 50
    with pytest.raises(ValueError, match="Cannot sample 100 examples"):
        sampler.sample(100)

    # Should also raise when requesting exactly reservoir size but buffer is smaller
    with pytest.raises(ValueError, match="Cannot sample"):
        sampler.sample(100)


def test_sample_returns_dataframe(tmp_path):
    """sample(n) returns a DataFrame with correct shape."""
    sampler = DriftAwareReservoirSampler(
        reservoir_size=100,
        flush_interval=100000,
        drift_detector=MockCUSUMDetector(alarm_state=False),
        buffer_path=str(tmp_path / "reservoir.parquet"),
    )

    # Fill reservoir
    for i in range(100):
        sampler.update({"value": float(i), "label": i % 2}, timestamp=float(i))

    # Sample
    df = sampler.sample(20)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert "value" in df.columns
    assert "label" in df.columns


# ---------------------------------------------------------------------------
# Test 4: Atomic write to disk
# ---------------------------------------------------------------------------


def test_atomic_write_on_flush(tmp_path):
    """Flush uses atomic write (temp file then rename)."""
    buffer_path = str(tmp_path / "reservoir.parquet")
    sampler = DriftAwareReservoirSampler(
        reservoir_size=100,
        flush_interval=10,  # Flush every 10 updates
        drift_detector=MockCUSUMDetector(alarm_state=False),
        buffer_path=buffer_path,
    )

    # Fill and trigger flush
    for i in range(15):
        sampler.update({"value": float(i)}, timestamp=float(i))

    # File should exist and be readable
    assert os.path.exists(buffer_path)

    df = pd.read_parquet(buffer_path)
    assert len(df) == 100
    assert "timestamp" in df.columns


def test_flush_empty_buffer_safe(tmp_path):
    """Flush on empty buffer should not error."""
    buffer_path = str(tmp_path / "empty.parquet")
    sampler = DriftAwareReservoirSampler(
        reservoir_size=100,
        flush_interval=100,
        drift_detector=MockCUSUMDetector(alarm_state=False),
        buffer_path=buffer_path,
    )

    sampler.flush()  # Should not raise
    assert not os.path.exists(buffer_path)


# ---------------------------------------------------------------------------
# Test 5: Manual reset
# ---------------------------------------------------------------------------


def test_manual_reset_clears_buffer_and_file(tmp_path):
    """Manual reset should clear the in-memory buffer and remove the parquet file."""
    buffer_path = str(tmp_path / "to_reset.parquet")
    sampler = DriftAwareReservoirSampler(
        reservoir_size=100,
        flush_interval=100000,
        drift_detector=MockCUSUMDetector(alarm_state=False),
        buffer_path=buffer_path,
    )

    # Fill reservoir
    for i in range(100):
        sampler.update({"value": float(i)}, timestamp=float(i))

    assert sampler.size == 100

    sampler.flush()
    assert os.path.exists(buffer_path)

    sampler.reset()
    assert sampler.size == 0
    assert not os.path.exists(buffer_path)


# ---------------------------------------------------------------------------
# Test 6: Drift mode entry/exit via CUSUM
# ---------------------------------------------------------------------------


def test_drift_mode_entry_via_cusum(tmp_path):
    """Sampler should enter drift mode when CUSUM alarm triggers."""
    detector = MockCUSUMDetector(alarm_state=False)
    sampler = DriftAwareReservoirSampler(
        reservoir_size=100,
        flush_interval=100000,
        drift_detector=detector,
        buffer_path=str(tmp_path / "reservoir.parquet"),
    )

    # Fill reservoir
    for i in range(100):
        sampler.update({"value": float(i)}, timestamp=float(i))

    # Trigger drift via detector
    detector.update(150.0)  # Will trigger alarm since value > 100

    # Apply updates - should use recency bias now
    for i in range(100, 200):
        sampler.update({"value": float(i)}, timestamp=float(i))

    # Verify drift mode is active
    assert sampler._is_drift_mode()


def test_drift_mode_exit_via_acknowledge(tmp_path):
    """Sampler should exit drift mode after acknowledge()."""
    detector = MockCUSUMDetector(alarm_state=True)
    sampler = DriftAwareReservoirSampler(
        reservoir_size=100,
        flush_interval=100000,
        drift_detector=detector,
        buffer_path=str(tmp_path / "reservoir.parquet"),
    )

    assert sampler._is_drift_mode()

    sampler.ack_drift()
    assert not sampler._is_drift_mode()


# ---------------------------------------------------------------------------
# Test 7: Invalid initialization parameters
# ---------------------------------------------------------------------------


def test_invalid_reservoir_size():
    """Negative or zero reservoir_size raises ValueError."""
    with pytest.raises(ValueError, match="reservoir_size must be positive"):
        DriftAwareReservoirSampler(reservoir_size=0)

    with pytest.raises(ValueError, match="reservoir_size must be positive"):
        DriftAwareReservoirSampler(reservoir_size=-100)


def test_invalid_flush_interval():
    """Negative or zero flush_interval raises ValueError."""
    with pytest.raises(ValueError, match="flush_interval must be positive"):
        DriftAwareReservoirSampler(flush_interval=0)

    with pytest.raises(ValueError, match="flush_interval must be positive"):
        DriftAwareReservoirSampler(flush_interval=-1)