"""Drift-aware reservoir sampling for maintaining a fixed-size training buffer.

This module implements a reservoir sampler that adapts its replacement strategy
based on drift detection signals. During stable periods, it uses standard
random replacement (Algorithm R). When drift is detected, it biases toward
recency to rapidly incorporate new patterns.

The reservoir is persisted as a Parquet file and supports atomic writes
to prevent corruption on crash.
"""

from __future__ import annotations

import os
import stat
from typing import Any

import numpy as np
import pandas as pd

from monitoring.cusum_detector import CUSUMDetector
from utils.logging import get_logger

logger = get_logger(__name__)

RESERVOIR_SIZE = int(os.getenv("RESERVOIR_SIZE", "10000"))
FLUSH_INTERVAL = int(os.getenv("RESERVOIR_FLUSH_INTERVAL", "1000"))


def _atomic_write_parquet(path: str, df: pd.DataFrame) -> None:
    """Write DataFrame to Parquet atomically (write temp, rename).

    Creates parent directories if needed. Sets file permissions to 0o600.
    """
    dir_path = os.path.dirname(os.path.abspath(path))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    tmp_path = os.path.join(dir_path, f".{os.path.basename(path)}.tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class DriftAwareReservoirSampler:
    """Fixed-size reservoir sampler with drift-biased replacement.

    Attributes:
        reservoir_size: Maximum number of examples to retain.
        flush_interval: Write to disk every N updates.
        drift_detector: CUSUMDetector instance to monitor for drift.
        buffer_path: Path to the Parquet file for persistence.

    The sampler operates in two modes:
        - Stable mode: Standard reservoir sampling (random replacement)
        - Drift mode: Recency-biased replacement (newer examples more likely to replace)
    """

    def __init__(
        self,
        reservoir_size: int = RESERVOIR_SIZE,
        flush_interval: int = FLUSH_INTERVAL,
        drift_detector: CUSUMDetector | None = None,
        buffer_path: str = "data/reservoir.parquet",
    ) -> None:
        if reservoir_size <= 0:
            raise ValueError("reservoir_size must be positive")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be positive")

        self.reservoir_size = reservoir_size
        self.flush_interval = flush_interval
        self.buffer_path = buffer_path
        self._drift_detector = drift_detector or CUSUMDetector()
        self._buffer: list[dict[str, Any]] = []
        self._timestamps: list[float] = []
        self._total_seen = 0
        self._updates_since_flush = 0

        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Load existing reservoir from disk if available."""
        if os.path.exists(self.buffer_path):
            try:
                df = pd.read_parquet(self.buffer_path)
                self._buffer = df.to_dict(orient="records")
                if "timestamp" in df.columns:
                    self._timestamps = df["timestamp"].tolist()
                else:
                    self._timestamps = [0.0] * len(self._buffer)
                self._total_seen = len(self._buffer)
                logger.info(
                    "Loaded reservoir with %d examples from %s",
                    len(self._buffer),
                    self.buffer_path,
                )
            except Exception as exc:
                logger.warning("Failed to load reservoir from %s: %s", self.buffer_path, exc)

    def _is_drift_mode(self) -> bool:
        """Check if CUSUM detector is in alarm state (drift detected)."""
        return self._drift_detector.is_alarm

    def _standard_replace(self, example: dict[str, Any], timestamp: float) -> None:
        """Standard reservoir sampling replacement (Algorithm R)."""
        idx = np.random.randint(0, self._total_seen)
        self._buffer[idx] = example
        self._timestamps[idx] = timestamp

    def _recency_biased_replace(self, example: dict[str, Any], timestamp: float) -> None:
        """Recency-biased replacement: newer examples replace older ones with higher probability."""
        assert len(self._buffer) == self.reservoir_size
        assert len(self._timestamps) == self.reservoir_size

        ages = np.array(self._timestamps)
        max_age = ages.max() if ages.max() > 0 else 1.0
        ages_norm = ages / max_age

        weights = 1.0 + ages_norm
        weights = weights / weights.sum()

        idx = np.random.choice(self.reservoir_size, p=weights)
        self._buffer[idx] = example
        self._timestamps[idx] = timestamp

    def update(self, example: dict[str, Any], timestamp: float | None = None) -> None:
        """Add an example to the reservoir.

        Args:
            example: The data example to add (dictionary of feature values).
            timestamp: Optional timestamp for recency weighting. Defaults to current time.
        """
        if timestamp is None:
            timestamp = np.datetime64("now").astype(np.int64) / 1e9

        self._total_seen += 1

        if len(self._buffer) < self.reservoir_size:
            self._buffer.append(example)
            self._timestamps.append(timestamp)
        else:
            if self._is_drift_mode():
                self._recency_biased_replace(example, timestamp)
            else:
                self._standard_replace(example, timestamp)

        self._updates_since_flush += 1
        if self._updates_since_flush >= self.flush_interval:
            self.flush()

    def sample(self, n: int) -> pd.DataFrame:
        """Return n examples from the reservoir without removal.

        Args:
            n: Number of examples to sample.

        Returns:
            DataFrame with n randomly sampled examples.

        Raises:
            ValueError: If n exceeds reservoir size.
        """
        if n > len(self._buffer):
            raise ValueError(
                f"Cannot sample {n} examples from reservoir of size {len(self._buffer)}"
            )

        indices = np.random.choice(len(self._buffer), size=n, replace=False)
        return pd.DataFrame([self._buffer[i] for i in indices])

    def flush(self) -> None:
        """Persist the reservoir to disk as Parquet (atomic write)."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        df["timestamp"] = self._timestamps

        _atomic_write_parquet(self.buffer_path, df)
        self._updates_since_flush = 0
        logger.debug(
            "Flushed reservoir (%d examples) to %s", len(self._buffer), self.buffer_path
        )

    def reset(self) -> None:
        """Clear the reservoir buffer (for manual reset)."""
        self._buffer.clear()
        self._timestamps.clear()
        self._total_seen = 0
        self._updates_since_flush = 0

        if os.path.exists(self.buffer_path):
            os.remove(self.buffer_path)
            logger.info("Reset reservoir and removed %s", self.buffer_path)

    @property
    def size(self) -> int:
        """Current number of examples in the reservoir."""
        return len(self._buffer)

    def ack_drift(self) -> None:
        """Acknowledge drift signal and exit drift mode."""
        self._drift_detector.acknowledge()