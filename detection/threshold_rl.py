"""Contextual bandit for adaptive risk-score alert threshold tuning.

ThresholdAgent maintains per-arm Q-values over discrete threshold values
[50, 55, 60, ..., 95] and updates them via F1-based rewards from analyst
annotation batches.  State is persisted to ``data/threshold_agent.json`` so
the agent survives restarts.

Operator override: set ``THRESHOLD_RL_PINNED=1`` (or any integer threshold
value) in the environment / config to disable the agent and hold the
threshold fixed.

Usage::

    agent = ThresholdAgent.load()
    threshold = agent.select_threshold({
        "fp_rate_7d": 0.12,
        "annotation_backlog": 42,
        "hour_of_day": 14,
    })
    # … run scoring at threshold …
    agent.update(threshold, f1_score)
    agent.save()
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from config import config

# Discrete arm values (multiples of 5 in [50, 95])
THRESHOLDS: list[int] = list(range(50, 100, 5))
_N_ARMS: int = len(THRESHOLDS)  # 10

_DEFAULT_STATE_PATH: str = "data/threshold_agent.json"
_ALPHA: float = 0.1  # learning rate
_GAMMA: float = 0.0  # no discounting for stationary bandit
_UCB_C: float = 1.0  # UCB exploration constant


def compute_f1(tp: int, fp: int, fn: int) -> float:
    """F1 = 2·precision·recall / (precision + recall), 0 if undefined."""
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 0.0


def _encode_context(context: dict[str, Any]) -> np.ndarray:
    """Normalise context dict to a fixed-length float array [fp_rate, backlog_norm, hour_norm]."""
    fp_rate = float(np.clip(context.get("fp_rate_7d", 0.0), 0.0, 1.0))
    backlog = float(np.clip(context.get("annotation_backlog", 0) / 200.0, 0.0, 1.0))
    hour = float(context.get("hour_of_day", 12)) / 23.0
    return np.array([fp_rate, backlog, hour], dtype=np.float32)


class ThresholdAgent:
    """Contextual bandit over THRESHOLDS = [50, 55, 60, ..., 95].

    Uses UCB1 with a context-independent Q-table for simplicity and
    interpretability.  The context vector is stored for future use with a
    linear UCB upgrade but does not affect arm selection in this version.

    Args:
        q_values:  Per-arm Q-value estimates (len == len(THRESHOLDS)).
        counts:    Number of updates per arm.
        state_path: Where to persist state.
    """

    def __init__(
        self,
        q_values: list[float] | None = None,
        counts: list[int] | None = None,
        state_path: str = _DEFAULT_STATE_PATH,
    ) -> None:
        self._q: np.ndarray = (
            np.array(q_values, dtype=float) if q_values is not None else np.zeros(_N_ARMS)
        )
        self._counts: np.ndarray = (
            np.array(counts, dtype=int) if counts is not None else np.zeros(_N_ARMS, dtype=int)
        )
        self._state_path = state_path
        self._total_steps: int = int(self._counts.sum())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_threshold(self, context: dict[str, Any]) -> int:
        """Select a threshold arm using UCB1.

        If the operator has pinned the threshold via ``THRESHOLD_RL_PINNED``,
        that value is returned immediately without updating any state.

        Returns:
            An integer threshold from THRESHOLDS.
        """
        pinned = _get_pinned()
        if pinned is not None:
            return pinned

        _encode_context(context)  # validate / future use

        # UCB1: for un-tried arms use +∞ so they're tried first
        self._total_steps += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ucb_scores = np.where(
                self._counts == 0,
                np.inf,
                self._q + _UCB_C * np.sqrt(math.log(self._total_steps) / self._counts),
            )
        arm = int(np.argmax(ucb_scores))
        return THRESHOLDS[arm]

    def update(self, threshold: int, reward: float) -> None:
        """Update Q-value for the chosen arm using incremental mean.

        Args:
            threshold: The threshold value that was used (must be in THRESHOLDS).
            reward:    Scalar reward — typically F1 on the last annotation batch.
        """
        if threshold not in THRESHOLDS:
            raise ValueError(f"threshold {threshold} not in THRESHOLDS={THRESHOLDS}")
        arm = THRESHOLDS.index(threshold)
        self._counts[arm] += 1
        # Incremental mean update: Q ← Q + α·(R - Q)
        self._q[arm] += _ALPHA * (reward - self._q[arm])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist agent state to ``state_path``."""
        os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
        payload = {
            "q_values": self._q.tolist(),
            "counts": self._counts.tolist(),
            "thresholds": THRESHOLDS,
        }
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self._state_path)

    @classmethod
    def load(cls, state_path: str = _DEFAULT_STATE_PATH) -> ThresholdAgent:
        """Load agent from ``state_path``, or create a fresh one if absent."""
        if os.path.exists(state_path):
            with open(state_path) as f:
                data = json.load(f)
            return cls(
                q_values=data.get("q_values"),
                counts=data.get("counts"),
                state_path=state_path,
            )
        return cls(state_path=state_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def q_values(self) -> list[float]:
        return [float(v) for v in self._q.tolist()]

    @property
    def counts(self) -> list[int]:
        return [int(v) for v in self._counts.tolist()]


def _get_pinned() -> int | None:
    """Return the pinned threshold if THRESHOLD_RL_PINNED is set, else None."""
    raw = getattr(config, "THRESHOLD_RL_PINNED", None)
    if raw is None:
        return None
    try:
        val = int(raw)
        return val if val else None  # "0" / "" → disabled
    except (ValueError, TypeError):
        return None
