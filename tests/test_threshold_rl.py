"""Tests for detection/threshold_rl.py (ThresholdAgent contextual bandit).

Acceptance criteria covered:
  AC1  Agent converges to within 2 points of optimal F1 threshold in ≤50 episodes.
  AC2  Threshold updated at most once per annotation batch (update() call).
  AC3  State persists across restarts (save/load round-trip).
  AC4  Unit tests: reward (F1) computation, selection distribution, convergence sim.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from detection.threshold_rl import (
    THRESHOLDS,
    ThresholdAgent,
    _encode_context,
    _get_pinned,
    compute_f1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONTEXT = {"fp_rate_7d": 0.1, "annotation_backlog": 30, "hour_of_day": 10}


@pytest.fixture()
def tmp_path_agent(tmp_path):
    return str(tmp_path / "agent.json")


# ---------------------------------------------------------------------------
# AC4a: compute_f1 reward formula
# ---------------------------------------------------------------------------


def test_compute_f1_perfect():
    assert compute_f1(tp=10, fp=0, fn=0) == pytest.approx(1.0)


def test_compute_f1_zero_tp():
    assert compute_f1(tp=0, fp=5, fn=5) == pytest.approx(0.0)


def test_compute_f1_balanced():
    # 2*5 / (2*5 + 5 + 5) = 10/20 = 0.5
    assert compute_f1(tp=5, fp=5, fn=5) == pytest.approx(0.5)


def test_compute_f1_undefined_returns_zero():
    assert compute_f1(tp=0, fp=0, fn=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AC4b: context encoding
# ---------------------------------------------------------------------------


def test_encode_context_shape():
    vec = _encode_context(CONTEXT)
    assert vec.shape == (3,)


def test_encode_context_clipping():
    vec = _encode_context({"fp_rate_7d": 5.0, "annotation_backlog": 9999, "hour_of_day": 25})
    assert vec[0] == pytest.approx(1.0)
    assert vec[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AC4c: selection distribution — all thresholds in THRESHOLDS
# ---------------------------------------------------------------------------


def test_select_threshold_returns_valid_arm(tmp_path_agent):
    agent = ThresholdAgent(state_path=tmp_path_agent)
    t = agent.select_threshold(CONTEXT)
    assert t in THRESHOLDS


def test_all_arms_explored_before_exploitation(tmp_path_agent):
    """UCB1 must try every arm at least once before repeating."""
    agent = ThresholdAgent(state_path=tmp_path_agent)
    seen = set()
    # After N_ARMS selections each arm should have been tried once
    for _ in range(len(THRESHOLDS)):
        t = agent.select_threshold(CONTEXT)
        agent.update(t, 0.5)
        seen.add(t)
    assert seen == set(THRESHOLDS)


# ---------------------------------------------------------------------------
# AC2: threshold updated at most once per batch (one update() per batch)
# ---------------------------------------------------------------------------


def test_single_update_per_batch(tmp_path_agent):
    agent = ThresholdAgent(state_path=tmp_path_agent)
    t = agent.select_threshold(CONTEXT)
    before = list(agent.q_values)
    agent.update(t, 0.8)
    after = list(agent.q_values)
    # Only the chosen arm's Q-value changed
    arm = THRESHOLDS.index(t)
    diffs = [abs(a - b) for a, b in zip(before, after)]
    for i, d in enumerate(diffs):
        if i == arm:
            assert d > 0
        else:
            assert d == pytest.approx(0.0)


def test_update_invalid_threshold_raises(tmp_path_agent):
    agent = ThresholdAgent(state_path=tmp_path_agent)
    with pytest.raises(ValueError, match="not in THRESHOLDS"):
        agent.update(99, 0.9)


# ---------------------------------------------------------------------------
# AC3: state persists across restarts
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path_agent):
    agent = ThresholdAgent(state_path=tmp_path_agent)
    for _ in range(5):
        t = agent.select_threshold(CONTEXT)
        agent.update(t, 0.7)
    agent.save()

    restored = ThresholdAgent.load(tmp_path_agent)
    assert restored.q_values == pytest.approx(agent.q_values, abs=1e-9)
    assert restored.counts == agent.counts


def test_load_missing_file_returns_fresh(tmp_path_agent):
    agent = ThresholdAgent.load(tmp_path_agent)
    assert agent.q_values == pytest.approx([0.0] * len(THRESHOLDS))
    assert agent.counts == [0] * len(THRESHOLDS)


def test_save_is_atomic(tmp_path_agent):
    """save() should use a .tmp intermediate so it's crash-safe."""
    agent = ThresholdAgent(state_path=tmp_path_agent)
    agent.update(THRESHOLDS[0], 0.5)
    agent.save()
    assert os.path.exists(tmp_path_agent)
    assert not os.path.exists(tmp_path_agent + ".tmp")


# ---------------------------------------------------------------------------
# AC1: convergence simulation — within 2 points of optimal in 50 episodes
# ---------------------------------------------------------------------------


def test_convergence_within_50_episodes(tmp_path_agent):
    """AC1: agent converges to within 2 threshold points of the optimal arm.

    Simulation: optimal threshold is 70 (arm index 4).  All other arms yield
    lower F1.  After 50 update rounds the agent should preferentially select
    threshold 70 (or an adjacent arm ≤ 2 points away).
    """
    rng = np.random.default_rng(42)
    optimal = 70
    agent = ThresholdAgent(state_path=tmp_path_agent)

    for _ in range(50):
        t = agent.select_threshold(CONTEXT)
        # Reward peaks at optimal=70; falls off with distance
        noise = float(rng.normal(0, 0.03))
        reward = max(0.0, 1.0 - abs(t - optimal) / 50.0 + noise)
        agent.update(t, reward)

    # After 50 episodes, select greedily (counts > 0 for all arms → no ∞)
    # and check we're near optimal
    greedy_arm = int(np.argmax(agent.q_values))
    best_threshold = THRESHOLDS[greedy_arm]
    assert abs(best_threshold - optimal) <= 2, (
        f"Agent converged to {best_threshold}, expected within 2 of {optimal}. "
        f"Q-values: {agent.q_values}"
    )


# ---------------------------------------------------------------------------
# Pin override: THRESHOLD_RL_PINNED bypasses agent
# ---------------------------------------------------------------------------


def test_pinned_threshold_bypasses_agent(tmp_path_agent, monkeypatch):
    """When THRESHOLD_RL_PINNED is set, select_threshold returns the pinned value."""
    from config import config

    monkeypatch.setattr(config, "THRESHOLD_RL_PINNED", 85)
    agent = ThresholdAgent(state_path=tmp_path_agent)
    t = agent.select_threshold(CONTEXT)
    assert t == 85


def test_pinned_zero_enables_agent(tmp_path_agent, monkeypatch):
    """THRESHOLD_RL_PINNED=0 means agent is active (not pinned)."""
    from config import config

    monkeypatch.setattr(config, "THRESHOLD_RL_PINNED", 0)
    agent = ThresholdAgent(state_path=tmp_path_agent)
    t = agent.select_threshold(CONTEXT)
    assert t in THRESHOLDS
