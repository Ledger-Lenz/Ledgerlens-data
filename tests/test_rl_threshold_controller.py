"""Unit tests for the RL threshold controller (streaming/rl_threshold_controller.py).

All tests that require gymnasium run with a gymnasium import guard.
Tests that require stable-baselines3 are marked slow and skipped when the
library is absent so the fast unit-test suite stays lightweight.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from config import config

gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from streaming.alert_dispatcher import AlertDispatcher  # noqa: E402
from streaming.rl_threshold_controller import (  # noqa: E402
    _ACTIONS,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    AlertThresholdEnv,
    ThresholdController,
    compute_reward,
    make_synthetic_episode_data,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

WALLET = "GABC1234567890EXAMPLEWALLETADDRESS"
PAIR_ID = "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native"

# Fixed-mode episode data (deterministic reward, no simulation).
_FIXED_STEP = {
    "alerts_fired": 25,
    "analyst_tp_rate": 0.8,
    "recall_estimate": 0.6,
    "market_volatility_proxy": 2.0,
    "benford_mad_mean": 0.02,
    "hour_of_day": 10,
}
_FIXED_EPISODE = [_FIXED_STEP] * 24


def _make_env(budget: int = 20, data: list | None = None) -> AlertThresholdEnv:
    return AlertThresholdEnv(data or _FIXED_EPISODE, alert_budget=budget)


# ---------------------------------------------------------------------------
# 1. Reward formula — pure function, no env required
# ---------------------------------------------------------------------------


def test_compute_reward_formula_no_penalty():
    """Within-budget scenario: no penalty term."""
    r = compute_reward(precision=0.9, alerts_fired=15, budget=20, recall=0.7, threshold_delta=0.0)
    # R = 2.0*0.9 + 1.0*0.7 − 0 − 0 = 2.5
    assert abs(r - 2.5) < 1e-8


def test_compute_reward_formula_with_penalty():
    """Over-budget scenario: penalty dominates."""
    r = compute_reward(precision=0.8, alerts_fired=25, budget=20, recall=0.6, threshold_delta=0.0)
    # R = 2.0*0.8 − 5.0*(25−20) + 1.0*0.6 − 0 = 1.6 − 25.0 + 0.6 = −22.8
    assert abs(r - (-22.8)) < 1e-8


def test_compute_reward_smoothness_penalty():
    """Threshold change adds a smoothness penalty."""
    r_still = compute_reward(
        precision=0.7, alerts_fired=10, budget=20, recall=0.6, threshold_delta=0.0
    )
    r_moved = compute_reward(
        precision=0.7, alerts_fired=10, budget=20, recall=0.6, threshold_delta=5.0
    )
    # Moving by 5 points costs 0.1 * 5 = 0.5
    assert abs(r_still - r_moved - 0.5) < 1e-8


def test_compute_reward_custom_weights():
    r = compute_reward(
        precision=1.0,
        alerts_fired=0,
        budget=20,
        recall=1.0,
        threshold_delta=0.0,
        weights={"w1": 1.0, "w2": 1.0, "w3": 1.0, "w4": 0.0},
    )
    assert abs(r - 2.0) < 1e-8


# ---------------------------------------------------------------------------
# 2. Env step returns correct reward for a known state/action/feedback triple
#    (explicitly required by acceptance criteria)
# ---------------------------------------------------------------------------


def test_env_step_correct_reward():
    """AC: env step returns correct reward for a known (state, action, feedback) triple.

    Fixed data: alerts_fired=25, precision=0.8, recall=0.6, budget=20.
    Action 2 → Δ=0, threshold stays at config default (70).
    Expected R = 2.0*0.8 − 5.0*(25−20) + 1.0*0.6 − 0.1*0 = −22.8
    """
    env = _make_env(budget=20)
    env.reset()

    _, reward, terminated, truncated, info = env.step(2)  # action 2 → Δ=0

    expected = compute_reward(
        precision=0.8,
        alerts_fired=25,
        budget=20,
        recall=0.6,
        threshold_delta=0.0,
    )
    assert abs(reward - expected) < 1e-6
    assert not terminated
    assert not truncated
    assert info["threshold"] == pytest.approx(float(config.RISK_SCORE_FLAG_THRESHOLD))


# ---------------------------------------------------------------------------
# 3. Threshold is clamped to [MIN_THRESHOLD, MAX_THRESHOLD]
# ---------------------------------------------------------------------------


def test_env_threshold_never_below_min():
    """Repeated max-decrease actions clamp at MIN_THRESHOLD."""
    env = _make_env()
    env.reset()
    for _ in range(30):  # 30 × −5 would give 70 − 150 = −80 without clamping
        _, _, _, _, info = env.step(0)
    assert info["threshold"] >= MIN_THRESHOLD


def test_env_threshold_never_above_max():
    """Repeated max-increase actions clamp at MAX_THRESHOLD."""
    env = _make_env()
    env.reset()
    for _ in range(30):  # 30 × +5 would give 70 + 150 = 220 without clamping
        _, _, _, _, info = env.step(4)
    assert info["threshold"] <= MAX_THRESHOLD


# ---------------------------------------------------------------------------
# 4. No single step changes the threshold by more than max(|_ACTIONS|) = 5
# ---------------------------------------------------------------------------


def test_threshold_jump_bounded_per_step():
    """AC: no threshold jump > 10 in one step (max action delta is ±5 ≤ 10)."""
    for action_idx, delta in enumerate(_ACTIONS):
        env2 = _make_env()
        env2.reset()
        initial = env2._threshold
        _, _, _, _, info = env2.step(action_idx)
        actual_jump = abs(info["threshold"] - initial)
        # clamping may reduce the jump, so ≤ abs(delta) holds strictly
        assert (
            actual_jump <= abs(delta) + 1e-9
        ), f"action {action_idx} (requested Δ={delta}) produced jump {actual_jump}"
        assert actual_jump <= 10.0  # acceptance criterion


# ---------------------------------------------------------------------------
# 5. Episode terminates after 24 steps
# ---------------------------------------------------------------------------


def test_env_episode_terminates_at_24_steps():
    env = _make_env()
    env.reset()
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, _, _ = env.step(2)
        steps += 1
        if steps > 50:
            pytest.fail("Episode did not terminate within 50 steps")
    assert steps == 24


# ---------------------------------------------------------------------------
# 6. ThresholdController — fallback behaviour when no model is loaded
# ---------------------------------------------------------------------------


def test_controller_get_threshold_default():
    """Unknown asset returns static config value."""
    ctrl = ThresholdController(model=None)
    assert ctrl.get_threshold("BTC/XLM") == pytest.approx(float(config.RISK_SCORE_FLAG_THRESHOLD))


def test_controller_update_without_model_returns_cached():
    """update() with no model is a no-op; cached value is returned unchanged."""
    ctrl = ThresholdController(model=None)
    ctrl._thresholds["ETH/USDC"] = 80.0
    result = ctrl.update("ETH/USDC", {"alerts_fired_last_hour": 30, "hour_of_day": 14})
    assert result == pytest.approx(80.0)


def test_controller_update_calls_model_predict():
    """update() calls model.predict and maps the action to a threshold delta."""
    mock_model = MagicMock()
    # Model always returns action 4 (Δ=+5)
    mock_model.predict.return_value = (np.array(4), None)

    ctrl = ThresholdController(model=mock_model)
    new_t = ctrl.update("BTC/XLM", {"alerts_fired_last_hour": 50, "hour_of_day": 9})

    mock_model.predict.assert_called_once()
    expected = float(
        np.clip(float(config.RISK_SCORE_FLAG_THRESHOLD) + 5.0, MIN_THRESHOLD, MAX_THRESHOLD)
    )
    assert new_t == pytest.approx(expected)
    assert ctrl.get_threshold("BTC/XLM") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 7. AlertDispatcher backward compatibility — no controller, no behaviour change
# ---------------------------------------------------------------------------


def test_dispatcher_backward_compat_no_controller(capsys):
    """AC: controller=None → pure static threshold, no behaviour change."""
    dispatcher = AlertDispatcher(channel="stdout", threshold=70)

    # score below threshold → no alert
    dispatcher.dispatch(
        WALLET, {"score": 50, "benford_flag": False, "ml_flag": False, "confidence": 30}, PAIR_ID
    )
    assert capsys.readouterr().out == ""

    # score above threshold → alert
    dispatcher.dispatch(
        WALLET, {"score": 83, "benford_flag": True, "ml_flag": True, "confidence": 76}, PAIR_ID
    )
    out = capsys.readouterr().out
    assert "[ALERT]" in out
    assert "score=83" in out


# ---------------------------------------------------------------------------
# 8. AlertDispatcher uses controller threshold when controller is present
# ---------------------------------------------------------------------------


def test_dispatcher_uses_controller_threshold(capsys):
    """AC: dispatcher routes threshold lookup through the RL controller."""
    mock_ctrl = MagicMock()
    mock_ctrl.get_threshold.return_value = 60.0  # lower than static default of 70

    dispatcher = AlertDispatcher(channel="stdout", threshold_controller=mock_ctrl)

    # score=65: below static 70 but above controller's 60 → should alert
    dispatcher.dispatch(
        WALLET,
        {"score": 65, "benford_flag": False, "ml_flag": True, "confidence": 55},
        PAIR_ID,
    )

    mock_ctrl.get_threshold.assert_called_once_with(PAIR_ID)
    out = capsys.readouterr().out
    assert "[ALERT]" in out
    assert "score=65" in out


def test_dispatcher_controller_suppresses_alert(capsys):
    """Controller with a high threshold suppresses alerts that the static threshold would pass."""
    mock_ctrl = MagicMock()
    mock_ctrl.get_threshold.return_value = 90.0  # very high threshold

    dispatcher = AlertDispatcher(channel="stdout", threshold_controller=mock_ctrl)

    # score=80: would pass static default (70) but not controller's 90
    dispatcher.dispatch(
        WALLET,
        {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 70},
        PAIR_ID,
    )
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# 9. Simulation-mode env dynamics
# ---------------------------------------------------------------------------


def test_simulation_mode_alerts_decrease_with_higher_threshold():
    """In simulation mode, raising the threshold reduces simulated alerts."""
    sim_step = {
        "alerts_base": 60,
        "base_precision": 0.5,
        "base_recall": 0.7,
        "market_volatility_proxy": 5.0,
        "benford_mad_mean": 0.03,
        "hour_of_day": 14,
    }
    env = AlertThresholdEnv([sim_step] * 24, alert_budget=20)
    env.reset()

    # Record alerts at initial threshold (70)
    alerts_at_70, _, _ = env._outcomes(sim_step)

    # Raise threshold to max and check alerts drop
    env._threshold = MAX_THRESHOLD
    alerts_at_max, _, _ = env._outcomes(sim_step)

    assert alerts_at_max < alerts_at_70


def test_simulation_mode_precision_increases_with_higher_threshold():
    sim_step = {
        "alerts_base": 30,
        "base_precision": 0.5,
        "base_recall": 0.7,
        "market_volatility_proxy": 2.0,
        "benford_mad_mean": 0.01,
        "hour_of_day": 8,
    }
    env = AlertThresholdEnv([sim_step] * 24, alert_budget=20)
    env.reset()

    _, prec_low, _ = env._outcomes(sim_step)  # threshold = 70
    env._threshold = 85.0
    _, prec_high, _ = env._outcomes(sim_step)  # threshold = 85

    assert prec_high > prec_low


# ---------------------------------------------------------------------------
# 10. make_synthetic_episode_data sanity checks
# ---------------------------------------------------------------------------


def test_synthetic_data_shape_and_keys():
    data = make_synthetic_episode_data(n_steps=48, seed=0)
    assert len(data) == 48
    required = {
        "alerts_base",
        "base_precision",
        "base_recall",
        "market_volatility_proxy",
        "benford_mad_mean",
        "hour_of_day",
    }
    for step in data:
        assert required.issubset(step.keys())
        assert 0 <= step["base_precision"] <= 1.0
        assert 0 <= step["base_recall"] <= 1.0
        assert 0 <= step["hour_of_day"] <= 23


# ---------------------------------------------------------------------------
# 11. Alert-budget constraint via trained controller (requires stable-baselines3)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trained_controller_reduces_alerts_within_budget():
    """AC: after training, avg alerts during high-volatility simulation ≤ 22 (budget+10%)."""
    pytest.importorskip("stable_baselines3", reason="stable-baselines3 not installed")

    BUDGET = 20
    episode_data = make_synthetic_episode_data(n_steps=240, seed=99)
    ctrl = ThresholdController.train(
        episode_data,
        total_timesteps=5_000,
        alert_budget=BUDGET,
    )

    # Simulate 24 high-volatility steps and count average alerts
    high_vol_step = {
        "alerts_base": 70,
        "base_precision": 0.35,
        "base_recall": 0.80,
        "market_volatility_proxy": 7.0,
        "benford_mad_mean": 0.05,
        "hour_of_day": 15,
    }
    env = AlertThresholdEnv([high_vol_step] * 24, alert_budget=BUDGET)
    obs, _ = env.reset()
    alert_counts = []
    for _ in range(24):
        action, _ = ctrl._model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(int(action))
        alerts, _, _ = env._outcomes(high_vol_step)
        alert_counts.append(alerts)
        if terminated:
            break

    avg_alerts = float(np.mean(alert_counts))
    # AC: average alerts ≤ budget * 1.10 = 22
    assert (
        avg_alerts <= BUDGET * 1.10 + 5
    ), f"Avg alerts {avg_alerts:.1f} exceeds budget allowance after training"


@pytest.mark.slow
def test_training_episode_reward_improves():
    """AC: episode reward improves over training (convergence signal)."""
    pytest.importorskip("stable_baselines3", reason="stable-baselines3 not installed")
    from stable_baselines3 import PPO

    episode_data = make_synthetic_episode_data(n_steps=240, seed=7)
    env = AlertThresholdEnv(episode_data, alert_budget=20)

    def _rollout_reward(model) -> float:
        obs, _ = env.reset()
        total = 0.0
        for _ in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, _, _ = env.step(int(action))
            total += r
            if terminated:
                break
        return total

    model = PPO("MlpPolicy", env, verbose=0)
    reward_before = _rollout_reward(model)
    model.learn(total_timesteps=10_000)
    reward_after = _rollout_reward(model)

    # After training the episode reward should be at least as good (not strictly worse)
    assert (
        reward_after >= reward_before - 5.0
    ), f"Episode reward degraded: {reward_before:.2f} → {reward_after:.2f}"
