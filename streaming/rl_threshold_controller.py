"""RL-based adaptive alert threshold controller using PPO (stable-baselines3).

ThresholdController wraps a trained PPO policy that adjusts per-asset alert
thresholds within a configurable alert budget.  AlertDispatcher uses
``controller.get_threshold(asset)`` when a controller is injected and falls
back to ``config.RISK_SCORE_FLAG_THRESHOLD`` otherwise.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO

    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SB3_AVAILABLE = False

from config import config

logger = logging.getLogger(__name__)

MIN_THRESHOLD: float = 40.0
MAX_THRESHOLD: float = 95.0
# Discrete action deltas: index 0-4 → {-5, -2, 0, +2, +5}
_ACTIONS: list[int] = [-5, -2, 0, 2, 5]
_DEFAULT_WEIGHTS: dict[str, float] = {"w1": 2.0, "w2": 5.0, "w3": 1.0, "w4": 0.1}
# Temperature for exponential alert-volume simulation (simulation mode only)
_SIM_SCALE: float = 10.0


def compute_reward(
    precision: float,
    alerts_fired: int,
    budget: int,
    recall: float,
    threshold_delta: float,
    weights: dict[str, float] | None = None,
) -> float:
    """R = w1·precision − w2·max(0, alerts−budget) + w3·recall − w4·|Δthreshold|"""
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    return (
        w["w1"] * precision
        - w["w2"] * max(0.0, float(alerts_fired) - float(budget))
        + w["w3"] * recall
        - w["w4"] * abs(threshold_delta)
    )


def make_synthetic_episode_data(n_steps: int = 240, seed: int = 42) -> list[dict[str, Any]]:
    """Return synthetic hourly step data for offline PPO training.

    Uses *simulation mode* (``alerts_base`` / ``base_precision`` / ``base_recall``)
    so the env can model the causal effect of threshold adjustments on alert volume.
    High-volatility hours (30 % of steps) have inflated alert bases and lower
    baseline precision, creating a clear incentive to raise the threshold.
    """
    rng = np.random.default_rng(seed)
    steps: list[dict[str, Any]] = []
    for i in range(n_steps):
        high_vol = rng.random() < 0.3
        steps.append(
            {
                "alerts_base": int(rng.integers(50, 90) if high_vol else rng.integers(5, 20)),
                "base_precision": float(
                    rng.uniform(0.25, 0.45) if high_vol else rng.uniform(0.60, 0.85)
                ),
                "base_recall": float(rng.uniform(0.70, 0.90)),
                "market_volatility_proxy": float(
                    rng.uniform(4.0, 8.0) if high_vol else rng.uniform(0.5, 2.0)
                ),
                "benford_mad_mean": float(
                    rng.uniform(0.02, 0.07) if high_vol else rng.uniform(0.0, 0.025)
                ),
                "hour_of_day": i % 24,
            }
        )
    return steps


class AlertThresholdEnv(gym.Env):
    """Gymnasium environment for PPO-based alert threshold optimisation.

    **Episode** = 24 hourly steps; **step** = one 1-hour bucket.

    Two episode-data formats are supported:

    *Simulation mode* (training) — episode dicts contain ``alerts_base``,
    ``base_precision``, ``base_recall``.  Alert volume and precision are
    computed as a function of the current threshold so the agent receives a
    meaningful causal reward signal.

    *Fixed mode* (unit testing) — episode dicts contain ``alerts_fired``,
    ``analyst_tp_rate``, ``recall_estimate``.  Values are taken verbatim,
    making the reward deterministic for a given (state, action, feedback)
    triple.

    **Observation space** (6 normalised floats):
        [threshold_norm, alerts_norm, precision, volatility_norm, mad_norm, hour_norm]

    **Action space**: Discrete(5) → Δthreshold ∈ {−5, −2, 0, +2, +5},
    clamped to [MIN_THRESHOLD, MAX_THRESHOLD].
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        episode_data: list[dict[str, Any]],
        alert_budget: int = 20,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required: pip install gymnasium")
        super().__init__()
        if not episode_data:
            raise ValueError("episode_data must be non-empty")

        self._episode_data = episode_data
        self._alert_budget = alert_budget
        self._weights: dict[str, float] = {**_DEFAULT_WEIGHTS, **(reward_weights or {})}

        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(_ACTIONS))

        self._step_idx: int = 0
        self._threshold: float = float(config.RISK_SCORE_FLAG_THRESHOLD)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_idx = 0
        self._threshold = float(config.RISK_SCORE_FLAG_THRESHOLD)
        return self._make_obs(self._episode_data[0]), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_data = self._episode_data[self._step_idx % len(self._episode_data)]

        delta = float(_ACTIONS[int(action)])
        old_threshold = self._threshold
        self._threshold = float(np.clip(self._threshold + delta, MIN_THRESHOLD, MAX_THRESHOLD))
        actual_delta = self._threshold - old_threshold

        alerts_fired, precision, recall = self._outcomes(step_data)
        reward = compute_reward(
            precision=precision,
            alerts_fired=alerts_fired,
            budget=self._alert_budget,
            recall=recall,
            threshold_delta=actual_delta,
            weights=self._weights,
        )

        self._step_idx += 1
        terminated = self._step_idx >= 24
        next_data = self._episode_data[self._step_idx % len(self._episode_data)]
        return (
            self._make_obs(next_data),
            reward,
            terminated,
            False,
            {"threshold": self._threshold},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _outcomes(self, step_data: dict) -> tuple[int, float, float]:
        """Return (alerts_fired, precision, recall) at the current threshold."""
        if "alerts_base" in step_data:
            # Simulation mode: threshold causally reduces alert volume
            factor = float(np.exp(-(self._threshold - 70.0) / _SIM_SCALE))
            alerts = max(0, int(float(step_data["alerts_base"]) * factor))
            precision = float(
                np.clip(
                    float(step_data["base_precision"]) + (self._threshold - 70.0) * 0.008,
                    0.0,
                    1.0,
                )
            )
            recall = float(
                np.clip(
                    float(step_data["base_recall"]) - (self._threshold - 70.0) * 0.005,
                    0.0,
                    1.0,
                )
            )
        else:
            # Fixed mode: use verbatim feedback from episode data
            alerts = int(step_data.get("alerts_fired", 0))
            precision = float(step_data.get("analyst_tp_rate", 0.5))
            recall = float(step_data.get("recall_estimate", 0.5))
        return alerts, precision, recall

    def _make_obs(self, step_data: dict) -> np.ndarray:
        alerts, precision, _ = self._outcomes(step_data)
        return np.array(
            [
                (self._threshold - MIN_THRESHOLD) / (MAX_THRESHOLD - MIN_THRESHOLD),
                min(float(alerts) / 100.0, 1.0),
                float(np.clip(precision, 0.0, 1.0)),
                min(float(step_data.get("market_volatility_proxy", 0.0)) / 10.0, 1.0),
                min(float(step_data.get("benford_mad_mean", 0.0)) / 0.1, 1.0),
                float(step_data.get("hour_of_day", 12)) / 23.0,
            ],
            dtype=np.float32,
        )


class ThresholdController:
    """Wraps a trained PPO policy to provide per-asset adaptive alert thresholds.

    When ``model`` is ``None``, ``get_threshold`` returns the static config
    value — identical to ``AlertDispatcher`` without a controller attached.

    Typical usage::

        controller = ThresholdController.train(make_synthetic_episode_data())
        dispatcher = AlertDispatcher(threshold_controller=controller)
    """

    def __init__(
        self,
        model: Any = None,
        alert_budget: int = 20,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        self._model = model
        self._alert_budget = alert_budget
        self._weights: dict[str, float] = {**_DEFAULT_WEIGHTS, **(reward_weights or {})}
        self._thresholds: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_threshold(self, asset: str) -> float:
        """Return the cached threshold for *asset*, or the static config default."""
        return self._thresholds.get(asset, float(config.RISK_SCORE_FLAG_THRESHOLD))

    def update(self, asset: str, obs_dict: dict[str, Any]) -> float:
        """Run the policy to select a new threshold for *asset* and cache it.

        ``obs_dict`` keys mirror the state-space description in the issue:
        ``alerts_fired_last_hour``, ``analyst_tp_rate_last_24h``,
        ``market_volatility_proxy``, ``benford_mad_mean_across_pairs``,
        ``hour_of_day``.  Missing keys fall back to neutral defaults.

        Returns the updated threshold.  If no model is loaded the existing
        cached value (or config default) is returned unchanged.
        """
        if self._model is None:
            return self.get_threshold(asset)

        current = self.get_threshold(asset)
        obs = self._encode_obs(current, obs_dict)
        action, _ = self._model.predict(obs, deterministic=True)
        delta = float(_ACTIONS[int(action)])
        new_threshold = float(np.clip(current + delta, MIN_THRESHOLD, MAX_THRESHOLD))
        self._thresholds[asset] = new_threshold
        logger.debug("RL threshold %s: %.1f → %.1f (Δ%+.0f)", asset, current, new_threshold, delta)
        return new_threshold

    @classmethod
    def train(
        cls,
        episode_data: list[dict[str, Any]],
        total_timesteps: int = 10_000,
        alert_budget: int = 20,
        reward_weights: dict[str, float] | None = None,
    ) -> ThresholdController:
        """Train a PPO policy on *episode_data* and return a ready controller.

        Uses a 2-layer MLP policy (stable-baselines3 default for ``"MlpPolicy"``).
        For quick convergence tests use a small ``total_timesteps``; for
        production use ≥ 100 000 steps.

        Raises ``ImportError`` if stable-baselines3 is not installed.
        """
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for training: pip install stable-baselines3"
            )
        env = AlertThresholdEnv(
            episode_data, alert_budget=alert_budget, reward_weights=reward_weights
        )
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=total_timesteps)
        return cls(model=model, alert_budget=alert_budget, reward_weights=reward_weights)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_obs(current_threshold: float, obs_dict: dict[str, Any]) -> np.ndarray:
        return np.array(
            [
                (current_threshold - MIN_THRESHOLD) / (MAX_THRESHOLD - MIN_THRESHOLD),
                min(float(obs_dict.get("alerts_fired_last_hour", 0)) / 100.0, 1.0),
                float(np.clip(obs_dict.get("analyst_tp_rate_last_24h", 0.5), 0.0, 1.0)),
                min(float(obs_dict.get("market_volatility_proxy", 0.0)) / 10.0, 1.0),
                min(float(obs_dict.get("benford_mad_mean_across_pairs", 0.0)) / 0.1, 1.0),
                float(obs_dict.get("hour_of_day", 12)) / 23.0,
            ],
            dtype=np.float32,
        )
