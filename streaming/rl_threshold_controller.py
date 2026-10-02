"""RL-based adaptive alert threshold controller using PPO (stable-baselines3).

ThresholdController wraps a trained PPO policy that adjusts per-asset alert
thresholds within a configurable alert budget.  AlertDispatcher uses
``controller.get_threshold(asset)`` when a controller is injected and falls
back to ``config.RISK_SCORE_FLAG_THRESHOLD`` otherwise.
"""

from __future__ import annotations

import json
import logging
import math
import os
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
# Circuit-breaker safe range defaults (Issue #898). Outside this band the RL
# policy is considered runaway and the controller reverts to the safe static
# threshold until an operator calls ``reset_circuit_breaker``.
DEFAULT_MAX_ALERTS_PER_HOUR: float = float(os.getenv("RL_MAX_ALERTS_PER_HOUR", "100"))
DEFAULT_MIN_ALERTS_PER_HOUR: float = float(os.getenv("RL_MIN_ALERTS_PER_HOUR", "0"))
DEFAULT_MIN_TP_RATE: float = float(os.getenv("RL_MIN_TP_RATE", "0.05"))
DEFAULT_OVERRIDE_PATH: str = os.getenv("RL_THRESHOLD_OVERRIDE_PATH", "data/threshold_override.json")
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
        min_threshold: float = MIN_THRESHOLD,
        max_threshold: float = MAX_THRESHOLD,
        safe_threshold: float | None = None,
        max_alerts_per_hour: float = DEFAULT_MAX_ALERTS_PER_HOUR,
        min_alerts_per_hour: float = DEFAULT_MIN_ALERTS_PER_HOUR,
        min_tp_rate: float = DEFAULT_MIN_TP_RATE,
        override_path: str | None = None,
    ) -> None:
        if min_threshold > max_threshold:
            raise ValueError("min_threshold must be <= max_threshold")
        self._model = model
        self._alert_budget = alert_budget
        self._weights: dict[str, float] = {**_DEFAULT_WEIGHTS, **(reward_weights or {})}
        self._thresholds: dict[str, float] = {}
        # Hard bounds — no policy output can move a threshold outside these.
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.safe_threshold = self._clamp(
            float(config.RISK_SCORE_FLAG_THRESHOLD) if safe_threshold is None else safe_threshold
        )
        self.max_alerts_per_hour = max_alerts_per_hour
        self.min_alerts_per_hour = min_alerts_per_hour
        self.min_tp_rate = min_tp_rate
        self.circuit_open: bool = False
        self.circuit_reason: str | None = None
        # Operator overrides: asset -> pinned threshold ("*" pins every asset).
        self._overrides: dict[str, float] = {}
        self._override_path = override_path
        self._override_mtime: float | None = None

    # ------------------------------------------------------------------
    # Safety: bounds, circuit breaker, operator override (Issue #898)
    # ------------------------------------------------------------------

    def _clamp(self, value: float) -> float:
        if not math.isfinite(value):
            return self.safe_threshold if hasattr(self, "safe_threshold") else self.min_threshold
        return float(min(max(value, self.min_threshold), self.max_threshold))

    def pin_threshold(self, value: float, asset: str = "*") -> None:
        """Manually pin *asset*'s threshold; takes precedence over the RL policy."""
        self._overrides[asset] = self._clamp(float(value))
        logger.warning("Threshold override pinned: %s=%.1f", asset, self._overrides[asset])

    def release_override(self, asset: str = "*") -> None:
        """Release a manual pin, handing control back to the RL policy."""
        self._overrides.pop(asset, None)
        logger.warning("Threshold override released: %s", asset)

    def _override_for(self, asset: str) -> float | None:
        self._sync_override_file()
        return self._overrides.get(asset, self._overrides.get("*"))

    def _sync_override_file(self) -> None:
        """Pick up pins written by ``scripts/threshold_override.py``."""
        path = self._override_path
        if not path:
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            if self._override_mtime is not None:
                self._overrides.clear()
                self._override_mtime = None
            return
        if mtime == self._override_mtime:
            return
        self._override_mtime = mtime
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._overrides = {k: self._clamp(float(v)) for k, v in data.items()}
        except (OSError, ValueError, TypeError) as exc:
            logger.error("Ignoring unreadable threshold override file %s: %s", path, exc)

    def _check_circuit(self, obs_dict: dict[str, Any]) -> None:
        alerts = float(obs_dict.get("alerts_fired_last_hour", 0.0))
        tp_rate = obs_dict.get("analyst_tp_rate_last_24h")
        reason = None
        if alerts > self.max_alerts_per_hour:
            reason = f"alert volume {alerts:.0f}/h above {self.max_alerts_per_hour:.0f}/h"
        elif alerts < self.min_alerts_per_hour:
            reason = f"alert volume {alerts:.0f}/h below {self.min_alerts_per_hour:.0f}/h"
        elif tp_rate is not None and float(tp_rate) < self.min_tp_rate:
            reason = f"TP rate {float(tp_rate):.2f} below {self.min_tp_rate:.2f}"
        if reason and not self.circuit_open:
            self.circuit_open = True
            self.circuit_reason = reason
            logger.error("RL threshold circuit breaker tripped: %s", reason)

    def reset_circuit_breaker(self) -> None:
        """Close the breaker and resume RL control (operator action)."""
        self.circuit_open = False
        self.circuit_reason = None
        self._thresholds.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_threshold(self, asset: str) -> float:
        """Return the effective threshold for *asset*.

        Precedence: operator override > circuit-breaker safe value > RL
        cached value > static config default; always within hard bounds.
        """
        pinned = self._override_for(asset)
        if pinned is not None:
            return pinned
        if self.circuit_open:
            return self.safe_threshold
        return self._clamp(self._thresholds.get(asset, float(config.RISK_SCORE_FLAG_THRESHOLD)))

    def update(self, asset: str, obs_dict: dict[str, Any]) -> float:
        """Run the policy to select a new threshold for *asset* and cache it.

        ``obs_dict`` keys mirror the state-space description in the issue:
        ``alerts_fired_last_hour``, ``analyst_tp_rate_last_24h``,
        ``market_volatility_proxy``, ``benford_mad_mean_across_pairs``,
        ``hour_of_day``.  Missing keys fall back to neutral defaults.

        Returns the updated threshold.  If no model is loaded the existing
        cached value (or config default) is returned unchanged.
        """
        if self._override_for(asset) is not None:
            return self.get_threshold(asset)
        self._check_circuit(obs_dict)
        if self._model is None or self.circuit_open:
            return self.get_threshold(asset)

        current = self.get_threshold(asset)
        obs = self._encode_obs(current, obs_dict)
        action, _ = self._model.predict(obs, deterministic=True)
        try:
            idx = int(np.asarray(action).reshape(-1)[0])
        except (TypeError, ValueError, OverflowError):
            idx = _ACTIONS.index(0)
        # Adversarial/out-of-range actions are clamped to the extreme valid delta.
        delta = float(_ACTIONS[min(max(idx, 0), len(_ACTIONS) - 1)])
        new_threshold = self._clamp(current + delta)
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
