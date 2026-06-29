"""Differential privacy primitives for the SHAP explanation layer (Issue #59).

Exposing exact per-feature SHAP contributions lets an adversary mount a model
inversion attack: by querying a wallet, then re-querying it with one trade
removed, the delta in SHAP values reveals which individual trade was most
anomalous. The Gaussian mechanism here bounds that delta so individual trade
contributions are indistinguishable, while keeping the explanation useful for
legitimate audit.

References
----------
- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014)
- Mironov, "Rényi Differential Privacy of the Gaussian Mechanism" (CSF 2017)
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Standard deviation of the Gaussian mechanism for (epsilon, delta)-DP.

    ``sigma = Δ * sqrt(2 * ln(1.25 / delta)) / epsilon``

    where ``Δ`` (`sensitivity`) is the max change in the SHAP value when one
    trade is added/removed.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if sensitivity < 0:
        raise ValueError("sensitivity must be >= 0")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def renyi_noise_multiplier(
    query_count: int,
    threshold: int | None = None,
    multiplier: float | None = None,
) -> float:
    """Noise scaling factor from Rényi (moments-accountant) composition.

    Repeated SHAP queries against the same wallet compose and consume privacy
    budget. Once `query_count` exceeds `threshold` the per-query noise sigma is
    scaled by `multiplier` so the cumulative (epsilon, delta) guarantee still
    holds. Returns ``1.0`` below the threshold.
    """
    threshold = config.DP_RENYI_QUERY_THRESHOLD if threshold is None else threshold
    multiplier = config.DP_RENYI_NOISE_MULTIPLIER if multiplier is None else multiplier
    return multiplier if query_count > threshold else 1.0


def load_shap_sensitivity(path: str | None = None) -> dict:
    """Load the per-model / per-feature sensitivity map from JSON.

    Returns an empty dict if the file does not exist (callers then fall back to
    `config.DP_DEFAULT_SENSITIVITY`).
    """
    path = path or config.SHAP_SENSITIVITY_PATH
    if not os.path.exists(path):
        logger.warning("SHAP sensitivity file not found at %s — using default sensitivity", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def feature_sensitivity(sensitivities: dict, feature: str, default: float | None = None) -> float:
    """Resolve the sensitivity for `feature`, falling back to the configured default."""
    default = config.DP_DEFAULT_SENSITIVITY if default is None else default
    value = sensitivities.get(feature, default) if sensitivities else default
    return float(value)


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Scale parameter for Laplace mechanism: Δ / ε.
    
    Laplace noise N(0, scale) with this scale is (ε, 0)-DP.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if sensitivity < 0:
        raise ValueError("sensitivity must be >= 0")
    return sensitivity / epsilon


def add_laplace_noise(value: float, scale: float, rng: np.random.Generator | None = None) -> float:
    """Add Laplace noise to a value for output perturbation.
    
    Args:
        value: The score to perturb
        scale: Laplace scale parameter (Δ / ε)
        rng: numpy random generator (uses default if None)
    
    Returns:
        Perturbed score
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.laplace(loc=0.0, scale=scale)
    return value + noise
