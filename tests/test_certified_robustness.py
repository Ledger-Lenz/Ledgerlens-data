"""Unit tests for detection.certified_robustness (IBP) — Issue #245."""

import numpy as np
import pytest

from detection.certified_robustness import (
    Layer,
    certify_ibp,
    _ibp_linear,
    _ibp_relu,
    _ibp_batchnorm,
    _propagate,
    layers_from_neural_process,
)


# ---------------------------------------------------------------------------
# Helper: build a trivial all-class-0 model (always outputs score < threshold)
# ---------------------------------------------------------------------------

def _constant_model(output: float, feature_dim: int = 4) -> list[Layer]:
    """One Linear layer that maps any input to a constant scalar."""
    W = np.zeros((1, feature_dim))  # zero weights
    b = np.array([output])
    return [{"type": "linear", "W": W, "b": b}]


def _identity_model(feature_dim: int = 4) -> list[Layer]:
    """One Linear layer y = x[0] (first feature as output)."""
    W = np.zeros((1, feature_dim))
    W[0, 0] = 1.0
    b = np.zeros(1)
    return [{"type": "linear", "W": W, "b": b}]


# ---------------------------------------------------------------------------
# 1. IBP layer helpers
# ---------------------------------------------------------------------------


def test_ibp_linear_positive_weights():
    W = np.array([[1.0, 2.0]])
    b = np.array([0.5])
    lo, hi = _ibp_linear(np.array([1.0, 0.0]), np.array([2.0, 1.0]), W, b)
    assert lo[0] < hi[0]
    assert hi[0] == pytest.approx(1 * 2 + 2 * 1 + 0.5)


def test_ibp_relu_preserves_positive():
    lo, hi = _ibp_relu(np.array([1.0, -1.0]), np.array([3.0, 2.0]))
    assert lo[0] == pytest.approx(1.0)
    assert lo[1] == pytest.approx(0.0)  # max(0, -1) = 0
    assert hi[1] == pytest.approx(2.0)


def test_ibp_batchnorm_positive_scale():
    gamma = np.array([2.0])
    beta = np.array([0.0])
    mean = np.array([1.0])
    var = np.array([1.0])
    lo, hi = _ibp_batchnorm(np.array([0.0]), np.array([2.0]), gamma, beta, mean, var)
    # scale = 2/sqrt(2) ≈ 1.414, bias = -1.414
    assert lo[0] < hi[0]


def test_ibp_batchnorm_negative_scale_flips_interval():
    """Negative gamma flips lo/hi."""
    gamma = np.array([-1.0])
    beta = np.array([0.0])
    mean = np.array([0.0])
    var = np.array([1.0])
    lo0, hi0 = np.array([-1.0]), np.array([1.0])
    lo, hi = _ibp_batchnorm(lo0, hi0, gamma, beta, mean, var)
    assert lo[0] <= hi[0]


# ---------------------------------------------------------------------------
# 2. All-class-0 model is trivially certified at ε=0
# ---------------------------------------------------------------------------


def test_all_class0_model_certified_at_eps0():
    """A model outputting 0 is certified for label=0 at ε=0 (trivially)."""
    layers = _constant_model(output=10.0)  # score=10 < 50 threshold → class 0
    x = np.array([1.0, 2.0, 3.0, 4.0])
    radius = certify_ibp(layers, x, epsilon=0.0, label=0)
    assert radius == pytest.approx(0.0)


def test_constant_class1_model_certified_at_eps0():
    """A model outputting 80 (fraud) for any input is certified for label=1 at ε=0."""
    layers = _constant_model(output=80.0, feature_dim=4)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    radius = certify_ibp(layers, x, epsilon=0.0, label=1)
    assert radius == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Robust model certified at positive ε
# ---------------------------------------------------------------------------


def test_constant_fraud_model_certified_at_large_eps():
    """A model always outputting 80 is certified at any ε for label=1."""
    layers = _constant_model(output=80.0, feature_dim=4)
    x = np.zeros(4)
    radius = certify_ibp(layers, x, epsilon=1.0, label=1)
    assert radius == pytest.approx(1.0), "Constant model should be certified at any ε"


# ---------------------------------------------------------------------------
# 4. Sensitive model produces small certified radius
# ---------------------------------------------------------------------------


def test_sensitive_model_small_certified_radius():
    """A model with high Lipschitz constant (large weights) has a tiny certified radius."""
    # y = 1000 * x[0] — very sensitive to perturbations of x[0]
    W = np.array([[1000.0, 0.0, 0.0, 0.0]])
    b = np.array([50.0])  # at x=0 the score is exactly 50.0 (on the boundary)
    layers = [{"type": "linear", "W": W, "b": b}]
    x = np.array([0.0, 0.0, 0.0, 0.0])  # score = 50.0 — exactly at threshold, not certified
    # At x[0] = 0 the score is 50 = threshold, so both labels fail clean classification.
    # Use x[0] = 0.01 so score = 60 (correctly classified as fraud).
    x2 = np.array([0.01, 0.0, 0.0, 0.0])
    radius = certify_ibp(layers, x2, epsilon=0.1, label=1, binary_search_steps=20)
    # With W[0]=1000, even ε=0.001 shifts the output by 1 unit; certified radius must be tiny.
    assert radius < 0.05, f"Expected small radius for high-Lipschitz model, got {radius}"


# ---------------------------------------------------------------------------
# 5. Misclassified input returns 0
# ---------------------------------------------------------------------------


def test_misclassified_input_returns_zero():
    """If the model misclassifies the clean input, certify_ibp returns 0."""
    layers = _constant_model(output=10.0, feature_dim=4)  # always benign
    x = np.zeros(4)
    # label=1 (fraud) but model scores 10 (benign) → misclassified → radius = 0
    radius = certify_ibp(layers, x, epsilon=0.5, label=1)
    assert radius == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. layers_from_neural_process returns a non-empty list
# ---------------------------------------------------------------------------


def test_layers_from_neural_process():
    from detection.neural_process import NeuralProcess
    model = NeuralProcess(feature_dim=8, seed=0)
    layers = layers_from_neural_process(model)
    assert len(layers) > 0
    types = {l["type"] for l in layers}
    assert "linear" in types
    assert "relu" in types
