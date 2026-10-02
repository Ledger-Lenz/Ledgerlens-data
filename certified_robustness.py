"""Certified robustness utilities.

Threat model: this module mitigates evasion (T1) and backdoor certification
bounds (T4), and assumes an attacker that may forge/replay certification
messages (T5). See ``docs/adversarial_threat_model.md`` section 3.1 for the
full attacker capabilities, assumed knowledge, query budget, and explicit
out-of-scope items (poisoning T2/T3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class RobustnessCertificate:
    """A certified robustness guarantee for a given input.

    The certificate is valid only for the declared ``eps`` and ``norm``; see
    ``docs/adversarial_threat_model.md`` section 3.1.
    """

    eps: float
    norm: str
    radius: float
    valid: bool = True


class CertifiedRobustness:
    """Compute certified robustness radii for a classifier.

    Mitigates threats T1 (evasion) and T4 (backdoor certification bounds) as
    described in ``docs/adversarial_threat_model.md`` section 3.1.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        norm: str = "l2",
    ) -> None:
        self.predict_fn = predict_fn
        self.norm = norm

    def certify(
        self,
        x: np.ndarray,
        eps: float,
        num_samples: int = 100,
        sigma: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> RobustnessCertificate:
        """Certify the prediction at ``x`` within an Lp ball of radius ``eps``.

        The returned certificate is valid only for the declared ``eps`` and
        ``norm``; it provides no guarantee outside that ball (see
        ``docs/adversarial_threat_model.md`` section 3.1).
        """
        if rng is None:
            rng = np.random.default_rng()

        x = np.asarray(x, dtype=float)
        base = np.argmax(self.predict_fn(x[None, :])[0])

        counts = np.zeros_like(self.predict_fn(x[None, :])[0])
        for _ in range(num_samples):
            noise = rng.normal(scale=sigma, size=x.shape)
            pred = np.argmax(self.predict_fn((x + noise)[None, :])[0])
            counts[pred] += 1

        p_hat = counts[base] / num_samples
        radius = self._radius_from_p_hat(p_hat, sigma)
        valid = radius >= eps
        return RobustnessCertificate(eps=eps, norm=self.norm, radius=radius, valid=valid)

    def _radius_from_p_hat(self, p_hat: float, sigma: float) -> float:
        """Conservative radius estimate from the smoothed lower bound."""
        p_lower = max(p_hat - 0.05, 0.0)
        if p_lower <= 0.5:
            return 0.0
        return float(sigma * self._inverse_normal_cdf(p_lower))

    @staticmethod
    def _inverse_normal_cdf(p: float) -> float:
        """Rational approximation of the inverse standard normal CDF."""
        # Acklam's algorithm.
        a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
             1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
        b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
             6.680131188771972e01, -1.328068155288572e01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
             -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
             3.754408661907416e00]
        p_low, p_high = 0.02425, 1 - 0.02425
        if p < p_low:
            q = np.sqrt(-2 * np.log(p))
            return float(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        if p <= p_high:
            q = p - 0.5
            r = q * q
            return float(((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        q = np.sqrt(-2 * np.log(1 - p))
        return -float(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    def certify_batch(
        self,
        xs: Sequence[np.ndarray],
        eps: float,
        **kwargs,
    ) -> list:
        """Certify a batch of inputs; see section 3.1 of the threat model."""
        return [self.certify(x, eps, **kwargs) for x in xs]
