"""Gradient-based evasion attacks against the risk-scoring ensemble.

The ensemble (`RiskScorer`) is tree-based, so there is no analytic gradient
to back-propagate through. Both attacks here estimate the gradient of the
*continuous* risk score (`RiskScorer.score_continuous`, 0-100) with respect
to each feature via central finite differences, then take signed steps that
lower the score while staying inside an L-infinity perturbation budget.

  - `FGSMAttack` — single signed step (Goodfellow et al., 2015).
  - `PGDAttack` — iterated FGSM with projection back into the L-inf ball
    after every step (Madry et al., 2018).

Perturbations are applied only to the numeric feature columns; the `wallet`
and `label` columns (`FEATURE_COLUMNS_EXCLUDE`) are passed through
untouched. An optional `feature_scale` expresses the L-inf budget `epsilon`
and `step_size` in standardized units, so a single `epsilon` is comparable
across features that span very different raw magnitudes (e.g. a Benford MAD
of ~0.01 vs. an account age of ~1000 days). A feature whose scale is `0` is
frozen (never perturbed), which is how `robustness.minimum_epsilon_per_feature`
isolates a single feature.

The full finite-difference gradient is evaluated in a single batched
`score_continuous_batch` call per step (all `2 * n_features` probes at once)
to keep the attacks cheap enough for per-wallet sweeps.
"""

import numpy as np
import pandas as pd

from detection.model_training import FEATURE_COLUMNS_EXCLUDE

DEFAULT_TARGET_SCORE = 40

# Default finite-difference probe radius, as a fraction of the per-feature
# L-inf budget (`epsilon * scale`). Tree ensembles are piecewise-constant and,
# on well-separated data, *saturated* (flat) over large neighbourhoods, so a
# tiny probe sees a flat region and yields a useless zero gradient. Probing at
# the budget scale (fraction 1.0) instead asks "if I spend my whole budget on
# this feature, does the score drop?" — i.e. greedy, budget-aware coordinate
# direction finding, which is what makes the attack effective against trees.
DEFAULT_PROBE_FRACTION = 1.0


class _GradientAttack:
    """Shared finite-difference machinery for FGSM/PGD.

    `scorer` is any object exposing `score_continuous_batch(pd.DataFrame)`
    and `score_continuous(pd.Series)` (in practice a
    `detection.model_inference.RiskScorer`). `feature_scale`, when given, is
    aligned to the perturbable feature columns and scales both the L-inf
    budget and the finite-difference probe per feature; it defaults to
    all-ones (raw units). A scale of `0` freezes that feature.
    """

    def __init__(
        self,
        scorer,
        epsilon: float = 0.1,
        *,
        feature_scale: np.ndarray | dict | None = None,
        clip_min: float | None = 0.0,
        clip_max: float | None = None,
        probe_fraction: float = DEFAULT_PROBE_FRACTION,
    ):
        self.scorer = scorer
        self.epsilon = float(epsilon)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.probe_fraction = float(probe_fraction)
        self._feature_scale = feature_scale

    def _feature_columns(self, feature_row: pd.Series) -> list[str]:
        return [c for c in feature_row.index if c not in FEATURE_COLUMNS_EXCLUDE]

    def _scale_vector(self, feature_cols: list[str]) -> np.ndarray:
        """Per-feature scale aligned to `feature_cols` (defaults to ones).

        Negative scales are coerced to `1.0`; an explicit `0.0` is preserved
        so the feature stays frozen.
        """
        scale = self._feature_scale
        if scale is None:
            return np.ones(len(feature_cols))
        if isinstance(scale, dict):
            scale = np.array([scale.get(c, 1.0) for c in feature_cols], dtype=float)
        else:
            scale = np.asarray(scale, dtype=float)
        return np.where(scale < 0, 1.0, scale)

    def _score(self, x: np.ndarray, template: pd.Series, feature_cols: list[str]) -> float:
        row = template.copy()
        row[feature_cols] = x
        return self.scorer.score_continuous(row)

    def _score_gradient(
        self,
        x: np.ndarray,
        template: pd.Series,
        feature_cols: list[str],
        scale: np.ndarray,
    ) -> np.ndarray:
        """Central finite-difference gradient of the continuous score,
        evaluated for all features in a single batched scoring call.

        Frozen features (`scale == 0`) get a zero gradient and are skipped.
        """
        n = len(x)
        # Probe at the budget scale so the gradient reflects a move the attack
        # could actually make (see DEFAULT_PROBE_FRACTION).
        h = self.probe_fraction * self.epsilon * scale
        active = h > 0
        if not active.any():
            return np.zeros(n)

        # Build the +h / -h probe batch only for active features.
        idx = np.flatnonzero(active)
        probes = np.repeat(x[np.newaxis, :], 2 * len(idx), axis=0)
        for k, i in enumerate(idx):
            probes[2 * k, i] += h[i]
            probes[2 * k + 1, i] -= h[i]

        batch = pd.DataFrame(probes, columns=feature_cols)
        scores = self.scorer.score_continuous_batch(batch)

        grad = np.zeros(n)
        for k, i in enumerate(idx):
            grad[i] = (scores[2 * k] - scores[2 * k + 1]) / (2.0 * h[i])
        return grad

    def _clip_to_budget(self, x: np.ndarray, x0: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Project `x` into the L-inf ball of radius `epsilon * scale` around
        `x0`, then into the global `[clip_min, clip_max]` feature domain."""
        lo = x0 - self.epsilon * scale
        hi = x0 + self.epsilon * scale
        x = np.clip(x, lo, hi)
        if self.clip_min is not None:
            x = np.maximum(x, self.clip_min)
        if self.clip_max is not None:
            x = np.minimum(x, self.clip_max)
        return x

    def _to_row(self, x: np.ndarray, template: pd.Series, feature_cols: list[str]) -> pd.Series:
        row = template.copy()
        row[feature_cols] = x
        return row


class FGSMAttack(_GradientAttack):
    """Fast Gradient Sign Method: one signed step of size `epsilon`.

    Lowers the score by stepping each feature against the sign of its
    gradient, then projecting back into the feature domain.
    """

    def perturb(
        self, feature_row: pd.Series, target_score: float = DEFAULT_TARGET_SCORE
    ) -> pd.Series:
        """Return a perturbed copy of `feature_row` (the `target_score` is
        accepted for API symmetry with `PGDAttack` but unused — FGSM always
        takes a single full-budget step)."""
        feature_cols = self._feature_columns(feature_row)
        scale = self._scale_vector(feature_cols)
        x0 = feature_row[feature_cols].to_numpy(dtype=float)

        grad = self._score_gradient(x0, feature_row, feature_cols, scale)
        x = x0 - self.epsilon * scale * np.sign(grad)
        x = self._clip_to_budget(x, x0, scale)

        return self._to_row(x, feature_row, feature_cols)


class PGDAttack(_GradientAttack):
    """Projected Gradient Descent: iterated FGSM with L-inf projection.

    Each step moves every feature `step_size` (in scaled units) against the
    gradient sign, then projects back into the `epsilon` L-inf ball. Stops
    early once the continuous score drops below `target_score`.
    """

    def __init__(
        self,
        scorer,
        epsilon: float = 0.1,
        steps: int = 40,
        step_size: float = 0.01,
        *,
        feature_scale: np.ndarray | dict | None = None,
        clip_min: float | None = 0.0,
        clip_max: float | None = None,
        probe_fraction: float = DEFAULT_PROBE_FRACTION,
    ):
        super().__init__(
            scorer,
            epsilon,
            feature_scale=feature_scale,
            clip_min=clip_min,
            clip_max=clip_max,
            probe_fraction=probe_fraction,
        )
        self.steps = int(steps)
        self.step_size = float(step_size)

    def perturb(
        self, feature_row: pd.Series, target_score: float = DEFAULT_TARGET_SCORE
    ) -> pd.Series:
        """Return a minimally perturbed copy of `feature_row` that scores
        below `target_score` (or the best found within `steps`)."""
        feature_cols = self._feature_columns(feature_row)
        scale = self._scale_vector(feature_cols)
        x0 = feature_row[feature_cols].to_numpy(dtype=float)
        x = x0.copy()

        for _ in range(self.steps):
            if self._score(x, feature_row, feature_cols) < target_score:
                break
            grad = self._score_gradient(x, feature_row, feature_cols, scale)
            x = x - self.step_size * scale * np.sign(grad)
            x = self._clip_to_budget(x, x0, scale)

        return self._to_row(x, feature_row, feature_cols)
