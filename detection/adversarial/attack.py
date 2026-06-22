"""Gradient-based adversarial attacks against the wash-trade ensemble.

The LedgerLens risk score is produced by a tree ensemble (Random Forest,
XGBoost, LightGBM — see `detection.model_inference`). Tree models are
non-differentiable, so we estimate gradients with *central finite
differences* (Madry et al., 2018, adapted for non-differentiable models)
and run FGSM / PGD on the continuous ensemble probability.

Two subtleties that this module exists to get right:

* **Use the continuous score, not the rounded one.** `RiskScorer.score`
  rounds the ensemble probability to an `int` 0-100. Differencing a rounded
  score yields zero gradients for small perturbations, so the attack would
  stall immediately. `EnsembleScoreFunction` returns the *unrounded*
  ``avg_prob * 100`` so the gradient signal survives.

* **The probe step must straddle tree split thresholds.** A finite-difference
  probe that is too small lands inside the same leaf for every tree and the
  gradient vanishes. The probe is therefore scaled per feature (a fraction of
  each feature's natural scale, e.g. its standard deviation) rather than a
  fixed tiny epsilon.

`epsilon`, `step_size`, and the probe are all expressed in *feature-scale
units*: the perturbation budget for feature ``i`` is ``epsilon * scale[i]``.
Pass ``feature_scale`` (typically the per-feature training std) to make the
L-infinity budget meaningful across features that span very different
magnitudes (a Benford MAD ~0.05 vs a volume ratio ~10000).
"""

import numpy as np
import pandas as pd


class EnsembleScoreFunction:
    """Maps a feature vector to the continuous ensemble risk score (0-100).

    Wraps the same per-model averaging that `RiskScorer` uses, but returns
    the *unrounded* score so it can be differenced. `feature_columns` fixes
    the column order of the vectors passed to the attack.
    """

    def __init__(self, models: dict, feature_columns: list[str]):
        if not models:
            raise ValueError("EnsembleScoreFunction requires at least one model")
        self.models = dict(models)
        self.feature_columns = list(feature_columns)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score a 2-D array (n_samples, n_features) -> (n_samples,) of 0-100."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        frame = pd.DataFrame(X, columns=self.feature_columns)
        probs = np.zeros(len(frame), dtype=float)
        for model in self.models.values():
            probs += model.predict_proba(frame)[:, 1]
        probs /= len(self.models)
        return probs * 100.0

    def __call__(self, x: np.ndarray) -> float:
        """Score a single feature vector -> scalar 0-100."""
        return float(self.score_batch(np.asarray(x, dtype=float).reshape(1, -1))[0])


def feature_scale_from_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Per-feature scale (population std) used to size epsilon/step/probe.

    Zero-variance features fall back to a scale of ``1.0`` so they still get a
    finite, non-degenerate perturbation budget.
    """
    arr = np.asarray(X, dtype=float)
    scale = arr.std(axis=0)
    scale[scale == 0.0] = 1.0
    return scale


class GradientAttack:
    """Base class: central-finite-difference gradient + L-infinity projection.

    Subclasses implement `perturb`. `score_fn` must expose `score_batch` and
    `feature_columns` (see `EnsembleScoreFunction`).
    """

    def __init__(
        self,
        score_fn: EnsembleScoreFunction,
        *,
        epsilon: float = 0.1,
        feature_scale: np.ndarray | None = None,
        probe_fraction: float = 0.05,
        clip_min: np.ndarray | None = None,
        clip_max: np.ndarray | None = None,
        mutable_mask: np.ndarray | None = None,
    ):
        self.score_fn = score_fn
        self.feature_columns = list(score_fn.feature_columns)
        n = len(self.feature_columns)

        self.epsilon = float(epsilon)
        self.scale = np.ones(n) if feature_scale is None else np.asarray(feature_scale, dtype=float)
        if self.scale.shape != (n,):
            raise ValueError(f"feature_scale must have length {n}, got {self.scale.shape}")
        self.probe = probe_fraction * self.scale

        self.clip_min = None if clip_min is None else np.asarray(clip_min, dtype=float)
        self.clip_max = None if clip_max is None else np.asarray(clip_max, dtype=float)
        # 1.0 = adversary may move this feature, 0.0 = immutable (e.g. account age).
        self.mutable_mask = (
            np.ones(n) if mutable_mask is None else np.asarray(mutable_mask, dtype=float)
        )

    def _as_vector(self, feature_row: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(feature_row, pd.Series):
            return feature_row.reindex(self.feature_columns).to_numpy(dtype=float)
        vec = np.asarray(feature_row, dtype=float)
        if vec.shape != (len(self.feature_columns),):
            raise ValueError(
                f"feature_row must have length {len(self.feature_columns)}, got {vec.shape}"
            )
        return vec

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Central-difference gradient of the score w.r.t. each feature.

        Computed in a single batched scoring call over the ``2 * n_features``
        probe points. Immutable features get a zero gradient so they never move.
        """
        n = len(x)
        diag = np.diag(self.probe)
        plus = x + diag
        minus = x - diag
        scores = self.score_fn.score_batch(np.vstack([plus, minus]))
        f_plus, f_minus = scores[:n], scores[n:]
        grad = (f_plus - f_minus) / (2.0 * self.probe)
        return grad * self.mutable_mask

    def _project(self, x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        x = np.clip(x, lower, upper)
        if self.clip_min is not None or self.clip_max is not None:
            x = np.clip(x, self.clip_min, self.clip_max)
        return x

    def perturb(self, feature_row, target_score: float = 40.0):  # pragma: no cover - abstract
        raise NotImplementedError


class FGSMAttack(GradientAttack):
    """Fast Gradient Sign Method — a single ``epsilon``-sized descent step.

    A cheap one-shot baseline: the strongest perturbation reachable in one
    gradient step at the full L-infinity budget.
    """

    def perturb(
        self, feature_row: pd.Series | np.ndarray, target_score: float = 40.0
    ) -> np.ndarray:
        x0 = self._as_vector(feature_row)
        grad = self.gradient(x0)
        x = x0 - self.epsilon * self.scale * np.sign(grad)
        lower = x0 - self.epsilon * self.scale
        upper = x0 + self.epsilon * self.scale
        return self._project(x, lower, upper)


class PGDAttack(GradientAttack):
    """Projected Gradient Descent (Madry et al., 2018).

    Iteratively descends the score by ``step_size`` and projects back into the
    ``epsilon`` L-infinity ball around the original row, stopping early once the
    score drops below ``target_score``.
    """

    def __init__(
        self,
        score_fn: EnsembleScoreFunction,
        *,
        epsilon: float = 0.1,
        steps: int = 40,
        step_size: float = 0.01,
        **kwargs,
    ):
        super().__init__(score_fn, epsilon=epsilon, **kwargs)
        self.steps = int(steps)
        self.step_size = float(step_size)

    def perturb(
        self, feature_row: pd.Series | np.ndarray, target_score: float = 40.0
    ) -> np.ndarray:
        """Return a minimally perturbed feature row that scores below ``target_score``.

        Falls back to the best (lowest-scoring) iterate found if the target is
        not reached within ``steps``.
        """
        x0 = self._as_vector(feature_row)
        lower = x0 - self.epsilon * self.scale
        upper = x0 + self.epsilon * self.scale

        x = x0.copy()
        best_x, best_score = x0.copy(), self.score_fn(x0)
        for _ in range(self.steps):
            if self.score_fn(x) < target_score:
                break
            grad = self.gradient(x)
            x = x - self.step_size * self.scale * np.sign(grad)
            x = self._project(x, lower, upper)
            current = self.score_fn(x)
            if current < best_score:
                best_x, best_score = x.copy(), current
        return x if self.score_fn(x) < target_score else best_x

    def steps_to_target(
        self, feature_row: pd.Series | np.ndarray, target_score: float = 40.0
    ) -> int | None:
        """Number of PGD steps needed to cross below ``target_score``.

        Returns ``None`` if the attack does not succeed within ``steps``. Used
        by the evaluator to report the ``<= 40 steps`` acceptance criterion.
        """
        x0 = self._as_vector(feature_row)
        lower = x0 - self.epsilon * self.scale
        upper = x0 + self.epsilon * self.scale

        x = x0.copy()
        if self.score_fn(x) < target_score:
            return 0
        for step in range(1, self.steps + 1):
            grad = self.gradient(x)
            x = x - self.step_size * self.scale * np.sign(grad)
            x = self._project(x, lower, upper)
            if self.score_fn(x) < target_score:
                return step
        return None
