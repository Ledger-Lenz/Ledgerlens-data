"""Pair-specific anomaly score normalisation using rolling percentile calibration."""

from dataclasses import dataclass

import redis

SCORE_NORM_WINDOW_SIZE = 1000
SCORE_NORM_MIN_SAMPLES = 50

# Keep this list aligned with detection.model_training.MODEL_REGISTRY.  The
# normaliser is deliberately model-agnostic: all registered estimators emit a
# raw anomaly score that is calibrated with the same per-pair percentile rule.
REGISTERED_MODEL_TYPES = ("random_forest", "xgboost", "lightgbm")

# Percentile calibration preserves ordering within one model/pair/window. It
# does not make independently calibrated model outputs comparable in absolute
# terms across models or asset pairs.
DIRECT_COMPARISON_MATRIX = {
    model: {other: model == other for other in REGISTERED_MODEL_TYPES}
    for model in REGISTERED_MODEL_TYPES
}


def is_directly_comparable(model_a: str, model_b: str) -> bool:
    """Return whether normalized scores may be compared directly.

    Direct comparison is safe only for the same registered model type (and
    only when the pair/window calibration context is also the same).
    """
    if model_a not in REGISTERED_MODEL_TYPES or model_b not in REGISTERED_MODEL_TYPES:
        raise ValueError(f"unknown model type: {model_a!r} or {model_b!r}")
    return DIRECT_COMPARISON_MATRIX[model_a][model_b]


ASSET_PAIR_ALLOWLIST = {
    "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
}


@dataclass
class NormalisedScore:
    """Result of :meth:`PerPairScoreNormaliser.normalise`.

    Attributes:
        normalised_risk_score: When ``normalisation_skipped`` is ``False`` this
            is the rolling-window percentile rank of the input score. For a
            window of ``n`` samples it is a ``float`` in ``[0.5 / n,
            (n + 0.5) / n]`` — always strictly positive, inside the ``(0.0,
            1.0)`` unit interval for any score within the observed sample
            range, and at most marginally above ``1.0`` for a score that
            exceeds every sample in the window. When ``normalisation_skipped``
            is ``True`` the input ``score`` is passed through unchanged, so the
            value carries whatever range the caller's raw score had.
        normalisation_skipped: ``True`` when the pair's rolling window held
            fewer than ``SCORE_NORM_MIN_SAMPLES`` samples and calibration was
            therefore not applied.
    """

    normalised_risk_score: float
    normalisation_skipped: bool


class PerPairScoreNormaliser:
    """Calibrate raw anomaly scores into per-pair percentile ranks via Redis.

    Each asset pair keeps a rolling window of the most recent
    ``SCORE_NORM_WINDOW_SIZE`` raw scores in a Redis sorted set. ``normalise``
    maps a raw score to its percentile position within that window.

    Range contract:
        * Input ``score`` is an unbounded raw anomaly score (any finite
          ``float`` in ``R``); larger means more anomalous.
        * Output ``NormalisedScore.normalised_risk_score`` is a percentile
          rank once the window has at least ``SCORE_NORM_MIN_SAMPLES``
          samples: strictly positive, within ``(0.0, 1.0)`` for scores inside
          the observed sample range, and bounded above by ``(n + 0.5) / n``
          for a window of ``n`` samples. Until the window fills the raw score
          is returned unchanged with ``normalisation_skipped=True``.

    The output is a percentile fraction, not a 0-100 ``RiskScore``.
    """

    def __init__(self, redis_client: redis.Redis):
        """Bind the normaliser to a Redis client holding the rolling windows."""
        self.redis = redis_client
        self.window_size = SCORE_NORM_WINDOW_SIZE
        self.min_samples = SCORE_NORM_MIN_SAMPLES

    def _validate_asset_pair(self, asset_pair: str) -> None:
        if asset_pair not in ASSET_PAIR_ALLOWLIST:
            raise ValueError(f"Invalid asset pair: {asset_pair}")

    def _get_key(self, asset_pair: str) -> str:
        return f"score_window:{asset_pair}"

    def add_score(self, asset_pair: str, score: float) -> None:
        """Append a raw score to ``asset_pair``'s rolling calibration window.

        Args:
            asset_pair: An allowlisted asset-pair identifier; anything else
                raises ``ValueError``.
            score: The raw, unbounded anomaly score (any finite ``float``) to
                record. The window is trimmed to the newest
                ``SCORE_NORM_WINDOW_SIZE`` scores.
        """
        self._validate_asset_pair(asset_pair)
        key = self._get_key(asset_pair)
        pipe = self.redis.pipeline()
        pipe.zadd(key, {str(score): score})
        pipe.zremrangebyrank(key, 0, -self.window_size - 1)
        pipe.execute()

    def normalise(self, asset_pair: str, score: float) -> NormalisedScore:
        """Map a raw score to its percentile rank within the rolling window.

        Args:
            asset_pair: An allowlisted asset-pair identifier; anything else
                raises ``ValueError``.
            score: The raw, unbounded anomaly score (any finite ``float``) to
                calibrate.

        Returns:
            A ``NormalisedScore``. If the window holds at least
            ``SCORE_NORM_MIN_SAMPLES`` samples, ``normalised_risk_score`` is the
            percentile rank of ``score`` as a strictly positive ``float`` —
            inside ``(0.0, 1.0)`` for a score within the observed sample range
            and at most ``(n + 0.5) / n`` for a window of ``n`` samples — and
            ``normalisation_skipped`` is ``False``. Otherwise the raw ``score``
            is returned unchanged with ``normalisation_skipped`` set to
            ``True``.
        """
        self._validate_asset_pair(asset_pair)
        key = self._get_key(asset_pair)
        window = self.redis.zrange(key, 0, -1, withscores=True)

        if len(window) < self.min_samples:
            return NormalisedScore(normalised_risk_score=score, normalisation_skipped=True)

        scores = [s for _, s in window]
        scores.sort()
        rank = sum(1 for s in scores if s < score)
        percentile = (rank + 0.5) / len(scores)
        return NormalisedScore(normalised_risk_score=percentile, normalisation_skipped=False)
