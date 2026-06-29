"""Pair-specific anomaly score normalisation using rolling percentile calibration."""

from dataclasses import dataclass

import redis

SCORE_NORM_WINDOW_SIZE = 1000
SCORE_NORM_MIN_SAMPLES = 50

ASSET_PAIR_ALLOWLIST = {
    "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
}


@dataclass
class NormalisedScore:
    normalised_risk_score: float
    normalisation_skipped: bool


class PerPairScoreNormaliser:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.window_size = SCORE_NORM_WINDOW_SIZE
        self.min_samples = SCORE_NORM_MIN_SAMPLES

    def _validate_asset_pair(self, asset_pair: str) -> None:
        if asset_pair not in ASSET_PAIR_ALLOWLIST:
            raise ValueError(f"Invalid asset pair: {asset_pair}")

    def _get_key(self, asset_pair: str) -> str:
        return f"score_window:{asset_pair}"

    def add_score(self, asset_pair: str, score: float) -> None:
        self._validate_asset_pair(asset_pair)
        key = self._get_key(asset_pair)
        pipe = self.redis.pipeline()
        pipe.zadd(key, {str(score): score})
        pipe.zremrangebyrank(key, 0, -self.window_size - 1)
        pipe.execute()

    def normalise(self, asset_pair: str, score: float) -> NormalisedScore:
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
