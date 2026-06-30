"""Feature store integration to serve precomputed wallet features via Redis Hash."""

import hashlib
import logging
from typing import Any, Callable

import msgpack
import redis
from prometheus_client import Counter

from config import config

FEATURE_STORE_TTL_SECONDS = 300
SCHEMA_VERSION = "1"

ledgerlens_feature_cache_hits_total = Counter(
    "ledgerlens_feature_cache_hits_total", "Total feature cache hits"
)
ledgerlens_feature_cache_misses_total = Counter(
    "ledgerlens_feature_cache_misses_total", "Total feature cache misses"
)

logger = logging.getLogger(__name__)


class WalletFeatureStore:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = FEATURE_STORE_TTL_SECONDS

    def _hash_wallet(self, wallet_address: str) -> str:
        return hashlib.sha256(wallet_address.encode()).hexdigest()

    def _get_key(self, wallet_address: str, asset_pair: str) -> str:
        hashed_wallet = self._hash_wallet(wallet_address)
        return f"feat:{hashed_wallet}:{asset_pair}"

    def get_or_compute(
        self,
        wallet: str,
        pair: str,
        compute_fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        key = self._get_key(wallet, pair)
        try:
            cached = self.redis.hget(key, "data")
            if cached:
                data = msgpack.unpackb(cached, raw=False)
                if data.get("schema_version") == SCHEMA_VERSION:
                    ledgerlens_feature_cache_hits_total.inc()
                    return data["features"]
                else:
                    self.redis.delete(key)
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}, falling back to compute")

        ledgerlens_feature_cache_misses_total.inc()
        features = compute_fn()
        payload = {"schema_version": SCHEMA_VERSION, "features": features}
        try:
            self.redis.hset(key, "data", msgpack.packb(payload))
            self.redis.expire(key, self.ttl)
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")
        return features

    def prefetch(self, wallet_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any] | None]:
        results: dict[tuple[str, str], dict[str, Any] | None] = {}
        pipe = self.redis.pipeline()
        keys = {self._get_key(w, p): (w, p) for w, p in wallet_pairs}
        for key in keys:
            pipe.hget(key, "data")
        cached_values = pipe.execute()

        for key, cached in zip(keys.keys(), cached_values):
            wallet, pair = keys[key]
            if cached:
                try:
                    data = msgpack.unpackb(cached, raw=False)
                    if data.get("schema_version") == SCHEMA_VERSION:
                        results[(wallet, pair)] = data["features"]
                        ledgerlens_feature_cache_hits_total.inc()
                        continue
                except Exception:
                    pass
            results[(wallet, pair)] = None
            ledgerlens_feature_cache_misses_total.inc()
        return results
