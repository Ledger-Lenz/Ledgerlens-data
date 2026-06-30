"""Streaming feature store backed by Redis (#183).

Caches precomputed per-wallet feature vectors in Redis so the streaming
scorer can serve them in < 1 ms instead of recomputing all 37+ features on
every trade event.

Design choices
--------------
- **msgpack serialisation** (not pickle): deterministic, fast, and safe
  against code-execution attacks from a compromised Redis instance.
- **Connection pooling**: a single ``redis.ConnectionPool`` is shared across
  all ``RedisFeatureStore`` instances in the same process.
- **Per-window TTLs**: 1-hour features expire in 1 h, 7-day features in 7 d,
  etc.  Defaults come from ``config.FEATURE_STORE_WINDOW_TTLS``.
- **TLS support**: set ``FEATURE_STORE_REDIS_TLS=true`` (or use a
  ``rediss://`` URL) to encrypt the connection.
- **Fallback**: when Redis is unavailable, ``get()`` returns ``None`` and
  ``put()`` is a no-op.  Callers fall back to direct feature computation.
- **Prometheus counters**: ``ledgerlens_feature_cache_hits_total`` /
  ``ledgerlens_feature_cache_misses_total``.

Key schema
----------
``feat:{sha256(wallet)[:16]}:{pair_id}``
Value: msgpack-encoded ``{"v": 1, "features": {...}}``

The key uses a truncated SHA-256 of the wallet address to avoid embedding
full Stellar public keys (56 chars) in Redis key space while keeping them
unique.  The truncation length (16 hex chars = 64 bits) gives negligible
collision probability across millions of wallets.
"""

from __future__ import annotations

import hashlib
import logging
import ssl
from typing import Any, Callable

import msgpack
import redis
import redis.connection

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters (optional)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter

    _cache_hits = Counter(
        "ledgerlens_feature_cache_hits_total",
        "Total Redis feature store cache hits",
        ["store"],
    )
    _cache_misses = Counter(
        "ledgerlens_feature_cache_misses_total",
        "Total Redis feature store cache misses",
        ["store"],
    )
    _fallback_total = Counter(
        "ledgerlens_feature_store_fallback_total",
        "Number of times feature store fell back to direct computation",
        ["store"],
    )
except Exception:  # pragma: no cover
    _cache_hits = None  # type: ignore[assignment]
    _cache_misses = None  # type: ignore[assignment]
    _fallback_total = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Serialisation format version — bump when the wire format changes
# ---------------------------------------------------------------------------
_FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# Shared connection pool (one per (url, pool_size) pair)
# ---------------------------------------------------------------------------
_pool_registry: dict[str, redis.ConnectionPool] = {}


def _get_pool(url: str, pool_size: int, tls: bool, ca_cert: str) -> redis.ConnectionPool:
    """Return a (cached) connection pool for the given URL."""
    key = f"{url}:{pool_size}:{tls}:{ca_cert}"
    if key not in _pool_registry:
        kwargs: dict[str, Any] = {
            "max_connections": pool_size,
            "decode_responses": False,  # we handle bytes ourselves
        }
        if tls or url.startswith("rediss://"):
            ssl_context = ssl.create_default_context()
            if ca_cert:
                ssl_context.load_verify_locations(ca_cert)
            kwargs["ssl"] = True
            kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED if ca_cert else ssl.CERT_NONE
            kwargs["ssl_ca_certs"] = ca_cert or None
        _pool_registry[key] = redis.ConnectionPool.from_url(url, **kwargs)
    return _pool_registry[key]


# ---------------------------------------------------------------------------
# TTL helpers
# ---------------------------------------------------------------------------

def _parse_window_ttls(raw: str) -> dict[int, int]:
    """Parse ``"1:3600,4:14400,24:86400"`` → ``{1: 3600, 4: 14400, 24: 86400}``."""
    ttls: dict[int, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        hours_str, _, secs_str = part.partition(":")
        try:
            ttls[int(hours_str)] = int(secs_str)
        except ValueError:
            logger.warning("Ignoring malformed FEATURE_STORE_WINDOW_TTLS entry: %r", part)
    return ttls


_DEFAULT_TTL = 300  # seconds — fallback when window is not in the map


def _ttl_for_window(window_hours: int | None, window_ttls: dict[int, int]) -> int:
    if window_hours is None:
        return _DEFAULT_TTL
    return window_ttls.get(window_hours, _DEFAULT_TTL)


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------

def _make_key(wallet_id: str, pair_id: str) -> str:
    """Build a Redis key for the (wallet, pair) feature entry."""
    wallet_hash = hashlib.sha256(wallet_id.encode()).hexdigest()[:16]
    return f"feat:{wallet_hash}:{pair_id}"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RedisFeatureStore:
    """Sub-millisecond feature lookup for the streaming pipeline.

    Parameters
    ----------
    redis_url : str
        Redis connection URL.  Defaults to ``config.FEATURE_STORE_REDIS_URL``.
    pool_size : int
        Maximum number of pooled connections.
    tls : bool
        Enable TLS.  Implied when ``redis_url`` starts with ``rediss://``.
    ca_cert : str
        Path to CA certificate for TLS verification.
    window_ttls : dict[int, int] | None
        Override per-window TTLs ``{hours: seconds}``.  When ``None``, parsed
        from ``config.FEATURE_STORE_WINDOW_TTLS``.
    fallback_enabled : bool
        When ``True`` (default), ``get()`` returns ``None`` on any Redis error
        instead of raising.  Callers should fall back to direct computation.
    store_name : str
        Label used in Prometheus counter ``store`` dimension.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        pool_size: int | None = None,
        tls: bool | None = None,
        ca_cert: str | None = None,
        window_ttls: dict[int, int] | None = None,
        fallback_enabled: bool | None = None,
        store_name: str = "default",
    ) -> None:
        self._url = redis_url or config.FEATURE_STORE_REDIS_URL
        self._pool_size = pool_size if pool_size is not None else config.FEATURE_STORE_REDIS_POOL_SIZE
        self._tls = tls if tls is not None else config.FEATURE_STORE_REDIS_TLS
        self._ca_cert = ca_cert if ca_cert is not None else config.FEATURE_STORE_REDIS_TLS_CA_CERT
        self._window_ttls = window_ttls or _parse_window_ttls(config.FEATURE_STORE_WINDOW_TTLS)
        self._fallback = fallback_enabled if fallback_enabled is not None else config.FEATURE_STORE_FALLBACK_ENABLED
        self._store_name = store_name
        self._client: redis.Redis | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_client(self) -> redis.Redis:
        """Return the shared Redis client (lazy initialisation)."""
        if self._client is None:
            pool = _get_pool(self._url, self._pool_size, self._tls, self._ca_cert)
            self._client = redis.Redis(connection_pool=pool)
        return self._client

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            return self._get_client().ping()
        except Exception:
            return False

    def close(self) -> None:
        """Close the connection pool (call on application shutdown)."""
        key = f"{self._url}:{self._pool_size}:{self._tls}:{self._ca_cert}"
        pool = _pool_registry.pop(key, None)
        if pool:
            pool.disconnect()
        self._client = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(
        self,
        wallet_id: str,
        pair_id: str,
    ) -> dict[str, Any] | None:
        """Return the cached feature dict for ``(wallet_id, pair_id)``.

        Returns ``None`` on a cache miss or Redis error (fallback mode).
        The caller is responsible for recomputing features on a miss.

        Only msgpack-deserialised content is returned.  Pickle is never used.
        """
        key = _make_key(wallet_id, pair_id)
        try:
            raw = self._get_client().get(key)
        except Exception as exc:
            if self._fallback:
                logger.warning("RedisFeatureStore.get failed (falling back): %s", exc)
                _inc(_fallback_total, self._store_name)
                return None
            raise

        if raw is None:
            _inc(_cache_misses, self._store_name)
            return None

        try:
            payload = msgpack.unpackb(raw, raw=False)
            if payload.get("v") != _FORMAT_VERSION:
                # Stale format — treat as a miss and let the caller refresh
                logger.debug("Feature store: stale format version, treating as miss.")
                _inc(_cache_misses, self._store_name)
                return None
            _inc(_cache_hits, self._store_name)
            return payload["features"]
        except Exception as exc:
            logger.warning("RedisFeatureStore: deserialization failed for key %s: %s", key, exc)
            _inc(_cache_misses, self._store_name)
            return None

    def put(
        self,
        wallet_id: str,
        pair_id: str,
        features: dict[str, Any],
        ttl_seconds: int | None = None,
        window_hours: int | None = None,
    ) -> bool:
        """Store ``features`` for ``(wallet_id, pair_id)`` with a TTL.

        The TTL is resolved as follows (first match wins):
        1. ``ttl_seconds`` argument
        2. ``window_ttls[window_hours]`` from the instance config
        3. ``_DEFAULT_TTL`` (300 s)

        Only msgpack is used for serialisation (never pickle).

        Returns ``True`` on success, ``False`` on error (silent in fallback mode).
        """
        if ttl_seconds is None:
            ttl_seconds = _ttl_for_window(window_hours, self._window_ttls)

        key = _make_key(wallet_id, pair_id)
        # Validate: only serialise float/int/str/list/dict types — reject anything
        # that msgpack cannot round-trip safely.
        _validate_features(features)

        payload = {"v": _FORMAT_VERSION, "features": features}
        try:
            raw = msgpack.packb(payload, use_bin_type=True)
        except Exception as exc:
            logger.warning("RedisFeatureStore: serialization failed: %s", exc)
            return False

        try:
            self._get_client().setex(key, ttl_seconds, raw)
            return True
        except Exception as exc:
            if self._fallback:
                logger.warning("RedisFeatureStore.put failed (non-fatal): %s", exc)
                return False
            raise

    def get_or_compute(
        self,
        wallet_id: str,
        pair_id: str,
        compute_fn: Callable[[], dict[str, Any]],
        ttl_seconds: int | None = None,
        window_hours: int | None = None,
    ) -> dict[str, Any]:
        """Return cached features or compute them and populate the cache.

        This is the primary call-site in the streaming scorer:

        .. code-block:: python

            features = store.get_or_compute(
                wallet, pair, lambda: build_feature_row(wallet, trades)
            )

        Parameters
        ----------
        wallet_id, pair_id : str
            Cache key components.
        compute_fn : callable
            Zero-argument callable that computes and returns the feature dict
            when there is a cache miss.
        ttl_seconds, window_hours : optional
            TTL override (see ``put()``).
        """
        cached = self.get(wallet_id, pair_id)
        if cached is not None:
            return cached

        features = compute_fn()
        self.put(wallet_id, pair_id, features, ttl_seconds=ttl_seconds, window_hours=window_hours)
        return features

    def delete(self, wallet_id: str, pair_id: str) -> None:
        """Explicitly evict a cached entry."""
        key = _make_key(wallet_id, pair_id)
        try:
            self._get_client().delete(key)
        except Exception as exc:
            if self._fallback:
                logger.warning("RedisFeatureStore.delete failed (non-fatal): %s", exc)
            else:
                raise

    def pipeline_get(
        self,
        wallet_pair_list: list[tuple[str, str]],
    ) -> dict[tuple[str, str], dict[str, Any] | None]:
        """Batch GET using a single Redis pipeline round-trip.

        Returns a mapping ``{(wallet_id, pair_id): features | None}``.
        Useful for pre-warming the streaming scorer's per-batch lookups.
        """
        if not wallet_pair_list:
            return {}

        keys = [_make_key(w, p) for w, p in wallet_pair_list]
        try:
            pipe = self._get_client().pipeline(transaction=False)
            for k in keys:
                pipe.get(k)
            raw_values = pipe.execute()
        except Exception as exc:
            if self._fallback:
                logger.warning("RedisFeatureStore.pipeline_get failed (falling back): %s", exc)
                return {wp: None for wp in wallet_pair_list}
            raise

        result: dict[tuple[str, str], dict[str, Any] | None] = {}
        for (w, p), raw in zip(wallet_pair_list, raw_values):
            if raw is None:
                _inc(_cache_misses, self._store_name)
                result[(w, p)] = None
            else:
                try:
                    payload = msgpack.unpackb(raw, raw=False)
                    if payload.get("v") == _FORMAT_VERSION:
                        _inc(_cache_hits, self._store_name)
                        result[(w, p)] = payload["features"]
                    else:
                        _inc(_cache_misses, self._store_name)
                        result[(w, p)] = None
                except Exception:
                    _inc(_cache_misses, self._store_name)
                    result[(w, p)] = None
        return result


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

_ALLOWED_TYPES = (int, float, str, bool, list, dict, type(None))


def _validate_features(features: dict[str, Any]) -> None:
    """Raise ``TypeError`` if any feature value cannot be safely msgpack'd.

    Rejects numpy scalars, bytes blobs, and arbitrary Python objects that
    could smuggle executable code when unpacked on another host.
    """
    for key, val in features.items():
        if not isinstance(val, _ALLOWED_TYPES):
            raise TypeError(
                f"Feature '{key}' has type {type(val).__name__!r} which cannot be "
                "safely serialised with msgpack.  Convert to float/int/str/list/dict first."
            )


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------

def _inc(counter, label: str) -> None:
    if counter is not None:
        try:
            counter.labels(store=label).inc()
        except Exception:
            pass
