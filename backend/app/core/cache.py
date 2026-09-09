"""Caching, with Redis when it is configured and an in-process fallback when not.

Two rules shape this module:

1. **The app must run without Redis.** Local development has no services, so an
   unset ``REDIS_URL`` transparently gives an in-process dictionary instead. It
   is per-worker and dies with the process, which is fine for what it caches.
2. **A cache failure is never a request failure.** If Redis goes away
   mid-request, the call returns a miss and the caller recomputes. Losing a
   cache should make the site slow, not broken.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Protocol

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Cache(Protocol):
    """What the rest of the app is allowed to assume about caching."""

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...

    def delete(self, key: str) -> None: ...

    def clear_prefix(self, prefix: str) -> int: ...


class InProcessCache:
    """Dictionary with expiry. Per-worker, lost on restart, good enough locally.

    Deliberately not an LRU: what gets cached here is bounded by the number of
    users and menu items, so it cannot grow without limit in the way a cache
    keyed by arbitrary user input could.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[float | None, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and expires_at < time.monotonic():
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds else None
        with self._lock:
            self._store[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [key for key in self._store if key.startswith(prefix)]
            for key in keys:
                del self._store[key]
            return len(keys)


class RedisCache:
    """Redis-backed cache storing JSON.

    Every operation is wrapped: a cache that is down must degrade to a miss, not
    raise into the request handler.
    """

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
        except Exception:
            logger.warning("cache get failed for %s; treating as a miss", key, exc_info=True)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # A value written by an older release with a different shape.
            logger.warning("discarding undecodable cache entry %s", key)
            return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        try:
            payload = json.dumps(value, default=str)
        except (TypeError, ValueError):
            logger.warning("value for %s is not JSON-serialisable; not caching", key)
            return
        try:
            if ttl_seconds:
                self._client.setex(key, ttl_seconds, payload)
            else:
                self._client.set(key, payload)
        except Exception:
            logger.warning("cache set failed for %s; continuing", key, exc_info=True)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception:
            logger.warning("cache delete failed for %s", key, exc_info=True)

    def clear_prefix(self, prefix: str) -> int:
        """Delete every key under a prefix.

        Uses SCAN rather than KEYS: KEYS blocks the whole server while it walks
        the keyspace, which is a genuine outage on a large database.
        """
        removed = 0
        try:
            for key in self._client.scan_iter(match=f"{prefix}*", count=500):
                self._client.delete(key)
                removed += 1
        except Exception:
            logger.warning("cache prefix clear failed for %s", prefix, exc_info=True)
        return removed


_cache: Cache | None = None
_cache_lock = threading.Lock()


def get_cache() -> Cache:
    """Process-wide cache singleton, chosen from settings."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                settings = get_settings()
                if settings.redis_url:
                    logger.info("using Redis cache")
                    _cache = RedisCache(settings.redis_url)
                else:
                    logger.info("REDIS_URL is unset; using the in-process cache")
                    _cache = InProcessCache()
    return _cache


def reset_cache() -> None:
    """Drop the singleton. For tests, and for settings changes at runtime."""
    global _cache
    with _cache_lock:
        _cache = None


# ---------------------------------------------------------------------------
# Key naming
# ---------------------------------------------------------------------------
# Keys carry the model version so that retraining invalidates recommendations
# implicitly. Without it, a new model would keep serving the previous one's
# cached output until each key happened to expire.

RECOMMENDATIONS_PREFIX = "reco:for-me:"
SIMILAR_PREFIX = "reco:similar:"
POPULAR_PREFIX = "reco:popular:"


def recommendations_key(user_id: int, model_version: str, limit: int) -> str:
    return f"{RECOMMENDATIONS_PREFIX}{model_version}:{user_id}:{limit}"


def similar_key(item_id: int, model_version: str, limit: int) -> str:
    return f"{SIMILAR_PREFIX}{model_version}:{item_id}:{limit}"


def popular_key(limit: int) -> str:
    return f"{POPULAR_PREFIX}{limit}"
