"""Token-bucket rate limiting (contract §3.4).

Two implementations: an in-process bucket (thread-safe, used in dev and as the Redis
fallback) and a Redis bucket whose read-modify-write is a single atomic Lua script.
No ``slowapi``/``limits`` -- the bucket is ~30 lines.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

__all__ = [
    "MemoryRateLimiter",
    "RateLimiter",
    "RedisRateLimiter",
    "build_rate_limiter",
    "normalize_key",
]

#: Key used whenever the caller has no API key *and* no usable client IP. Sharing one bucket
#: means a missing address can never bypass the limiter.
ANONYMOUS_KEY = "__anonymous__"

# Atomic token bucket. Server-side TIME makes the bucket correct across app replicas.
_LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = burst
  ts = now
end
local delta = now - ts
if delta < 0 then delta = 0 end
tokens = math.min(burst, tokens + delta * rate)
local allowed = 0
local retry = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry = (1 - tokens) / rate
end
redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', key, ttl)
return {tostring(allowed), tostring(retry)}
"""


def normalize_key(key: str | None) -> str:
    """Map empty/missing client identifiers onto one shared bucket."""
    value = (key or "").strip()
    if not value or value.lower() == "unknown":
        return ANONYMOUS_KEY
    return value


class RateLimiter(Protocol):
    """``allow`` returns ``(allowed, retry_after_seconds)``; retry_after is 0.0 if allowed."""

    def allow(self, key: str) -> tuple[bool, float]: ...


class MemoryRateLimiter:
    """Thread-safe in-process token bucket."""

    def __init__(
        self,
        rate_limit_rps: float = 20.0,
        rate_limit_burst: int = 40,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_limit_rps <= 0:
            raise ValueError("rate_limit_rps must be > 0")
        if rate_limit_burst < 1:
            raise ValueError("rate_limit_burst must be >= 1")
        self.rps = float(rate_limit_rps)
        self.burst = float(rate_limit_burst)
        self.clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> tuple[bool, float]:
        bucket_key = normalize_key(key)
        now = self.clock()
        with self._lock:
            tokens, ts = self._buckets.get(bucket_key, (self.burst, now))
            tokens = min(self.burst, tokens + max(0.0, now - ts) * self.rps)
            if tokens >= 1.0:
                self._buckets[bucket_key] = (tokens - 1.0, now)
                return True, 0.0
            self._buckets[bucket_key] = (tokens, now)
            retry = (1.0 - tokens) / self.rps
            return False, retry


class RedisRateLimiter:
    """Redis token bucket (atomic Lua); any error degrades to the in-process bucket."""

    prefix = "tmm:rl:"

    def __init__(
        self,
        client: Any,
        rate_limit_rps: float = 20.0,
        rate_limit_burst: int = 40,
        fallback: RateLimiter | None = None,
    ) -> None:
        if rate_limit_rps <= 0:
            raise ValueError("rate_limit_rps must be > 0")
        if rate_limit_burst < 1:
            raise ValueError("rate_limit_burst must be >= 1")
        self.rps = float(rate_limit_rps)
        self.burst = float(rate_limit_burst)
        self._client = client
        self._fallback = fallback or MemoryRateLimiter(rate_limit_rps, rate_limit_burst)
        self._script = client.register_script(_LUA_TOKEN_BUCKET)
        self._ttl = max(1, int(math.ceil(self.burst / self.rps)) + 1)

    def allow(self, key: str) -> tuple[bool, float]:
        bucket_key = self.prefix + normalize_key(key)
        try:
            raw = self._script(
                keys=[bucket_key],
                args=[str(self.rps), str(self.burst), str(self._ttl)],
            )
            allowed = self._as_bool(raw[0])
            retry = float(self._as_text(raw[1]))
            return allowed, (0.0 if allowed else max(0.0, retry))
        except Exception:  # noqa: BLE001 - any Redis problem must not take the API down
            return self._fallback.allow(key)

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return str(value)

    @classmethod
    def _as_bool(cls, value: Any) -> bool:
        return cls._as_text(value) not in ("0", "0.0", "", "false", "False")


def build_rate_limiter(settings: Any, client: Any | None = None) -> RateLimiter:
    """In-process bucket by default; the shared Redis bucket is opt-in (see below)."""
    memory = MemoryRateLimiter(settings.rate_limit_rps, settings.rate_limit_burst)
    if client is not None:
        return RedisRateLimiter(client, settings.rate_limit_rps, settings.rate_limit_burst,
                                fallback=memory)
    backend = getattr(settings, "rate_limit_backend", "memory")
    if backend != "redis" or not settings.redis_url:
        # Default: one bucket per process. The Redis bucket exists to share the limit across
        # replicas and is opted into with TMM_RATE_LIMIT_BACKEND=redis. Keeping memory the
        # default means an injected in-process app never secretly depends on a live Redis,
        # and unit tests cannot leak limiter state into each other through a real server.
        return memory
    try:  # pragma: no cover - depends on the environment
        import redis

        client = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=0.3, socket_timeout=0.3)
        return RedisRateLimiter(client, settings.rate_limit_rps, settings.rate_limit_burst,
                                fallback=memory)
    except Exception:  # noqa: BLE001
        return memory
