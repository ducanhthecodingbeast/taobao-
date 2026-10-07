"""Session storage with idempotent appends (contract §3.1).

``RedisSessionStore``  -- LPUSH + LTRIM + EXPIRE, raw 8-byte little-endian ints,
                          ``SET idem:{event_id} 1 NX EX ttl`` for dedup.
``MemorySessionStore``  -- identical semantics, ``deque`` + expiring seen-set.
``FallbackSessionStore``-- wraps a Redis store and transparently degrades to memory; it
                          re-probes the primary so a restarted Redis is picked up *without*
                          restarting the API.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

from .resilience import CircuitBreaker

__all__ = [
    "FallbackSessionStore",
    "MemorySessionStore",
    "RedisSessionStore",
    "SessionStore",
    "build_session_store",
    "decode_item",
    "encode_item",
]

SESSION_PREFIX = "tmm:sess:"
IDEMPOTENCY_PREFIX = "idem:"
SESSION_TTL_S = 3600


def encode_item(item_id: int) -> bytes:
    """Raw signed 64-bit little-endian -- never a JSON list."""
    return int(item_id).to_bytes(8, "little", signed=True)


def decode_item(raw: bytes) -> int:
    return int.from_bytes(raw, "little", signed=True)


class SessionStore(Protocol):
    #: Read-only so both a plain class attribute ("memory"/"redis") and a computed property
    #: (the fallback store, whose value tracks live health) satisfy the protocol.
    @property
    def backend(self) -> str: ...

    def append(self, user_id: int, item_id: int,
               event_id: str | None = None) -> tuple[int, bool]: ...
    def get(self, user_id: int) -> list[int]: ...
    def is_seen(self, event_id: str) -> bool: ...
    def ping(self) -> bool: ...


class MemorySessionStore:
    """In-process sessions + idempotency. Used in dev and whenever Redis is unreachable."""

    backend = "memory"

    def __init__(
        self,
        session_max: int = 50,
        idempotency_ttl_s: int = SESSION_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_max = int(session_max)
        self.idempotency_ttl_s = int(idempotency_ttl_s)
        self.clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[int, deque[int]] = {}
        self._seen: dict[str, float] = {}

    def append(self, user_id: int, item_id: int,
               event_id: str | None = None) -> tuple[int, bool]:
        with self._lock:
            if event_id is not None and self._is_seen_locked(event_id):
                return len(self._sessions.get(int(user_id), ())), False
            if event_id is not None:
                self._seen[event_id] = self.clock() + self.idempotency_ttl_s
            session = self._sessions.get(int(user_id))
            if session is None:
                session = deque(maxlen=self.session_max)
                self._sessions[int(user_id)] = session
            session.appendleft(int(item_id))
            return len(session), True

    def get(self, user_id: int) -> list[int]:
        with self._lock:
            return list(self._sessions.get(int(user_id), ()))

    def is_seen(self, event_id: str) -> bool:
        with self._lock:
            return self._is_seen_locked(event_id)

    def _is_seen_locked(self, event_id: str) -> bool:
        expiry = self._seen.get(event_id)
        if expiry is None:
            return False
        if expiry <= self.clock():
            self._seen.pop(event_id, None)
            return False
        return True

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass


class RedisSessionStore:
    """Redis-backed sessions. Methods raise on connection errors; callers handle fallback."""

    backend = "redis"

    def __init__(
        self,
        client: Any,
        session_max: int = 50,
        idempotency_ttl_s: int = SESSION_TTL_S,
        session_ttl_s: int = SESSION_TTL_S,
    ) -> None:
        self._client = client
        self.session_max = int(session_max)
        self.idempotency_ttl_s = int(idempotency_ttl_s)
        self.session_ttl_s = int(session_ttl_s)

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> RedisSessionStore:
        import redis

        store_keys = ("session_max", "idempotency_ttl_s", "session_ttl_s")
        store_kwargs = {k: kwargs.pop(k) for k in store_keys if k in kwargs}
        connect_timeout = kwargs.pop("socket_connect_timeout", 0.5)
        socket_timeout = kwargs.pop("socket_timeout", 0.5)
        client = redis.Redis.from_url(
            url, socket_connect_timeout=connect_timeout, socket_timeout=socket_timeout,
            decode_responses=False, **kwargs)
        return cls(client, **store_kwargs)

    @staticmethod
    def _session_key(user_id: int) -> str:
        return f"{SESSION_PREFIX}{int(user_id)}"

    @staticmethod
    def _idem_key(event_id: str) -> str:
        return f"{IDEMPOTENCY_PREFIX}{event_id}"

    def append(self, user_id: int, item_id: int,
               event_id: str | None = None) -> tuple[int, bool]:
        key = self._session_key(user_id)
        if event_id is not None:
            claimed = self._client.set(self._idem_key(event_id), "1", nx=True,
                                       ex=self.idempotency_ttl_s)
            if not claimed:
                return int(self._client.llen(key)), False
        pipe = self._client.pipeline()
        pipe.lpush(key, encode_item(item_id))
        pipe.ltrim(key, 0, self.session_max - 1)
        pipe.expire(key, self.session_ttl_s)
        pipe.llen(key)
        result = pipe.execute()
        return int(result[-1]), True

    def get(self, user_id: int) -> list[int]:
        raw = self._client.lrange(self._session_key(user_id), 0, self.session_max - 1)
        return [decode_item(b) for b in raw]

    def is_seen(self, event_id: str) -> bool:
        return bool(self._client.exists(self._idem_key(event_id)))

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        try:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001
            pass


class FallbackSessionStore:
    """Redis primary with an in-process fallback and a health probe that drives recovery.

    ``backend`` reflects the *currently live* primary, which is what ``/health`` and the
    degradation contract read. A failed primary is re-probed at most once per
    ``probe_interval_s`` so restarting Redis flips the app back to ``redis`` on its own.
    """

    def __init__(
        self,
        primary: SessionStore | None = None,
        fallback: MemorySessionStore | None = None,
        breaker: CircuitBreaker | None = None,
        probe_interval_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._primary = primary
        self._fallback = fallback or MemorySessionStore()
        self._breaker = breaker
        self._probe_interval_s = float(probe_interval_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._live = False
        self._last_probe = float("-inf")

    # -- health ------------------------------------------------------------------------
    @property
    def primary_backend(self) -> str:
        return getattr(self._primary, "backend", "memory")

    @property
    def backend(self) -> str:
        return "redis" if self.probe() else "memory"

    @property
    def is_live(self) -> bool:
        return self._live

    def probe(self, force: bool = False) -> bool:
        """Cheap Reachability check for the primary, throttled to ``probe_interval_s``."""
        if self._primary is None:
            self._live = False
            return False
        now = self._clock()
        with self._lock:
            if not force and now - self._last_probe < self._probe_interval_s:
                return self._live
            self._last_probe = now
        try:
            ok = bool(self._primary.ping())
        except Exception:  # noqa: BLE001
            ok = False
        self._live = ok
        if ok and self._breaker is not None:
            self._breaker.record_success()
        return ok

    def _mark_down(self) -> None:
        with self._lock:
            self._live = False
            self._last_probe = self._clock()

    def _call(self, method: str, *args: Any) -> Any:
        fn = getattr(self._primary, method)
        if self._breaker is not None:
            return self._breaker.call(fn, *args)
        return fn(*args)

    def _with_fallback(self, method: str, *args: Any) -> Any:
        if self.probe():
            try:
                return self._call(method, *args)
            except Exception:  # noqa: BLE001 - any primary failure degrades to memory
                self._mark_down()
        return getattr(self._fallback, method)(*args)

    # -- SessionStore protocol ---------------------------------------------------------
    def append(self, user_id: int, item_id: int,
               event_id: str | None = None) -> tuple[int, bool]:
        return self._with_fallback("append", user_id, item_id, event_id)

    def get(self, user_id: int) -> list[int]:
        return self._with_fallback("get", user_id)

    def is_seen(self, event_id: str) -> bool:
        return bool(self._with_fallback("is_seen", event_id))

    def ping(self) -> bool:
        if self._primary is None:
            return self._fallback.ping()
        return self.probe()

    def close(self) -> None:
        for store in (self._primary, self._fallback):
            close = getattr(store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass


def build_session_store(
    settings: Any,
    breaker: CircuitBreaker | None = None,
    client: Any | None = None,
) -> SessionStore:
    """Redis-backed store when configured; a plain in-memory store otherwise."""
    memory = MemorySessionStore(
        session_max=settings.session_max,
        idempotency_ttl_s=settings.idempotency_ttl_s,
    )
    primary: SessionStore | None = None
    if client is not None:
        primary = RedisSessionStore(
            client, session_max=settings.session_max,
            idempotency_ttl_s=settings.idempotency_ttl_s)
    elif settings.redis_url:
        try:  # pragma: no cover - environment dependent
            primary = RedisSessionStore.from_url(
                settings.redis_url, session_max=settings.session_max,
                idempotency_ttl_s=settings.idempotency_ttl_s)
        except Exception:  # noqa: BLE001 - Redis is optional
            primary = None
    if primary is None:
        return memory
    return FallbackSessionStore(primary, memory, breaker=breaker,
                                probe_interval_s=min(1.0, max(0.1, float(settings.cache_ttl_s))))
