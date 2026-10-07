"""Runtime settings for the hardened serving path (contract §1).

Everything is environment driven and :func:`load_settings` is a *pure* function of
``os.environ`` -- no caching -- so tests can monkeypatch the environment and call it
repeatedly without leaking state between cases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: Sentinel that lets ``auth_required`` default to ``bool(api_keys)`` in a frozen dataclass.
_MISSING = object()

_TRUTHY = {"1", "true", "yes", "on", "y"}
_FALSY = {"0", "false", "no", "off", "n"}


def _env_raw(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _env_str(name: str, default: str | None) -> str | None:
    value = _env_raw(name)
    return default if value is None else value


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_raw(name))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_raw(name))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    """Frozen view of the serving configuration."""

    #: Comma separated keys from ``TMM_API_KEYS``; ``()`` means dev mode.
    api_keys: tuple[str, ...] = ()
    #: ``TMM_AUTH_REQUIRED``; defaults to ``bool(api_keys)`` when not given explicitly.
    auth_required: bool = field(default=_MISSING)  # type: ignore[assignment]
    rate_limit_rps: float = 20.0
    rate_limit_burst: int = 40
    #: ``TMM_RATE_LIMIT_BACKEND``: "memory" (per-process, default) or "redis" (shared).
    rate_limit_backend: str = "memory"
    redis_url: str | None = None
    kafka_bootstrap: str | None = None
    kafka_topic: str = "tmm.clickstream"
    kafka_dlq_topic: str = "tmm.clickstream.dlq"
    idempotency_ttl_s: int = 3600
    session_max: int = 50
    cache_ttl_s: int = 5
    stage_timeout_s: float = 0.5
    circuit_fail_threshold: int = 5
    circuit_reset_s: float = 10.0

    def __post_init__(self) -> None:
        if self.auth_required is _MISSING:
            object.__setattr__(self, "auth_required", bool(self.api_keys))


def load_settings() -> Settings:
    """Build :class:`Settings` from the environment. No caching, no side effects."""
    raw_keys = os.environ.get("TMM_API_KEYS", "")
    api_keys = tuple(k.strip() for k in raw_keys.split(",") if k.strip())
    auth_env = os.environ.get("TMM_AUTH_REQUIRED")
    auth_required = bool(api_keys) if auth_env is None or auth_env.strip() == "" \
        else _env_bool("TMM_AUTH_REQUIRED", bool(api_keys))

    return Settings(
        api_keys=api_keys,
        auth_required=auth_required,
        rate_limit_rps=_env_float("TMM_RATE_LIMIT_RPS", 20.0),
        rate_limit_burst=_env_int("TMM_RATE_LIMIT_BURST", 40),
        rate_limit_backend=str(
            _env_str("TMM_RATE_LIMIT_BACKEND", "memory") or "memory").lower(),
        redis_url=_env_str("REDIS_URL", None),
        kafka_bootstrap=_env_str("KAFKA_BOOTSTRAP", None),
        kafka_topic=_env_str("TMM_KAFKA_TOPIC", "tmm.clickstream") or "tmm.clickstream",
        kafka_dlq_topic=_env_str("TMM_KAFKA_DLQ_TOPIC", "tmm.clickstream.dlq")
        or "tmm.clickstream.dlq",
        idempotency_ttl_s=_env_int("TMM_IDEMPOTENCY_TTL_S", 3600),
        session_max=_env_int("TMM_SESSION_MAX", 50),
        cache_ttl_s=_env_int("TMM_CACHE_TTL_S", 5),
        stage_timeout_s=_env_float("TMM_STAGE_TIMEOUT_S", 0.5),
        circuit_fail_threshold=_env_int("TMM_CIRCUIT_FAILS", 5),
        circuit_reset_s=_env_float("TMM_CIRCUIT_RESET_S", 10.0),
    )
