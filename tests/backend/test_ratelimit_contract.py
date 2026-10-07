"""Contract §3.4 — token bucket, service-free (``MemoryRateLimiter``).

Boundary cases are checked with a refill rate of 1e-9 tokens/s so the bucket cannot refill
while the test runs: an exact ``burst``-then-reject boundary is observable without sleeping.
"""

from __future__ import annotations

import threading

from tests.backend import _harness as H

SLOW = 1e-9  # effectively no refill during the test


def limiter_spec(burst: int, rate: float = SLOW):
    settings = H.contract_settings(rate_limit_rps=rate, rate_limit_burst=burst)
    return {
        "settings": settings,
        "rate_limit_rps": rate, "rate": rate, "rps": rate, "refill_rate": rate,
        "refill_per_second": rate, "tokens_per_second": rate, "fill_rate": rate,
        "rate_limit_burst": burst, "burst": burst, "capacity": burst, "max_tokens": burst,
        "bucket_size": burst, "size": burst,
        "redis_url": None,
    }


def memory_limiter(burst: int = 5, rate: float = SLOW):
    cls = H.module("tmm.serve.ratelimit").MemoryRateLimiter
    return H.construct(cls, limiter_spec(burst, rate), fallback_args=(rate, burst))


def test_allow_returns_tuple_with_zero_retry_after_when_allowed():
    lim = memory_limiter()
    allowed, retry_after = lim.allow("k")
    assert allowed is True
    assert retry_after == 0.0


# --------------------------------------------------------------------------------------
# adversarial: exact burst boundary
# --------------------------------------------------------------------------------------
def test_exactly_burst_requests_allowed_then_burst_plus_one_rejected():
    burst = 4
    lim = memory_limiter(burst=burst)
    results = [lim.allow("key-a") for _ in range(burst)]
    assert all(a is True for a, _ in results), results
    assert all(r == 0.0 for _, r in results), results

    allowed, retry_after = lim.allow("key-a")
    assert allowed is False
    assert retry_after > 0.0
    assert isinstance(retry_after, float)


def test_buckets_are_per_key():
    lim = memory_limiter(burst=2)
    assert lim.allow("a")[0] is True
    assert lim.allow("a")[0] is True
    assert lim.allow("a")[0] is False
    assert lim.allow("b")[0] is True


def test_refill_after_waiting():
    """A fast bucket refills: the 3rd immediate call is rejected, then allowed after a pause."""
    import time

    lim = memory_limiter(burst=2, rate=50.0)
    assert lim.allow("z")[0] is True
    assert lim.allow("z")[0] is True
    assert lim.allow("z")[0] is False
    time.sleep(0.1)  # 5 tokens worth of refill
    assert lim.allow("z")[0] is True


# --------------------------------------------------------------------------------------
# adversarial: a request with no resolvable client IP must still be limited
# --------------------------------------------------------------------------------------
def test_empty_key_is_still_limited():
    lim = memory_limiter(burst=3)
    assert [lim.allow("")[0] for _ in range(3)] == [True, True, True]
    assert lim.allow("")[0] is False, "an empty key bypassed the limiter"


def test_unknown_key_is_still_limited():
    lim = memory_limiter(burst=3)
    assert [lim.allow("unknown")[0] for _ in range(3)] == [True, True, True]
    assert lim.allow("unknown")[0] is False, "'unknown' bypassed the limiter"


def test_empty_and_unknown_share_one_bucket():
    """Contract §3.4: empty/unknown keys "use a shared bucket"."""
    lim = memory_limiter(burst=3)
    assert [lim.allow("")[0] for _ in range(3)] == [True, True, True]
    assert lim.allow("unknown")[0] is False, (
        "empty and 'unknown' keys have separate buckets, so a request with no resolvable "
        "client IP can multiply its own budget"
    )


# --------------------------------------------------------------------------------------
# adversarial: thread safety (Starlette runs sync handlers in a threadpool)
# --------------------------------------------------------------------------------------
def test_memory_limiter_is_thread_safe_and_never_over_admits():
    burst = 10
    lim = memory_limiter(burst=burst)
    granted: list[bool] = []
    lock = threading.Lock()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(20):
                ok, _retry = lim.allow("shared")
                with lock:
                    granted.append(ok)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(granted) == 320
    assert sum(granted) == burst, f"limiter admitted {sum(granted)} of a burst of {burst}"


# --------------------------------------------------------------------------------------
# RedisRateLimiter: must fall back to memory when Redis is unavailable
# --------------------------------------------------------------------------------------
def test_redis_limiter_falls_back_to_memory_on_connection_error():
    """Contract §3.4: RedisRateLimiter falls back to Memory on error, never raising.

    The client points at a closed port, so every script call fails and the in-process bucket
    must take over.
    """
    import redis

    mod = H.module("tmm.serve.ratelimit")
    dead_url = "redis://127.0.0.1:1/0"
    client = redis.Redis.from_url(dead_url, socket_connect_timeout=0.2, socket_timeout=0.2)
    spec = limiter_spec(burst=2)
    spec.update({"client": client, "redis_url": dead_url, "url": dead_url,
                 "fallback": None,
                 "settings": H.contract_settings(rate_limit_rps=SLOW, rate_limit_burst=2,
                                                 redis_url=dead_url)})
    lim = H.construct(mod.RedisRateLimiter, spec)
    # No Redis is reachable here: allow() must survive and behave like the memory bucket.
    assert lim.allow("x")[0] is True
    assert lim.allow("x")[0] is True
    assert lim.allow("x")[0] is False


def test_unknown_limiter_class_is_reported_not_silently_skipped():
    mod = H.module("tmm.serve.ratelimit")
    for name in ("MemoryRateLimiter", "RedisRateLimiter"):
        assert hasattr(mod, name), f"contract §3.4 names {name}, which is missing from {mod.__name__}"
