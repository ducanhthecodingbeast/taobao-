"""Contract §2/§3.1 — session store semantics, service-free (``MemorySessionStore``).

Redis-specific checks (raw binary encoding, TTLs, Lua idempotency) live in the integration
module; everything here must pass with no Redis running.
"""

from __future__ import annotations

from tests.backend import _harness as H

SESSION_SPEC = {
    "session_max": 50, "max_len": 50, "maxlen": 50, "max_size": 50, "capacity": 50, "limit": 50,
    "ttl_s": 3600, "idempotency_ttl_s": 3600, "expiry_s": 3600, "seen_ttl_s": 3600,
    "cache_ttl_s": 5, "settings": H.contract_settings(),
}


def memory_store(**overrides):
    cls = getattr(H.module("tmm.serve.session"), "MemorySessionStore")
    spec = dict(SESSION_SPEC)
    spec.update(overrides)
    return H.construct(cls, spec)


def test_backend_attribute_is_memory():
    assert memory_store().backend == "memory"


def test_append_returns_size_and_accepted_and_get_returns_history():
    s = memory_store()
    assert s.append(7, 100) == (1, True)
    assert s.append(7, 101) == (2, True)
    assert sorted(s.get(7)) == [100, 101]
    # existing behaviour (LPUSH) keeps the most recent item first
    assert s.get(7)[0] == 101
    assert s.get(999) == []


def test_users_are_isolated():
    s = memory_store()
    s.append(1, 10)
    s.append(2, 20)
    assert s.get(1) == [10]
    assert s.get(2) == [20]


def test_session_is_capped_at_session_max():
    s = memory_store()
    for i in range(65):
        s.append(3, i)
    assert len(s.get(3)) == 50
    # the cap drops the oldest, not the newest
    assert s.get(3)[0] == 64
    assert 0 not in s.get(3)


def test_ping_true_for_memory_store():
    assert memory_store().ping() is True


# --------------------------------------------------------------------------------------
# adversarial: duplicate replay
# --------------------------------------------------------------------------------------
def test_duplicate_event_id_is_deduplicated_and_not_double_appended():
    s = memory_store()
    size, accepted = s.append(5, 42, event_id="evt-1")
    assert (size, accepted) == (1, True)
    size2, accepted2 = s.append(5, 42, event_id="evt-1")
    assert (size2, accepted2) == (1, False)
    assert s.get(5) == [42]
    assert s.is_seen("evt-1") is True
    assert s.is_seen("evt-nope") is False

    # a *different* event id with the same payload is a genuine second click
    size3, accepted3 = s.append(5, 42, event_id="evt-2")
    assert (size3, accepted3) == (2, True)


def test_duplicate_is_scoped_by_event_id_not_by_session():
    s = memory_store()
    s.append(5, 42, event_id="evt-1")
    # the same event id seen again for a *different* user is still a duplicate (it is the
    # idempotency key that is global, per §3.1 `SET idem:{event_id} 1 NX`)
    _, accepted = s.append(6, 42, event_id="evt-1")
    assert accepted is False
    assert s.get(6) == []


# --------------------------------------------------------------------------------------
# adversarial: event_id=None must NOT deduplicate
# --------------------------------------------------------------------------------------
def test_none_event_id_never_deduplicates():
    s = memory_store()
    r1 = s.append(8, 1, event_id=None)
    r2 = s.append(8, 1, event_id=None)
    r3 = s.append(8, 1)
    assert r1 == (1, True)
    assert r2 == (2, True)
    assert r3 == (3, True)
    assert s.get(8) == [1, 1, 1]


def test_memory_idempotency_expires_with_the_injected_clock():
    """§3.1: the memory store keeps seen ids *with expiry* (same TTL as the Redis NX key)."""

    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    spec = dict(SESSION_SPEC)
    spec.update({"idempotency_ttl_s": 10, "ttl_s": 10, "expiry_s": 10, "clock": clock})
    s = H.construct(H.module("tmm.serve.session").MemorySessionStore, spec)
    assert s.append(8, 1, event_id="ttl-1") == (1, True)
    assert s.append(8, 1, event_id="ttl-1") == (1, False)
    clock.now += 11
    assert s.is_seen("ttl-1") is False
    assert s.append(8, 1, event_id="ttl-1") == (2, True)
