"""Contract §4/§2 — HTTP surface, degradation modes, auth, rate limiting, lifespan.

Service-free: every dependency is injected through ``create_app``'s documented keywords, and
the heavy recommender is replaced by ``_harness.StubRecommender``.  Expected responses come
from the contract tables (§2 and §4), not from the implementation.
"""

from __future__ import annotations

import io
import json
import logging
import time

import pytest
from fastapi.testclient import TestClient

from tests.backend import _harness as H

EVENT_KEYS = {"ok", "event_id", "accepted", "session_size", "mode", "duplicate"}


def _client(h: H.AppHarness) -> TestClient:
    return TestClient(h.app)


def _minimal_app(monkeypatch: pytest.MonkeyPatch, **over) -> H.AppHarness:
    """No Redis, no Kafka -> contract mode ``minimal``."""
    settings = H.contract_settings(redis_url=None, kafka_bootstrap=None, **over)
    return H.build_app(monkeypatch, settings=settings)


def _full_app(monkeypatch: pytest.MonkeyPatch, *, store=None, publisher=None, **over):
    settings = H.contract_settings(redis_url="redis://127.0.0.1:6399/0",
                                   kafka_bootstrap="127.0.0.1:9199", **over)
    store = store if store is not None else H.FakeSessionStore(backend="redis")
    publisher = publisher if publisher is not None else H.FakePublisher(kind="kafka", healthy=True)
    h = H.build_app(monkeypatch, settings=settings, store=store, publisher=publisher)
    return h, store, publisher


def _degraded_app(monkeypatch: pytest.MonkeyPatch, **over):
    """Redis reachable, Kafka absent."""
    settings = H.contract_settings(redis_url="redis://127.0.0.1:6399/0",
                                   kafka_bootstrap=None, **over)
    store = H.FakeSessionStore(backend="redis")
    publisher = H.FakePublisher(kind="memory", healthy=True)
    h = H.build_app(monkeypatch, settings=settings, store=store, publisher=publisher)
    return h, store, publisher


# ======================================================================================
# §2 degradation modes
# ======================================================================================
def test_livez_is_always_200(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        r = c.get("/livez")
        assert r.status_code == 200, r.text


def test_minimal_mode_health_readyz_and_recommend_still_serves(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        health = c.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["mode"] == "minimal", body
        assert body["redis"] is False and body["kafka"] is False, body

        ready = c.get("/readyz")
        assert ready.status_code == 503, f"/readyz must be 503 in minimal, got {ready.status_code}"
        assert ready.json()["mode"] == "minimal"
        assert H.sample_value("tmm_ready") == 0.0

        rec = c.get("/recommend", params={"user_id": 1, "k": 3})
        assert rec.status_code == 200, rec.text
        assert isinstance(rec.json(), list) and rec.json(), rec.text


def test_full_mode_health_and_readyz_are_ready(monkeypatch):
    h, _store, _pub = _full_app(monkeypatch)
    with _client(h) as c:
        health = c.get("/health").json()
        assert health["mode"] == "full", health
        assert health["redis"] is True and health["kafka"] is True, health
        assert c.get("/readyz").status_code == 200
        assert H.sample_value("tmm_ready") == 1.0


def test_degraded_mode_health_and_readyz_are_ready(monkeypatch):
    h, _store, _pub = _degraded_app(monkeypatch)
    with _client(h) as c:
        health = c.get("/health").json()
        assert health["mode"] == "degraded", health
        assert health["redis"] is True and health["kafka"] is False, health
        assert c.get("/readyz").status_code == 200


# ======================================================================================
# §4 event ingestion
# ======================================================================================
def test_post_events_returns_202_with_documented_body(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        r = c.post("/events", json={"user_id": 5, "item_id": 9, "event_type": "click"})
        assert r.status_code == 202, r.text
        body = r.json()
        assert EVENT_KEYS <= set(body), body
        assert body["ok"] is True
        assert body["accepted"] is True
        assert body["duplicate"] is False
        assert body["session_size"] == 1
        assert body["event_id"]
        assert body["mode"] == "minimal"


def test_event_id_is_generated_server_side_when_absent(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        before = int(time.time() // 60)
        body = c.post("/events", json={"user_id": 7, "item_id": 8,
                                       "event_type": "click"}).json()
        after = int(time.time() // 60)
        generated = body["event_id"]
        assert generated in {f"7:8:click:{b}" for b in (before, after)}, generated


def test_explicit_event_id_is_echoed(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        body = c.post("/events", json={"user_id": 7, "item_id": 8, "event_type": "click",
                                       "event_id": "my-event"}).json()
        assert body["event_id"] == "my-event"


def test_duplicate_replay_over_http_is_not_double_appended(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        payload = {"user_id": 5, "item_id": 42, "event_type": "click", "event_id": "http-dup"}
        first = c.post("/events", json=payload).json()
        second = c.post("/events", json=payload).json()
        assert first["accepted"] is True and first["duplicate"] is False, first
        assert second["accepted"] is False and second["duplicate"] is True, second
        assert second["session_size"] == first["session_size"] == 1, (first, second)
        rec = c.get("/recommend", params={"user_id": 5, "k": 2})
        assert rec.status_code == 200


def test_legacy_event_endpoint_is_the_same_handler(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        body = c.post("/event", json={"user_id": 3, "item_id": 4, "event_type": "cart"})
        # §4 lists /event as the legacy alias of /events; the coder kept the historical 200
        # status for backwards compatibility (§0.5).  Both are accepted here; the *body* is
        # what must match the canonical handler.
        assert body.status_code in (200, 202), body.text
        assert EVENT_KEYS <= set(body.json()), body.json()


# ======================================================================================
# adversarial: no double write when Kafka is healthy (§4, acceptance 7)
# ======================================================================================
def test_full_mode_does_not_double_write_the_session(monkeypatch):
    h, store, publisher = _full_app(monkeypatch)
    with _client(h) as c:
        r = c.post("/events", json={"user_id": 5, "item_id": 9, "event_type": "click",
                                    "event_id": "no-double-write"})
        assert r.status_code == 202, r.text
        assert r.json()["mode"] == "full"
        assert len(publisher.published) == 1, (
            "the event was not handed to Kafka, so the test proves nothing about double-write"
        )
        assert publisher.published[0]["event_id"] == "no-double-write"
        assert store.append_calls == [], (
            f"full mode also wrote the session directly: {store.append_calls}"
        )
        assert store.get(5) == []


def test_degraded_mode_falls_back_to_a_direct_session_write(monkeypatch):
    h, store, publisher = _degraded_app(monkeypatch)
    with _client(h) as c:
        r = c.post("/events", json={"user_id": 5, "item_id": 9, "event_type": "click",
                                    "event_id": "degraded-write"})
        assert r.status_code == 202, r.text
        assert r.json()["mode"] == "degraded"
        assert store.get(5) == [9], (
            f"degraded mode did not write the session directly: {store.get(5)}"
        )
        assert [e["event_id"] for e in publisher.published] == ["degraded-write"]


def test_full_mode_publish_failure_falls_back_to_a_direct_write(monkeypatch):
    """§4: a failed publish must still land in the session, so no accepted click is lost."""
    publisher = H.FakePublisher(kind="kafka", healthy=True, publish_result=False)
    h, store, _ = _full_app(monkeypatch, publisher=publisher)
    with _client(h) as c:
        body = c.post("/events", json={"user_id": 5, "item_id": 9, "event_type": "click",
                                       "event_id": "publish-failed"}).json()
        assert body["mode"] == "full"
        assert body["accepted"] is True, body
        assert store.get(5) == [9], (
            f"the event was dropped after a failed publish: session={store.get(5)}"
        )


def test_full_mode_replay_is_reported_duplicate_without_republishing(monkeypatch):
    """§4 response includes `duplicate`; in full mode it is answered by a read-only probe.

    The idempotency key is global (§3.1), so an event the consumer has already ingested must
    come back as `duplicate=true`, must not be re-published to Kafka, and must not be written
    to the session on the request path.
    """
    h, store, publisher = _full_app(monkeypatch)
    with _client(h) as c:
        store.append(999, 1, event_id="seen-already")   # the consumer got there first
        assert store.append_calls == [(999, 1, "seen-already")]
        body = c.post("/events", json={"user_id": 5, "item_id": 9, "event_type": "click",
                                       "event_id": "seen-already"}).json()
        assert body["accepted"] is False and body["duplicate"] is True, body
        assert publisher.published == [], "a replayed event was published to Kafka again"
        assert store.append_calls == [(999, 1, "seen-already")], (
            f"the replay was appended on the request path: {store.append_calls}"
        )
        assert store.get(5) == []


# ======================================================================================
# adversarial: auth
# ======================================================================================
def test_missing_api_key_is_401_with_www_authenticate(monkeypatch):
    h = _minimal_app(monkeypatch, api_keys=("good-key",), auth_required=True)
    with _client(h) as c:
        r = c.post("/events", json={"user_id": 1, "item_id": 2})
        assert r.status_code == 401, r.text
        assert r.headers.get("WWW-Authenticate") == "X-API-Key", dict(r.headers)
        r2 = c.get("/items/sample")
        assert r2.status_code == 401


def test_wrong_api_key_is_401(monkeypatch):
    h = _minimal_app(monkeypatch, api_keys=("good-key",), auth_required=True)
    with _client(h) as c:
        r = c.get("/items/sample", headers={"X-API-Key": "bad-key"})
        assert r.status_code == 401, r.text


def test_valid_api_key_is_accepted(monkeypatch):
    h = _minimal_app(monkeypatch, api_keys=("good-key",), auth_required=True)
    with _client(h) as c:
        r = c.post("/events", json={"user_id": 1, "item_id": 2},
                   headers={"X-API-Key": "good-key"})
        assert r.status_code == 202, r.text
        assert c.get("/items/sample", headers={"X-API-Key": "good-key"}).status_code == 200


def test_health_and_metrics_need_no_auth(monkeypatch):
    h = _minimal_app(monkeypatch, api_keys=("good-key",), auth_required=True)
    with _client(h) as c:
        for path in ("/livez", "/readyz", "/health", "/metrics"):
            assert c.get(path).status_code in (200, 503), path


def test_auth_disabled_logs_one_loud_warning(monkeypatch):
    """Contract §4: when auth_required is false, log a loud *one-time* warning."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(H.module("tmm.serve.obs").JsonFormatter())
    app_logger = logging.getLogger("tmm.serve.app")
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.DEBUG)
    try:
        h = _minimal_app(monkeypatch)
        with _client(h) as c:
            for _ in range(3):
                assert c.get("/items/sample").status_code == 200
    finally:
        app_logger.removeHandler(handler)

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert lines, "running without auth logged nothing at all"
    parsed = [json.loads(ln) for ln in lines]  # §6.8: every log line is valid single-line JSON
    warnings = [p for p in parsed if p.get("level") == "WARNING"]
    loud = [w for w in warnings
            if "auth" in json.dumps(w).lower() or "api_key" in json.dumps(w).lower()]
    assert len(loud) == 1, f"expected exactly one loud auth warning, got {warnings}"


# ======================================================================================
# adversarial: rate limiting
# ======================================================================================
def test_rate_limit_allows_exactly_burst_then_429_with_retry_after(monkeypatch):
    h = _minimal_app(monkeypatch, rate_limit_rps=1e-9, rate_limit_burst=3)
    with _client(h) as c:
        before, _, _ = H.delta("tmm_rate_limited_total", {"scope": "ip"})
        for i in range(3):
            r = c.get("/items/sample")
            assert r.status_code == 200, f"request {i + 1} of the burst was rejected: {r.status_code}"
        r = c.get("/items/sample")
        assert r.status_code == 429, f"burst+1 must be 429, got {r.status_code}"
        retry_after = r.headers.get("Retry-After")
        assert retry_after is not None, "429 without a Retry-After header"
        assert int(retry_after) >= 1, retry_after
        after, _, _ = H.delta("tmm_rate_limited_total", {"scope": "ip"})
        assert after > before, "tmm_rate_limited_total{scope='ip'} was not incremented"


def test_rate_limit_metric_uses_api_key_scope_when_key_present(monkeypatch):
    h = _minimal_app(monkeypatch, api_keys=("good-key",), auth_required=True,
                     rate_limit_rps=1e-9, rate_limit_burst=2)
    with _client(h) as c:
        headers = {"X-API-Key": "good-key"}
        assert c.get("/items/sample", headers=headers).status_code == 200
        assert c.get("/items/sample", headers=headers).status_code == 200
        before, after, r = H.delta("tmm_rate_limited_total", {"scope": "api_key"},
                                   lambda: c.get("/items/sample", headers=headers))
        assert r.status_code == 429
        assert after > before, "tmm_rate_limited_total{scope='api_key'} was not incremented"


@pytest.mark.asyncio
async def test_request_without_any_client_ip_is_still_rate_limited(monkeypatch):
    """A request whose ASGI scope carries no client at all must not bypass the limiter."""
    h = _minimal_app(monkeypatch, rate_limit_rps=1e-9, rate_limit_burst=2)
    statuses = []
    for _ in range(3):
        status, _headers, _body = await H.asgi_call(h.app, "GET", "/items/sample")
        statuses.append(status)
    assert statuses[:2] == [200, 200], statuses
    assert statuses[2] == 429, f"a request with no client IP bypassed the limiter: {statuses}"


@pytest.mark.asyncio
async def test_request_with_none_client_is_still_rate_limited(monkeypatch):
    h = _minimal_app(monkeypatch, rate_limit_rps=1e-9, rate_limit_burst=2)
    statuses = []
    for _ in range(3):
        status, _headers, _body = await H.asgi_call(h.app, "GET", "/items/sample", client=None)
        statuses.append(status)
    assert statuses[:2] == [200, 200], statuses
    assert statuses[2] == 429, f"client=None bypassed the limiter: {statuses}"


# ======================================================================================
# §3.6 stage instrumentation + §6.4 metrics surface
# ======================================================================================
def test_recommend_records_all_five_documented_stages(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        assert c.get("/recommend", params={"user_id": 1, "k": 3}).status_code == 200
    for stage in ("decode", "session_read", "tower", "search", "total"):
        count = H.sample_value("tmm_stage_duration_seconds", {"stage": stage}, suffix="_count")
        assert count is not None and count >= 1.0, (
            f"stage {stage!r} was not recorded (count={count}); contract §3.6 names it explicitly"
        )


class _RecordingKafkaProducer:
    """Minimal stand-in for ``kafka.KafkaProducer`` so the real publisher needs no broker."""

    instances: list["_RecordingKafkaProducer"] = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.sent: list[tuple[str, object, object]] = []
        self.fail = False
        _RecordingKafkaProducer.instances.append(self)

    def send(self, topic, value=None, key=None):
        if self.fail:
            from kafka.errors import KafkaError

            raise KafkaError("simulated")
        self.sent.append((topic, value, key))
        return _DoneFuture()

    def bootstrap_connected(self) -> bool:
        return not self.fail

    def flush(self, timeout=None):
        pass

    def close(self, timeout=None):
        pass


class _DoneFuture:
    def get(self, timeout=None):
        return None


def _real_kafka_publisher(monkeypatch: pytest.MonkeyPatch, topic: str = "tmm.clickstream.test"):
    from tmm.serve import events as events_mod
    from tmm.serve.events import KafkaEventPublisher

    _RecordingKafkaProducer.instances.clear()
    monkeypatch.setattr(events_mod, "KafkaProducer", _RecordingKafkaProducer)
    return KafkaEventPublisher("127.0.0.1:9199", topic)


def test_metrics_endpoint_exposes_every_contract_metric(monkeypatch):
    """Acceptance 4: every metric named in §3.6 is exposed by GET /metrics.

    Each metric is first *made to exist* through the real code path (real KafkaEventPublisher
    against a double producer, the real consumer processing function for the DLQ, a real 429),
    then the exposition is parsed.
    """
    from tmm.serve.consumer import process_message_detail

    real_pub = _real_kafka_publisher(monkeypatch)
    # burst 3 == the three rate-limited calls below (/events, /items/sample, /recommend)
    h, store, _pub = _full_app(monkeypatch, publisher=real_pub, api_keys=("k",),
                               auth_required=True, rate_limit_rps=1e-9, rate_limit_burst=3)
    headers = {"X-API-Key": "k"}
    with _client(h) as c:
        c.get("/livez")
        c.get("/health")
        c.get("/readyz")
        ok = c.post("/events", json={"user_id": 1, "item_id": 2, "event_type": "click"},
                    headers=headers)
        assert ok.status_code == 202, ok.text
        assert c.get("/items/sample", headers=headers).status_code == 200
        assert c.get("/recommend", params={"user_id": 1, "k": 2}, headers=headers).status_code == 200
        limited = c.get("/items/sample", headers=headers)
        assert limited.status_code == 429, limited.text
        # exercise the DLQ counter through the consumer's real processing function
        process_message_detail(b"not-json", store, None)
        r = c.get("/metrics")
        assert r.status_code == 200, r.text
        assert "text/plain" in r.headers.get("content-type", ""), r.headers

    exposition = H.parse_exposition(r.text)
    names = set(exposition)
    missing = []
    for name in H.REQUIRED_METRICS:
        present = name in names or f"{name}_count" in names or f"{name}_bucket" in names
        if not present:
            missing.append(name)
    assert not missing, (
        f"contract §3.6 metrics absent from the /metrics exposition: {missing}\n"
        f"exposed: {sorted(names)}"
    )
    # and the raw values really moved (not just empty families)
    assert H.counter_value(exposition, "tmm_kafka_publish_total", {"result": "ok"}) >= 1.0
    assert H.counter_value(exposition, "tmm_dlq_total", {"reason": "decode"}) >= 1.0
    assert H.counter_value(exposition, "tmm_rate_limited_total", {"scope": "api_key"}) >= 1.0
    assert H.counter_value(exposition, "tmm_events_ingested_total", {"result": "accepted"}) >= 1.0


def test_metrics_exposition_is_prometheus_text_and_has_no_default_registry_leak(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        text = c.get("/metrics").text
    exposition = H.parse_exposition(text)
    assert exposition, text
    helpers = {n for n in exposition if n.startswith("python_") or n.startswith("process_")}
    assert not helpers, f"default-registry collectors leaked into /metrics: {helpers}"


# ======================================================================================
# §4 graceful shutdown + §0.5 backwards compatibility
# ======================================================================================
def test_lifespan_closes_publisher_and_store(monkeypatch):
    h, store, publisher = _full_app(monkeypatch)
    with _client(h) as c:
        assert c.get("/livez").status_code == 200
    assert publisher.closed is True, "lifespan did not close the event publisher"
    assert store.closed is True, "lifespan did not close the session store"


def test_health_keeps_the_pre_existing_fields(monkeypatch):
    h = _minimal_app(monkeypatch)
    with _client(h) as c:
        body = c.get("/health").json()
    for field in ("status", "session_backend", "index_version", "catalogue", "index_mb",
                  "embedding_table_mb", "user_tower", "uptime_s"):
        assert field in body, f"existing /health field {field!r} disappeared: {body}"
    for field in ("mode", "kafka", "redis", "circuits"):
        assert field in body, f"new /health field {field!r} missing: {body}"


def test_no_secret_is_ever_logged(monkeypatch):
    secret = "s3cr3t-key-xyz"
    h = _minimal_app(monkeypatch, api_keys=(secret,), auth_required=True)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(H.module("tmm.serve.obs").JsonFormatter())
    app_logger = logging.getLogger("tmm.serve.app")
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.DEBUG)
    try:
        with _client(h) as c:
            c.get("/items/sample", headers={"X-API-Key": "wrong-key-attempt"})
            c.get("/items/sample", headers={"X-API-Key": secret})
            c.post("/events", json={"user_id": 1, "item_id": 2}, headers={"X-API-Key": secret})
    finally:
        app_logger.removeHandler(handler)

    output = buf.getvalue()
    assert secret not in output, f"the API key leaked into the logs: {output}"
    assert "wrong-key-attempt" not in output, f"a rejected key leaked into the logs: {output}"
    for line in [ln for ln in output.splitlines() if ln.strip()]:
        json.loads(line)
