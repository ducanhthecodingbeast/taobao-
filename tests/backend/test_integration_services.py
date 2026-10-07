"""Integration layer: real Redis + single-node KRaft Kafka.

Stack:
    docker compose -f docker/docker-compose.test.yml up -d --wait
    PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest tests/backend -q
    docker compose -f docker/docker-compose.test.yml down -v

Every test here is marked ``integration`` and is *skipped* (with a reason) when 127.0.0.1:6399
or 127.0.0.1:9199 is not listening, so the service-free layer never hangs on a missing broker.
Topics/keys are unique per test run, so repeat runs cannot see each other's offsets.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest

from tests.backend import _harness as H

pytestmark = pytest.mark.integration

GROUP = "tmm-ingest"
CONTAINER_REDIS = "tmmtest-redis"


# ======================================================================================
# helpers
# ======================================================================================
def _redis_client(url: str):
    import redis

    return redis.Redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=2.0,
                                decode_responses=False)


def _unique(prefix: str) -> str:
    return f"tmmtest-{prefix}-{uuid.uuid4().hex[:10]}"


def _create_topics(bootstrap: str, count: int = 1, partitions: int = 3) -> list[str]:
    from kafka.admin import KafkaAdminClient, NewTopic

    names = [_unique("topic") for _ in range(count)]
    admin = KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=15000)
    try:
        admin.create_topics([NewTopic(n, num_partitions=partitions, replication_factor=1)
                             for n in names], validate_only=False)
        assert H.wait_until(lambda: len(admin.describe_topics(names)) == len(names),
                            timeout_s=25), f"new topics never became visible: {names}"
    finally:
        admin.close()
    return names


def _drain(bootstrap: str, topic: str, expected: int, timeout_s: float = 25.0,
           group_id: str | None = None) -> list:
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap,
                             group_id=group_id or _unique("drain"),
                             auto_offset_reset="earliest", enable_auto_commit=False)
    out: list = []
    deadline = time.monotonic() + timeout_s
    try:
        while len(out) < expected and time.monotonic() < deadline:
            for messages in consumer.poll(timeout_ms=500).values():
                out.extend(messages)
    finally:
        consumer.close()
    return out


def _wait_redis(url: str, timeout_s: float = 30.0) -> bool:
    return H.wait_until(lambda: bool(_redis_client(url).ping()), timeout_s=timeout_s)


def _docker(*args: str, timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _container_exists(name: str) -> bool:
    import shutil

    if shutil.which("docker") is None:
        return False
    proc = _docker("inspect", "--format", "{{.State.Status}}", name)
    return proc.returncode == 0


# ======================================================================================
# Redis store (§3.1)
# ======================================================================================
def test_redis_session_store_raw_encoding_ttl_dedup_and_cap(redis_test_url):
    from tmm.serve.session import RedisSessionStore

    r = _redis_client(redis_test_url)
    r.flushdb()
    store = RedisSessionStore(r, session_max=50, idempotency_ttl_s=120, session_ttl_s=120)
    assert store.backend == "redis"
    assert store.ping() is True

    item = 123456789
    assert store.append(1, item, event_id="ev-1") == (1, True)
    raw = r.lrange("tmm:sess:1", 0, -1)
    assert raw == [int(item).to_bytes(8, "little", signed=True)], (
        f"session values must be raw 8-byte little-endian ints, got {raw!r}"
    )
    assert r.ttl("tmm:sess:1") > 0, "redis session key has no TTL"
    assert r.ttl("idem:ev-1") > 0, "idempotency key has no TTL"

    # duplicate replay
    assert store.append(1, 999, event_id="ev-1") == (1, False)
    assert store.get(1) == [item]
    assert store.is_seen("ev-1") is True and store.is_seen("nope") is False

    # event_id=None never deduplicates
    store.append(1, 5, event_id=None)
    store.append(1, 5, event_id=None)
    assert store.get(1)[:2] == [5, 5], store.get(1)

    # cap
    for i in range(70):
        store.append(2, i)
    assert len(store.get(2)) == 50
    assert store.get(2)[0] == 69


def test_redis_rate_limiter_is_atomic_and_shared_across_instances(redis_test_url):
    from tmm.serve.ratelimit import RedisRateLimiter

    r = _redis_client(redis_test_url)
    r.flushdb()
    a = RedisRateLimiter(r, 1e-6, 3)
    b = RedisRateLimiter(r, 1e-6, 3)
    assert [a.allow("shared")[0] for _ in range(3)] == [True, True, True]
    allowed, retry = b.allow("shared")
    assert allowed is False, "a second app instance did not see the shared Redis bucket"
    assert retry > 0.0


# ======================================================================================
# Kafka publisher / consumer (§3.2, §3.3)
# ======================================================================================
def test_kafka_producer_real_config_matches_contract(kafka_test_bootstrap):
    from tmm.serve.events import KafkaEventPublisher

    topic = _create_topics(kafka_test_bootstrap)[0]
    pub = KafkaEventPublisher(kafka_test_bootstrap, topic)
    try:
        assert pub.kind == "kafka"
        assert pub.healthy() is True
        cfg = pub._producer.config
        # kafka-python normalises acks="all" to -1 internally (both mean "all in-sync replicas")
        assert cfg["acks"] in ("all", -1), cfg
        assert cfg["retries"] == 3, cfg
        assert cfg["linger_ms"] == 5, cfg
        assert cfg["max_in_flight_requests_per_connection"] == 1, cfg
    finally:
        pub.close()


def test_kafka_publish_then_consume_preserves_key_and_partition(kafka_test_bootstrap):
    from tmm.serve.events import KafkaEventPublisher

    topic = _create_topics(kafka_test_bootstrap)[0]
    pub = KafkaEventPublisher(kafka_test_bootstrap, topic)
    events = [{"user_id": 4242, "item_id": i, "event_type": "click", "event_id": f"e{i}"}
              for i in range(3)]
    try:
        assert all(pub.publish(e) for e in events)
    finally:
        pub.close()

    from kafka import KafkaConsumer

    consumer = KafkaConsumer(topic, bootstrap_servers=kafka_test_bootstrap,
                             group_id=_unique("rt"), auto_offset_reset="earliest",
                             enable_auto_commit=False)
    received: list = []
    deadline = time.monotonic() + 25
    try:
        while len(received) < 3 and time.monotonic() < deadline:
            for messages in consumer.poll(timeout_ms=500).values():
                received.extend(messages)
    finally:
        consumer.close()

    assert len(received) == 3, f"expected 3 messages, got {len(received)}"
    keys = {m.key for m in received}
    assert len(keys) == 1, f"one user's events landed on different keys: {keys}"
    key = next(iter(keys))
    as_text = key.decode(errors="replace") if isinstance(key, bytes) else str(key)
    # accept both the literal contract spelling (str(user_id.encode()) == "b'4242'", which needs
    # key_serializer=str.encode) and the equivalent bytes key str(user_id).encode() == b"4242"
    assert "4242" in as_text and len(as_text) <= 12, f"key {key!r} is not a per-user key"
    partitions = {m.partition for m in received}
    assert len(partitions) == 1, f"keyed writes scattered across partitions: {partitions}"
    assert {json.loads(m.value)["item_id"] for m in received} == {0, 1, 2}


def test_end_to_end_consume_forever_dedup_and_dlq(kafka_test_bootstrap, redis_test_url):
    """Acceptance 2: published event consumed exactly once; replay deduped; malformed -> DLQ."""
    from tmm.serve.consumer import consume_forever
    from tmm.serve.events import KafkaEventPublisher
    from tmm.serve.session import RedisSessionStore

    topic, dlq_topic = _create_topics(kafka_test_bootstrap, 2)
    r = _redis_client(redis_test_url)
    r.flushdb()

    pub = KafkaEventPublisher(kafka_test_bootstrap, topic)
    valid = {"user_id": 777, "item_id": 555, "event_type": "click", "event_id": "e-1"}
    try:
        assert pub.publish(valid) is True
        assert pub.publish(valid) is True                        # replay -> duplicate
        assert pub.publish({"user_id": 777, "item_id": 9,
                            "event_type": "click"}) is True      # no event_id -> appended
        # poison / undecodable record; a key is required because the producer has a
        # key_serializer configured (str.encode)
        pub._producer.send(topic, key="b'poison'", value=b"\xff\xfe not json")
        pub._producer.flush()
    finally:
        pub.close()

    settings = H.contract_settings(kafka_bootstrap=kafka_test_bootstrap, kafka_topic=topic,
                                   kafka_dlq_topic=dlq_topic, idempotency_ttl_s=120,
                                   circuit_fail_threshold=3)
    store = RedisSessionStore(r, session_max=50, idempotency_ttl_s=120)
    dlq_pub = KafkaEventPublisher(kafka_test_bootstrap, dlq_topic)

    summary: dict = {}

    def run():
        try:
            summary.update(consume_forever(settings, store, publisher=dlq_pub, max_messages=4))
        except BaseException as exc:  # pragma: no cover
            summary["error"] = repr(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(45)
    dlq_pub.close()
    assert not thread.is_alive(), "consume_forever did not stop after max_messages=4"
    assert "error" not in summary, summary

    session = store.get(777)
    assert session.count(555) == 1, f"the replayed event was appended twice: {session}"
    assert session.count(9) == 1, session
    assert summary.get("accepted", 0) == 2, summary
    assert summary.get("duplicate", 0) == 1, summary
    assert summary["dlq"]["decode"] == 1, summary

    dlq_messages = _drain(kafka_test_bootstrap, dlq_topic, 1)
    assert dlq_messages, "the malformed message never reached the DLQ topic"
    payload = json.loads(dlq_messages[0].value)
    assert payload["reason"] == "decode", payload
    assert payload["raw"], payload


def test_consumer_sigterm_shuts_down_gracefully(kafka_test_bootstrap, redis_test_url):
    """§3.3: SIGTERM finishes the in-flight message, commits and exits cleanly."""
    topic, dlq_topic = _create_topics(kafka_test_bootstrap, 2)
    r = _redis_client(redis_test_url)
    r.flushdb()
    user_id = int(uuid.uuid4().int % 1_000_000) + 1

    from tmm.serve.events import KafkaEventPublisher

    pub = KafkaEventPublisher(kafka_test_bootstrap, topic)
    try:
        assert pub.publish({"user_id": user_id, "item_id": 4242, "event_type": "click",
                            "event_id": f"sig-{user_id}"}) is True
    finally:
        pub.close()

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(H.ROOT_SRC) + os.pathsep + env.get("PYTHONPATH", ""),
        "KAFKA_BOOTSTRAP": kafka_test_bootstrap,
        "REDIS_URL": redis_test_url,
        "TMM_KAFKA_TOPIC": topic,
        "TMM_KAFKA_DLQ_TOPIC": dlq_topic,
        "TMM_IDEMPOTENCY_TTL_S": "120",
        "TMM_LOG_LEVEL": "INFO",
        "MPLCONFIGDIR": "/tmp/mpl",
    })
    proc = subprocess.Popen([sys.executable, "-m", "tmm.serve.consumer"],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        consumed = H.wait_until(lambda: r.exists(f"tmm:sess:{user_id}"), timeout_s=30)
        assert consumed, "consumer never wrote the session before SIGTERM"
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=20)
        assert proc.returncode == 0, (
            f"consumer exited {proc.returncode} on SIGTERM\nstdout={out}\nstderr={err}"
        )
        assert "consumer_stopped" in err, f"no graceful-shutdown log line:\n{err}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)
    assert r.llen(f"tmm:sess:{user_id}") == 1


# ======================================================================================
# acceptance 6: Redis dies and comes back without restarting the API
# ======================================================================================
def test_api_survives_redis_kill_and_recovers_without_restart(redis_test_url, monkeypatch):
    if not _container_exists(CONTAINER_REDIS):
        pytest.skip(f"container {CONTAINER_REDIS} not running (compose stack not up)")

    from fastapi.testclient import TestClient

    from tmm.serve.app import create_app

    assert _wait_redis(redis_test_url), "redis was not reachable before the test"

    settings = H.contract_settings(redis_url=redis_test_url, kafka_bootstrap=None,
                                   cache_ttl_s=1)
    app = create_app(settings=settings, recommender=H.StubRecommender())
    restored = False
    try:
        with TestClient(app) as client:
            assert client.get("/health").json()["mode"] == "degraded"
            assert client.get("/readyz").status_code == 200

            assert _docker("stop", CONTAINER_REDIS).returncode == 0
            deadline = time.monotonic() + 20
            mode = None
            while time.monotonic() < deadline:
                mode = client.get("/health").json()["mode"]
                if mode == "minimal":
                    break
                time.sleep(0.2)
            assert mode == "minimal", f"/health did not move to minimal after Redis died: {mode}"
            ready = client.get("/readyz")
            assert ready.status_code == 503, f"/readyz must be 503 in minimal, got {ready.status_code}"
            rec = client.get("/recommend", params={"user_id": 1, "k": 3})
            assert rec.status_code == 200, f"/recommend broke while Redis was down: {rec.text}"
            assert rec.json()

            assert _docker("start", CONTAINER_REDIS).returncode == 0
            assert _wait_redis(redis_test_url), "redis did not come back"
            recovered = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                recovered = client.get("/health").json()["mode"]
                if recovered != "minimal":
                    break
                time.sleep(0.3)
            restored = True
            assert recovered in ("full", "degraded"), (
                f"app did not recover to full/degraded without a restart: {recovered}"
            )
            assert client.get("/readyz").status_code == 200
    finally:
        if not restored:
            _docker("start", CONTAINER_REDIS)
            _wait_redis(redis_test_url)
