"""Contract §3.2 — ``EventPublisher`` implementations, service-free.

The Kafka producer is verified by intercepting the ``KafkaProducer`` constructor, so the
contract's producer settings (``acks=all``, ``retries=3``, ``linger_ms=5``,
``max_in_flight=1``), the message key rule and the "never raise on broker failure" rule can all
be checked with **no broker running**.
"""

from __future__ import annotations

import json
import threading

import pytest

from tests.backend import _harness as H

BOOTSTRAP = "127.0.0.1:9199"


class RecordingProducer:
    """Stand-in for ``kafka.KafkaProducer`` that records config and sent records."""

    instances: list["RecordingProducer"] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.sent: list[tuple[str, bytes | str | None, bytes | str | None]] = []
        self.closed = False
        self.flushed = 0
        self.fail_with: BaseException | None = None
        RecordingProducer.instances.append(self)

    def send(self, topic, value=None, key=None):
        if self.fail_with is not None:
            raise self.fail_with
        self.sent.append((topic, value, key))
        return _Future()

    def flush(self, timeout=None):
        self.flushed += 1

    def close(self, timeout=None):
        self.closed = True

    def partitions_for_topic(self, topic):
        return {0}


class _Future:
    def get(self, timeout=None):
        return None


def _patch_producer(monkeypatch: pytest.MonkeyPatch, mod) -> list[str]:
    patched = []
    if isinstance(getattr(mod, "KafkaProducer", None), type):
        monkeypatch.setattr(mod, "KafkaProducer", RecordingProducer)
        patched.append(f"{mod.__name__}.KafkaProducer")
    kafka_mod = getattr(mod, "kafka", None)
    if kafka_mod is not None and isinstance(getattr(kafka_mod, "KafkaProducer", None), type):
        monkeypatch.setattr(kafka_mod, "KafkaProducer", RecordingProducer)
        patched.append("kafka.KafkaProducer")
    return patched


@pytest.fixture
def recording_producer(monkeypatch: pytest.MonkeyPatch):
    RecordingProducer.instances.clear()
    mod = H.module("tmm.serve.events")
    patched = _patch_producer(monkeypatch, mod)
    if not patched:
        pytest.fail("tmm.serve.events does not expose KafkaProducer in a patchable way")
    return RecordingProducer


def kafka_publisher(settings=None):
    mod = H.module("tmm.serve.events")
    settings = settings or H.contract_settings(kafka_bootstrap=BOOTSTRAP,
                                               kafka_topic="tmm.clickstream.test",
                                               kafka_dlq_topic="tmm.clickstream.test.dlq")
    spec = {
        "settings": settings,
        "bootstrap": BOOTSTRAP, "bootstrap_servers": BOOTSTRAP, "servers": BOOTSTRAP,
        "kafka_bootstrap": BOOTSTRAP,
        "topic": settings.kafka_topic, "kafka_topic": settings.kafka_topic,
        "dlq_topic": settings.kafka_dlq_topic, "kafka_dlq_topic": settings.kafka_dlq_topic,
    }
    return H.construct(mod.KafkaEventPublisher, spec)


# ======================================================================================
# MemoryEventPublisher
# ======================================================================================
def test_memory_publisher_kind_health_and_publish():
    pub = H.construct(H.module("tmm.serve.events").MemoryEventPublisher, {"settings": H.contract_settings()})
    assert pub.kind == "memory"
    assert pub.healthy() is True
    assert pub.publish({"user_id": 1, "item_id": 2, "event_type": "click"}) is True
    pub.close()


def test_memory_publisher_is_thread_safe():
    pub = H.construct(H.module("tmm.serve.events").MemoryEventPublisher, {"settings": H.contract_settings()})
    errors: list[BaseException] = []

    def worker(offset: int) -> None:
        try:
            for i in range(100):
                pub.publish({"user_id": offset, "item_id": i, "event_type": "click"})
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # contract §3.2: MemoryEventPublisher "appends to a thread-safe list" -- find that list
    stored = None
    for val in vars(pub).values():
        if isinstance(val, (list, tuple)) or type(val).__name__ in ("deque", "Queue", "LifoQueue"):
            stored = val
            break
    assert stored is not None, f"no append-only container found on MemoryEventPublisher: {vars(pub)}"
    assert len(stored) == 800, f"MemoryEventPublisher dropped events under threads: {len(stored)}"


# ======================================================================================
# KafkaEventPublisher
# ======================================================================================
def test_kafka_publisher_kind(recording_producer):
    assert kafka_publisher().kind == "kafka"


def test_kafka_producer_settings_match_contract(recording_producer):
    settings = H.contract_settings(kafka_bootstrap=BOOTSTRAP)
    pub = kafka_publisher(settings)
    # force lazy producer construction if the publisher defers it
    pub.publish({"user_id": 1, "item_id": 2, "event_type": "click"})
    assert RecordingProducer.instances, "no KafkaProducer was constructed by the publisher"
    kwargs = RecordingProducer.instances[0].kwargs
    assert kwargs.get("acks") == "all", kwargs
    assert kwargs.get("retries") == 3, kwargs
    assert kwargs.get("linger_ms") == 5, kwargs
    # Contract §3.2 says `max_in_flight=1`; kafka-python names that setting
    # `max_in_flight_requests_per_connection`. Accept either spelling; both mean the same
    # thing, and the semantic requirement (ordering under retry) is what matters.
    in_flight = kwargs.get("max_in_flight", kwargs.get("max_in_flight_requests_per_connection"))
    assert in_flight == 1, kwargs
    # bootstrap must be wired from settings
    boot = kwargs.get("bootstrap_servers")
    assert boot in (BOOTSTRAP, [BOOTSTRAP], (BOOTSTRAP,)), kwargs


def test_kafka_partition_key_is_accepted_by_the_real_kafka_python_contract(recording_producer):
    """The key handed to ``KafkaProducer.send`` must be bytes unless a serializer is set.

    ``KafkaProducer.send()` raises ``AssertionError`` for a ``str`` key when no
    ``key_serializer`` is configured (verified against the live broker), i.e. a publisher that
    passes a str key can never deliver an event.  The contract's routing intent — one key per
    user — only holds if the key survives this boundary.
    """
    pub = kafka_publisher()
    assert pub.publish({"user_id": 4242, "item_id": 7, "event_type": "click"}) is True
    producer = RecordingProducer.instances[0]
    topic, value, key = producer.sent[0]
    serializer = producer.kwargs.get("key_serializer")
    assert isinstance(key, (bytes, bytearray)) or serializer is not None, (
        f"publish() handed key={key!r} (type {type(key).__name__}) to KafkaProducer with no "
        "key_serializer; KafkaProducer.send() rejects a str key with AssertionError, so every "
        "publish fails against a real broker"
    )
    if serializer is None:
        assert key == b"4242", f"unexpected bytes key {key!r}"


def test_kafka_publish_message_is_compact_json(recording_producer):
    settings = H.contract_settings(kafka_bootstrap=BOOTSTRAP, kafka_topic="tmm.clickstream.test")
    pub = kafka_publisher(settings)
    event = {"user_id": 4242, "item_id": 7, "event_type": "click"}
    assert pub.publish(event) is True
    producer = RecordingProducer.instances[0]
    assert producer.sent, "publish() did not hand the record to the producer"
    topic, value, key = producer.sent[0]
    assert topic == settings.kafka_topic
    raw = value.decode() if isinstance(value, (bytes, bytearray)) else value
    assert isinstance(raw, str) and "\n" not in raw, "message must be compact single-line JSON"
    assert json.loads(raw) == event
    # §3.2 writes the rule as str(user_id.encode()); the routing requirement is what matters,
    # and the key must still be a deterministic function of the user id.
    assert key is not None


def test_kafka_partition_key_is_stable_per_user_and_distinct_between_users(recording_producer):
    """The *purpose* of the §3.2 key rule: one partition per user, order preserved."""
    settings = H.contract_settings(kafka_bootstrap=BOOTSTRAP)
    pub = kafka_publisher(settings)
    for user_id in (11, 11, 12):
        pub.publish({"user_id": user_id, "item_id": 1, "event_type": "click"})
    keys = [k for _t, _v, k in RecordingProducer.instances[0].sent]
    assert keys[0] == keys[1], f"same user produced different keys: {keys}"
    assert keys[0] != keys[2], f"different users shared a key: {keys}"


def test_kafka_publish_returns_false_and_does_not_raise_on_broker_failure(recording_producer):
    from kafka.errors import KafkaError

    pub = kafka_publisher()
    pub.publish({"user_id": 1, "item_id": 2, "event_type": "click"})
    producer = RecordingProducer.instances[0]
    producer.fail_with = KafkaError("simulated broker failure")

    before, after, result = H.delta(
        "tmm_kafka_publish_total", {"result": "error"},
        lambda: pub.publish({"user_id": 1, "item_id": 3, "event_type": "click"}))
    assert result is False, "publish() must return False on broker failure, not raise"
    assert after > before, "tmm_kafka_publish_total{result='error'} was not incremented"


def test_kafka_publish_ok_metric(recording_producer):
    pub = kafka_publisher()
    before, after, result = H.delta(
        "tmm_kafka_publish_total", {"result": "ok"},
        lambda: pub.publish({"user_id": 1, "item_id": 2, "event_type": "click"}))
    assert result is True
    assert after > before, "tmm_kafka_publish_total{result='ok'} was not incremented"


def test_kafka_publisher_close_flushes_and_closes_producer(recording_producer):
    pub = kafka_publisher()
    pub.publish({"user_id": 1, "item_id": 2, "event_type": "click"})
    pub.close()
    assert RecordingProducer.instances[0].closed is True
