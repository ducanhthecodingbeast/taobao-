"""Contract §3.3 — the ingest worker ``consume_forever``.

A fake ``KafkaConsumer`` supplies the messages, so the whole decode / schema / dedup / DLQ /
circuit path is verified with no broker.  ``consume_forever`` runs on a **daemon thread with a
join timeout**, so a loop that refuses to honour ``max_messages`` fails the test instead of
hanging the suite.
"""

from __future__ import annotations

import json
import threading

import pytest

from tests.backend import _harness as H


class FakeMessage:
    def __init__(self, value, *, topic="tmm.clickstream.test", partition=0, offset=0, key=None):
        self.value = value
        self.key = key
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.headers: list[tuple[str, bytes]] = []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakeMessage(offset={self.offset}, value={self.value!r})"


class FakeConsumer:
    """Supports both ``for msg in consumer`` and ``consumer.poll()`` styles."""

    instances: list["FakeConsumer"] = []
    queue: list[list[FakeMessage]] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.messages: list[FakeMessage] = FakeConsumer.queue.pop(0) if FakeConsumer.queue else []
        self.committed: list[object] = []
        self.closed = False
        self.subscribed: list[str] | None = None
        self._i = 0
        FakeConsumer.instances.append(self)

    # -- consumption -------------------------------------------------------------------
    def __iter__(self):
        while self._i < len(self.messages):
            msg = self.messages[self._i]
            self._i += 1
            yield msg

    def poll(self, timeout_ms: int = 0, max_records: int | None = None):
        from kafka.structs import TopicPartition

        out: dict = {}
        while self._i < len(self.messages):
            msg = self.messages[self._i]
            self._i += 1
            out.setdefault(TopicPartition(msg.topic, msg.partition), []).append(msg)
            if max_records and sum(len(v) for v in out.values()) >= max_records:
                break
        return out

    # -- offsets -----------------------------------------------------------------------
    def commit(self, offsets=None):
        self.committed.append(offsets)

    def commit_async(self, offsets=None):
        self.committed.append(offsets)

    def seek(self, *a, **k):
        pass

    def assign(self, *a, **k):
        pass

    def subscribe(self, topics, **k):
        self.subscribed = list(topics) if isinstance(topics, (list, tuple)) else [topics]

    def close(self):
        self.closed = True

    def topics(self):
        return ["tmm.clickstream.test"]


def _patch_kafka_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_consumer`` imports KafkaConsumer inside the function, so patch the kafka package."""
    import kafka

    monkeypatch.setattr(kafka, "KafkaConsumer", FakeConsumer)


def _settings(**over):
    base = dict(kafka_bootstrap="127.0.0.1:9199", kafka_topic="tmm.clickstream.test",
                kafka_dlq_topic="tmm.clickstream.test.dlq")
    base.update(over)
    return H.contract_settings(**base)


def _run(messages, *, store=None, publisher=None, settings=None, max_messages=None,
         timeout_s: float = 25.0):
    """Run consume_forever on a daemon thread; never hang the suite."""
    mod = H.module("tmm.serve.consumer")
    FakeConsumer.instances.clear()
    FakeConsumer.queue = [list(messages)]
    store = store if store is not None else H.FakeSessionStore(backend="memory")
    publisher = publisher if publisher is not None else H.FakePublisher(kind="memory")
    settings = settings or _settings()
    result: dict = {}

    def target():
        try:
            result["value"] = mod.consume_forever(settings, store, publisher=publisher,
                                                  max_messages=max_messages)
        except BaseException as exc:  # pragma: no cover - reported below
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    assert not thread.is_alive(), (
        f"consume_forever did not return within {timeout_s}s "
        f"(max_messages={max_messages}, {len(messages)} messages) -- it is not honouring "
        f"max_messages or is blocking"
    )
    assert "error" not in result, f"consume_forever raised: {result.get('error')!r}"
    consumer = FakeConsumer.instances[0] if FakeConsumer.instances else None
    return consumer, result.get("value")


@pytest.fixture(autouse=True)
def _fast_patch(monkeypatch: pytest.MonkeyPatch):
    FakeConsumer.queue = []
    _patch_kafka_consumer(monkeypatch)


def _msg(payload: dict | bytes | str, offset: int = 0) -> FakeMessage:
    if isinstance(payload, (bytes, str)):
        raw = payload if isinstance(payload, bytes) else payload.encode()
    else:
        raw = json.dumps(payload).encode()
    return FakeMessage(raw, offset=offset)


def _valid(user_id=1, item_id=10, event_id="e1", offset=0, event_type="click"):
    return _msg({"user_id": user_id, "item_id": item_id, "event_type": event_type,
                 "event_id": event_id}, offset=offset)


# ======================================================================================
# happy path
# ======================================================================================
def test_consume_forever_returns_a_summary_dict():
    _consumer, summary = _run([_valid()], max_messages=1)
    assert isinstance(summary, dict)


def test_valid_event_is_appended_once_and_counted_accepted():
    store = H.FakeSessionStore(backend="memory")
    before, after, _ = H.delta(
        "tmm_events_ingested_total", {"result": "accepted"},
        lambda: _run([_valid(item_id=10, event_id="a1")], store=store, max_messages=1))
    assert store.get(1) == [10]
    assert after > before, "tmm_events_ingested_total{result='accepted'} not incremented"


def test_consumer_group_id_is_tmm_ingest():
    consumer, _ = _run([_valid()], max_messages=1)
    assert consumer.kwargs.get("group_id") == "tmm-ingest", consumer.kwargs


# ======================================================================================
# adversarial: duplicate replay
# ======================================================================================
def test_replayed_event_is_deduplicated_and_not_double_appended():
    store = H.FakeSessionStore(backend="memory")
    messages = [_valid(item_id=10, event_id="dup", offset=0),
                _valid(item_id=10, event_id="dup", offset=1)]
    before, after, _ = H.delta(
        "tmm_events_ingested_total", {"result": "duplicate"},
        lambda: _run(messages, store=store, max_messages=2))
    assert store.get(1) == [10], f"duplicate replay double-appended: {store.get(1)}"
    assert len(store.append_calls) == 2, store.append_calls
    assert after > before, "tmm_events_ingested_total{result='duplicate'} not incremented"


def test_events_without_event_id_are_not_deduplicated():
    store = H.FakeSessionStore(backend="memory")
    messages = [_msg({"user_id": 2, "item_id": 20, "event_type": "click"}, offset=i)
                for i in range(2)]
    _run(messages, store=store, max_messages=2)
    assert store.get(2) == [20, 20], store.get(2)


# ======================================================================================
# adversarial: poison message -> DLQ, partition not blocked
# ======================================================================================
def test_undecodable_message_goes_to_dlq_with_reason_decode_and_does_not_block_partition():
    store = H.FakeSessionStore(backend="memory")
    dlq = H.FakePublisher(kind="memory")
    messages = [FakeMessage(b"\xff\xfe not json at all", offset=0),
                _valid(item_id=99, event_id="after-poison", offset=1)]
    before, after, _ = H.delta(
        "tmm_dlq_total", {"reason": "decode"},
        lambda: _run(messages, store=store, publisher=dlq, max_messages=2))

    assert after > before, "tmm_dlq_total{reason='decode'} not incremented"
    reasons = [e.get("reason") for e in dlq.published]
    assert "decode" in reasons, f"decode failure did not reach the DLQ: {dlq.published}"
    # the partition kept moving: the message after the poison one was ingested
    assert store.get(1) == [99], f"poison message blocked the partition: session={store.get(1)}"


def test_poison_message_offset_is_committed():
    """Contract §3.3: commit the offset anyway so the partition advances."""
    store = H.FakeSessionStore(backend="memory")
    dlq = H.FakePublisher(kind="memory")
    consumer, _ = _run([FakeMessage(b"not-json", offset=0)], store=store, publisher=dlq,
                       max_messages=1)
    auto = bool(consumer.kwargs.get("enable_auto_commit"))
    assert consumer.committed or auto, (
        "the decode-poison offset was neither committed explicitly nor auto-committed; the "
        f"partition would replay it forever (consumer kwargs: {consumer.kwargs})"
    )


def test_schema_invalid_message_goes_to_dlq_with_reason_schema():
    store = H.FakeSessionStore(backend="memory")
    dlq = H.FakePublisher(kind="memory")
    bad = _msg({"user_id": "not-an-int", "item_id": 5, "event_type": "click",
                "event_id": "s1"}, offset=0)
    before, after, _ = H.delta(
        "tmm_dlq_total", {"reason": "schema"},
        lambda: _run([bad], store=store, publisher=dlq, max_messages=1))
    assert after > before, "tmm_dlq_total{reason='schema'} not incremented"
    assert any(e.get("reason") == "schema" for e in dlq.published), dlq.published
    assert store.append_calls == [], f"schema-invalid event reached the store: {store.append_calls}"


def test_missing_required_field_goes_to_dlq_with_reason_schema():
    dlq = H.FakePublisher(kind="memory")
    missing = _msg({"user_id": 1, "event_type": "click", "event_id": "m1"}, offset=0)
    _run([missing], publisher=dlq, max_messages=1)
    assert any(e.get("reason") == "schema" for e in dlq.published), dlq.published


def test_events_ingested_rejected_metric_counts_poison_messages():
    dlq = H.FakePublisher(kind="memory")
    before, after, _ = H.delta(
        "tmm_events_ingested_total", {"result": "rejected"},
        lambda: _run([FakeMessage(b"}", offset=0)], publisher=dlq, max_messages=1))
    assert after > before, "tmm_events_ingested_total{result='rejected'} not incremented"


# ======================================================================================
# adversarial: store failures -> circuit -> DLQ reason=store
# ======================================================================================
def test_store_failures_open_the_circuit_and_send_to_dlq():
    store = H.FakeSessionStore(backend="memory", fail_with=RuntimeError("redis down"))
    dlq = H.FakePublisher(kind="memory")
    messages = [_valid(item_id=i, event_id=f"c{i}", offset=i) for i in range(6)]
    before, after, _ = H.delta(
        "tmm_dlq_total", {"reason": "store"},
        lambda: _run(messages, store=store, publisher=dlq,
                     settings=_settings(circuit_fail_threshold=3), max_messages=6))
    assert after > before, "tmm_dlq_total{reason='store'} not incremented after store failures"
    assert any(e.get("reason") == "store" for e in dlq.published), dlq.published


def test_out_of_int64_range_id_goes_to_dlq_with_reason_schema():
    store = H.FakeSessionStore(backend="memory")
    dlq = H.FakePublisher(kind="memory")
    bad = _msg({"user_id": 1, "item_id": 2**70, "event_type": "click", "event_id": "r1"},
               offset=0)
    _run([bad], store=store, publisher=dlq, max_messages=1)
    assert any(e.get("reason") == "schema" for e in dlq.published), dlq.published
    assert store.append_calls == [], store.append_calls
