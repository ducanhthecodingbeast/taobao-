"""Event publishing (contract §3.2).

The request path hands events to Kafka when a broker is configured; otherwise events are
kept in a thread-safe in-process list so the app is fully functional with zero services.
``publish`` never raises: it returns ``False`` and bumps ``tmm_kafka_publish_total{error}``
so the caller can fall back to a direct session write.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Protocol

from .obs import DEFAULT_METRICS, Metrics, log_warning

#: Anonymised ids are arbitrary int64 values, negatives included. Anything outside that range
#: cannot be stored (Redis encodes ids as int64) or looked up, so it is rejected at the edge.
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1

try:  # kafka is optional at import time; a missing install must not break the API
    from kafka import KafkaProducer
except Exception:  # noqa: BLE001 - any import failure means "no broker available"
    KafkaProducer = None  # type: ignore[assignment,misc]

__all__ = [
    "EventPublisher",
    "KafkaEventPublisher",
    "MemoryEventPublisher",
    "build_dlq_publisher",
    "build_publisher",
]

logger = logging.getLogger("tmm.serve.events")


def _compact(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")


def _partition_key(event: dict[str, Any]) -> str:
    """One key per user so a user's events stay on one partition and in order.

    Contract §3.2 writes this as ``str(user_id.encode())``. ``user_id`` is an ``int`` on the
    wire, so the type-safe spelling of the same value is ``str(str(user_id).encode())`` --
    i.e. the Python string ``"b'4242'"``.
    """
    user_id = event.get("user_id")
    if user_id is None:
        return str(b"dlq")
    return str(str(user_id).encode("utf-8"))


class EventPublisher(Protocol):
    kind: str

    def publish(self, event: dict[str, Any]) -> bool: ...
    def healthy(self) -> bool: ...
    def close(self) -> None: ...


class MemoryEventPublisher:
    """In-process append-only event log (dev mode, DLQ fallback and tests)."""

    kind = "memory"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> bool:
        with self._lock:
            self._events.append(dict(event))
            if len(self._events) > 100_000:
                del self._events[: len(self._events) - 100_000]
        return True

    def healthy(self) -> bool:
        return True

    def close(self) -> None:
        pass

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)


class KafkaEventPublisher:
    """Thin kafka-python-ng producer wrapper with contract-mandated ordering settings."""

    kind = "kafka"

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        metrics: Metrics | None = None,
        send_timeout_s: float = 2.0,
        **producer_kwargs: Any,
    ) -> None:
        self.topic = topic
        self._metrics = metrics if metrics is not None else DEFAULT_METRICS
        self._send_timeout_s = float(send_timeout_s)
        self._healthy_ttl_s = 30.0
        self._unhealthy = False
        self._last_ok: float | None = None
        self._producer = self._build_producer(bootstrap_servers, producer_kwargs)

    @staticmethod
    def _build_producer(bootstrap_servers: str, producer_kwargs: dict[str, Any]) -> Any:
        """Construct the producer, tolerating the two spellings of ``max_in_flight``.

        Two real-broker quirks are handled here:

        * ``key_serializer=str.encode`` -- :func:`_partition_key` returns the contract-literal
          Python string ``"b'4242'"``; kafka-python-ng asserts unless the key is bytes, so the
          serializer turns it into ``b"b'4242'"`` while the value stays deterministic per user.
        * The contract names the setting ``max_in_flight``; kafka-python-ng's real config key
          is ``max_in_flight_requests_per_connection`` and it rejects unknown keys. We pass the
          contract spelling first and retry with the library spelling only when the class
          refuses it.
        """
        producer_cls = KafkaProducer
        if producer_cls is None:  # pragma: no cover - import guard
            raise RuntimeError("kafka-python-ng is not installed")
        base = dict(
            bootstrap_servers=bootstrap_servers,
            acks="all",
            retries=3,
            linger_ms=5,
            max_block_ms=1000,
            api_version_auto_timeout_ms=1000,
            key_serializer=str.encode,
            **producer_kwargs,
        )
        conf = dict(base)
        conf.setdefault("max_in_flight", 1)
        try:
            return producer_cls(**conf)
        except (TypeError, AssertionError):
            conf.pop("max_in_flight", None)
            conf["max_in_flight_requests_per_connection"] = 1
            return producer_cls(**conf)

    def publish(self, event: dict[str, Any]) -> bool:
        try:
            future = self._producer.send(
                self.topic, key=_partition_key(event), value=_compact(event))
            future.get(timeout=self._send_timeout_s)
        except Exception as exc:  # noqa: BLE001 - broker problems must not 500 the API
            self._unhealthy = True
            self._last_ok = None
            self._metrics.kafka_publish.labels(result="error").inc()
            log_warning(logger, "kafka_publish_failed", topic=self.topic,
                        error=type(exc).__name__)
            return False
        self._unhealthy = False
        self._last_ok = time.monotonic()
        self._metrics.kafka_publish.labels(result="ok").inc()
        return True

    def healthy(self) -> bool:
        """Is the broker usable?

        A recent successful publish is the strongest evidence and short-circuits the probe.
        Otherwise we ask the producer for the topic's partitions: that is a metadata call
        which raises when the broker is unreachable. ``bootstrap_connected()`` is deliberately
        NOT the primary signal -- with kafka-python it flips to ``False`` as soon as the client
        migrates to the advertised listener, while publishing keeps working.
        """
        if self._unhealthy:
            return False
        if self._last_ok is not None and time.monotonic() - self._last_ok <= self._healthy_ttl_s:
            return True
        probe = getattr(self._producer, "partitions_for", None)
        if not callable(probe):
            probe = getattr(self._producer, "partitions_for_topic", None)
        if callable(probe):
            try:
                partitions = probe(self.topic)
            except Exception:  # noqa: BLE001 - unreachable broker
                return False
            # ``None`` means "broker answered, topic not known locally" -> still reachable.
            return True if partitions is None else bool(partitions)
        connected = getattr(self._producer, "bootstrap_connected", None)
        if callable(connected):
            try:
                return bool(connected())
            except Exception:  # noqa: BLE001
                return False
        return True

    def close(self) -> None:
        try:
            self._producer.close(timeout=5)
        except Exception:  # noqa: BLE001
            pass


def build_publisher(settings: Any, metrics: Metrics | None = None) -> EventPublisher:
    """Kafka publisher when ``KAFKA_BOOTSTRAP`` is set and reachable, else memory."""
    if not settings.kafka_bootstrap:
        return MemoryEventPublisher()
    try:  # pragma: no cover - environment dependent
        return KafkaEventPublisher(
            settings.kafka_bootstrap, settings.kafka_topic, metrics=metrics)
    except Exception as exc:  # noqa: BLE001 - Kafka is optional
        log_warning(logger, "kafka_unavailable", bootstrap=settings.kafka_bootstrap,
                    error=type(exc).__name__)
        return MemoryEventPublisher()


def build_dlq_publisher(settings: Any, metrics: Metrics | None = None) -> EventPublisher:
    """Producer bound to the DLQ topic; memory when no broker is configured."""
    if not settings.kafka_bootstrap:
        return MemoryEventPublisher()
    try:  # pragma: no cover - environment dependent
        return KafkaEventPublisher(
            settings.kafka_bootstrap, settings.kafka_dlq_topic, metrics=metrics)
    except Exception as exc:  # noqa: BLE001
        log_warning(logger, "kafka_dlq_unavailable", bootstrap=settings.kafka_bootstrap,
                    error=type(exc).__name__)
        return MemoryEventPublisher()
