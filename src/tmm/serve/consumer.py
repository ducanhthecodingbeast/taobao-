"""The ingest worker (contract §3.3).

Runs as its own process -- ``python -m tmm.serve.consumer`` -- never inside the API. It
consumes ``tmm.clickstream``, validates and deduplicates via the session store, and routes
poison messages to the DLQ while committing past them so one bad record cannot block a
partition.

``process_message`` is deliberately a standalone function: unit tests exercise the
decode/schema/dedup rules without a broker, while ``consume_forever`` owns the Kafka loop,
offset commits and SIGTERM handling.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from typing import Any

from .events import EventPublisher, build_dlq_publisher
from .obs import DEFAULT_METRICS, Metrics, configure_logging, log_event, log_warning
from .session import SessionStore, build_session_store
from .settings import Settings, load_settings

__all__ = [
    "GROUP_ID",
    "build_consumer",
    "consume_forever",
    "main",
    "process_message",
    "process_message_detail",
    "validate_event",
]

logger = logging.getLogger("tmm.serve.consumer")

GROUP_ID = "tmm-ingest"
POLL_TIMEOUT_MS = 500
REQUIRED_FIELDS: tuple[str, ...] = ("user_id", "item_id", "event_type")
DLQ_REASONS: tuple[str, ...] = ("decode", "schema", "store")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_event(event: Any) -> str | None:
    """Return a human-readable reason when the event violates the schema, else ``None``."""
    if not isinstance(event, dict):
        return "message is not a JSON object"
    missing = [field for field in REQUIRED_FIELDS if field not in event]
    if missing:
        return f"missing required field(s): {','.join(missing)}"
    if not _is_int(event["user_id"]):
        return "user_id must be an int"
    if not _is_int(event["item_id"]):
        return "item_id must be an int"
    if not isinstance(event["event_type"], str):
        return "event_type must be a str"
    return None


def to_dlq(
    dlq: EventPublisher | None,
    raw: Any,
    reason: str,
    error: str,
    metrics: Metrics | None = None,
) -> None:
    """Best-effort dead-letter write plus the ``tmm_dlq_total{reason}`` counter."""
    bundle = metrics if metrics is not None else DEFAULT_METRICS
    bundle.dlq.labels(reason=reason).inc()
    if dlq is None:
        return
    payload = {
        "reason": reason,
        "error": error,
        "raw": raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw),
    }
    try:
        dlq.publish(payload)
    except Exception as exc:  # noqa: BLE001 - the DLQ must never kill the worker
        log_warning(logger, "dlq_publish_failed", reason=reason, error=type(exc).__name__)


def process_message_detail(
    raw: Any,
    store: SessionStore,
    dlq: EventPublisher | None = None,
    metrics: Metrics | None = None,
) -> tuple[str, str | None]:
    """Handle one raw message -> ``(outcome, dlq_reason)``.

    ``outcome`` is ``accepted``/``duplicate``/``rejected``; ``dlq_reason`` is set only when
    the message was dead-lettered. Store exceptions propagate so the caller can retry and,
    past the threshold, DLQ with ``reason="store"``.
    """
    bundle = metrics if metrics is not None else DEFAULT_METRICS
    try:
        event = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - poison message
        to_dlq(dlq, raw, "decode", f"{type(exc).__name__}: {exc}", bundle)
        bundle.events_ingested.labels(result="rejected").inc()
        return "rejected", "decode"

    reason = validate_event(event)
    if reason is not None:
        to_dlq(dlq, raw, "schema", reason, bundle)
        bundle.events_ingested.labels(result="rejected").inc()
        return "rejected", "schema"

    _size, accepted = store.append(
        int(event["user_id"]), int(event["item_id"]), event_id=event.get("event_id"))
    if accepted:
        bundle.events_ingested.labels(result="accepted").inc()
        return "accepted", None
    bundle.events_ingested.labels(result="duplicate").inc()
    return "duplicate", None


def process_message(
    raw: Any,
    store: SessionStore,
    dlq: EventPublisher | None = None,
    metrics: Metrics | None = None,
) -> str:
    """``process_message_detail`` reduced to the outcome string."""
    return process_message_detail(raw, store, dlq, metrics)[0]


def build_consumer(settings: Settings) -> Any:
    """Create the kafka-python-ng consumer (manual commits, group ``tmm-ingest``)."""
    from kafka import KafkaConsumer

    return KafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=None,
    )


def _install_signal_handlers(stop: threading.Event) -> dict[Any, Any]:
    previous: dict[Any, Any] = {}

    def handler(_signum: int, _frame: Any) -> None:
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread (tests)
            pass
    return previous


def _restore_signal_handlers(previous: dict[Any, Any]) -> None:
    for sig, old in previous.items():
        try:
            signal.signal(sig, old)
        except (ValueError, OSError, TypeError):  # pragma: no cover
            pass


def _new_summary() -> dict[str, Any]:
    return {
        "processed": 0, "accepted": 0, "duplicate": 0, "rejected": 0,
        "store_failures": 0,
        "dlq": {reason: 0 for reason in DLQ_REASONS},
        "stopped": "eof",
    }


def consume_forever(
    settings: Settings,
    store: SessionStore,
    publisher: EventPublisher | None = None,
    max_messages: int | None = None,
) -> dict[str, Any]:
    """Blocking consume loop. Returns a summary dict (also when ``max_messages`` is hit)."""
    metrics = DEFAULT_METRICS
    stop = threading.Event()
    previous = _install_signal_handlers(stop)
    owned_dlq = publisher is None
    dlq: EventPublisher | None = (
        publisher if publisher is not None else build_dlq_publisher(settings, metrics))
    summary = _new_summary()
    limit = int(max_messages) if max_messages else None
    threshold = max(1, int(settings.circuit_fail_threshold))
    consecutive_store_failures = 0
    consumer: Any = None
    try:
        consumer = build_consumer(settings)
        log_event(logger, "consumer_started", group=GROUP_ID, topic=settings.kafka_topic)
        while not stop.is_set():
            batches = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)
            if not batches:
                continue
            done = False
            for messages in batches.values():
                for message in messages:
                    consecutive_store_failures = _handle_with_retry(
                        message.value, store, dlq, metrics, summary,
                        consecutive_store_failures, threshold)
                    summary["processed"] += 1
                    # Commit past the message even when it was poison: a DLQ entry must
                    # not stall the partition.
                    consumer.commit()
                    if stop.is_set():
                        summary["stopped"] = "sigterm"
                        done = True
                        break
                    if limit is not None and summary["processed"] >= limit:
                        summary["stopped"] = "max_messages"
                        done = True
                        break
                if done:
                    break
            if done:
                break
    finally:
        _restore_signal_handlers(previous)
        if consumer is not None:
            try:
                consumer.close()
            except Exception:  # noqa: BLE001
                pass
        if owned_dlq and dlq is not None:
            dlq.close()
    return summary


def _handle_with_retry(
    raw: Any,
    store: SessionStore,
    dlq: EventPublisher | None,
    metrics: Metrics,
    summary: dict[str, Any],
    consecutive_store_failures: int,
    threshold: int,
) -> int:
    """Process one message, retrying store failures; returns consecutive failure count."""
    for attempt in range(1, threshold + 1):
        try:
            outcome, dlq_reason = process_message_detail(raw, store, dlq, metrics)
        except Exception as exc:  # noqa: BLE001 - store problem, not a schema problem
            consecutive_store_failures += 1
            summary["store_failures"] += 1
            if consecutive_store_failures >= threshold:
                to_dlq(dlq, raw, "store", f"{type(exc).__name__}: {exc}", metrics)
                summary["dlq"]["store"] += 1
                summary["rejected"] += 1
                return 0
            time.sleep(min(0.05 * attempt, 0.25))
            continue
        summary[outcome] = summary.get(outcome, 0) + 1
        if dlq_reason is not None:
            summary["dlq"][dlq_reason] += 1
        return 0
    return consecutive_store_failures


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m tmm.serve.consumer``."""
    configure_logging(os.environ.get("TMM_LOG_LEVEL", "INFO"))
    settings = load_settings()
    if not settings.kafka_bootstrap:
        log_warning(logger, "consumer_no_broker", hint="set KAFKA_BOOTSTRAP")
        return 2
    store = build_session_store(settings)
    try:
        summary = consume_forever(settings, store)
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        log_warning(logger, "consumer_crashed", error=type(exc).__name__, detail=str(exc))
        return 1
    log_event(logger, "consumer_stopped", **summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
