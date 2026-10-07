"""Observability: JSON logging, Prometheus metrics and per-stage latency (contract §3.6).

Design notes
------------
* Stdlib ``logging`` plus a small JSON formatter -- no ``structlog``.
* Collectors are registered on a **dedicated** :class:`CollectorRegistry` owned by this
  module, never on ``prometheus_client.REGISTRY``. Re-importing the module therefore builds a
  brand-new registry and can never raise ``Duplicated timeseries``; ``build_metrics()`` reuses
  that one registry so every component in a process observes the same samples.
* The nine contract metrics are also exposed as module-level objects
  (``HTTP_REQUESTS``, ``DLQ_TOTAL``, ...) so they can be inspected directly without scraping
  ``/metrics``.
* :class:`StageTimer` takes an optional ``metrics`` object; without one it records into the
  process-wide defaults, so the consumer process gets stage/ingest metrics for free.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any, Literal

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

__all__ = [
    "CONTENT_TYPE_LATEST",
    "DEFAULT_METRICS",
    "DEFAULT_REGISTRY",
    "Metrics",
    "StageTimer",
    "build_metrics",
    "configure_logging",
    "generate_latest",
    "log_event",
    "log_warning",
]

#: Exact stage names required by the contract for the recommend path.
STAGES: tuple[str, ...] = ("decode", "session_read", "tower", "search", "total")

_SECRET_HINTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
    "x-api-key",
)


def redact(field: str, value: Any) -> Any:
    """Replace obviously secret-looking values with ``***`` before they hit a log line."""
    lowered = field.lower()
    if any(hint in lowered for hint in _SECRET_HINTS):
        return "***"
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line; exception text is newline-escaped so lines stay atomic."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", None) or record.getMessage(),
        }
        fields = getattr(record, "tmm_fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if key in payload:
                    continue
                payload[key] = redact(str(key), value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info).replace("\n", "\\n")
        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return json.dumps({"ts": payload["ts"], "level": payload["level"],
                               "event": str(payload["event"])})


_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    global _configured
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = JsonFormatter()
    if not any(getattr(h, "_tmm_json", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._tmm_json = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    for existing in root.handlers:
        if getattr(existing, "_tmm_json", False):
            existing.setFormatter(formatter)
    _configured = True


def _emit(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    scrubbed = {key: redact(key, value) for key, value in fields.items()}
    logger.log(level, event, extra={"event": event, "tmm_fields": scrubbed})


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit exactly one JSON log line for ``event`` with ``fields`` as structured data."""
    _emit(logger, logging.INFO, event, **fields)


def log_warning(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Same as :func:`log_event` but at WARNING level."""
    _emit(logger, logging.WARNING, event, **fields)


class Metrics:
    """The nine contract metrics, bound to one registry."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        reg = self.registry
        self.http_requests = Counter(
            "tmm_http_requests_total", "HTTP requests by endpoint and status",
            ["endpoint", "status"], registry=reg)
        self.http_duration = Histogram(
            "tmm_http_request_duration_seconds", "HTTP request latency",
            ["endpoint"], registry=reg)
        self.stage_duration = Histogram(
            "tmm_stage_duration_seconds", "Recommend-path stage latency",
            ["stage"], registry=reg)
        self.events_ingested = Counter(
            "tmm_events_ingested_total", "Ingested clickstream events by result",
            ["result"], registry=reg)
        self.kafka_publish = Counter(
            "tmm_kafka_publish_total", "Kafka publish attempts by result",
            ["result"], registry=reg)
        self.dlq = Counter(
            "tmm_dlq_total", "Dead-lettered messages by reason",
            ["reason"], registry=reg)
        self.circuit_state = Gauge(
            "tmm_circuit_state", "Circuit state (0 closed, 1 half-open, 2 open)",
            ["backend"], registry=reg)
        self.rate_limited = Counter(
            "tmm_rate_limited_total", "Rate-limited requests by scope",
            ["scope"], registry=reg)
        self.ready = Gauge(
            "tmm_ready", "1 when the replica is ready to take traffic", registry=reg)

    def render(self) -> bytes:
        return generate_latest(self.registry)


#: Dedicated process registry -- deliberately not ``prometheus_client.REGISTRY``.
DEFAULT_REGISTRY = CollectorRegistry()
DEFAULT_METRICS = Metrics(DEFAULT_REGISTRY)

# Module-level handles so any component (and the contract test harness) can read samples.
HTTP_REQUESTS = DEFAULT_METRICS.http_requests
HTTP_DURATION = DEFAULT_METRICS.http_duration
STAGE_DURATION = DEFAULT_METRICS.stage_duration
EVENTS_INGESTED = DEFAULT_METRICS.events_ingested
KAFKA_PUBLISH = DEFAULT_METRICS.kafka_publish
DLQ_TOTAL = DEFAULT_METRICS.dlq
CIRCUIT_STATE = DEFAULT_METRICS.circuit_state
RATE_LIMITED = DEFAULT_METRICS.rate_limited
READY = DEFAULT_METRICS.ready


def build_metrics(registry: CollectorRegistry | None = None) -> Metrics:
    """Return the shared metrics bundle, or build a private one on ``registry``.

    Passing ``registry`` rebinds the module-level handles so both ``/metrics`` and direct
    metric lookups see the same collectors.
    """
    global CIRCUIT_STATE, DEFAULT_METRICS, DLQ_TOTAL, EVENTS_INGESTED
    global HTTP_DURATION, HTTP_REQUESTS, KAFKA_PUBLISH, RATE_LIMITED, READY, STAGE_DURATION
    if registry is None:
        return DEFAULT_METRICS
    metrics = Metrics(registry)
    DEFAULT_METRICS = metrics
    HTTP_REQUESTS = metrics.http_requests
    HTTP_DURATION = metrics.http_duration
    STAGE_DURATION = metrics.stage_duration
    EVENTS_INGESTED = metrics.events_ingested
    KAFKA_PUBLISH = metrics.kafka_publish
    DLQ_TOTAL = metrics.dlq
    CIRCUIT_STATE = metrics.circuit_state
    RATE_LIMITED = metrics.rate_limited
    READY = metrics.ready
    return metrics


class StageTimer:
    """Context manager observing one stage into ``tmm_stage_duration_seconds``.

    ``with StageTimer("tower"):`` records into the process defaults; pass ``metrics=`` to
    bind it to a specific app instance.
    """

    __slots__ = ("_t0", "duration", "metrics", "stage")

    def __init__(self, stage: str, metrics: Metrics | None = None) -> None:
        self.stage = stage
        self.metrics = metrics
        self.duration = 0.0
        self._t0 = 0.0

    def __enter__(self) -> StageTimer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        self.duration = time.perf_counter() - self._t0
        bundle = self.metrics if self.metrics is not None else DEFAULT_METRICS
        bundle.stage_duration.labels(stage=self.stage).observe(self.duration)
        return False
