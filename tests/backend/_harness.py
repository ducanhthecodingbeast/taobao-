"""Contract-only test harness for the backend hardening work.

There is deliberately **no expected behaviour encoded here**.  This module only knows how to
*reach* the components that ``docs/BACKEND_CONTRACT.md`` names, plus enough parsing helpers to
read Prometheus text.  Every assertion lives in the ``test_*`` modules and is derived from the
frozen contract, never from the coder's implementation.

Guessing is contained and visible:

* :func:`construct` fills a constructor's required parameters **by name** from a caller-supplied
  synonym table.  If it cannot, it raises :class:`HarnessError` -- which surfaces as a test
  error, never as a silent pass.
* :func:`build_app` patches the app module's factories and *records what it patched*
  (``AppHarness.patched``), so a test can tell the difference between "the component under test
  behaved" and "my fake was never wired in".
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import socket
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SERVE = "tmm.serve"

ROOT = Path(__file__).resolve().parents[2]
ROOT_SRC = ROOT / "src"


class HarnessError(AssertionError):
    """Raised when a contract-named component cannot be reached as documented."""


# ======================================================================================
# settings (§1)
# ======================================================================================
#: The exact defaults the frozen contract §1 prescribes.
CONTRACT_SETTINGS_DEFAULTS: dict[str, Any] = {
    "api_keys": (),
    "auth_required": False,
    "rate_limit_rps": 20.0,
    "rate_limit_burst": 40,
    "redis_url": None,
    "kafka_bootstrap": None,
    "kafka_topic": "tmm.clickstream",
    "kafka_dlq_topic": "tmm.clickstream.dlq",
    "idempotency_ttl_s": 3600,
    "session_max": 50,
    "cache_ttl_s": 5,
    "stage_timeout_s": 0.5,
    "circuit_fail_threshold": 5,
    "circuit_reset_s": 10.0,
}


def module(dotted: str) -> Any:
    return importlib.import_module(dotted)


def settings_cls() -> type:
    return getattr(module(f"{SERVE}.settings"), "Settings")


def contract_settings(**overrides: Any) -> Any:
    """Build a ``Settings`` carrying the contract's field values."""
    cls = settings_cls()
    fields = {f.name: f for f in dataclasses.fields(cls)}
    missing_fields = [n for n in CONTRACT_SETTINGS_DEFAULTS if n not in fields]
    if missing_fields:
        raise HarnessError(f"Settings is missing contract fields: {missing_fields}")
    required_extra = [
        n for n, f in fields.items()
        if n not in CONTRACT_SETTINGS_DEFAULTS
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    ]
    if required_extra:
        raise HarnessError(
            f"Settings added required fields the contract does not define: {required_extra}"
        )
    kw = dict(CONTRACT_SETTINGS_DEFAULTS)
    kw.update(overrides)
    return cls(**kw)


def apply_env(monkeypatch: Any, settings: Any) -> None:
    """Mirror a Settings object into the environment (belt-and-braces for code that reads env)."""
    env = {
        "TMM_API_KEYS": ",".join(settings.api_keys) if settings.api_keys else "",
        "TMM_AUTH_REQUIRED": "1" if settings.auth_required else "0",
        "TMM_RATE_LIMIT_RPS": repr(settings.rate_limit_rps),
        "TMM_RATE_LIMIT_BURST": str(settings.rate_limit_burst),
        "TMM_KAFKA_TOPIC": settings.kafka_topic,
        "TMM_KAFKA_DLQ_TOPIC": settings.kafka_dlq_topic,
        "TMM_IDEMPOTENCY_TTL_S": str(settings.idempotency_ttl_s),
        "TMM_SESSION_MAX": str(settings.session_max),
        "TMM_CACHE_TTL_S": str(settings.cache_ttl_s),
        "TMM_STAGE_TIMEOUT_S": repr(settings.stage_timeout_s),
        "TMM_CIRCUIT_FAILS": str(settings.circuit_fail_threshold),
        "TMM_CIRCUIT_RESET_S": repr(settings.circuit_reset_s),
    }
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    for key, val in (("REDIS_URL", settings.redis_url), ("KAFKA_BOOTSTRAP", settings.kafka_bootstrap)):
        if val is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, val)


# ======================================================================================
# generic construction
# ======================================================================================
def construct(cls: type, spec: dict[str, Any], fallback_args: Sequence[Any] = ()) -> Any:
    """Instantiate ``cls``, filling required parameters by name from ``spec``.

    ``spec`` may contain several synonyms for the same concept; only parameters the real
    signature declares are used.  Falls back to ``fallback_args`` positionally, then raises.
    """
    try:
        params = [
            p for p in list(inspect.signature(cls.__init__).parameters.values())[1:]
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
    except (TypeError, ValueError):  # not a Python-level class
        if fallback_args:
            return cls(*fallback_args)
        raise HarnessError(f"cannot introspect {cls!r}") from None

    required = [p for p in params if p.default is inspect.Parameter.empty and p.kind != p.VAR_KEYWORD]
    unknown = [p.name for p in required if p.name not in spec]
    if not unknown:
        kwargs = {p.name: spec[p.name] for p in params if p.name in spec}
        return cls(**kwargs)
    if fallback_args:
        try:
            return cls(*fallback_args)
        except TypeError:
            pass
    raise HarnessError(
        f"cannot construct {getattr(cls, '__name__', cls)} as documented in the contract: "
        f"required parameter(s) {unknown} not covered by {sorted(spec)}"
    )


class _Factory:
    """Stand-in for a class: always returns the same pre-built object."""

    def __init__(self, obj: Any):
        self._obj = obj

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._obj


# ======================================================================================
# contract-derived fakes (test doubles, not re-implementations)
# ======================================================================================
class FakeSessionStore:
    """In-memory ``SessionStore`` with *controllable* health and failure injection."""

    def __init__(self, backend: str = "redis", session_max: int = 50,
                 ping_ok: bool = True, fail_with: Exception | None = None):
        self.backend = backend
        self.session_max = session_max
        self.ping_ok = ping_ok
        self.fail_with = fail_with
        self._sessions: dict[int, list[int]] = {}
        self._seen: set[str] = set()
        self.append_calls: list[tuple[int, int, str | None]] = []
        self.ping_calls = 0
        self.closed = False

    # -- SessionStore protocol ---------------------------------------------------------
    def append(self, user_id: int, item_id: int, event_id: str | None = None) -> tuple[int, bool]:
        self.append_calls.append((user_id, item_id, event_id))
        if self.fail_with is not None:
            raise self.fail_with
        if event_id is not None:
            if event_id in self._seen:
                return len(self._sessions.get(user_id, [])), False
            self._seen.add(event_id)
        lst = self._sessions.setdefault(user_id, [])
        lst.insert(0, int(item_id))          # newest first, like LPUSH
        del lst[self.session_max:]
        return len(lst), True

    def get(self, user_id: int) -> list[int]:
        return list(self._sessions.get(user_id, []))

    def is_seen(self, event_id: str) -> bool:
        return event_id in self._seen

    def ping(self) -> bool:
        self.ping_calls += 1
        return self.ping_ok

    # -- teardown ----------------------------------------------------------------------
    def close(self) -> None:
        self.closed = True


class FakePublisher:
    """``EventPublisher`` double that records publishes without a broker."""

    def __init__(self, kind: str = "kafka", healthy: bool = True,
                 publish_result: bool | None = None):
        self.kind = kind
        self._healthy = healthy
        self.publish_result = publish_result
        self.published: list[dict] = []
        self.closed = False

    def publish(self, event: dict) -> bool:
        self.published.append(event)
        if self.publish_result is not None:
            return self.publish_result
        return self._healthy

    def healthy(self) -> bool:
        return self._healthy

    def close(self) -> None:
        self.closed = True


class StubRecommender:
    """Cheap stand-in for :class:`tmm.serve.app.Recommender` (the real one costs ~GBs).

    Implements the same surface ``/recommend`` and ``/health`` use: ``user_vector``, ``search``,
    ``sample_item_ids`` plus the health attributes.  It returns *fixed* results, so HTTP-layer
    tests assert on status/metrics/plumbing rather than on model quality.
    """

    def __init__(self, n_items: int = 8, dim: int = 4, version: str = "v1-test"):
        self.n_items = n_items
        self.dim = dim
        self.version = version
        self.index_bytes = 1024
        self.emb_mb = 0.1
        self.onnx_session = None
        self.tower = None
        self.item_ids = list(range(1, n_items + 1))
        self.calls: list[tuple[str, object]] = []

    def sample_item_ids(self, n: int = 200, seed: int = 0) -> list[int]:
        self.calls.append(("sample_item_ids", n))
        return self.item_ids[:n]

    def user_vector(self, session: list[int]):
        import numpy as np

        self.calls.append(("user_vector", list(session)))
        return np.ones(self.dim, dtype="float32")

    def search(self, user_vec, session: list[int], k: int = 10,
               exclude_seen: bool = True) -> list[dict]:
        self.calls.append(("search", (list(session), k)))
        seen = set(session) if exclude_seen else set()
        out = [{"item_id": i, "score": 1.0, "cold_start": False}
               for i in self.item_ids if i not in seen]
        return out[:k]

    def recommend(self, session: list[int], k: int = 10, exclude_seen: bool = True) -> list[dict]:
        return self.search(self.user_vector(session), session, k=k, exclude_seen=exclude_seen)


# ======================================================================================
# app construction (§4)
# ======================================================================================
_STORE_NAMES = ("RedisSessionStore", "MemorySessionStore", "SessionStore",
                "build_session_store", "make_session_store", "get_session_store",
                "build_store", "make_store")
_PUBLISHER_NAMES = ("KafkaEventPublisher", "MemoryEventPublisher", "EventPublisher",
                    "build_publisher", "make_publisher", "build_event_publisher",
                    "get_publisher")
_RECOMMENDER_NAMES = ("Recommender", "build_recommender", "make_recommender", "load_recommender")


@dataclasses.dataclass
class AppHarness:
    app: Any
    module: Any
    settings: Any
    store: Any = None
    publisher: Any = None
    recommender: Any = None
    patched: tuple[str, ...] = ()

    @property
    def patched_names(self) -> set[str]:
        return set(self.patched)


def build_app(monkeypatch: Any, *, settings: Any | None = None, store: Any = None,
              publisher: Any = None, recommender: Any = "stub",
              create_kwargs: dict[str, Any] | None = None) -> AppHarness:
    """Create the FastAPI app with the contract's dependencies injected.

    ``store``/``publisher`` are injected by replacing the app module's factories, so the
    *app's own* mode logic runs against a double.  ``recommender="stub"`` avoids loading the
    real FAISS/ONNX stack; pass ``None`` to keep the real one.
    """
    mod = module(f"{SERVE}.app")
    settings = settings if settings is not None else contract_settings()
    patched: list[str] = []

    if hasattr(mod, "load_settings"):
        monkeypatch.setattr(mod, "load_settings", lambda: settings)
        patched.append("load_settings")
    elif hasattr(mod, "get_settings"):
        monkeypatch.setattr(mod, "get_settings", lambda: settings)
        patched.append("get_settings")
    else:
        raise HarnessError("tmm.serve.app exposes neither load_settings nor get_settings")
    apply_env(monkeypatch, settings)

    if store is not None:
        hits = [n for n in _STORE_NAMES if isinstance(getattr(mod, n, None), type)]
        if not hits:
            raise HarnessError(f"cannot inject a SessionStore: none of {_STORE_NAMES} exist in app")
        for name in hits:
            monkeypatch.setattr(mod, name, _Factory(store))
            patched.append(name)

    if publisher is not None:
        hits = [n for n in _PUBLISHER_NAMES if isinstance(getattr(mod, n, None), type)]
        if not hits:
            raise HarnessError(f"cannot inject an EventPublisher: none of {_PUBLISHER_NAMES} exist")
        for name in hits:
            monkeypatch.setattr(mod, name, _Factory(publisher))
            patched.append(name)

    rec = recommender
    if isinstance(rec, str) and rec == "stub":
        rec = StubRecommender()
    if rec is not None:
        hits = [n for n in _RECOMMENDER_NAMES if isinstance(getattr(mod, n, None), type)]
        if not hits:
            raise HarnessError(f"cannot inject a Recommender: none of {_RECOMMENDER_NAMES} exist")
        for name in hits:
            monkeypatch.setattr(mod, name, _Factory(rec))
            patched.append(name)

    kwargs = dict(create_kwargs or {})
    sig = inspect.signature(mod.create_app)
    for key, val in (("settings", settings), ("store", store), ("publisher", publisher),
                     ("recommender", rec)):
        if key in sig.parameters and val is not None and key not in kwargs:
            kwargs[key] = val
    try:
        app = mod.create_app(**kwargs)
    except TypeError:
        app = mod.create_app()
    return AppHarness(app=app, module=mod, settings=settings, store=store,
                      publisher=publisher, recommender=rec, patched=tuple(patched))


# ======================================================================================
# Prometheus text helpers
# ======================================================================================
def parse_exposition(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus text exposition into ``{metric_name: [(labels, value), ...]}``."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name_and_labels, _, value = line.rpartition(" ")
        name, labels = _split_name_labels(name_and_labels)
        try:
            val = float(value)
        except ValueError:
            continue
        out.setdefault(name, []).append((labels, val))
    return out


def _split_name_labels(chunk: str) -> tuple[str, dict[str, str]]:
    if "{" not in chunk:
        return chunk, {}
    name, _, rest = chunk.partition("{")
    labels: dict[str, str] = {}
    for part in rest.rstrip("}").split(","):
        if not part:
            continue
        k, _, v = part.partition("=")
        labels[k.strip()] = v.strip().strip('"')
    return name, labels


def counter_value(exposition: dict[str, list[tuple[dict[str, str], float]]], name: str,
                  labels: dict[str, str] | None = None) -> float:
    total = 0.0
    for lbl, val in exposition.get(name, []):
        if labels is None or all(lbl.get(k) == v for k, v in labels.items()):
            total += val
    return total


_METRIC_MODULES = (f"{SERVE}.obs", f"{SERVE}.app", f"{SERVE}.consumer", f"{SERVE}.events",
                   f"{SERVE}.session", f"{SERVE}.ratelimit", f"{SERVE}.resilience")


def _metric_objects() -> list[Any]:
    from prometheus_client.metrics import MetricWrapperBase

    seen: list[Any] = []
    for mod_name in _METRIC_MODULES:
        try:
            mod = module(mod_name)
        except Exception:
            continue
        for val in vars(mod).values():
            if isinstance(val, MetricWrapperBase):
                seen.append(val)
    return seen


def metric_names(metric: Any) -> set[str]:
    """Names this metric object is exposed under.

    ``prometheus_client`` strips the ``_total`` suffix from :class:`Counter` ``_name``, so both
    spellings have to be accepted when matching a contract metric name.
    """
    names = {metric._name}
    if getattr(metric, "_type", "") == "counter":
        names.add(f"{metric._name}_total")
    return names


def all_metric_names() -> set[str]:
    """Every Prometheus metric name exposed by the ``tmm.serve`` modules."""
    out: set[str] = set()
    for metric in _metric_objects():
        out |= metric_names(metric)
    return out


def find_metric(name: str) -> Any:
    for metric in _metric_objects():
        if name in metric_names(metric):
            return metric
    return None


def registry_metric_names(registry: Any) -> set[str]:
    """Names registered on a ``CollectorRegistry`` (including families with no samples yet)."""
    names: set[str] = set()
    collectors = getattr(registry, "_names_to_collectors", {})
    for key, collector in collectors.items():
        names.add(key)
        for extra in getattr(collector, "_names", ()) or ():
            names.add(extra)
    return names


def sample_value(name: str, labels: dict[str, str] | None = None,
                 suffix: str = "") -> float | None:
    """Current value of a metric sample, or None if the sample does not exist.

    ``name`` is the exposed name (e.g. ``tmm_dlq_total``); ``suffix`` selects e.g. ``_count``
    on a histogram.
    """
    metric = find_metric(name)
    if metric is None:
        return None
    wanted = f"{name}{suffix}"
    value: float | None = None
    for family in metric.collect():
        for sample in family.samples:
            if sample.name != wanted:
                continue
            if labels is not None and not all(sample.labels.get(k) == v for k, v in labels.items()):
                continue
            value = (value or 0.0) + float(sample.value)
    return value


def delta(name: str, labels: dict[str, str] | None = None, fn: Callable[[], Any] | None = None,
          suffix: str = "") -> tuple[float, float, Any]:
    """Run ``fn`` and return ``(before, after, result)`` for a metric sample."""
    before = sample_value(name, labels, suffix)
    result = fn() if fn is not None else None
    after = sample_value(name, labels, suffix)
    return (0.0 if before is None else before, 0.0 if after is None else after, result)


REQUIRED_METRICS: dict[str, tuple[str, set[str]]] = {
    # name -> (type, label names) from contract §3.6
    "tmm_http_requests_total": ("counter", {"endpoint", "status"}),
    "tmm_http_request_duration_seconds": ("histogram", {"endpoint"}),
    "tmm_stage_duration_seconds": ("histogram", {"stage"}),
    "tmm_events_ingested_total": ("counter", {"result"}),
    "tmm_kafka_publish_total": ("counter", {"result"}),
    "tmm_dlq_total": ("counter", {"reason"}),
    "tmm_circuit_state": ("gauge", {"backend"}),
    "tmm_rate_limited_total": ("counter", {"scope"}),
    "tmm_ready": ("gauge", set()),
}


# ======================================================================================
# service probing (integration layer)
# ======================================================================================
def tcp_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def redis_url() -> str:
    import os
    return os.environ.get("TMM_TEST_REDIS_URL", "redis://127.0.0.1:6399/0")


def kafka_bootstrap() -> str:
    import os
    return os.environ.get("TMM_TEST_KAFKA_BOOTSTRAP", "127.0.0.1:9199")


def wait_until(pred: Callable[[], bool], timeout_s: float = 15.0, interval_s: float = 0.2) -> bool:
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


# ======================================================================================
# raw ASGI invocation (needed to send a scope with no client IP at all)
# ======================================================================================
_MISSING = object()


async def asgi_call(app: Any, method: str, path: str, *, headers: Iterable[tuple[bytes, bytes]] = (),
                    json_body: Any = _MISSING, query_string: bytes = b"",
                    client: Any = _MISSING) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    """Call an ASGI app directly so the test controls ``scope['client']`` exactly."""
    import json as _json

    body = b"" if json_body is _MISSING else _json.dumps(json_body).encode()
    hdrs = list(headers)
    if json_body is not _MISSING:
        hdrs.append((b"content-type", b"application/json"))
    if body:
        hdrs.append((b"content-length", str(len(body)).encode()))

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query_string,
        "root_path": "",
        "headers": hdrs,
        "server": ("testserver", 80),
    }
    if client is not _MISSING:
        scope["client"] = client

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if not getattr(receive, "_done", False):
            receive._done = True  # type: ignore[attr-defined]
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)

    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), 0)
    out_headers = next((m.get("headers", []) for m in sent if m["type"] == "http.response.start"), [])
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, list(out_headers), payload
