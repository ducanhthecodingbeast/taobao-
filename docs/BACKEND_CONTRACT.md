# Backend Interface Contract (FROZEN)

**Status:** frozen by the Lead. Do not change a signature in here without telling the Lead —
the coder and the verifier work in parallel against this document, so a silent change is a
merge conflict in behaviour.

**Scope:** backend hardening for the TAOBAO-MM real-time demo. Tasks B1–B5:

| # | Task |
|---|---|
| B1 | Kafka ingestion path: producer → consumer → Redis session, idempotency, DLQ |
| B2 | Auth (API key) + rate limiting |
| B3 | Observability: JSON logs, Prometheus metrics, per-stage latency |
| B4 | Resilience: `/livez` + `/readyz`, circuit breaker, timeouts, graceful shutdown |
| B5 | CI workflow + lint/type gates |

**Out of scope (do not build):** Terraform/IaC, Feast/feature store, Dagster/Airflow, Triton,
Redis vector index, retraining automation.

---

## 0. Non-negotiable constraints

1. **Zero hard dependencies on running services.** Redis and Kafka must each be *optional*. If the
   broker or cache is absent the app must still start and serve in a documented degraded mode.
2. **Every service-backed component needs an in-process implementation** so unit tests need no
   containers. Real Redis/Kafka are used only in integration tests (Docker is available).
3. **No new heavyweight frameworks.** Stdlib `logging` + a custom JSON formatter (no `structlog`);
   hand-rolled token bucket and circuit breaker (no `slowapi`/`limits`/`pybreaker`). Only new runtime
   dep is `prometheus-client`.
4. **Never block the event loop.** CPU-bound work stays in sync `def` handlers (FastAPI threadpool).
5. **Backwards compatibility.** The existing endpoints `/health`, `/event`, `/recommend`,
   `/items/sample`, `/metrics` must keep working; `/metrics` changes format to Prometheus text.
6. **CPU-bound demo stays light** — the reduced embedding table (212 MB) is already in place; do not
   regress it.

---

## 1. Settings — `src/tmm/serve/settings.py`

```python
@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...]          # from TMM_API_KEYS (comma separated); () => dev mode
    auth_required: bool                # TMM_AUTH_REQUIRED, default: bool(api_keys)
    rate_limit_rps: float              # TMM_RATE_LIMIT_RPS, default 20.0
    rate_limit_burst: int              # TMM_RATE_LIMIT_BURST, default 40
    redis_url: str | None              # REDIS_URL
    kafka_bootstrap: str | None        # KAFKA_BOOTSTRAP
    kafka_topic: str                   # TMM_KAFKA_TOPIC, default "tmm.clickstream"
    kafka_dlq_topic: str               # TMM_KAFKA_DLQ_TOPIC, default "tmm.clickstream.dlq"
    idempotency_ttl_s: int             # TMM_IDEMPOTENCY_TTL_S, default 3600
    session_max: int                   # TMM_SESSION_MAX, default 50
    cache_ttl_s: int                   # TMM_CACHE_TTL_S, default 5
    stage_timeout_s: float             # TMM_STAGE_TIMEOUT_S, default 0.5
    circuit_fail_threshold: int        # TMM_CIRCUIT_FAILS, default 5
    circuit_reset_s: float             # TMM_CIRCUIT_RESET_S, default 10.0

def load_settings() -> Settings: ...          # reads os.environ
```

`load_settings()` must be pure w.r.t. the environment (no caching that breaks tests) — tests will
monkeypatch `os.environ` and call it repeatedly.

---

## 2. Degradation contract

Exactly three modes, reported by `/health` as `"mode"`:

| mode | condition | behaviour |
|---|---|---|
| `full` | Redis **and** Kafka reachable | events go through Kafka; sessions in Redis |
| `degraded` | Redis reachable, Kafka **not** | events written **directly** to Redis (dual-write fallback) |
| `minimal` | Redis **not** reachable | in-process session store; events stored in memory; **must still serve `/recommend`** |

`/readyz` returns 200 in `full` and `degraded`, and **503 in `minimal`** (it can serve, but it is not
replica-safe). `/livez` always 200 while the process is up.

---

## 3. Components

### 3.1 `src/tmm/serve/session.py` — `SessionStore`

Keep the existing behaviour and add idempotency + health:

```python
class SessionStore(Protocol):
    backend: str                                    # "redis" | "memory"
    def append(self, user_id: int, item_id: int, event_id: str | None = None) -> tuple[int, bool]:
        """Returns (session_size, accepted). accepted=False when event_id was already seen."""
    def get(self, user_id: int) -> list[int]: ...
    def is_seen(self, event_id: str) -> bool: ...
    def ping(self) -> bool: ...
```

* `RedisSessionStore` — LPUSH + LTRIM + EXPIRE; idempotency via `SET idem:{event_id} 1 NX EX ttl`;
  sessions stored as raw 8-byte little-endian signed ints (never JSON lists).
* `MemorySessionStore` — same semantics with a `deque` + a `dict` of seen event ids with expiry.
* Idempotency applies **only** when `event_id` is not `None`.
* Session keys must include the TTL (Redis) and be capped by `session_max`.

### 3.2 `src/tmm/serve/events.py` — `EventPublisher`

```python
class EventPublisher(Protocol):
    kind: str                                       # "kafka" | "memory"
    def publish(self, event: dict) -> bool: ...     # True if handed to the broker
    def healthy(self) -> bool: ...
    def close(self) -> None: ...

class KafkaEventPublisher: ...                       # kafka-python-ng, key = str(user_id)
class MemoryEventPublisher: ...                      # appends to a thread-safe list
```

* Messages are compact JSON bytes; **key = `str(user_id.encode())`** so a user's events land on one
  partition and keep order.
* `publish()` must not raise on broker failure: return `False` after recording a metric, so the caller
  can fall back.
* Producer settings: `acks="all"`, `retries=3`, `linger_ms=5`, `max_in_flight=1` (ordering under retry).

### 3.3 `src/tmm/serve/consumer.py` — the ingest worker

Runs as a **separate process** (`python -m tmm.serve.consumer`), never in the API process.

```python
def consume_forever(settings, store, publisher=None, max_messages: int | None = None) -> dict:
    """Blocking loop. Returns a summary dict when max_messages is reached (for tests)."""
```

Rules:
* Decode JSON; on failure → DLQ with `reason="decode"`, metric, **commit the offset anyway**
  (a poison message must not block the partition).
* Validate required fields `user_id:int`, `item_id:int`, `event_type:str`; on failure → DLQ
  `reason="schema"`.
* Deduplicate through `store.append(..., event_id=...)`; duplicates increment a
  `duplicate` counter and do **not** re-append.
* After `circuit_fail_threshold` consecutive append failures → DLQ `reason="store"`.
* Must handle SIGTERM: finish the in-flight message, then commit and exit (graceful shutdown).
* Consumer group id: `tmm-ingest`.

### 3.4 `src/tmm/serve/ratelimit.py` — token bucket

```python
class RateLimiter(Protocol):
    def allow(self, key: str) -> tuple[bool, float]:
        """(allowed, retry_after_seconds). retry_after is 0.0 when allowed."""

class MemoryRateLimiter: ...
class RedisRateLimiter: ...        # atomic via a Lua script; falls back to Memory on error
```

* Token bucket: `rate_limit_rps` refill, `rate_limit_burst` capacity.
* Limit key = API key if present else client IP. Empty/`unknown` keys must still be limited
  (use a shared bucket) — do **not** let a missing IP bypass the limiter.
* Must be thread-safe (Starlette runs sync handlers in a threadpool).

### 3.5 `src/tmm/serve/resilience.py`

```python
class CircuitBreaker:
    def __init__(self, fail_threshold: int, reset_s: float, clock=time.monotonic): ...
    state: str                       # "closed" | "open" | "half_open"
    def call(self, fn, *args, **kwargs): ...   # raises CircuitOpen when open

class CircuitOpen(RuntimeError): ...

def call_with_timeout(fn, timeout_s: float, *args, **kwargs):
    """Run fn in a worker thread, raise TimeoutError if it exceeds timeout_s."""
```

* State machine: N consecutive failures → `open` for `reset_s` → one `half_open` probe → closed on
  success, re-open on failure. Injectable `clock` so tests need no `sleep()`.

### 3.6 `src/tmm/serve/obs.py` — observability

```python
def configure_logging(level: str = "INFO") -> None      # JSON formatter on the root logger
def log_event(logger, event: str, **fields) -> None     # one JSON line, no secrets
class StageTimer:                                       # context manager
    def __init__(self, stage: str): ...
```

**Metric names (exact — tests assert on them):**

| Metric | Type | Labels |
|---|---|---|
| `tmm_http_requests_total` | Counter | `endpoint`, `status` |
| `tmm_http_request_duration_seconds` | Histogram | `endpoint` |
| `tmm_stage_duration_seconds` | Histogram | `stage` |
| `tmm_events_ingested_total` | Counter | `result` = `accepted`\|`duplicate`\|`rejected` |
| `tmm_kafka_publish_total` | Counter | `result` = `ok`\|`error` |
| `tmm_dlq_total` | Counter | `reason` = `decode`\|`schema`\|`store` |
| `tmm_circuit_state` | Gauge | `backend` = `redis`\|`kafka` (0 closed, 1 half-open, 2 open) |
| `tmm_rate_limited_total` | Counter | `scope` = `api_key`\|`ip` |
| `tmm_ready` | Gauge | — (1 ready / 0 not) |

Stages recorded by the recommend path, **exactly these names**: `decode`, `session_read`,
`tower`, `search`, `total`.

`/metrics` returns the Prometheus text exposition (`CONTENT_TYPE_LATEST`). Use a **dedicated
CollectorRegistry** per app instance, or tests will see duplicate-registration errors on re-import.

---

## 4. HTTP surface — `src/tmm/serve/app.py`

| Method | Path | Auth | Rate-limited | Notes |
|---|---|---|---|---|
| GET | `/livez` | no | no | always 200 while alive |
| GET | `/readyz` | no | no | 200 `full`/`degraded`, **503** `minimal` |
| GET | `/health` | no | no | keeps existing fields + `mode`, `kafka`, `redis`, `circuits` |
| GET | `/metrics` | no | no | Prometheus text |
| GET | `/items/sample` | yes | yes | existing |
| POST | `/events` | yes | yes | **new canonical**; 202 Accepted |
| POST | `/event` | yes | yes | legacy alias, same handler |
| GET | `/recommend` | yes | yes | existing behaviour |

* Auth: header `X-API-Key`. Missing/wrong → **401** with `WWW-Authenticate: X-API-Key`.
  When `auth_required` is false, allow and log a loud one-time warning — never silently.
* Rate limited → **429** with `Retry-After: <int seconds>`.
* `/events` body: `{user_id:int, item_id:int, event_type:str="click", event_id:str|None=None,
  ts:str|None=None}`. Response 202: `{ok, event_id, accepted, session_size, mode, duplicate}`.
* `event_id` is generated server-side when absent: `f"{user_id}:{item_id}:{event_type}:{ts_bucket}"`
  where `ts_bucket = int(time.time() // 60)`.
* Publish path: if `publisher.healthy()` → publish, and **also** write the session directly only when
  in `degraded` mode (Kafka absent). In `full` mode the consumer owns the session write, so
  `/events` must **not** double-write. This must be covered by a test.
* All handlers that touch CPU/IO stay sync `def`.
* Graceful shutdown: a `lifespan` handler closes the producer and the store on exit.

---

## 5. Files and ownership (strict — no cross-writing)

**Coder owns** (create/edit only here):
```
src/tmm/serve/settings.py
src/tmm/serve/session.py
src/tmm/serve/events.py
src/tmm/serve/consumer.py
src/tmm/serve/ratelimit.py
src/tmm/serve/resilience.py
src/tmm/serve/obs.py
src/tmm/serve/app.py            (modify)
src/tmm/serve/__init__.py
docker/Dockerfile
docker/docker-compose.yml
.github/workflows/ci.yml
pyproject.toml                  (ruff + mypy config)
requirements.txt
```

**Verifier owns** (create/edit only here):
```
tests/backend/                  (all files)
tests/conftest.py               (if needed)
docs/BACKEND_DESIGN.md
docker/docker-compose.test.yml
```

**Neither may edit** `TASKS.md`, `README.md`, `artifacts/reports/*`, or `src/tmm/*.py` outside
`serve/` — the Lead integrates those.

---

## 6. Acceptance criteria (what "done" means)

1. `pytest -q` passes with **no Redis and no Kafka running** (unit/contract tests).
2. With `docker compose -f docker/docker-compose.test.yml up -d` (real Redis + single-node KRaft
   Kafka), the integration tests pass: a published event is consumed exactly once into the session,
   a replayed event is deduplicated, a malformed event lands in the DLQ.
3. `ruff check src tests` and `mypy src/tmm/serve` are clean.
4. `GET /metrics` exposes every metric named in §3.6.
5. `POST /events` returns 401 without a key when auth is enabled, and 429 after the bucket empties.
6. Killing Redis at runtime moves `/health` to `minimal`, `/readyz` to 503, and `/recommend` keeps
   responding; restoring Redis returns the app to `full`/`degraded` **without a restart**.
7. `/events` does not double-write the session when Kafka is healthy.
8. No secret is ever logged; log lines are valid single-line JSON.

## 7. Definition of done for the verifier

* Every acceptance criterion above has an automated test or a scripted probe, and the verifier
  reports the **observed** result, including failures.
* At least one **adversarial** test per component: duplicate replay, poison message, clock advance
  through open→half-open→closed, rate-limit boundary (burst-1 vs burst), missing client IP, and
  `event_id=None` (must not deduplicate).
* A written `docs/BACKEND_DESIGN.md` covering: component diagram, degradation modes, failure-mode
  table, capacity notes, and an explicit "known gaps" section.
* Verification must be run against the **coder's actual code on disk**, and the report must state
  the exact commands and their raw results. If something fails, say so — do not paper over it.

---

# Amendment v1.1 — deviations accepted during verification

**There is no repo reorg and no git worktree plan.** That instruction was never given by the Lead
and both teammates should ignore it; work continues in the existing working tree. (Recorded here
so the confusion is not repeated.)

Four contract points were found to be wrong or ambiguous while the implementation was verified
against a **real** broker. All four are accepted, and the contract text is superseded as follows.

### A1 — §3.2 `max_in_flight=1` names a key that does not exist
`kafka-python-ng` rejects it (`AssertionError: Unrecognized configs`). The correct key is
`max_in_flight_requests_per_connection=1`.
**Accepted implementation:** pass `max_in_flight` first, fall back to
`max_in_flight_requests_per_connection`. Semantically compliant (one in-flight request per
connection preserves ordering under retry).
**Superseded text:** replace `max_in_flight=1` with `max_in_flight_requests_per_connection=1`.

### A2 — §3.2 partition key was written as `str(user_id.encode())`
This is type-nonsense: `int` has no `.encode()`, and `KafkaProducer.send()` **asserts on non-bytes
keys**. Implemented literally, it made **every publish against a real broker fail**
(`AssertionError`), the consumer never saw client events, and the first failure flipped `/health`
to `degraded`. Unit doubles could not detect this; the integration test did.
**Accepted implementation:** `key_serializer=str.encode`, serialising `str(user_id)` to bytes.
**Superseded text:** the partition key is `str(user_id).encode()` (bytes), or equivalently
`str(user_id)` with `key_serializer=str.encode`.

### A3 — §4 `/event` status code
The contract said "legacy alias, same handler", which is ambiguous about the status. `/events`
returns **202 Accepted** (correct — the write is asynchronous once Kafka owns it); `/event` keeps
its historical **200** for backwards compatibility, with an identical body shape.
**Accepted implementation:** as built. Both are documented in the OpenAPI schema.

### A4 — `acks="all"` normalisation
`kafka-python` normalises `acks="all"` to `-1` in `producer.config`. Tests assert membership of
`("all", -1)`. No behavioural change; noted so the assertion is not mistaken for a bug.

### Consequences for the definition of done
* Acceptance §6.2 is **PASS** as of revision `b9ce2463a4681d34a0c93fde42fb07e4`, re-run required
  after the two pending `docker/Dockerfile` edits.
* The Dockerfile is **UNVERIFIED** until it is rebuilt: the image must copy `artifacts/stats/`
  (not just `profile.json`) and default to `--workers 1` with a `TMM_WORKERS` override, because two
  workers would exceed the 2 GB compose memory limit.
* Carried forward as unverified: in-container memory under the 2 GB limit, multi-replica behaviour,
  consumer throughput/p95, graceful shutdown of in-flight requests, and the GitHub Actions workflow
  (no runner available — only `docker compose config --quiet` was checked).
