# Backend design — hardened TAOBAO-MM real-time demo (B1–B5)

**Sources of truth.** `docs/BACKEND_CONTRACT.md` (frozen) defines the interfaces; this document
describes *how* the implementation on disk realises them, plus the verification evidence.
**Status:** verified against the code at the time of writing; see `tests/backend/` for the
executable form of every claim here and §7 of the contract for what "verified" means.

**Owners.** Implementation lives under `src/tmm/serve/**` (coder). The test suite
(`tests/backend/**`), this document and `docker/docker-compose.test.yml` are the verifier's.

**Contract amendments (v1.1).** The Lead added `docs/BACKEND_CONTRACT.md` §"Amendment v1.1"
after this verification run. It accepts four points observed here: A1 `max_in_flight` must be
`max_in_flight_requests_per_connection`; A2 the partition key is `str(user_id).encode()` bytes,
or `str(user_id)` plus `key_serializer=str.encode`; A3 `/event` keeps its historical 200 while
`/events` is 202; A4 `acks="all"` is normalised to `-1` by kafka-python. The tests in
`tests/backend/` already assert the amended forms (either spelling where both are accepted), so
this document and the suite are consistent with v1.1.

---

## 1. Component diagram

```
                    ┌──────────────────────────────────────────────────────────────┐
   client ──HTTP──► │  FastAPI app  (src/tmm/serve/app.py, create_app)             │
   X-API-Key        │                                                              │
                    │  middleware: tmm_http_requests_total / _duration_seconds     │
                    │      │                                                       │
                    │      ├─ guard():  auth (§4)  →  token bucket (§3.4)          │
                    │      │              └─ MemoryRateLimiter | RedisRateLimiter   │
                    │      │                     (Lua, atomic; memory fallback)    │
                    │      ├─ POST /events ──► EventPublisher (§3.2)               │
                    │      │        │            KafkaEventPublisher ──► broker    │
                    │      │        │            MemoryEventPublisher (no broker)  │
                    │      │        └─ SessionStore.append (§3.1) *only* when not  │
                    │      │           `full` (degraded/minimal) or publish failed │
                    │      ├─ GET /recommend ─► StageTimer(decode, session_read,   │
                    │      │        tower, search, total) ─► Recommender (FAISS +  │
                    │      │        ONNX/torch, reduced 212 MB table)              │
                    │      └─ GET /livez /readyz /health /metrics                  │
                    └──────────────────────────────────────────────────────────────┘
                              │                        ▲
              SessionStore (Redis LPUSH/LTRIM/EXPIRE   │  tmm_ready gauge
              + SET idem:{id} NX EX, raw 8-byte ints)  │
                              │                        │
                    ┌─────────▼─────────┐    ┌─────────┴───────────────────────┐
                    │  Redis (optional) │    │ Prometheus CollectorRegistry     │
                    └─────────▲─────────┘    │ (dedicated per process, obs.py)  │
                              │              └──────────────────────────────────┘
                              │
   broker ── consume ─►  python -m tmm.serve.consumer  (separate process)
                          consume_forever → validate → SessionStore.append(event_id)
                                             ├─ decode failure ─► DLQ reason=decode
                                             ├─ schema failure ─► DLQ reason=schema
                                             └─ store failure ×N ─► DLQ reason=store
                          CircuitBreaker(redis) + commit-past-poison offset handling
```

Key structural decisions:

* **CQRS read path.** `/recommend` only reads the session store; it never touches Kafka, so
  p95 latency does not depend on broker health.
* **The consumer owns the session write in `full` mode.** `/events` publishes and stops there;
  a duplicate is therefore impossible to create twice (idempotency is per `event_id`).
* **Every service-backed component has an in-process twin** (`MemorySessionStore`,
  `MemoryEventPublisher`, `MemoryRateLimiter`), so the unit layer needs no containers.
* **Sync `def` handlers.** CPU-bound inference stays in FastAPI's threadpool; only the metrics
  middleware is `async` and it does no CPU work.
* **The Recommender is memory-flat by default.** It prefers the precomputed demo artifact
  (`artifacts/models/demo/`: fp16 table + sorted vocab ids + catalogue) so the request path never
  maps the 4.5 GB int8 table nor builds the 35.46 M-entry vocabulary. Measured 791 MB peak /
  1.5 s startup vs 7 028 MB / 8.8 s for the legacy layout (see §5).

---

## 2. The three degradation modes (contract §2)

Mode is derived on every `/health`, `/readyz` and `/metrics` call
(`health_snapshot()` in `app.py`):

```
redis_ok = session_store.backend == "redis"        # FallbackSessionStore probes the primary
kafka_ok = publisher.kind == "kafka" and publisher.healthy()

redis_ok and kafka_ok → "full"
redis_ok and not kafka_ok → "degraded"
not redis_ok → "minimal"
```

| mode | condition | session writes | events | `/readyz` | `/recommend` |
|---|---|---|---|---|---|
| `full` | Redis **and** Kafka reachable | consumer process only (no double write) | published to Kafka | 200 | served |
| `degraded` | Redis reachable, Kafka **not** | written **directly** to Redis on the request path | kept in `MemoryEventPublisher` (bounded 10 000) | 200 | served |
| `minimal` | Redis **not** reachable | in-process `MemorySessionStore` | kept in memory | **503** | **still served** |

`/livez` is always 200 while the process serves. `/readyz` is 503 in `minimal` because
in-process sessions are not replica-safe: a load balancer must stop sending traffic to a node
that cannot see other replicas' sessions.

Mode transitions happen **without a restart**: `FallbackSessionStore.probe()` re-checks the
primary at most once per `min(cache_ttl_s, 1s)` and on every forced health probe, and
`record_success()` closes the Redis circuit breaker, so a restarted Redis is picked up on the
next request. Kafka, by contrast, is chosen once at startup
(`build_publisher`) — see known gaps.

---

## 3. Failure-mode table

| # | Failure | Detection | Behaviour | Recovery |
|---|---|---|---|---|
| 1 | Redis absent at startup | `RedisSessionStore.from_url` / connect failure in `build_session_store` | app starts on `MemorySessionStore`; mode `minimal`; `/readyz` 503 | set `REDIS_URL`, restart, or start Redis and the fallback probe flips the mode |
| 2 | Redis dies at runtime | `FallbackSessionStore.probe()` → `ping()` fails; `SessionStore` ops raise → `_mark_down()`; `call_with_timeout` on session read | session ops transparently use memory; `/health` → `minimal`; `/readyz` 503; `/recommend` keeps answering with the degraded session | Redis returns → `probe(force=True)` succeeds → `redis_ok` true again, no restart (verified, acceptance 6) |
| 3 | Redis read hangs | `call_with_timeout(session_store.get, stage_timeout_s)` (default 0.5 s) | `TimeoutError` → `record_failure()` on the Redis breaker, empty session, `/recommend` returns cold-start items instead of 500 | next reads succeed → `record_success()` |
| 4 | Redis append fails ×N (`circuit_fail_threshold`) | consumer's consecutive-failure counter | message is DLQ'd with `reason="store"`; `tmm_dlq_total{reason="store"}` | counter resets after a successful append; UV/Redis restored |
| 5 | Kafka absent at startup | `KafkaEventPublisher` construction raises `NoBrokersAvailable` | `build_publisher` logs `kafka_unavailable` and returns `MemoryEventPublisher`; mode `degraded` (Redis up) | restart the API with the broker up (no hot re-probe) |
| 6 | Kafka publish fails mid-request | `future.get(timeout=send_timeout_s)` raises | `publish()` returns `False`, `tmm_kafka_publish_total{result="error"}`; `/events` falls back to a direct session write | next publish succeeds (`record_success`); breaker closes |
| 7 | Kafka producer unhealthy ×N | `CircuitBreaker(kafka)` over `publish()`; `tmm_circuit_state{backend="kafka"}=2` | requests skip the broker attempt and write the session directly | breaker half-opens after `circuit_reset_s`, one probe closes it |
| 8 | Poison / undecodable Kafka message | `json.loads` raises in `process_message_detail` | DLQ `reason="decode"`, `tmm_dlq_total{reason="decode"}`, **offset committed anyway** | none needed — partition keeps moving (verified) |
| 9 | Schema-invalid event (`user_id`/`item_id` not int, `event_type` not str, field missing) | `validate_event()` | DLQ `reason="schema"`; never reaches the store | producer fixes the payload |
| 10 | Replayed event (at-least-once broker) | `SET idem:{event_id} NX EX ttl` claim fails / seen-set hit | `append()` returns `accepted=False`; `tmm_events_ingested_total{result="duplicate"}`; **no second session entry** | none needed (idempotent); after `idempotency_ttl_s` a replay is accepted again |
| 11 | Rate-limit bucket empty | `allow()` returns `(False, retry_after)` | HTTP 429 + `Retry-After: <int s>`; `tmm_rate_limited_total{scope}` | bucket refills at `rate_limit_rps` |
| 12 | Client IP cannot be resolved | `_client_ip()` → `"unknown"` → `normalize_key` → one shared anonymous bucket | still limited (no bypass); `scope="ip"` | n/a |
| 13 | Redis rate-limiter error | Lua call raises | falls back to the in-process bucket for that call | next call uses Redis again if it succeeds |
| 14 | SIGTERM to the consumer | signal handler sets a `threading.Event` | finishes the in-flight message, commits, closes the consumer, exits 0 (`consumer_stopped` log) | process supervision |
| 15 | SIGTERM to the API | FastAPI `lifespan` finally-block | closes the producer (flush) and the session store | process supervision |
| 16 | Model artifacts missing | `Recommender.__init__` raises during `create_app()` | module falls back to `_fallback_app`: `/health`, `/livez`, `/readyz`(503), `/metrics` still answer | mount `artifacts/`, restart |
| 17 | Duplicate Prometheus registration on re-import | `ValueError: Duplicated timeseries` | prevented: `obs.py` owns a dedicated `CollectorRegistry`, never `prometheus_client.REGISTRY` | n/a (verified in a subprocess reload test) |

---

## 4. Observability (contract §3.6)

All nine metrics are created in `obs.Metrics` on a dedicated registry and re-exported at module
level (`HTTP_REQUESTS`, `DLQ_TOTAL`, `STAGE_DURATION`, …) so both `/metrics` and direct lookups
see the same samples.

| metric | type | labels | emitted by |
|---|---|---|---|
| `tmm_http_requests_total` | counter | `endpoint`, `status` | HTTP middleware |
| `tmm_http_request_duration_seconds` | histogram | `endpoint` | HTTP middleware |
| `tmm_stage_duration_seconds` | histogram | `stage` | `StageTimer`; stages are exactly `decode`, `session_read`, `tower`, `search`, `total` |
| `tmm_events_ingested_total` | counter | `result`=`accepted`\|`duplicate`\|`rejected` | `/events` and the consumer |
| `tmm_kafka_publish_total` | counter | `result`=`ok`\|`error` | `KafkaEventPublisher.publish` |
| `tmm_dlq_total` | counter | `reason`=`decode`\|`schema`\|`store` | consumer |
| `tmm_circuit_state` | gauge | `backend`=`redis`\|`kafka` (0 closed, 1 half-open, 2 open) | breakers in `app.py`/`session.py` |
| `tmm_rate_limited_total` | counter | `scope`=`api_key`\|`ip` | `guard()` |
| `tmm_ready` | gauge | — | `/health`, `/readyz`, `/metrics` |

Logging is stdlib `logging` + `obs.JsonFormatter` on the root logger: one JSON object per line,
`event` plus structured `tmm_fields`, exception text newline-escaped, and values whose field name
looks like a credential redacted to `***`. `configure_logging()` is idempotent. Latency stages
are recorded without logging per request, so log volume tracks events, not traffic.

---

## 5. Capacity notes

### Measured on this host (2026-10-07, `.venv` python 3.13)

Two `Recommender` layouts exist. The **demo artifact** (`artifacts/models/demo/`, ~218 MB on
disk, built offline by `python -m tmm.cli build-demo-artifact`) is preferred; the legacy layout
(`artifacts/data` + `TMM_DATA/feature_map`, scatter-gathering out of the 4.5 GB int8 memmap) is
kept as a fallback behind `use_demo_artifact=False`.

| quantity | demo artifact | legacy layout |
|---|---|---|
| startup peak RSS (`python -m tmm.cli measure-memory --mode …`) | **790.7 MB** | **7 027.6 MB** |
| startup wall time (measured by the same command) | **1.5 s** | 8.8 s |
| `create_app()` wall time, incl. ONNX + FAISS | ~2–3 s | ~12 s |
| embedding table reported by `/health.embedding_table_mb` | **210.2 MB** | 212.6 MB |
| FAISS HNSW index (`/health.index_mb`) | **7.32 MB** | 7.32 MB |
| user tower | `onnx` (`user_tower_fp32.onnx`) | same |
| Redis test container | 128 MB maxmemory, `allkeys-lru`, no persistence | — |
| Kafka test container | single node, `-Xmx512m` | — |

**This table is the worked example of a sizing claim that was only corrected by measuring.**
The project asserted a 2 vCPU / 4 GB VPS was comfortable, and the reduced table really is only
210 MB — but the *measured* startup peak was **7 027 MB**, ~2× the whole VPS budget, so the
claim was wrong. Root cause: `load_embeddings_subset()` scatter-gathers ~870 k scattered rows
out of a 4.5 GB memory-mapped int8 file, and mmap-resident pages count towards RSS; on top of
that `load_prepared()` materialises the full 35.46 M-entry item vocabulary plus both splits
(~1.1 GB) even though the service only needs "which row is this item id?". Moving that work
offline into the demo artifact cut the peak **8.9×** (7 027 → 791 MB) and startup from 8.8 s to
1.5 s. Neither number is a guess: both were produced by `measure-memory` and by
`tests/backend/test_recommend_real_model.py::test_artifact_mode_startup_peak_rss_is_bounded`,
which fails if the artifact path ever exceeds 1.5 GB again.

At 791 MB peak the 2 vCPU / 4 GB / 2 GB-container sizing is defensible for one worker; the
legacy fallback is only viable on a ≥8 GB host. The Dockerfile therefore defaults to
`--workers 1`.

### Steady-state sizing rules

* **Sessions** — one Redis list per user, capped at `session_max` (50) raw 8-byte values plus
  list overhead ≈ 0.8–1.2 KB per active user; 100 000 users ≈ 100 MB (matches the compose note).
* **Idempotency keys** — `idem:{event_id}` for `idempotency_ttl_s` (3 600 s). At 5 000 events/s
  that is 18 M keys ≈ 1.5–2 GB; size `idempotency_ttl_s` against the peak replay window.
* **Events endpoint** — one Lua call (Redis limiter) or one dict operation (memory limiter) plus
  one producer `send().get()` bounded by `send_timeout_s` (2 s) and `max_block_ms=1000`.
  Bounded worst case per request in `degraded/full` ≈ 2 s; the session read on `/recommend` is
  bounded by `stage_timeout_s` (0.5 s).
* **Rate limiting** — default 20 rps refill / 40 burst per key. The bucket is per API key when a
  key is presented, else per client IP; unidentifiable clients share one bucket, which is the
  safe failure direction (one anonymous budget, not unlimited).
* **Consumer** — single-threaded, `poll(timeout_ms=500)`, commits after every message (a
  per-record commit, i.e. trade throughput for a bounded replay window). Scale by running more
  replicas in the same group `tmm-ingest` (partition count is the ceiling) — three partitions in
  the test stack; use a partition count that matches the expected replica fan-out.
* **Read path** — inference is CPU-bound; `faiss.omp_set_num_threads(4)` and the ONNX session
  use 4 threads, so plan ≥ 4 vCPU per API replica for a saturated single request, or lower the
  thread counts when packing many replicas on one box.
* **Metrics** — one registry per process; with N replicas, aggregate with `sum by (...)` at the
  scrape layer.

---

## 6. Security properties

* `X-API-Key` header, compared against `TMM_API_KEYS`; missing/wrong → 401 with
  `WWW-Authenticate: X-API-Key`. Auth runs **before** rate limiting so a bad key is always 401,
  never a 429 that leaks limiter state.
* With `TMM_API_KEYS` empty, `auth_required` defaults to false and the app logs a single loud
  `auth_disabled` warning at construction — never silently.
* No secret is ever placed in a log field, and `JsonFormatter` additionally redacts values whose
  field name matches `api_key|apikey|authorization|credential|password|secret|token`.
* The test stack runs Redis and Kafka in PLAINTEXT with no auth on `127.0.0.1` only, and is torn
  down with `docker compose -f docker/docker-compose.test.yml down -v`.

---

## 7. Known gaps

Explicit list of what this design does **not** guarantee, in rough priority order.

1. **Only Redis recovers without a restart.** Kafka is selected once at startup in
   `build_publisher`; if the broker is down then, the process stays `degraded` until restarted
   (contract criterion 6 only requires Redis hot recovery).
2. **Degraded/minimal events are bounded in memory, not durable.** `MemoryEventPublisher` keeps
   at most 10 000 events and drops the oldest; a restart loses them. There is no disk spool.
3. **`X-Forwarded-For` is trusted unconditionally** (`_client_ip`). Behind a proxy that appends
   correctly this is fine; exposed directly, a client can spoof the header and mint a fresh
   per-IP bucket. Only API-key-scoped limiting is spoof-proof.
4. **Rate limiting is per process in memory mode** (`MemoryRateLimiter`) and per Redis instance
   otherwise; it is not a global quota across replicas unless Redis is shared.
5. **Idempotency is time-bounded.** A replay after `idempotency_ttl_s` is appended again;
   exactly-once is not claimed, only de-duplication inside the window.
6. **The legacy Recommender fallback still needs ~7 GB at startup.** The default demo-artifact
   path peaks at 791 MB and is guarded by a test, but `use_demo_artifact=False` (or a missing
   `artifacts/models/demo/`) still scatter-gathers out of the 4.5 GB memmap and peaks at
   ~7 028 MB — more than the 2 GB compose limit and more than a 4 GB VPS. Falling back without
   also raising the memory limit will OOM. Build the artifact (`python -m tmm.cli
   build-demo-artifact`) as part of deployment, and see known gap 19 for the mount layout.
7. **No Redis HA.** Single instance, no Sentinel/Cluster; Redis loss is survivable but
   `minimal` mode is not replica-safe by construction (hence `/readyz` 503).
8. **DLQ records are unversioned.** `{reason, error, raw}` with `raw` truncated to a lossy utf-8
   decode; there is no replay tooling and no DLQ retry path.
9. **Consumer commits per message.** Throughput is bounded by a commit round-trip per event;
   batch commits would be faster but widen the replay window on crash.
10. **Circuit-breaker state is per process** and resets on restart; the gauge is only meaningful
    per replica.
11. **`/event` still returns 200 while `/events` returns 202.** The contract §4 calls `/event` a
    "legacy alias, same handler"; keeping the historical status was a deliberate
    backwards-compatibility choice (§0.5). The response *body* is identical.
12. **`tmm_ready` is only refreshed when `/health`, `/readyz` or `/metrics` is called**; a
    deployment that only scrapes `/metrics` is fine, but the gauge is not updated by data-plane
    requests.
13. **No load/latency test.** No p95/p99 SLO was measured for `/recommend` or `/events`; the
    capacity numbers above are budget arithmetic, not benchmark results.
14. **Histogram buckets** are `prometheus_client` defaults (5 ms … 10 s); single-stage timings
    below 5 ms land in the first bucket, so stage latency percentiles are coarse. A dedicated
    bucket set per stage would be needed for fine-grained latency work.
15. **Signal handling covers SIGTERM/SIGINT only.** SIGKILL or OOM still drops the in-flight
    message (at-least-once semantics apply, dedup handles the replay).
16. **The `max_in_flight` fallback constructs the producer twice on first use** when the broker
    is live (`AssertionError: Unrecognized configs: {'max_in_flight': 1}` from kafka-python-ng,
    then the retry with `max_in_flight_requests_per_connection`). Functionally correct, but it
    makes startup validation noisier than necessary.
17. **Test coverage is behavioural, not exhaustive.** The unit layer intercepts
    `KafkaProducer`/`KafkaConsumer` with doubles; the real broker path is covered only by the
    integration file, which requires Docker and the isolated stack.
18. **In `full` mode `duplicate`/`accepted` are best-effort (CQRS eventual consistency).**
    `/events` answers them with a read-only `store.is_seen(event_id)` probe — no double write —
    so a replay is reported as a duplicate only once the consumer has written the idempotency
    key. A replay that arrives *before* the consumer catches up is re-published and then
    deduplicated by the consumer; `session_size` is the pre-ingest size. Verified: pre-seeded
    idempotency key → `202 {accepted: false, duplicate: true}`, nothing re-published; fresh key
    with a healthy broker → published, consumer dedupes. A client must not read `duplicate:
    false` in `full` mode as "this is the first time this event exists".
19. **The Docker image is deliberately not self-sufficient.** It ships `artifacts/models/` and
    (since the v1.1 review) all of `artifacts/stats/`, but `load_prepared()` also reads
    `artifacts/data/{train,test}.npz`, `item_vocab.npz`, `hist.*`, and the embedding gather needs
    `TMM_DATA/feature_map`. `docker-compose.yml` mounts both, so the compose stack is fine; a bare
    `docker run` of the image falls back to `_fallback_app` (health endpoints only, no
    `/recommend`). The Dockerfile documents this as a deliberate trade-off to keep the image small.
20. **Worker count is a memory knob.** `docker/Dockerfile` now defaults to `--workers 1` with a
    `TMM_WORKERS` override, because each worker builds its own Recommender (~12 s, ~1.4 GB steady,
    ~6.9 GB transient peak) and two workers would exceed the compose 2 GB limit. In `minimal` mode
    sessions, breakers and the memory rate-limit bucket are per worker, so raising `TMM_WORKERS`
    also multiplies the number of independent anonymous buckets and session stores.

---

## 8. Verification map

| contract acceptance | where it is checked |
|---|---|
| 1. `pytest -q` clean with no Redis/Kafka | whole `tests/backend` unit layer (integration tests skip with a reason) |
| 2. real Redis+Kafka round trip, dedup, DLQ | `test_integration_services.py::test_end_to_end_consume_forever_dedup_and_dlq` |
| 3. `ruff check src tests`, `mypy src/tmm/serve` | run directly; both clean |
| 4. `/metrics` exposes every §3.6 metric | `test_http_contract.py::test_metrics_endpoint_exposes_every_contract_metric`, `test_obs_contract.py::test_every_contract_metric_exists_with_the_right_type_and_labels` |
| 5. 401 without key, 429 when the bucket empties | `test_http_contract.py::test_missing_api_key_is_401_with_www_authenticate`, `…rate_limit_allows_exactly_burst_then_429_with_retry_after` |
| 6. Redis kill → minimal/503/still serving → recovery | `test_integration_services.py::test_api_survives_redis_kill_and_recovers_without_restart` |
| 7. no double write when Kafka is healthy | `test_http_contract.py::test_full_mode_does_not_double_write_the_session` |
| 8. no secrets logged, single-line JSON | `test_http_contract.py::test_no_secret_is_ever_logged`, `test_obs_contract.py::test_log_event_emits_one_valid_json_line` |
| §0.1 app still starts with no services / no model artifacts | `test_http_contract.py::test_minimal_mode_health_readyz_and_recommend_still_serves`, `test_no_artifacts_startup.py::test_app_starts_and_serves_with_no_model_artifacts` |

---

## 9. Verification log (observed results)

Final run: 2026-10-07 against **TREE FROZEN v3**, revision
`md5sum src/tmm/serve/*.py src/tmm/cli.py src/tmm/demo_artifact.py pyproject.toml
requirements.txt docker/Dockerfile docker/docker-compose.yml .github/workflows/ci.yml | md5sum` =
**`fd96821755880db9869e3b7bff5f5bf6`**. (An earlier pass against
`b9ce2463a4681d34a0c93fde42fb07e4` produced the same verdict but predates the demo-artifact
wiring and the Dockerfile edits.) **The log is only valid for the cited revision.**

```
$ PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest tests/backend -q   # stack UP
108 passed, 1 warning in 11.05s

$ PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest tests/backend -q   # stack STOPPED
101 passed, 7 skipped, 1 warning in 5.60s
SKIPPED [7] tests/backend/test_integration_services.py: integration stack unavailable:
            redis not listening on 127.0.0.1:6399 (start docker/docker-compose.test.yml)

$ PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest -q                 # stack STOPPED
112 passed, 7 skipped, 1 warning in 7.76s

$ .venv/bin/ruff check src tests
All checks passed!
$ .venv/bin/mypy
Success: no issues found in 9 source files

$ PYTHONPATH=src .venv/bin/python -m tmm.cli measure-memory --mode artifact
  "peak_rss_mb": "791.15625"          # 1.6 s
$ PYTHONPATH=src .venv/bin/python -m tmm.cli measure-memory --mode legacy
  "peak_rss_mb": "7027.65234375"      # 9.8 s
```

Raw probes (live stack, real Redis + Kafka, stub recommender where the model is irrelevant):

```
/health            {'mode': 'full', 'redis': True, 'kafka': True,
                    'circuits': {'redis': 'closed', 'kafka': 'closed'}}
/readyz            200
POST /events       202 {... 'mode': 'minimal', 'session_size': 1, 'duplicate': False}   (minimal app)
BURST 4 calls:     [200, 200, 200, 429]  -> 5th 429, Retry-After: 1     (20 rps / 3-burst app)
POST /events no key 401, WWW-Authenticate: X-API-Key
/recommend         stages recorded: decode, search, session_read, total, tower
/metrics           all nine §3.6 names present once the nine paths are exercised
full-mode replay, idempotency key already written by the consumer:
                   202 {'accepted': False, 'duplicate': True}   (no re-publish, no request-path append)
/health (artifact mode) embedding_table_mb=210.2, index_mb=7.32, catalogue=10000, tower=onnx
/items/sample (artifact mode) real int64 anonymised ids, all present in
                    artifacts/models/demo/item_ids.npy
```

Headline bug caught by the integration layer (unit doubles could not see it): the literal §3.2
partition key `str(user_id.encode())` is a `str`, and `KafkaProducer.send()` asserts on non-bytes
keys, so **every publish against a real broker failed**. Minimal reproduction, raw output:

```
$ python -c "KafkaProducer(...).send('tmmtest-repro', value=b'{\"user_id\": 4242}', key=\"b'4242'\")"
FAIL contract literal key=str(uid.encode()) -> str: AssertionError:
$ ... same call with key=b"4242"
OK   bytes key=str(uid).encode()
```

Fixed with `key_serializer=str.encode` (contract amendment A2) and re-verified end-to-end.

Container teardown (run at the end of the final pass):

```
$ docker compose -f docker/docker-compose.test.yml down -v
 Volume tmmtest_tmmtest-redis-data Removed
 Volume tmmtest_tmmtest-kafka-data Removed
 Network tmmtest_default Removed
$ docker ps -a --filter name=tmmtest     # empty
$ docker volume ls --filter name=tmmtest # empty
$ docker network ls --filter name=tmmtest# empty
```
