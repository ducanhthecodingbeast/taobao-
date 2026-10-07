# TAOBAO-MM Portfolio Project — Work Breakdown Structure (WBS)

**Objective.** Turn section (2) *"Tổng quan bộ dữ liệu TAOBAO-MM"* and section (3) *"Hai hướng triển khai"*
of the project brief into **verified research + running code + a report with graphs**, sized to the
hardware that actually exists in this machine.

**Verified hardware baseline** (`nproc`, `free`, `df`, `nvidia-smi`):

| Resource | Value | Consequence for the design |
|---|---|---|
| CPU | Intel Core Ultra 9 285K, 24 threads | data engineering + Spark; parallelism via threads |
| RAM | 156 GiB total, ~41 GiB **available** | Full 139 GB dataset cannot be held in RAM |
| Disk | 101 GB free on `/` (NVMe) | Cannot materialize another full copy of the dataset |
| **GPU** | **NVIDIA RTX 5880 Ada Generation, 47.4 GiB VRAM, CC 8.9** | **GPU training is viable → Direction 2 is a real distributed-scale build, not a mock** |
| Driver / CUDA | 580.178.04 / CUDA 13.0, torch 2.14.1+cu130 | bf16 + Flash-Attention-2 (sm89) available |
| GPU throughput | measured **102.6 TFLOP/s** fp16 matmul, 110 SMs | two-tower + DIN/MUSE training is minutes, not hours |
| VRAM headroom | **37.9 GiB free** (7 foreign processes hold ~9 GiB) | full 35.4 M × 128 item embedding table (9 GiB bf16) fits **entirely in VRAM** |
| Java | OpenJDK 17 | PySpark 3.5+ compatible |
| Sandbox | `/dev` is a private tmpfs (`CapEff=0`) | GPU commands require one-shot `danger-full-access` escalation; batch GPU work into few long jobs |

> **Key architectural consequence of the GPU.** The whole 35.4 M-item SCL embedding table fits in VRAM
> as bf16 (35 458 467 × 128 × 2 B = **8.45 GiB**). So the item tower can be a *lookup over the entire
> catalogue on-device*, and the MUSE-style "search the 1 000-item history with the target item as query"
> becomes a single batched `torch.bmm` + `topk` on GPU. That is what makes the two-stage
> retrieval→ranking design measurable rather than hypothetical.

**Verified dataset baseline** (local files at `/home/aiface/Taobao-MM`):

| Table | Rows | Size | Note |
|---|---|---|---|
| `train/*.parquet` (161 shards) | 76,015,123 | 49 GB | **already joined**, 13 columns |
| `test/*.parquet` (48 shards) | 22,979,465 | 15 GB | same schema |
| `raw/train_samples.parquet` | 76,015,123 | 714 MB | `label_0`, `129_1`, `205` |
| `raw/train_user_features.parquet` | 6,929,671 | **44.6 GB** | 7 row-groups, 1k-long sequences |
| `raw/scl_embedding_int8_p90.parquet` | 35,458,467 | 5.25 GB | `205`, `205_c` = 128×int8 |
| `raw/item_features.parquet` | 4,164,497 | 48 MB | category/city/province |
| `feature_map/scl_emb_int8_p90_values.npy` | 35,458,467 × 128 | 4.54 GB | **memmap-able**, the fast path |
| measured positive rate | **13.9 %** (50 k sampled rows) | | class imbalance is real |
| measured sequence length | mean **987**, p50 = p90 = **1000** | | truncation must be designed for |

---

## Task graph (execution order)

```
T0 Environment & data contracts ─┐
                                 ├─> T1 Data profiling / EDA ──┐
T2 Vocab + memmap embedding store┘                             │
                                                               v
                     ┌──────────── T3 Data Engineering (the crux) ────────────┐
                     │  T3a DuckDB/Arrow single-node  |  T3b PySpark cluster  │
                     └───────────────┬────────────────────────┬───────────────┘
                                     v                        v
                            T4 Baselines (pop / ItemCF / eASE)
                                     │
                     ┌───────────────┴────────────────┐
                     v                                v
        T5a Two-Tower retrieval (Dir. 2)     T6 Lightweight realtime path (Dir. 1)
        T5b Ranker: DIN + MUSE-style search   T6a ONNX int8 + FAISS index
                     │                        T6b FastAPI + Redis + Kafka
                     │                        T6c Docker/Caddy HTTPS + load test
                     └────────────┬───────────┘
                                  v
                  T7 Evaluation harness (HR@K / NDCG@K / AUC, leakage-free)
                                  v
                  T8 Figures  ──>  T9 Report + pitch
```

---

## T0 — Environment, contracts, and reproducibility
**Method.** Isolated `.venv` (CPU torch), `requirements.txt` pinned, `configs/default.yaml` holding
dataset paths + subsample budget, every artifact written under `artifacts/`.
**Problems.** (a) The host `python3` is another project's venv → polluting it would break that
project. (b) Only 101 GB disk free. (c) `/home/aiface/.cache` is not writable → pip cache errors.
**Solutions.** Dedicated `venv`; all intermediates capped by a `MAX_ROWS` budget and written as
Parquet with zstd; `PIP_CACHE_DIR=/tmp/pipcache`; `MPLCONFIGDIR=/tmp/mpl`.
**Acceptance.** `pytest -q` passes; `python -m tmm.cli doctor` prints hardware + file inventory.

## T1 — Data profiling / EDA grounded in the real files
**Method.** Stream Parquet **row-group by row-group** (never `read_table` on the 44 GB file);
compute label balance, sequence-length histogram, item popularity (long tail), category
distribution, embedding norm/quantization stats, coverage of `scl_emb` over history items,
per-shard sanity checks, and the user/item overlap between train and test.
**Problems.** (a) `raw/train_user_features.parquet` is 44.6 GB with only 7 row-groups → a naive
`to_pandas()` OOMs. (b) `label_0` is `list<int8>` of length 2, not an int, so the label is not a
scalar anywhere. (c) Anonymized ids span the full int64 range including negatives, so they cannot be
used as array indices.
**Solutions.** `iter_batches(batch_size=…)` + reservoir sampling; `label_0[1]` → int8 label;
reservoir-sampled id sets; every figure regenerable from `artifacts/stats/*.json`.

**Correction (measured after the first draft).** An earlier version of this document claimed that
rows are *not* grouped by user and that a partial shard scan would give most users ~1 candidate
instead of ~11. **That claim is false.** Measured on `test-shard-000000`: 500 000 rows →
44 570 distinct users → **11.22 rows/user**, and **99.9 %** of those users have their complete
candidate set inside that single shard. The release *is* user-grouped at shard granularity. The
decision to source candidate sets from `raw/{split}_samples.parquet` nevertheless stands, but for
robustness reasons (exactness, independence from an undocumented locality property, safety against
row-group-level reads and early stops), not because a shard scan is broken.
**Deliverables.** `artifacts/stats/profile_*.json`, `artifacts/figures/01_*.png` … `05_*.png`.
**Acceptance.** Numbers reproduce on re-run (seed 42) and match the README within tolerance.

## T2 — Vocabulary / id-remapping and the memmap embedding store
**Method.** Use the provided `feature_map/*_sorted_map.npy` (a **sorted** array → `np.searchsorted`
gives O(log n) id→index). Memory-map `scl_emb_int8_p90_values.npy` (4.54 GB) read-only; dequantize
int8→float32 with the per-vector/per-tensor scale found empirically in T1.
**Problems.** (a) 4.54 GB cannot be duplicated on a 101 GB disk *and* kept in the 41 GB budget.
(b) `np.load(mmap_mode='r')` on a `.npy` needs the header parsed correctly for `searchsorted`.
(c) Real item ids are int64 including negatives — sorting and `searchsorted` must use `int64`.
**Solutions.** mmap + fancy-index gather into a small working matrix; `keys.npy` already aligned to
`values.npy` → build a **hash→row** index once and cache it; a `QuantSpec` dataclass records
scale/zero-point so the dequantization is not a magic number.
**Deliverables.** `src/tmm/vocab.py`, `artifacts/stats/embedding_stats.json`.
**Acceptance.** Random item id → correct 128-d vector; round-trip error `< 1e-2`.

## T3 — Data engineering: joining label ⟷ embedding ⟷ features (the real hard part)
> The brief calls this "the hardest part, not the algorithms". We implement it **twice**.

### T3a — Single-node Arrow/DuckDB path (Direction 1, subset)
**Method.** DuckDB reads `raw/train_samples.parquet` + a **zstd** copy of the embedding table,
`JOIN` on `205`, then materializes a 1–2 M row training table with history sequences.
**Problems.** (a) DuckDB must decode `LIST<INT8>` — supported, but the 128-wide list stays a list
until cast via `list_transform`. (b) Memory spikes on the 35 M-row join; DuckDB defaults to
`memory_limit=80% of RAM` and spills to `/tmp` (79 GB tmpfs — actually good).
**Solutions.** `PRAGMA memory_limit`, `temp_directory=/tmp`, `threads=24`, `preserve_insertion_order=false`;
write to Parquet zstd; verify row-count invariants (`joined == samples` since coverage is 100 % for
target items).
**Acceptance.** Byte-identical row count vs. `metadata.json`; join wall-time + peak RSS recorded.

### T3b — Distributed PySpark path (Direction 2, full scale)
**Method.** Spark `local[*]` reading the 161 train shards, broadcast-joining a bucketed embedding
lookup, `posexplode`-free handling of the 1000-length sequences, and a `Bucketizer`/repartition
strategy to avoid a 35 M-row × 128-col shuffle.
**Problems.** (a) `list<int8>` in Spark → `ArrayType(ByteType)` is handled, but casting to
`ArrayType(FloatType)` is the expensive step. (b) `broadcast()` of a 4.5 GB embedding table will
kill the driver → must use a **bucketed join** or `spark.sql.autoBroadcastJoinThreshold` tuning.
(c) Spark's default `spark.driver.memory` is 1 GB; the 44 GB `train_user_features` scan needs more.
(d) Skew: popular items create hot partitions.
**Solutions.** `spark.sql.shuffle.partitions=24×4`, 200 MB driver, `maxPartitionBytes=128m`,
AQE on, salted/bucketed join on `205`, embeddings stored as a Parquet table bucketed by `205`
at 256 buckets → **sort-merge join without shuffle**. Run a scaled version (N million rows)
locally and record the row/s throughput to extrapolate to the full 76 M.
**Deliverables.** `src/tmm/join_spark.py`, `artifacts/bench/spark_join.json`, Spark UI event log.
**Acceptance.** Produces the same schema as T3a; throughput extrapolates linearly (report R²).

## T4 — Baselines that must be beaten
**Method.** Popularity (global + category-conditional), Item-KNN/ItemCF on co-occurrence from the
history sequences, and **eASE** (closed-form linear auto-encoder, cheap on CPU).
**Problems.** Popularity baselines are **strong** here because the candidate set is heavily
head-biased; an untuned neural model loses to them, which is a classic portfolio trap.
**Solutions.** Report all baselines under the *same* leakage-free protocol before claiming wins;
keep the candidate set honest (see T7).
**Deliverables.** `artifacts/stats/baselines.json`, `artifacts/figures/06_baselines.png`.

## T5 — Models (Direction 2: two-stage retrieval + ranking)
### T5a — Two-Tower retrieval
**Method.** User tower = mean/attention-pooled SCL vectors of the history + demographic embeddings;
Item tower = linear/MLP over the 128-d SCL vector (+category/city). Trained with **sampled softmax
+ in-batch negatives**, temperature τ, L2-normalized outputs so that `dot == cosine`.
**Problems.** (a) 35 M items → full softmax impossible on CPU; (b) in-batch negatives alone give
poor calibration with a head-biased item distribution; (c) popularity bias in the index.
**Solutions.** logQ correction / `log_uniform` negative sampling; FAISS HNSW index over the item
tower output; optional cross-batch queue for more negatives.
**Acceptance.** Recall@200 beats ItemCF on the held-out temporal split.

### T5b — Ranker: DIN vs. MUSE-style search
**Method.** Implement **DIN** (attention over the history with the target item as query) as the
vanilla ranker, then **MUSE-style**: use the SCL similarity between the target item and each of the
1000 history items to *search* a top-k relevant sub-sequence, and run attention only over that k.
**Problems.** (a) DIN on 1000 tokens is O(1000) attention per sample and is slow on CPU;
(b) MUSE's search needs a per-sample similarity scan over 1000×128 — vectorizable but memory-heavy;
(c) the 900 dropped tokens lose information unless the retrieval is learned.
**Solutions.** Precompute the target embedding matrix once per batch (`[B,1000,128]` einsum),
`torch.topk(k=50)`, chunked accumulation; compare DIN-full vs. MUSE-k on latency *and* AUC to show
the tradeoff quantitatively (this is the portfolio story).
**Deliverables.** `src/tmm/models/{two_tower,din,muse}.py`, `artifacts/models/*.pt`, training curves.

## T6 — Direction 1: the interactive live demo
### T6a — Model compression + index
**Method.** Export the item tower to ONNX; INT8-quantize; build a FAISS **HNSW** index over 10 k
items (the brief's subset) and measure recall/latency vs. exact `IndexFlatIP`.
**Problems.** ONNX INT8 dynamic quantization of embedding-gather ops sometimes *increases* latency;
`faiss-cpu` HNSW build is single-threaded and can be slow.
**Solutions.** Benchmark FP32 vs. ONNX-FP32 vs. ONNX-INT8 and report the true winner; use
`IndexHNSWFlat` with `efConstruction=200, M=32`, then `efSearch` sweep for the recall/latency curve.

### T6b — Serving: FastAPI + Redis + Kafka
**Method.** `POST /event` → Kafka topic `clickstream`; consumer updates the user's Redis state
(`RPUSH`+`LTRIM` to a 50-item session, `EXPIRE`); `GET /recommend?user_id=` → HNSW search → cache
in Redis with a short TTL; `GET /health` + `/metrics`.
**Problems.** (a) Kafka in Docker on a laptop costs RAM; (b) cold-start users have no history;
(c) synchronization between the Kafka consumer and the request path.
**Solutions.** `KRaft` single-node Kafka (no ZooKeeper) with tight heap; cold-start fallback =
category-popularity; recommendation reads only from Redis (CQRS-ish) so latency is independent of
Kafka; a pure-Redis "degraded mode" flag if Kafka is down.

### T6c — Packaging, HTTPS, load test
**Method.** `docker-compose.yml` (api, kafka, redis, caddy), `Caddyfile` with automatic Let's
Encrypt, and a load test (locust/k6) reporting p50/p95/p99.
**Problems.** A "live demo on a cheap VPS" claim is unverifiable without numbers → measure locally
and state the tested ceiling.
**Acceptance.** `p95 < 20 ms` server-side at the tested concurrency, or the real measured number is
reported (no invented latency).

## T7 — Evaluation harness (where most portfolio projects leak)
**Method.** Temporal split; candidate generation reported **both** ways (sampled 1:1 and
full-corpus); HR@K, NDCG@K, MRR, AUC, logloss; per-user metric aggregation (not per-row, which
inflates NDCG when a user has many rows).
**Problems.** (a) The provided `test/` shards are already temporally split but a usermay appear in
both; (b) sampled-negative evaluation inflates HR@K by 10–50×; (c) NDCG with 1 positive is fine but
per-row averaging is not.
**Solutions.** Deduplicate by (user, item) using the user's *last* history timestamp proxy; always
report `full-corpus` numbers next to `sampled`; macro-average over users.
**Deliverables.** `src/tmm/metrics.py`, `artifacts/stats/eval_*.json`.

## T8 — Figures
Label balance; sequence-length CDF; item popularity long-tail (log-log); embedding int8 histogram;
recall/latency tradeoff curves; two-stage funnel diagram; training curves; baseline-vs-model bars;
architecture diagram of both directions. All as PNG at 160 dpi + the generating script.

## T9 — Report + pitch
`artifacts/reports/TAOBAO_MM_REPORT.md` — the single deliverable, containing: verified dataset
facts, the two architectures with diagrams, the **task table**, for each task **Method / Problems /
Solutions**, measured results, a "what I would do with more money" section, and a **Sources** table
of papers + GitHub repos.

---

## Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | OOM on the 44 GB `train_user_features` | High | High | row-group streaming only; never `read_table` |
| R2 | Disk exhaustion (101 GB) | Medium | High | `MAX_ROWS` budget; zstd; delete intermediates after use |
| R3 | Spark shuffles 4.5 GB embeddings | High | High | bucketed sort-merge join + AQE; subset run |
| R4 | GPU hidden by the sandbox → GPU runs need escalation | Certain | Low | one long background job per phase; CPU fallback path kept (`--device cpu`) |
| R4b | OOM on 47 GiB VRAM when gathering `[B,1000,128]` history | Medium | Medium | chunked gather + `torch.bfloat16`; keep the 9 GiB item table resident, not the histories |
| R4c | This GPU is **shared** (7 foreign processes, ~9 GiB) | High | Medium | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, batch-size probe, never assume all 48 GiB |
| R5 | Metrics inflated by sampled negatives | High | High | dual-protocol reporting (T7) |
| R6 | Neural ranker loses to popularity | Medium | Medium | baselines first, then ablation table |
| R7 | `faiss-cpu`/`onnxruntime` wheel missing for cp313 | Medium | Medium | **resolved**: faiss-cpu 1.15.1 + onnxruntime 1.30.0 installed OK |

## Time-box (single engineer, this machine)

| Block | Tasks | Est. |
|---|---|---|
| Day 1 | T0–T2, T3a | 6 h |
| Day 2 | T3b, T4, T7 | 6 h |
| Day 3 | T5a, T5b | 8 h |
| Day 4 | T6a–T6c | 6 h |
| Day 5 | T8, T9 | 4 h |

## T6d — Demo artifact: making the sizing claim true
**Found during backend verification.** The real `Recommender` peaked at **7,025 MB RSS**, which
contradicted the documented "2 vCPU / 4 GB VPS" sizing. Two causes, both measured:

1. `load_embeddings_subset` scatter-gathers ~870 k rows out of the 4.5 GB memmap. Each row is
   128 B, so neighbouring rows share no page; the kernel faults in a large fraction of the
   mapping, and **mmap-resident pages count toward RSS** — a 210 MB output cost ~1.5 GB.
2. `load_prepared()` loaded the full 35.46 M-entry item vocabulary and both splits (~1.5 GB) only
   to answer "which row is this item id?" and to read the cardinalities.

**Solution.** Precompute one self-sufficient offline artifact (`src/tmm/demo_artifact.py`,
`python -m tmm.cli build-demo-artifact`) holding the reduced fp16 table, the sorted raw ids, and
the catalogue rows/ids/categories. The service then needs neither the 4.5 GB memmap nor the
35.46 M vocabulary.

| path | peak RSS | startup |
|---|---|---|
| legacy (`load_prepared` + memmap gather) | **7,028 MB** | 8.9 s |
| demo artifact | **790 MB** | 1.1 s |

**8.9× lower peak, 8× faster startup.** (My first number was 745 MB from an ad-hoc measurement;
the authoritative figure from `python -m tmm.cli measure-memory` is **790 MB**, and the service
keeps the no-copy `np.ascontiguousarray(memmap)` form because copying the 210 MB table costs
+220 MB. The coder verified that no `state_dict` key writes into the embedding buffer, so the
read-only mapping is safe.)

Also fixed while building it: the catalogue was originally 10 k arbitrary vocabulary indices with
only **2.6 %** real item metadata (every other item got a constant category — the same class of bug
that corrupted the retrieval experiment). Rebuilt from the most frequently observed target items:
**100 % category coverage**, 1,694 distinct categories observed at serving time.

Reproduce: `python -m tmm.cli build-demo-artifact` then
`python -m tmm.cli measure-memory --mode artifact` / `--mode legacy`.

---

# Part B — Backend hardening (added after the ML build)

The original build produced a working but thin backend: FastAPI + Redis-or-memory sessions +
FAISS + ONNX, Docker Compose and Caddy, and a load test. A code audit found that the
architecture diagram **overpromised**: Kafka was declared in `docker-compose.yml` and described
in the app docstring, but there was **no producer and no consumer in the codebase**, and the
"idempotent per (user, item)" claim in the docstring was not implemented either. Part B fixes
that and hardens the rest.

Measured backend surface before Part B: **1,195 LOC** (34 % of the codebase), the largest single
slice — strongest on packaging/serving/performance, weakest on event-driven plumbing.

**Frozen interface contract:** [`docs/BACKEND_CONTRACT.md`](docs/BACKEND_CONTRACT.md).
Design + failure analysis: [`docs/BACKEND_DESIGN.md`](docs/BACKEND_DESIGN.md).

## Cockpit view

| # | Task | Status | Owner | Verification |
|---|---|---|---|---|
| B1 | Kafka producer + consumer + idempotency + DLQ | in progress | `backend-coder` | `backend-verifier` |
| B2 | API-key auth + token-bucket rate limiting | in progress | `backend-coder` | `backend-verifier` |
| B3 | JSON logs + Prometheus metrics + per-stage latency | in progress | `backend-coder` | `backend-verifier` |
| B4 | `/livez` + `/readyz`, circuit breaker, timeouts, graceful shutdown | in progress | `backend-coder` | `backend-verifier` |
| B5 | CI workflow + ruff/mypy gates | in progress | `backend-coder` | `backend-verifier` |
| B6 | IaC / deploy automation | **deferred** | — | — |
| B7 | Shared vector index (Redis/KNN) + feature store | **deferred** | — | — |
| B8 | Pipeline DAG + model registry | **deferred** | — | — |

## B1 — Kafka ingestion path
**Method.** `POST /events` publishes compact JSON to `tmm.clickstream` keyed by `user_id` so a
user's events stay on one partition and in order (`acks="all"`, `max_in_flight=1`). A **separate
process** (`python -m tmm.serve.consumer`) consumes into the Redis session list.
**Problems.** At-least-once delivery duplicates events; a poison message can block a partition
forever; a broker outage must not break the write path.
**Solutions.** Idempotency key (`SET idem:{event_id} 1 NX EX ttl`) makes replays harmless;
malformed events go to a DLQ and the offset is **committed anyway**; when the broker is unhealthy
the API dual-writes directly to the session store (`degraded` mode) and reports it in `/health`.

## B2 — Auth + abuse control
**Method.** `X-API-Key` required when `TMM_API_KEYS` is set; hand-rolled token bucket keyed by API
key, falling back to client IP.
**Problems.** A public demo URL is scraped within hours; a missing/unresolvable IP must not become
a bypass; Starlette runs sync handlers in a threadpool so the limiter must be thread-safe.
**Solutions.** 401 without a key, 429 + `Retry-After` when the bucket empties, a shared bucket for
unknown identities, and a loud one-time warning when auth is disabled rather than a silent bypass.

## B3 — Observability
**Method.** stdlib `logging` with a custom single-line JSON formatter; `prometheus-client` with a
**per-app CollectorRegistry**; a `StageTimer` instrumenting `decode`, `session_read`, `tower`,
`search`, `total`.
**Problems.** In-memory counters vanish on restart and cannot attribute p95 to a stage — which is
precisely why the earlier load test needed a `/health` control to separate inference cost from
harness cost.
**Solutions.** 9 named metrics (contract §3.6) exposed in Prometheus text format, so the next
latency question is answerable from data instead of by adding a control endpoint.

## B4 — Resilience
**Method.** `/livez` (liveness) separated from `/readyz` (readiness, 503 in `minimal` mode); a
hand-rolled circuit breaker with an **injectable clock**; per-stage timeouts; producer/store closed
on lifespan shutdown.
**Problems.** Silent degradation hides incidents; `sleep()`-based breaker tests are slow and flaky;
killed in-flight requests during redeploys.
**Solutions.** Explicit three-state machine (`closed`/`open`/`half_open`) testable by advancing a
fake clock, a `circuit_state` gauge for alerting, and graceful Shutdown that drains the in-flight
message before committing the offset.

## B5 — CI + quality gates
**Method.** GitHub Actions running `ruff`, `mypy src/tmm/serve`, the service-free test suite, and
a Docker build; config in `pyproject.toml`.
**Problems.** No automated verification means a silent regression (the class of bug this project
already hit three times) ships unnoticed.
**Solutions.** Gates that fail the build on lint, type, or test regressions.

## Acceptance (Part B)
1. `pytest -q` passes with **no Redis and no Kafka running**.
2. Integration tests pass against **real** Redis + KRaft Kafka (Docker is available on this host).
3. `ruff check src tests` and `mypy src/tmm/serve` clean.
4. `/metrics` exposes all 9 contract metrics.
5. 401 without a key; 429 after the bucket empties.
6. Killing Redis → `/health` `mode=minimal`, `/readyz` 503, `/recommend` still answers; recovery
   **without restart**.
7. `/events` does not double-write the session when Kafka is healthy.
8. No secrets logged; every log line is valid single-line JSON.
