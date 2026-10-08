# TAOBAO-MM — Real-time Next-Item Prediction (portfolio project)

Two deployment directions for one problem: predict the next product a user will engage with,
from up to **1 000** past interactions, using the **128-d multimodal (SCL) embeddings** shipped
with [TAOBAO-MM](https://huggingface.co/datasets/TaoBao-MM/TaoBao-MM).

| | Direction 1 — Lightweight live demo | Direction 2 — Enterprise / research |
|---|---|---|
| Catalogue | 10 k items, HNSW in-process | 35.46 M items, full vocabulary |
| Compute | CPU only, 2 vCPU / 4 GB VPS | RTX 5880 Ada 48 GB, PySpark |
| Retrieval | FAISS HNSW + ONNX int8 user tower | Two-tower trained on GPU |
| Ranking | popularity / content-KNN fallback | DIN vs MUSE-style search (measured A/B) |
| Goal | a URL you can click, p95 measured | honest AUC / NDCG@K vs strong baselines |

Full write-up: **[`artifacts/reports/TAOBAO_MM_REPORT.md`](artifacts/reports/TAOBAO_MM_REPORT.md)**
Task breakdown: **[`TASKS.md`](TASKS.md)**

---

## Verified dataset facts (measured locally, not copied from the README)

| Fact | Value | How |
|---|---|---|
| Train / test rows | 76,015,123 / 22,979,465 | parquet footers |
| Positive rate | **13.70 % / 13.71 %** (1 : 6.3) | exact DuckDB full scan |
| Candidates per user | **10.95 / 11.24** | exact |
| Sequence length | mean **978**, **96.3 %** hit exactly 1000 | 1.05 M users streamed |
| Item popularity Gini | **0.815** | exact |
| Top-100 items' share of **clicks** | **65.4 %** | exact |
| Embedding coverage of target items | **99.9994 %** (not 100 %) | 3.78 M ids checked |
| History vocab covered by embeddings | 35.46 M / 243.36 M = **14.6 %** | feature_map |
| SCL dequantisation scale | **1/127**, giving row norm 0.966 ≈ 1 | official `convert_scl_int8()` |

Three findings that change the design:

1. **The labels are not 1:1.** The shipped test set is already a hard-negative candidate set
   (~11 per user). That makes honest per-user ranking possible without inventing negatives —
   and makes the popularity baseline genuinely strong.
2. **There is no timestamp column.** A temporal split cannot be re-derived; the shipped split
   must be trusted, and that limitation is stated rather than papered over.
3. **Shards are user-grouped — verified, not assumed.** One test shard holds 44,570 users at
   11.22 rows each, and **99.9 %** of those users have their complete candidate set inside that
   shard. Histories and candidate sets are still sourced separately (histories from shards,
   candidates from the complete `*_samples` tables) so the pipeline does not depend on that
   undocumented locality.

## Measured results

| Stage | Engine | Result |
|---|---|---|
| Join labels x embeddings | DuckDB 1.5.6, single node | 2.0 M rows in **28.8 s** (69,491 rows/s) |
| Same join, distributed | PySpark 4.2 `local[4]` | 500 k rows in **6.45 s** (77,533 rows/s) → 16 min extrapolated for 76 M |
| Join safety | — | retention 99.9995 %, label rate shift < 1e-6 |
| Ranking (best) | DIN / MUSE | AUC **0.6103 / 0.6097** vs **0.461** popularity (below random), 0.5615 training-free cosine; HR@1 0.313 vs 0.238 random |
| Retrieval fix | two-tower + residual | Recall@10 **0.000 → 0.009** (1.57× the cosine baseline) |
| Demo quality | served path, held-out clicks | HR@10 **0.0435** (random 0.001; training-free 0.050) |
| Demo latency | ONNX-INT8 + HNSW, one request | tower **~13 µs**, index p95 **0.033 ms**, 98.1 % recall@10 |
| Frozen item table on GPU | torch 2.14 bf16 | 35,458,499 x 128 = **8.45 GiB**, 29.5 GiB VRAM left |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src

.venv/bin/python -m tmm.cli doctor          # hardware + dataset inventory
.venv/bin/python -m tmm.cli profile         # T1 EDA -> artifacts/stats/profile.json
.venv/bin/python -m tmm.cli join-duckdb     # T3a single-node join
.venv/bin/python -m tmm.cli join-spark      # T3b distributed join
.venv/bin/python -m tmm.cli prepare         # histories + complete candidate sets
.venv/bin/python -m tmm.cli train-retrieval # T5a two-tower (GPU)
.venv/bin/python -m tmm.cli train-ranker    # T5b DIN vs MUSE
.venv/bin/python -m tmm.cli evaluate        # T7 metrics vs baselines
.venv/bin/python -m tmm.cli export-onnx     # T6a int8 user tower (benchmark)
.venv/bin/python -m tmm.cli build-demo-artifact  # serving bundle: item vectors + ONNX tower, one checkpoint
.venv/bin/python -m tmm.cli index           # T6a FAISS recall/latency (needs the bundle)
.venv/bin/python -m tmm.cli serve-check     # T6c load test
.venv/bin/python -m tmm.cli figures         # T8 all plots
```

## Layout

```
src/tmm/
  config.py        paths, budgets, device resolution
  io.py            memmapped embedding store, sorted-key index, streaming readers
  vocab.py         item id <-> dense index <-> embedding row  (+ the table loader)
  profile.py       T1 EDA
  join_duckdb.py   T3a single-node join  (+ join-loss invariant check)
  join_spark.py    T3b distributed join  (co-partitioned, no 4.5 GB broadcast)
  prepare.py       histories from shards + COMPLETE candidate sets from *_samples
  models.py        TwoTower, DINRanker (din|muse)
  baselines.py     popularity / category / content-KNN
  metrics.py       HR@K, NDCG@K, MRR, AUC, LogLoss, GAUC (+ sampled-metric guard)
  train.py         GPU training loops
  evaluate.py      scoring + full-catalogue Recall@K
  index.py         FAISS flat vs HNSW, recall/latency sweep, memory table
  export_onnx.py   ONNX + dynamic INT8, FP32 vs INT8 benchmark
  serve/app.py     FastAPI + Redis/in-memory session store + Caddy-ready
  figures.py       all report plots
docker/            compose (demo|full|vector profiles) + Caddyfile
artifacts/         stats/*.json, figures/*.png, bench/*.json, models/, reports/
```

## Backend (Part B)

The Direction-1 service was hardened against a frozen interface contract
([`docs/BACKEND_CONTRACT.md`](docs/BACKEND_CONTRACT.md)); design and failure analysis in
[`docs/BACKEND_DESIGN.md`](docs/BACKEND_DESIGN.md).

| | |
|---|---|
| **Ingestion** | `POST /events` → Kafka (`tmm.clickstream`, keyed by user for partition ordering) → a separate consumer process writes the Redis session. Idempotency key makes at-least-once replay harmless; malformed events go to a DLQ and the offset is committed anyway so a poison message cannot block a partition. |
| **Degradation** | three explicit modes, reported by `/health`: `full` (Redis+Kafka), `degraded` (Redis only, direct write), `minimal` (no Redis, in-process). `/readyz` is 503 in `minimal`; `/livez` is always 200. Redis is recovered **without a restart**. |
| **Auth / abuse** | `X-API-Key` → 401 when missing; hand-rolled token bucket → 429 + `Retry-After`. A request with no resolvable IP is still limited (shared anonymous bucket). |
| **Observability** | single-line JSON logs with redaction, and 9 Prometheus metrics behind `/metrics`, including per-stage histograms (`decode`, `session_read`, `tower`, `search`, `total`). |
| **Resilience** | circuit breaker with an injectable clock, per-stage timeouts, SIGTERM drain, lifespan shutdown. |
| **CI** | `ruff` + `mypy` + the service-free suite + a Docker build. |

### The sizing claim was wrong until it was measured

Direction 1 is supposed to fit a **2 vCPU / 4 GB VPS**. Measured peak RSS of the real service was
**7,028 MB** — the claim was false. Two causes: `load_embeddings_subset` scatter-gathers ~870 k rows
out of the 4.5 GB memmap (each row is 128 B, so neighbouring rows share no page, and **touched mmap
pages count toward RSS**), and `load_prepared()` pulled in the full 35.46 M-entry vocabulary just to
map an id to a row.

Fixed by precomputing one self-sufficient artifact (`python -m tmm.cli build-demo-artifact`,
217 MB, 100 % category coverage). **790 MB and 1.1 s startup**, an 8.9× reduction. Reproduce:
`python -m tmm.cli measure-memory --mode artifact` / `--mode legacy`.

## Constraints honoured

* 41 GiB free RAM, 101 GB free disk, and a **shared** GPU — every stage has a row/user budget,
  streams Parquet row-group by row-group, and never materialises a whole table.
* No GPU is required for the CPU path (`--device cpu`); the GPU path is what makes
  Direction 2 a real build rather than a mock.

## References

See the Sources section of `artifacts/reports/TAOBAO_MM_REPORT.md` — MUSE (arXiv:2512.07216),
SCL (arXiv:2407.19467), the official MUSE implementation, DIN/SIM, FAISS, ONNX Runtime, and the
sampled-metrics critique (Rendle, arXiv:1912.02263).
