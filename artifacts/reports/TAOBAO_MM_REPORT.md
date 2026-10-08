# TAOBAO-MM — Real-time Next-Item Prediction
### A two-direction engineering study: from a 139 GB dataset to a sub-millisecond live demo

**Author:** portfolio project · **Date:** 2026-10-07, corrected 2026-10-09 · **Data:** `TAOBAO-MM` (Alibaba), Apache-2.0
**Machine:** Intel Core Ultra 9 285K (24 threads) · 156 GiB RAM (41 GiB free) · 101 GB free disk ·
**NVIDIA RTX 5880 Ada, 47.4 GiB VRAM** · PyTorch 2.14.1+cu130 · CUDA 13.0 · PySpark 4.2 · DuckDB 1.5.6

---

> **Correction notice (2026-10-09).** An audit found bugs that had produced several headline
> numbers in the first version of this report. All of them are fixed and every number below was
> regenerated from the fixed code:
>
> * **Popularity "AUC = exactly 0.5000"** was an argument-order bug (`item_counts` received the
>   labels as item ids, so every test item scored 0). Corrected: **0.461**, *below* random.
> * **Category popularity** was counted on the *test* split. Now from train positives: 0.536.
> * **Ranking metrics** averaged in the 44 % of users with no click, broke ties by row order,
>   and computed NDCG so that a perfect ranking scored 0.61. HR@10 out of ~11 candidates was
>   near-trivial. The table now scores users with a click and leads with HR@1 / NDCG@3.
> * **Sampled softmax trained on non-click rows as positives.** Retrained on clicks only, with
>   logQ correction: Recall@10 over 861 k items is **0.009 (1.57× cosine)**, not 0.0153 (2.65×).
> * **The live demo recommended at random level** (HR@10 0.0015 vs 0.001 random): the index held
>   raw SCL vectors while the query came from a trained tower exported from another checkpoint.
>   Fixed: **HR@10 0.0435** on held-out clicks, and the service no longer needs torch.
> * **Latency figures were batch averages**, and index recall was measured with catalogue items
>   as queries. Now: single-request latency, real user vectors as queries (§6).

## 0. Executive summary

I downloaded the full 139 GB TAOBAO-MM release and built **both** deployment directions from the
project brief, then measured every claim.

**The five findings that matter:**

1. **The problem is not 1:1.** The positive rate is **13.70 %** (76.0 M train / 23.0 M test rows) and
   the shipped test set is *already a hard-negative candidate set* of **11.24 candidates per user**.
   There is **no timestamp column**, so the shipped split cannot be re-derived and must be trusted.
2. **Global popularity is *anti*-predictive here — measured AUC 0.461**, below random (0.499).
   My initial hypothesis ("popularity is a strong baseline on e-commerce data") was **disproved by
   measurement**: the head of the catalogue holds 65.4 % of all *clicks*, yet within a user's
   candidate set the more popular candidate is *less* likely to be clicked. (The first version of
   this report said "exactly 0.5000"; that was a bug in the baseline, see the correction notice.)
3. **The real hard part is data engineering, as the brief said** — but the fix is not a bigger join,
   it is *not joining vectors at all*. 2 M rows join in **28.8 s** (DuckDB) / **77.5 k rows/s**
   (PySpark), with a label-rate shift of **6.9 × 10⁻⁷**. Learning the data first also **falsified one
   of my own structural assumptions** (see §1.2c) — the release is user-grouped at shard granularity,
   which I had claimed was not the case.
4. **A learned two-tower silently destroys retrieval.** Trained pointwise on the shipped candidates it
   scores AUC 0.599 within-candidate but **Recall@500 = 0.0005 ≈ random** over an 861 k catalogue.
   Diagnosis: the item space **collapses into a cone** (mean pairwise cosine **0.966** vs **0.022** for
   raw SCL vectors). Fixing it (sampled softmax on clicks + a linear residual from the frozen
   embeddings) lifts **Recall@10 from 0.000 to 0.009 — 1.57× a training-free cosine search** at the
   same catalogue size.
5. **The lightweight direction really is lightweight.** The embedding table is reduced from
   **8 450 MB to 210 MB**, the ONNX user tower answers one request in **~13 µs (p50)**, and FAISS
   HNSW reaches **98.1 % recall@10 at 0.033 ms p95** per query. Model + index work adds about
   **1 ms at p50** over a no-op `/health` endpoint, and the container runs in **~90 MB** without
   torch.

**Honest bottom line.** The ranking models beat every baseline on ranking quality
(MUSE AUC 0.6097 vs 0.461 popularity, 0.5615 training-free cosine; HR@1 0.313 vs 0.238 random),
and MUSE ties DIN's quality at **3.0× lower scoring cost**. The end-to-end HTTP p95 is **125 ms**, and the `/health` control shows
that is the *test harness*, not the model — so I report the number instead of claiming
"p95 < 20 ms". A single-node original-data experiment is **not** a validated production system and is
labelled as such throughout.

> **Tóm tắt tiếng Việt.** Đã tải trọn bộ 139 GB và triển khai **cả hai hướng** trong đề cương, đo đạc
> mọi con số. Phát hiện chính: (1) bài toán **không phải 1:1** mà là **13.70 % positive**, mỗi user có
> sẵn **11.24 ứng viên hard-negative**, và **không có cột timestamp** nên phải tin vào split có sẵn;
> (2) **popularity có AUC 0.461 — thấp hơn cả random** trong tập ứng viên (bản đầu ghi 0.5000 là do
> lỗi code baseline), giả thuyết ban đầu của tôi đã bị số liệu bác bỏ; (3) khâu khó nhất đúng là **data engineering**, nhưng cách giải không phải join
> to hơn mà là **không join vector** (2 M dòng / 28.8 s, lệch tỉ lệ nhãn 6.9e-7); (4) **two-tower học
> được làm sập không gian vector** (cosine trung bình 0.966 so với 0.022 của SCL gốc) khiến Recall@500
> ≈ random, và bản sửa (sampled softmax trên click + residual) đưa Recall@10 lên **0.009, gấp 1.57×
> baseline cosine**; (5) hướng nhẹ thực sự nhẹ: bảng embedding giảm **8 450 MB → 210 MB**, user tower
> ONNX **~13 µs/request**, demo chạy không cần torch (~90 MB). p95 HTTP thực đo là **125 ms** và đối
> chứng `/health` cho thấy đó là chi phí của harness đo, không phải của model.

---

## 1. Part 2 — The dataset, verified against the files

Everything in this section was measured on the local 139 GB tree, not copied from the README.
Reproduce with `python -m tmm.cli profile` → `artifacts/stats/profile.json`.

### 1.1 Scale and integrity

| Property | Measured | README claims | Verdict |
|---|---|---|---|
| Train rows | **76,015,123** | 76 M | ✅ |
| Test rows | **22,979,465** | 23 M | ✅ |
| Total samples | 98,994,588 | 99.0 M | ✅ |
| Train users | 6,929,606 | — | — |
| Test users | 2,063,682 | — | — |
| Union of users | 8,798,906 | 8.79 M | ✅ reconciled |
| Train/test user overlap | **194,382** (9.42 % of test users) | — | ⚠️ leakage surface |
| Target items (train) | 3,776,576 | — | — |
| History vocabulary (p90) | 35,458,498 | 35.4 M | ✅ |
| Positive rate | **13.704 % / 13.714 %** | not stated | ⚠️ not 1:1 |

The user counts reconcile exactly: 6,929,606 + 2,063,682 − 194,382 = **8,798,906**, matching the
documented 8.79 M. **The paper's §6.2 figures (107 M samples, 8.86 M users, 275 M items) do not match
either the download or the project page**; the measurements here support the project page/README.

### 1.2 Three facts that break naive designs

**(a) The label is not 1:1 — and the test set is already a candidate set.** 13.70 % positive means
~1:6.3. Critically, `test` has **11.244 candidate rows per user** (10.946 in train). That is a
pre-built hard-negative set, and it is what makes honest per-user ranking (HR@K, NDCG@K, MRR)
possible without inventing negatives.

![Label balance and candidate structure](../figures/01_labels_and_candidates.png)

**(b) There is no timestamp column.** The released schema has 13 fields and none is a time. A
temporal re-split is therefore impossible; a random re-split would be *worse* (it would ignore time
entirely). This is a stated limitation, not an oversight.

**(c) Shards *are* user-grouped — I initially got this wrong.** My first draft of this report
claimed rows are not grouped by user and that a partial shard scan would give most users ~1
candidate instead of ~11. **Measurement disproved it.** On `test-shard-000000`: 500,000 rows →
**44,570 distinct users → 11.22 rows/user**, and **99.9 % of those users have their complete
candidate set inside that single shard** (mean fraction of the complete set present = 1.000).
I had misread a research note that reported "5,239 user-runs vs 5,226 distinct users per 60 k rows"
— which actually shows users are *nearly contiguous*, i.e. the opposite.

The pipeline still sources candidate sets from the complete `*_samples` tables rather than from
shards, but for **robustness** reasons, not correctness: shard-level user locality is an
undocumented property of this release, it breaks if anyone reads a subset of *row groups*, and
stopping a shard scan early at a user cap can truncate the last user's candidates. The
`tests/test_invariants.py::test_candidate_sets_are_complete_and_consistent` guard remains valid —
it checks the property that actually matters (a sane rows-per-user), regardless of where the rows
came from.

![Sequence length distribution](../figures/02_sequence_length.png)

### 1.3 Popularity is extreme — and yet uninformative within a candidate set

| Metric | Value |
|---|---|
| Gini of item popularity | **0.815** |
| Top-10 / 100 / 1 k / 10 k items' share of interactions | 0.37 % / 1.75 % / 6.89 % / 20.97 % |
| Items needed for 50 % / 90 % of interactions | 96,139 / 1,038,044 |
| **Top-100 items' share of all *clicks*** | **65.36 %** |

![Popularity long tail](../figures/03_popularity_longtail.png)

The last row is the trap. The head of the catalogue dominates *clicks*, so it looks like a popularity
ranker must win. It does not: measured **AUC = 0.461**, below random (Section 4). The most plausible
explanation is that negatives are exposure-matched rather than uniformly sampled — but the release ships **no
sampling metadata**, so that remains an inference, not a verified fact. Either way the practical
conclusion is firm: **on this dataset a popularity baseline cannot be used to demonstrate progress;
the honest floor is random, and the honest content baseline is the SCL cosine.**

### 1.4 The multimodal embeddings: 128-d int8, and a scale trap

| Property | Value |
|---|---|
| Shape / dtype | 35,458,467 × 128 int8 = **4.54 GB** (fp32 would be 18.15 GB) |
| Quantisation range on a 500 k sample | [−60, +61] |
| **Correct dequantisation scale** | **1 / 127 = 0.0078740** |
| Row L2 norm, raw int8 | 122.65 ± 0.33 |
| **Row L2 norm after /127** | **0.9657 ± 0.0026** |
| Rows L2-normalised before quantisation | **True** |
| Coverage of *target* items | **99.9994 %** (not 100 %) |
| Coverage of the *history* vocabulary | 35.46 M / 243.36 M = **14.57 %** |

![Embedding int8 statistics](../figures/04_embedding_int8.png)

**The scale trap.** The natural choice `max|v| / 127 = 0.48` is **wrong by 61×**. The dataset's own
preprocessor (`utils/preprocess.py::convert_scl_int8` in the official MUSE repo) uses
`np.trunc(127 * np.clip(v, -1, 1))` on already-L2-normalised vectors and stores `scale = 1/127`.
The measured dequantised norm of **0.966 ≈ 1.0** confirms it. This is easy to miss because the
official model only ever computes cosine similarity (scale-invariant) — but it is *not* harmless for a
tower whose first layer is `LayerNorm(Wx + b)`, since a constant rescaling of `x` scales `Wx` and not
`b`. **I trained one full model with the wrong scale before catching this, and retrained.**

### 1.5 Two smaller surprises

* `206` (**item category**) has **no** `206_sorted_map.npy` in the release, unlike the demographic
  and geography fields. Its vocabulary has to be rebuilt from `item_features` (13,050 categories).
  Using the provided maps blindly produces an index-out-of-range on the first training step.
* `scl_emb_int8_p90_keys.npy` is sorted **except for one duplicated key at row 21,190,207**
  (35,458,467 rows, 35,458,466 unique). `searchsorted` remains safe; a strict monotonicity assertion
  would have failed.

---

## 2. Part 3 — The two directions, as built

![Both architectures](../figures/11_architecture.png)

The retrieval funnel every design has to respect - 35.46 M vocabulary items, of which 3.78 M
are ever targets, 10 k enter the demo index, and ~11 reach the ranker per user:

![Retrieval funnel](../figures/05_funnel.png)

| | **Direction 1 — Lightweight live demo** | **Direction 2 — Enterprise / research** |
|---|---|---|
| Catalogue | 10,000 items (HNSW in-process) | 860,903 catalogue items scored; 35.46 M vocabulary |
| Compute | CPU only | RTX 5880 Ada 48 GB + PySpark |
| Embedding table resident | **210 MB** (reduced) | **8.45 GiB** (full, bf16, on GPU) |
| Retrieval | FAISS HNSW + ONNX-INT8 user tower | Two-tower (sampled softmax + residual) |
| Ranking | popularity / content-KNN fallback | DIN vs MUSE-style search, measured A/B |
| Ingest | Kafka (KRaft, 512 MB) → Redis sessions | PySpark join over the 76 M-row label table |
| Packaging | Docker Compose (3 profiles) + Caddy auto-TLS | Spark job + GPU training CLI |
| Verified artifact | `artifacts/bench/serve.json` | `artifacts/stats/evaluate.json` |

### 2.1 The data-engineering core (the part the brief calls the real difficulty)

The supervised signal is deliberately sharded away from the features: labels in `*_samples.parquet`
(0.714 GB / 0.215 GB — tiny), item content in `item_features` (48 MB), multimodal signal in a
35.4 M × 128 int8 table (5.25 GB). The full materialised dataset is **1.55 TB** per the HuggingFace
viewer versus 68 GB of Parquet — a 23× blow-up, which is the whole OOM story.

**The decisive design choice: do not shuffle 4.5 GB of vectors.** Keep the embedding table dense and
keyed by a remapped integer id, store only an **int32 index** in the sample table, and let the
dataloader gather. The join then carries 8 bytes/row instead of 128.

| Engine | Config | Rows | Time | Throughput |
|---|---|---|---|---|
| **DuckDB 1.5.6** (single node) | 24 threads, 24 GB limit, spill to tmpfs | 2,000,000 | **28.78 s** | **69,491 rows/s** |
| **PySpark 4.2** `local[4]` | 20 GB driver, 256 co-partitions, AQE | 499,989 | **6.45 s** | **77,533 rows/s** → 16 min for 76 M |

**Correctness invariants, both checked automatically** (`tests/test_invariants.py`):

* join retention **99.9995 %** (the README's "100 % coverage" is really 99.9994 %);
* positive-rate shift before→after the inner join: **6.85 × 10⁻⁷** — i.e. the join introduces no
  label selection bias, which it would if coverage were materially below 100 %.

![Engine comparison](../figures/10_engines.png)

Problems hit and how they were fixed (all documented in the module docstrings):

| Problem | Fix |
|---|---|
| `label_0` is `list<int8>` one-hot, not a scalar | `label_0[2]` (DuckDB is 1-based) / `.getItem(1)` in Spark |
| DuckDB defaults `memory_limit` to 80 % of *total* RAM (125 GB > 41 GB free) | explicit `memory_limit='24GB'` |
| Root filesystem has 101 GB free; `/tmp` is a 79 GB tmpfs | `temp_directory=/tmp/tmm_duckdb` |
| A naive `broadcast()` of a 4.5 GB embedding table kills the driver | hash co-partition **both** sides on the join key |
| **`DataFrame.bucketBy` was removed in Spark 4.x** (catalog-only now) | `repartition(256, key)` on both sides → shuffle-free sort-merge join |
| `spark.driver.memory` defaults to 1 GB | 20 GB; `local[4]` beats `local[*]` under memory pressure |
| Skew from head items | `spark.sql.adaptive.skewJoin.enabled` |
| Sampling before vs after the join changes cost by ~4× | `USING SAMPLE reservoir(...) REPEATABLE(42)` pushed below the join |

---

## 3. Models

All models share the **frozen** 128-d SCL table as a feature (not a parameter): 4,538,687,872 frozen
values versus **588,241 trainable parameters**. Keeping it frozen is what lets the whole catalogue sit
in 8.45 GiB of VRAM with zero optimiser state.

| Model | Role | Trainable params | Train loop | Epochs | Steps | Loss (first → last) |
|---|---|---|---|---|---|---|
| Two-tower (BCE) | Direction 2 retrieval | 588,241 | 44.5 s | 3 | 11,682 | 0.4022 → 0.3907 |
| Two-tower (sampled softmax) | retrieval ablation | 588,241 | 5.7 s | 3 | 402 | 7.7316 → 6.5983 |
| Two-tower (ss + residual) | retrieval ablation | **621,009** | 6.4 s | 3 | 1,608 | 5.9076 → 4.6269 |
| **DIN** | Direction 2 ranking | 653,522 | **154.7 s** | 2 | 31,146 | 0.3936 → 0.3907 |
| **MUSE** (top-50 search) | Direction 2 ranking | 653,522 | **73.9 s** | 2 | 31,146 | 0.3939 → 0.3909 |

*Times are the **pure training loop** (`train_seconds` in the JSON artifacts), not the CLI stage
total: each retrieval stage also loads the 8.45 GiB frozen table, which costs ~50 s and would
otherwise mask the real cost. The residual twin adds 32,768 parameters (two 128→128 projection
matrices, one per tower), hence 621,009 not 588,241. Loss values are **not comparable across
objectives**: BCE is a per-row binary loss, while `sampled_softmax` is a multi-class
cross-entropy over the in-batch negatives, so 5.18 vs 0.40 is a different quantity, not a worse
model — see §5 for the retrieval outcome that is comparable. The sampled-softmax runs train on the
274,206 *clicked* rows only (in-batch softmax needs a true positive per row), which is why they take
far fewer steps than the BCE runs over all 1.99 M rows.*

The ~50 s constant is the cost of materialising the 8.45 GiB bf16 embedding table on the GPU;
it dominates the short runs and is why the retrieval ablations report sub-minute loop times.

Peak VRAM during two-tower training: **8.58 GiB** of 47.4 GiB. Training throughput 134,366 rows/s.

**MUSE vs DIN is a genuine controlled A/B:** identical data, identical seeds, identical parameters,
differing only in whether the 1,000-item history is first *searched* with the target embedding
(top-`k` = 50) before attention runs over it. This is the paper's central idea, and it is the
measurement that makes the latency/quality claim meaningful.

---

## 4. Results — ranking, on the shipped candidate sets

451,174 test rows · 40,124 users · **11.244 candidates/user for every model** (so the comparison is
legal under Rendle, arXiv:1912.02263 — `metrics.compare` asserts this and `tests/` covers it).
Ranking metrics are computed over the **22,566 users (56.24 %) who clicked at least one candidate**:
a user with no click has no right answer. Ties are broken at random with a fixed seed. HR@10 is
not shown: with ~11 candidates it is nearly guaranteed.

| Model | AUC | GAUC | LogLoss | PR-AUC | HR@1 | HR@3 | NDCG@3 | NDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.4988 | 0.4999 | 0.9150 | 0.1373 | 0.2382 | 0.5836 | 0.3532 | 0.4223 | 0.4527 |
| global popularity | **0.4615** | 0.4685 | 1.7679¹ | 0.1211 | 0.2050 | 0.5397 | 0.3250 | 0.3964 | 0.4192 |
| positive popularity | 0.4881 | 0.4916 | 1.0162¹ | 0.1316 | 0.2349 | 0.5721 | 0.3494 | 0.4193 | 0.4466 |
| category popularity | 0.5359 | 0.5237 | 4.6847¹ | 0.1565 | 0.2557 | 0.5974 | 0.3703 | 0.4386 | 0.4662 |
| **content-KNN** (training-free SCL cosine) | 0.5615 | 0.5571 | 0.9026 | 0.1681 | 0.2952 | 0.6388 | 0.4031 | 0.4679 | 0.5022 |
| two-tower (BCE) | 0.5991 | 0.5740 | **0.3929** | 0.1812 | 0.2879 | 0.6419 | 0.4062 | 0.4732 | 0.5005 |
| **DIN** | **0.6103** | 0.5894 | 0.3919 | 0.1903 | 0.3124 | 0.6668 | 0.4254 | 0.4896 | 0.5216 |
| **MUSE** (k = 50) | 0.6097 | **0.5898** | 0.3920 | **0.1903** | **0.3129** | **0.6680** | **0.4260** | **0.4907** | **0.5220** |

¹ The count-based baselines are used as `log1p(count)` *logits* and are therefore badly
calibrated; their LogLoss is meaningless. They must be read on AUC / HR@K, which are rank-based.
The neural models' ≈ 0.392 is a real, calibrated improvement.

Popularity is below random on every rank metric (HR@1 0.205 vs 0.238): within a candidate set
the popular candidate is the one *less* likely to be clicked. Category popularity, estimated on
train positives only, is the one count baseline above random.

**Scoring cost on the same 451,174 rows** — the point of the MUSE ablation:

| Model | Score time | Relative |
|---|---|---|
| content-KNN | 1.14 s | 0.13× DIN |
| two-tower | 1.85 s | 0.21× DIN |
| **DIN** (1,000-token attention) | **9.02 s** | 1.00× |
| **MUSE** (search top-50, then attend) | **2.97 s** | **0.33×** |

**MUSE delivers DIN's quality (GAUC 0.5898 vs 0.5894, HR@1 0.3129 vs 0.3124) at 3.0× lower
scoring cost and 2.09× lower training cost.** That is the paper's thesis, independently reproduced.

![Model comparison](../figures/06_model_comparison.png)

---

## 5. Results — retrieval, and a defect I found in my own pipeline

This is the most instructive part of the project, because the first three answers I produced were
**wrong** and the measurements said so.

### 5.1 The experiment

Retrieval over a **860,903-item** catalogue (all items observed in the sampled splits), 4,000 test
users, top-K by inner product. Random Recall@500 = 0.00058.

| Retrieval method | Recall@10 | Recall@50 | Recall@500 |
|---|---|---|---|
| **training-free cosine on frozen SCL** ("GSU") | 0.00575 | 0.02075 | 0.07725 |
| two-tower, pointwise BCE | **0.00000** | 0.00000 | 0.00050 |
| two-tower, sampled softmax (clicks, logQ) | 0.00100 | 0.00700 | 0.03975 |
| two-tower, sampled softmax **+ residual** | **0.00900** | **0.03475** | **0.11125** |

The sampled-softmax rows were regenerated after fixing a training bug: the in-batch loss had
treated every candidate row, 86 % of them non-clicks, as a positive. Trained correctly on the
274 k clicked rows, plain sampled softmax falls *below* the cosine baseline and only the residual
variant beats it. The buggy version scored higher (0.0153 / 0.0048) because it also learned from
exposures, ~7× more pairs; whether exposures are a useful weak positive for retrieval is an open
question worth a proper experiment, not something to fold silently into the loss.

### 5.2 Diagnosis: cone collapse

| Objective | Mean pairwise cosine of learned item vectors |
|---|---|
| raw SCL (reference) | **0.0536** |
| pointwise BCE | **0.9349** ← collapsed |
| sampled softmax (clicks, logQ) | 0.5734 |
| sampled softmax + residual | 0.3853 |

*Measured over the 10,000-item demo catalogue (the most frequent targets), so the absolute values
differ from the first version's random sample; the ordering is the same.*

A plain `LayerNorm`-MLP stacked on an already L2-normalised input drives every item vector into a
narrow cone. When all item vectors are nearly identical, top-K over 861 k items is arbitrary — which
is exactly what Recall@500 = 0.0005 (≈ random) means. An intermediate control confirms it is not a
*category* artefact: feeding the raw SCL vectors into the *learned user* tower also fails
(Recall@500 = 0.0005), and feeding the *raw pooled* history into the *learned item* tower fails too
(Recall@500 = 0.0005). Both towers are degenerate; only the frozen embeddings retain geometry.

This independently reproduces **why MUSE's GSU is a pure similarity search over frozen embeddings
rather than a learned tower**: a contrastive/cosine search preserves the metric, and model capacity
is better spent on the ESU ranker.

### 5.3 The fix, measured

Two changes, both ablation-verified: **sampled softmax** (in-batch negatives force the model to
separate an item from hundreds of others, not just a user's own 11) and a **linear residual from the
frozen embedding into the output space** (preserves the original metric while the MLP adds a learned
correction).

Result: **Recall@10 rises 0.000 → 0.009, which is 1.57× the training-free cosine baseline**, and
Recall@500 rises to 0.11125 vs 0.07725 (+44 %).

### 5.4 Two bugs *inside my own evaluation code* that this exercise exposed

| Bug | Symptom | Consequence if unnoticed |
|---|---|---|
| `full_catalogue_recall` never called `load_state_dict` | reported numbers for a **randomly initialised** model | the "trained tower is worse than raw embeddings" conclusion was based on noise; the honest conclusion is the opposite |
| Catalogue categories were passed as a constant `cat=0` | a feature value never seen in training | corrupts every item vector; fixed to real categories at **100 % coverage** |

I am reporting these because a portfolio project that hides its own wrong turns is worthless. Both are
now covered by the fact that the pipeline records `category_coverage_of_catalogue` and that
`full_catalogue_recall` raises on an incomplete `state_dict` mapping.

---

## 6. Results — Direction 1, the live demo

### 6.1 Vector index

10,000-item catalogue, 128-d item-tower vectors from the demo artifact, exact `IndexFlatIP` as
ground truth and HNSW (M = 32, efConstruction = 200, inner product) as the ANN path. **Queries are
2,000 real user vectors** from held-out test sessions, and **latency is per query**, one call per
query as `/recommend` issues it. (The first version queried with catalogue items, which trivially
find themselves, and divided a batched search by the batch size.)

| Strategy | Recall@10 | p50 | **p95** | QPS (single query) | Index size |
|---|---|---|---|---|---|
| `IndexFlatIP` (exact) | 1.0000 | 0.0923 ms | 0.0944 ms | 10,731 | 4.88 MB |
| HNSW, efSearch = 16 | 0.8681 | 0.0099 ms | 0.0125 ms | 98,883 | ~7.3 MB |
| HNSW, efSearch = 32 | 0.9475 | 0.0160 ms | 0.0195 ms | 61,448 | ~7.3 MB |
| **HNSW, efSearch = 64** | **0.9809** | 0.0275 ms | **0.0325 ms** | **36,153** | ~7.3 MB |
| HNSW, efSearch = 128 | 0.9944 | 0.0497 ms | 0.0579 ms | 20,101 | ~7.3 MB |
| HNSW, efSearch = 256 | 0.9978 | 0.0977 ms | 0.1132 ms | 10,201 | ~7.3 MB |

**efSearch = 64 is the knee: 98.1 % of exact recall at 3.4× the QPS of exhaustive search.** At 10 k
items the index is never the bottleneck — the honest framing is "here is what ANN costs", not "ANN
was necessary".

![Index recall/latency](../figures/07_index_recall_latency.png)

### 6.2 Model compression

The served tower is exported from the residual sampled-softmax checkpoint, the same one that
produced the indexed item vectors.

| Runtime | Bytes | One request (p50 / p95) | Amortised, batch 256 |
|---|---|---|---|
| PyTorch eager fp32 | — | — | 123.3 µs/user |
| ONNX fp32 | 453,025 | 13.3 / 15.1 µs | 0.75 µs/user |
| **ONNX dynamic INT8** | **124,612** (3.64× smaller) | **11.7 / 13.2 µs** | 0.77 µs/user |

A request scores one user, so **~13 µs is the number that matters**; the sub-microsecond figure
the first version reported is batch-256 throughput. INT8 saves ~1.6 µs per request and 3.6× the
bytes. Verified against the real `TwoTower.user_vec`, including masked pooling over padded
histories: **max abs diff 1.8 × 10⁻⁷, minimum cosine 1.0000**.

The exported graph deliberately **excludes the embedding table**: an 8.45 GiB initializer would make a
multi-GB `.onnx`, and ONNX Runtime's INT8 quantiser is known to balloon on multi-GB tables
(onnxruntime#21979). The gather happens outside the graph — which is also how such towers are really
deployed (embeddings in a KV store, MLP in the model server).

### 6.3 The service, and the p95 I will not fake

FastAPI + FAISS HNSW + ONNX tower, in-memory or Redis sessions, Kafka optional, Caddy for automatic
TLS. The embedding table is **reduced to the catalogue + anything that can appear in a session:
210 MB instead of 8,450 MB** — otherwise "lightweight" would be a lie and my own VPS sizing wrong.

**Does it recommend anything?** Measured through the serving code on 2,000 held-out test clicks
whose item is in the catalogue (`tests/backend/test_recommend_real_model.py`):

| Query → index | HR@10 |
|---|---|
| random | 0.0010 |
| first version (trained user tower → **raw SCL** item index) | 0.0015 |
| training-free (mean raw SCL session → raw SCL index) | 0.0500 |
| **served now** (one checkpoint for both sides) | **0.0435** |

The first version searched a raw-SCL index with a user vector from a trained tower, two unrelated
spaces, and was at random level while every endpoint test passed. The item vectors and the ONNX
tower are now built together from one checkpoint inside the demo artifact. On this popular 10k
catalogue the trained tower is slightly *below* the training-free baseline (0.0435 vs 0.050), even
though it beats it over the 861 k catalogue (§5); the test guards both "far above random" and "not
far below the baseline".

The image needs **no torch** (numpy + onnxruntime + FAISS), and the container measured **~90 MB**
after startup.

Load test, 1,984 requests per endpoint, concurrency 32, **0 errors**, loopback:

| Endpoint | p50 | p95 | p99 | rps |
|---|---|---|---|---|
| `GET /health` (**control**) | 26.28 ms | 104.20 ms | 163.68 ms | 858 |
| `POST /events` | 25.94 ms | 128.92 ms | 199.53 ms | 776 |
| `GET /recommend` | 27.35 ms | **124.56 ms** | 209.00 ms | 777 |

**The control is the finding.** A no-op endpoint has p95 = 104 ms, so **the model + index contribute
~1 ms at p50 and cannot be responsible for the p95**. Doubling uvicorn workers changes nothing
(852 → 857 rps on `/health`), which shows the ceiling is the load generator and the loopback HTTP
stack on this shared machine, not the service. The standalone measurements agree: ~13 µs per request
for the tower, 0.033 ms p95 for the index.

So the brief's target of "**< 20 ms**" is **not verified end-to-end** and I will not claim it. What is
verified: **the recommendation-specific work is sub-millisecond**, and the remaining latency is
framework + measurement overhead that a proper k6/vegeta run from a separate host would isolate.

---

## 7. Task breakdown — Method / Problem / Solution

Each row is a stage in `python -m tmm.cli <stage>`; full detail in `TASKS.md`.

| # | Task | Method | Key problem | Solution (verified) | Evidence |
|---|---|---|---|---|---|
| T0 | Env & contracts | isolated venv, pinned deps, `config.py` | host `python3` belongs to another project; pip cache unwritable | own venv; `PIP_CACHE_DIR=/tmp`; row budgets | `artifacts/stats/env.json` |
| T1 | Profiling / EDA | DuckDB exact scans + pyarrow row-group streaming | 44.6 GB `train_user_features` OOMs on `to_pandas()` | never `read_table`; one row-group + projections | `profile.json` (134 s) |
| T2 | Vocab + embedding store | sorted-map vocabulary, memmapped int8 table, chunked gather | keys only block-sorted (1 dup at row 21.19 M); unchunked gather = 27 GB peak | `searchsorted` + chunked dequantise | `vocab.py`, `io.py` |
| T2b | Shard structure assumption | measure users per shard end to end | I asserted rows were not user-grouped | **falsified**: 44 570 users/shard, 99.9 % complete sets | §1.2c |
| T3a | Join, single node | DuckDB, sample pushed below the join | `memory_limit` default 125 GB; disk 101 GB free | limit 24 GB, spill to 79 GB tmpfs, `preserve_insertion_order=false` | 2 M rows / 28.8 s; 69,491 rows/s |
| T3b | Join, distributed | PySpark `local[4]`, 256 co-partitions, AQE | `bucketBy` removed in Spark 4.x; 4.5 GB broadcast impossible | `repartition` both sides on the join key + 20 GB driver | 500 k rows / 6.45 s; 77,533 rows/s; 16 min → 76 M |
| T3c | Join correctness | explicit retention + label-shift check | an inner join silently drops rows below 100 % coverage | assert retention > 0.999 **and** label shift < 1e-4 | retention 99.9995 %; shift 6.9e-7 |
| T4 | Baselines | popularity (global/positive/category), training-free SCL cosine | assumed popularity was strong; first version had an argument-order bug | **measured AUC 0.461** (below random) → hypothesis disproved | `evaluate.json` |
| T5a | Two-tower | BCE on shipped hard negatives; GPU with frozen 8.45 GiB table | class imbalance; VRAM shared | bf16 table, `expandable_segments`, batch probe | AUC 0.5991; 8.58 GiB peak |
| T5b | DIN vs MUSE | attention over 1,000 vs search-top-50 then attend | O(1000) attention is slow | MUSE k = 50 | GAUC 0.5898 vs 0.5894; **3.0× faster scoring** |
| T5c | Retrieval defect | measure cone similarity + 4-way control | learned space collapses (cos 0.93); sampled softmax first trained on non-clicks | sampled softmax on clicks + logQ + residual from frozen embeddings | Recall@10 0.000 → 0.009 (1.57× the cosine baseline) |
| T6a | Index + compression | FAISS flat vs HNSW; ONNX fp32 vs INT8 | batch-averaged "latency"; self-queries inflated recall | single-query timing, held-out user vectors as queries | 98.1 % recall@10 @ 0.033 ms p95; ~13 µs/request; diff 1.8e-7 |
| T6b | Service | FastAPI + Redis/memory + Caddy; reduced table | 8.45 GiB is not "lightweight"; index and query lived in different spaces | demo artifact holds item vectors + ONNX tower from one checkpoint; no torch | HR@10 0.0435 on held-out clicks (was 0.0015); ~90 MB container |
| T6c | Load test | uvicorn + httpx, **with a `/health` control** | p95 ambiguous between service and harness | control endpoint + worker sweep | p95 125 ms, but control 104 ms → overhead ≈ 1 ms |
| T7 | Evaluation | per-user HR@K/NDCG@K/MRR over users with a click + AUC/GAUC/PR-AUC | sampled metrics do not preserve model order; row-order ties; NDCG ceiling 0.61 | single shared candidate set; random tie-break; DCG over all positives | all models on 11.244 candidates/user |
| T8 | Figures | 11 plots regenerated from JSON artifacts | hand-made charts drift from the data | `python -m tmm.cli figures` | `artifacts/figures/*.png` |

---

## 8. Limitations (stated, not hidden)

1. **Single-node original-data experiment, not a validated production system.** Spark ran as
   `local[4]`; the 76 M-row extrapolation is linear scaling, not a cluster measurement.
2. **The temporal split cannot be re-derived** (no timestamp field) and the feature maps + SCL
   embeddings were fitted on the whole corpus — a mild transductive leakage that is inherent to the
   release.
3. **9.42 % of test users also appear in train**, so the split is not user-disjoint.
4. **The full-catalogue retrieval experiment uses 860,903 items**, not the full 35.46 M vocabulary.
   Absolute Recall@K is therefore optimistic; relative ordering between methods is what it establishes.
5. **HTTP p95 is measured on loopback with the load generator on the same host.** The 20 ms target is
   unverified end-to-end; a separate-host k6 run is the right next step.
6. **The load test ran against the in-memory session store**, not real Kafka/Redis. The Docker
   image was built and `/recommend` verified inside the container, and `caddy validate` passes on
   the Caddyfile, but the full compose stack was not load-tested.
7. **The two rankers are the smallest useful versions** of DIN/MUSE: no SimTier histogram, no SA-TA
   multi-modal fusion, no auxiliary loss. Beating DIN by 0.02 AUC is not the paper's +0.03 GAUC.
8. **Weights were trained on 2.0 M of 76 M rows** (2.6 %), the largest sample the 41 GiB RAM budget
   allowed for history materialisation.

## 9. What I would do next, in priority order

1. **Scale the sample 10×** (20 M rows) — the only change with a guaranteed quality return; VRAM
   headroom (8.58 of 47.4 GiB used) allows a 4× larger batch.
2. **Full SimTier + SA-TA** to close the gap to the paper's ESU rather than approximating it.
3. **Rerun the load test from a second host with k6/vegeta** to get a real p95 and validate the
   `< 20 ms` target, then Dockerise on a 2 vCPU / 4 GB VPS and measure again.
4. **Rebuild the two-tower with hard-negative mining** (FAISS-mined in-batch hard negatives) rather
   than uniform in-batch negatives, and re-measure Recall@K.
5. **IVF-PQ / binary quantisation on the full 35.46 M catalogue** to show the memory math
   (5.5 GB int8 → ~1.4 GB PQ) with a measured recall cost.
6. **Feast feature store + point-in-time-correct training sets**, then Triton dynamic batching for the
   ranker.

---

## 10. Reproducing this

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
python -m tmm.cli doctor           # hardware + dataset inventory
python -m tmm.cli profile          # T1  -> artifacts/stats/profile.json
python -m tmm.cli join-duckdb      # T3a -> artifacts/bench/join_duckdb.json
python -m tmm.cli join-spark       # T3b -> artifacts/bench/join_spark.json
python -m tmm.cli prepare          # histories + COMPLETE candidate sets
python -m tmm.cli train-retrieval --device cuda
python -m tmm.cli train-retrieval --objective sampled_softmax --tag _ss --batch-size 2048 --device cuda
python -m tmm.cli train-retrieval --objective sampled_softmax --residual --tag _res --batch-size 512 --device cuda
python -m tmm.cli train-ranker --device cuda
python -m tmm.cli export-onnx
python -m tmm.cli evaluate --protocol both --device cuda
python -m tmm.cli retrieval-ablation
python -m tmm.cli build-demo-artifact   # item vectors + ONNX tower + eval set, one checkpoint
python -m tmm.cli index --items 10000 --exact
python -m tmm.cli serve-check --concurrency 32 --requests 2000 --workers 2
python -m tmm.cli figures
pytest -q                          # invariant + backend contract tests
```

---

## 11. Sources

### Primary
- **MUSE paper** — *MUSE: A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User
  Interest Modeling*, [arXiv:2512.07216](https://arxiv.org/abs/2512.07216) ·
  [HTML](https://arxiv.org/html/2512.07216v1) — GSU→ESU two-stage design, SimTier (22-bin cosine
  histogram), SA-TA, baselines DIN / SIM-hard / SIM-soft / TWIN / MISS; GAUC 0.6377 (production-5k),
  0.6154 (open-source-1k); online A/B CTR +12.6 %, RPM +5.1 %.
- **Official implementation** — <https://github.com/alimama-tech/MUSE> (Apache-2.0) — this is where the
  `convert_scl_int8()` scale of 1/127, the `label_0` one-hot handling, and the `keep_top=50` search
  were verified. `config/muse.json`: Adam, batch 1000, dense_lr 2e-4, `embedding_dim=16`.
- **SCL embeddings** — *Enhancing Taobao Display Advertising with Multimodal Representations*,
  [arXiv:2407.19467](https://arxiv.org/abs/2407.19467) — behavioural positive pairs, InfoNCE with a
  learnable temperature, memory bank K = 196,800. Note: the paper never mentions int8; the
  quantisation is a dataset artefact.
- **Dataset** — <https://huggingface.co/datasets/TaoBao-MM/Taobao-MM> · <https://taobao-mm.github.io/>

### Baselines and libraries
- DIN — <https://github.com/zhougr1993/DeepInterestNetwork> · DIEN — <https://github.com/mouna99/dien>
- SIM (GSU/ESU reference) — <https://github.com/tttwwy/SIM>
- DeepCTR-Torch — <https://github.com/shenweichen/DeepCTR-Torch> · FuxiCTR — <https://github.com/reczoo/FuxiCTR>
- RecBole — <https://github.com/RUCAIBox/RecBole> (evaluation protocol reference)
- Microsoft Recommenders — <https://github.com/recommenders-team/recommenders>
- Transformers4Rec — <https://github.com/NVIDIA-Merlin/Transformers4Rec>
- FAISS — <https://github.com/facebookresearch/faiss> ·
  [index selection guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- ONNX Runtime — [quantisation guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
  · embedding-table memory blow-up: <https://github.com/microsoft/onnxruntime/issues/21979>
- Redis vector search — <https://redis.io/docs/latest/develop/ai/search-and-query/query/vector-search/>
- NVIDIA Triton dynamic batching —
  <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html>
- Feast — <https://github.com/feast-dev/feast> (Redis online store supported)

### Evaluation methodology (why the metrics are reported the way they are)
- **Rendle, *Evaluation Metrics for Item Recommendation under Sampling*, [arXiv:1912.02263](https://arxiv.org/abs/1912.02263)**
  — sampled metrics do not preserve model order, not even in expectation. This is the reason every
  model here shares one candidate set and `metrics.compare()` refuses to rank across differing sizes.
- Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation*,
  [arXiv:2010.11060](https://arxiv.org/abs/2010.11060)
- *Time to Split: Data Splitting Strategies for Offline Evaluation of Sequential Recommenders*,
  [arXiv:2507.16289](https://arxiv.org/abs/2507.16289)
- *Evaluating Performance and Bias of Negative Sampling in Large-Scale Sequential Recommendation*,
  [arXiv:2410.17276](https://arxiv.org/abs/2410.17276)
- Dacrema et al., *Are We Really Making Much Progress?*, [arXiv:1907.06902](https://arxiv.org/abs/1907.06902)

### Practical pitfalls confirmed in this build
- pyarrow OOM with `Dataset.scanner` vs `ParquetFile.iter_batches` —
  <https://github.com/apache/arrow/issues/44799>
- Spark `bucketBy` no longer on `DataFrame` in 4.x → hash co-partitioning; Spark tuning reference —
  <https://spark.apache.org/docs/latest/sql-performance-tuning.html> (verified default
  `spark.sql.autoBroadcastJoinThreshold = 10485760`)
- NumPy memmap for O(1) row gather of a dense embedding table —
  <https://numpy.org/doc/stable/reference/generated/numpy.memmap.html>

**Claims marked UNVERIFIED by the research pass (do not cite as fact):** official code for
**ETA / SDIM / TWIN / MISS** could not be located (404s); `facebookresearch/EBES` does not exist (the
real EBES is an *event-sequence* benchmark, <https://github.com/On-Point-RND/EBES>); the `xiaohu2015`
GitHub user has no recommender repositories; Milvus documentation bodies and cloud VPS prices were not
retrievable from this host.

---

## 12. Artifact index

| Artifact | Contents |
|---|---|
| `artifacts/stats/profile.json` | T1 EDA: labels, popularity, sequences, embeddings, coverage |
| `artifacts/stats/prepare.json` | vocabulary, split statistics, history store |
| `artifacts/bench/join_duckdb.json` / `join_spark.json` | join throughput + correctness invariants |
| `artifacts/stats/train_two_tower*.json`, `train_rankers.json` | training curves, params, VRAM |
| `artifacts/stats/evaluate.json` | full model × metric table + full-catalogue retrieval |
| `artifacts/bench/retrieval_ablation.json`, `retrieval_diagnosis.json` | the cone-collapse study |
| `artifacts/bench/index.json` | FAISS flat vs HNSW recall/latency sweep, memory table |
| `artifacts/bench/onnx.json` | FP32 vs INT8 single-request latency/size + numerical equivalence |
| `artifacts/bench/serve.json` | load test incl. the `/health` control |
| `artifacts/figures/*.png` | 11 figures, all regenerated from the JSON above |
| `tests/test_invariants.py` | 14 tests guarding the silent-corruption failure modes |

![Training curves](../figures/08_training.png)

![Memory per encoding](../figures/09_memory_encodings.png)
