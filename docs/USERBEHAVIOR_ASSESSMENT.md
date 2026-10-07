# UserBehavior (Tianchi 649) — can we use it, and how?

**Verdict: yes, but as a *complement*, not a replacement — and it fixes the single biggest
methodological weakness in the TAOBAO-MM build.**

Measured on `UserBehavior.csv` (3.5 GB, extracted from `UserBehavior.csv.zip`, 905.80 MB,
2021-02-22). Profile JSON: `artifacts/stats/userbehavior_profile.json`,
`artifacts/stats/userbehavior_timestamps.json`.

---

## 1. What it actually is (measured, not quoted)

| Property | Value |
|---|---|
| Rows | **100,150,807** |
| Users | **987,994** |
| Items | **4,162,024** |
| Categories | **9,439** |
| Columns | `user_id, item_id, category_id, behavior_type, timestamp` (no header) |
| Date window | **9 days**, 2017-11-25 → 2017-12-03 (98.77 % of rows; +2017-11-24 and 325 rows on 12-04) |
| Behaviours | `pv` 89.58 %, `cart` 5.52 %, `fav` 2.88 %, **`buy` 2.01 %** |
| Rows / user | mean **101.4**, p50 75, p90 218, max 848 |
| Rows / item | mean 24.1, p50 3, **29.0 % singletons** |
| Item popularity Gini | **0.8309** (top-100 = 1.17 %, top-10k = 18.8 %) |
| Exact duplicate `(user,item,behaviour,ts)` groups | **49** — effectively clean |
| Profiling time with DuckDB | **8.9 s** |

### Data-quality note (measured, and smaller than it first looks)

The raw `min`/`max` timestamps are **1902-05-07 → 2037-04-09**, which looks broken. It is not:

| Check | Rows | Share |
|---|---|---|
| Timestamp outside calendar **2017** (genuinely corrupt) | **1,876** | **0.0019 %** |
| of which before 2017-01-01 | 782 | |
| of which after 2018-01-01 | 1,094 | |
| Distinct users affected | 181 of 987,994 | 0.02 % |
| Otherwise well-formed (valid `behavior_type`, ids present) | 1,876 / 1,876 | 100 % |
| Rows on **2017-11-24** (legitimate, just outside the usual quoted window) | ~1.23 M | 1.23 % |

So the correct handling is a **two-line filter** — drop the 1,876 impossible rows, and decide
explicitly whether 2017-11-24 belongs in the window (it is a timezone/window-boundary decision,
not corruption). Everything else is usable.

---

## 2. Side-by-side with TAOBAO-MM

| | TAOBAO-MM | UserBehavior 649 |
|---|---|---|
| Rows | 98,994,588 | 100,150,807 |
| Users | 8,798,906 | 987,994 |
| Items | 35.46 M (history vocab) / 3.78 M (targets) | 4.16 M |
| Labels | **shipped** candidate sets (~11/user, 13.70 % positive) | **you build them** |
| Behaviours | binary click only | 4-way: pv / cart / fav / buy |
| **Timestamps** | **NONE** | **yes — 9 days, per-event** |
| Sequence length | up to **1,000** | mean 101, max 848 |
| Multimodal embeddings | **128-d SCL int8** | **none** |
| Item metadata | category, city, province | category only |
| Disk / RAM to process | 139 GB / needs care | **3.5 GB / easy** |
| Item Gini | 0.815 | 0.831 |

**Item ids are a different anonymisation space.** They cannot be joined to TAOBAO-MM's ids, and
the dataset cannot be used to enrich TAOBAO-MM. Any use is *parallel*, not merged.

---

## 3. Four concrete uses, ranked by value

### U1 — Fix the biggest weakness in the current build: the temporal split ⭐ highest value
TAOBAO-MM ships **no timestamp**, so the shipped split has to be trusted and **9.42 % of test users
also appear in train**. Every leakage caveat in the report is therefore something we can only *cite*
(Ji et al. arXiv:2010.11060; arXiv:2507.16289), never *demonstrate*.

UserBehavior has per-event time, so we can implement a genuine **leave-last-day-out** split
(8 days train / 1 day test — the canonical Taobao protocol) and then measure the leakage directly:
compare a random split against the temporal split on the same harness. That converts a limitation
into an experiment.

### U2 — Behaviour-aware next-action prediction (the funnel)
`pv → cart → fav → buy` gives a 4-way target instead of a binary click, so the *same* two-stage
DIN/MUSE architecture can predict **conversion stage**, not just engagement. Buy is only
**2.01 %** — a harder imbalance than TAOBAO-MM's 13.70 %, which is itself a useful study
(calibration, PR-AUC vs AUC, per-stage GAUC). The `build`/`rank` code is unchanged; only the label
and the head change.

### U3 — Replay source for the Kafka pipeline (exercises the new backend)
The backend just built has a Kafka ingest path but **no realistic event stream**. UserBehavior's
real timestamps let us replay 9 days of events at controllable speed. Critically, this makes
**event-time vs processing-time** testable — late events, out-of-order arrival, duplicate replay —
which is exactly what the idempotency key, the DLQ and the consumer-offset logic exist for. The
`Event` model already carries an optional `ts` field, so no contract change is needed.

### U4 — Second benchmark to validate the evaluation harness
`metrics.py` (`per_user_ranking`, `gauc`, `recall_from_topk`, and the `compare()` guard against
sampled-metric abuse) is dataset-agnostic. Running it on a second dataset with a *differently
constructed* candidate set is real evidence the harness is correct rather than tuned to one split.

### What it cannot do
* It cannot carry the portfolio headline. No multimodal embeddings and 9-day histories mean the
  long-sequence + multimodal story stays with TAOBAO-MM.
* It ships **no candidate sets**, so negatives must be constructed — and that is precisely where
  the sampled-metric trap lives (Rendle, arXiv:1912.02263). The protocol must be published:
  candidate size, negative distribution, and full-vs-sampled mode, exactly as done for TAOBAO-MM.

---

## 4. Suggested integration shape (if we proceed)

1. `src/tmm/ub/` — loader: drop the 1,876 impossible timestamps, fix the window, build
   leave-last-day-out splits, encode `user_id`/`item_id`/`category_id` with `np.unique` +
   `searchsorted` (same pattern as `tmm/vocab.py`).
2. Reuse `tmm/metrics.py` **unchanged**; add a `--dataset {taobao-mm,ub}` switch to the CLI.
3. Candidate construction: leave-one-out for ranking plus a popularity-stratified negative sampler;
   report BOTH full and sampled modes.
4. Backend: a `python -m tmm.ub.replay --speed 1000` producer that publishes to
   `tmm.clickstream` preserving original `ts`, so the consumer's event-time handling is testable.
5. Cost: **far cheaper than TAOBAO-MM** — 3.5 GB CSV, mean 101 events/user, so the whole dataset
   fits in RAM comfortably. This makes it the right target for *fast iteration* while TAOBAO-MM
   stays the scale/multimodal story.

## 5. Recommendation

Do **U1 + U3 first**. U1 closes a real methodological hole in the report; U3 makes the
just-built Kafka path prove itself against event-time reality instead of injected stubs.
U2 and U4 are natural follow-ons once the loader exists.
