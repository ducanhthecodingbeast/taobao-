"""T6a — vector index: exact vs HNSW, memory math, and a recall/latency curve.

At the brief's demo scale (10 000 items) an exhaustive ``IndexFlatIP`` is trivially fast, so
the interesting question is not "can we use ANN" but "**what does ANN cost us**". This module
answers that with a measurement:

1. build ``IndexFlatIP`` (exact) and ``IndexHNSWFlat`` (M=32, efConstruction=200)
2. sweep ``efSearch`` and record recall@10 against the exact ground truth
3. record build time, index size, and query latency percentiles

Memory math printed by :func:`memory_table` is the justification for the compression story:
fp32 512 B/vector, int8 128 B/vector, PQ32 32 B/vector + 8 B id.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import PATHS


def _faiss():
    import faiss
    return faiss


def catalogue_items(vocab, n_items: int, mode: str = "popular") -> np.ndarray:
    """Return **vocabulary indices** for the demo catalogue.

    Not raw item ids: the popularity table in ``profile.json`` stores anonymised int64 ids in
    the full ``[-2^63, 2^63)`` range, and indexing the embedding table with those raises
    ``IndexError`` (or worse, silently wraps for small negative values). Everything downstream
    of this function speaks vocabulary indices, and only the API boundary decodes back.
    """
    v = vocab if hasattr(vocab, "encode") else None
    size = v.size if v is not None else int(vocab)
    if mode == "popular":
        try:
            res = json.loads((PATHS.stats / "profile.json").read_text())
            raw = np.asarray([int(i) for i, _ in res["popularity"]["top50"]], dtype=np.int64)
            if v is not None:
                idx = v.encode(raw)
                idx = idx[idx != v.pad_index]
            else:
                idx = np.arange(min(len(raw), size), dtype=np.int64)
            if len(idx) < n_items:            # pad with evenly spaced VOCAB INDICES
                extra = np.arange(1, size - 1, dtype=np.int64)
                extra = extra[:: max(1, len(extra) // (n_items * 2))]
                idx = np.unique(np.concatenate([idx, extra]))
            return idx[:n_items].astype(np.int64)
        except Exception:
            pass
    rng = np.random.default_rng(42)
    return rng.choice(size - 1, size=n_items, replace=False).astype(np.int64)


def run(n_items: int = 10_000, exact: bool = True, n_queries: int = 2000,
        ef_sweep: tuple[int, ...] = (16, 32, 64, 128, 256)) -> dict:
    """Benchmark the index the service actually builds, queried the way it is queried.

    * item vectors: the demo artifact's (the served search space)
    * queries: **real user vectors** from held-out test sessions, not catalogue items. Querying
      with indexed items made every query trivially find itself and inflated HNSW recall.
    * latency: one query per call, as a request issues it; the batched figure is reported
      separately as amortised throughput, never as a latency.
    """
    from .serve.app import Recommender

    faiss = _faiss()
    faiss.omp_set_num_threads(1)
    rec = Recommender(n_items=n_items)
    if rec.artifact is None:
        raise RuntimeError("demo artifact missing; run `python -m tmm.cli build-demo-artifact`")
    vecs = np.ascontiguousarray(rec.artifact.item_vecs[:n_items], dtype=np.float32)
    items = rec.item_ids
    d = vecs.shape[1]
    sessions, _ = rec.artifact.eval_sessions()
    Q = np.stack([rec.user_vector(sess) for sess in sessions[:n_queries]])
    Q = np.ascontiguousarray(Q[np.any(Q, axis=1)], dtype=np.float32)

    # ---- exact ground truth ----------------------------------------------------------
    t0 = time.time()
    flat = faiss.IndexFlatIP(d)
    flat.add(vecs)
    t_flat_build = time.time() - t0
    D_gt, I_gt = flat.search(Q, 10)

    res: dict = {
        "n_items": int(len(items)), "dim": int(d), "n_queries": int(len(Q)),
        "queries": "user vectors of held-out test sessions (demo artifact eval set)",
        "item_vectors": f"demo artifact ({rec.artifact.space})",
        "exact": {
            "index": "IndexFlatIP", "build_seconds": round(t_flat_build, 4),
            "bytes": int(flat.ntotal * d * 4),
            "mb": round(flat.ntotal * d * 4 / 2**20, 3),
        },
        "hnsw": {},
        "sweep": [],
    }

    # ---- latency of exact search -----------------------------------------------------
    res["exact"]["latency"] = _latency(lambda q: flat.search(q, 10), Q)

    if not exact:
        return res

    # ---- HNSW ------------------------------------------------------------------------
    t0 = time.time()
    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(vecs)
    t_hnsw_build = time.time() - t0
    res["hnsw"]["index"] = "IndexHNSWFlat(M=32, efConstruction=200)"
    res["hnsw"]["build_seconds"] = round(t_hnsw_build, 4)
    res["hnsw"]["speedup_vs_flat_build"] = round(t_flat_build / max(t_hnsw_build, 1e-9), 2)
    # HNSW storage = vectors + graph links (M*2 neighbours, 4 bytes each per level-0 node)
    graph_bytes = hnsw.ntotal * 32 * 2 * 4
    res["hnsw"]["bytes_vectors"] = int(hnsw.ntotal * d * 4)
    res["hnsw"]["bytes_graph_estimate"] = int(graph_bytes)
    res["hnsw"]["mb_total_est"] = round((hnsw.ntotal * d * 4 + graph_bytes) / 2**20, 3)

    for ef in ef_sweep:
        hnsw.hnsw.efSearch = ef
        lat = _latency(lambda q: hnsw.search(q, 10), Q)
        _, I = hnsw.search(Q, 10)
        rec = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I, I_gt)]))
        res["sweep"].append({"efSearch": ef, "recall@10": round(rec, 5),
                             "p50_ms": lat["p50_ms"], "p95_ms": lat["p95_ms"],
                             "p99_ms": lat["p99_ms"], "qps": lat["qps"]})
    res["hnsw"]["latency"] = _latency(lambda q: hnsw.search(q, 10), Q)

    res["memory_table"] = memory_table(len(items))
    PATHS.bench.mkdir(parents=True, exist_ok=True)
    (PATHS.bench / "index.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def _latency(fn, Q: np.ndarray) -> dict:
    """Per-request latency: each query is issued alone, as ``/recommend`` issues it."""
    for q in Q[:50]:                                   # warm-up
        fn(q[None, :])
    times = []
    for q in Q:
        t0 = time.perf_counter()
        fn(q[None, :])
        times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times)
    t0 = time.perf_counter()
    fn(Q)
    amortised = (time.perf_counter() - t0) / len(Q) * 1000
    return {"p50_ms": round(float(np.median(a)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "p99_ms": round(float(np.percentile(a, 99)), 4),
            "qps": int(1000 / max(a.mean(), 1e-9)), "queries": int(len(Q)),
            "batch_amortised_ms_per_query": round(amortised, 5)}


def memory_table(n_items: int = 10_000) -> list[dict]:
    """Bytes per vector and total footprint for each storage strategy."""
    rows = [
        ("fp32 raw (128-d)", 512, 0),
        ("fp16", 256, 0),
        ("int8 (as shipped)", 128, 0),
        ("PQ 32 B/vec (OPQ32_128,IVF4096,PQ32)", 32, 8),
        ("PQ 16 B/vec", 16, 8),
    ]
    out = []
    for name, code, idb in rows:
        total = n_items * (code + idb)
        out.append({"encoding": name, "bytes_per_vector": code + idb,
                    "total_bytes": total, "total_mb": round(total / 2**20, 3)})
    return out


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, default=str))
