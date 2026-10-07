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


def build_item_vectors(n_items: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Item-tower vectors for the demo catalogue (float32, L2-normalised for inner product)."""
    import torch

    from .models import TwoTower
    from .train import build_emb_table, load_prepared

    d = load_prepared("cpu")
    vocab, cards = d["vocab"], d["cards"]
    items = catalogue_items(vocab, n_items)          # vocabulary indices
    # float16 on CPU: 9.1 GiB instead of 18.2 GiB fp32. The towers cast to fp32 after the
    # gather, so storage dtype and compute dtype are decoupled.
    emb = build_emb_table(vocab, "cpu", dtype=torch.float16)

    # Prefer the residual checkpoint: the plain pointwise one has a collapsed item space
    # (pairwise cosine 0.966) and therefore produces a useless index. See retrieval_ablation.
    ck_path = next((p for p in (PATHS.models / "two_tower_res.pt",
                                PATHS.models / "two_tower.pt") if p.exists()),
                   PATHS.models / "two_tower.pt")
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = TwoTower(emb, cat_card=cards["206"],
                         demo_cards=[cards[c] for c in
                                     ("130_1", "130_2", "130_3", "130_4", "130_5")],
                         dim=ck["dim"], tower_seq_len=200,
                         residual=bool(ck.get("residual", False)))
        model.load_state_dict(ck["state_dict"])
        model.eval()
    else:                                     # fall back to the raw SCL table
        model = None

    with torch.no_grad():
        t = torch.from_numpy(items)
        c = torch.zeros_like(t)
        if model is not None:
            v = model.item_vec(t, c)
        else:
            v = torch.nn.functional.normalize(emb[t].float(), dim=-1)
    return items, v.numpy().astype(np.float32)


def run(n_items: int = 10_000, exact: bool = True, seed: int = 42,
        n_queries: int = 2000, ef_sweep: tuple[int, ...] = (16, 32, 64, 128, 256)) -> dict:
    faiss = _faiss()
    faiss.omp_set_num_threads(8)
    items, vecs = build_item_vectors(n_items)
    d = vecs.shape[1]
    Q = np.ascontiguousarray(vecs[:min(n_queries, len(vecs))])

    # ---- exact ground truth ----------------------------------------------------------
    t0 = time.time()
    flat = faiss.IndexFlatIP(d)
    flat.add(vecs)
    t_flat_build = time.time() - t0
    D_gt, I_gt = flat.search(Q, 10)

    res: dict = {
        "n_items": int(len(items)), "dim": int(d),
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
    hnsw = faiss.IndexHNSWFlat(d, 32)
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


def _latency(fn, Q: np.ndarray, repeats: int = 3) -> dict:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(Q)
        times.append((time.perf_counter() - t0) / len(Q) * 1000)
    a = np.asarray(times)
    return {"p50_ms": round(float(np.median(a)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "p99_ms": round(float(np.percentile(a, 99)), 4),
            "qps": int(1000 / max(a.mean(), 1e-9)),
            "repeats": repeats, "batch_size": int(len(Q))}


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
