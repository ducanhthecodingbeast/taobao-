"""T1 — dataset profiling grounded in the real files.

Profiling strategy (chosen for the 41 GiB RAM / 101 GB disk budget)
-------------------------------------------------------------------
* ``raw/train_samples.parquet`` is only **714 MB** for 76 M rows, so label balance,
  user/item cardinality and the popularity long-tail are computed **exactly** with DuckDB
  rather than estimated from a sample.
* ``raw/train_user_features.parquet`` is 44.6 GB with 7 row-groups: one row-group is read
  with pyarrow to characterise sequence length without touching the whole file.
* ``feature_map/*.npy`` (283 MB / 4.5 GB) is memory-mapped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import PATHS, SEED
from .io import EmbeddingStore, SortedKeyIndex, iter_batches


def _duck():
    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA threads=24")
    # /tmp is a 79 GB tmpfs on this host - a much better spill target than the 101 GB root.
    con.execute("PRAGMA temp_directory='/tmp/tmm_duckdb'")
    con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA preserve_insertion_order=false")
    return con


def inventory() -> dict:
    """File-level inventory: exact row counts from parquet footers (metadata only)."""
    from pyarrow.parquet import ParquetFile

    out: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tables": {}}
    for kind in ("train", "test"):
        files = sorted((PATHS.data_root / kind).glob("*.parquet"))
        rows = sum(ParquetFile(f).metadata.num_rows for f in files)
        size = sum(f.stat().st_size for f in files)
        out["tables"][kind] = {"files": len(files), "rows": rows, "bytes": size,
                               "gb": round(size / 2**30, 2)}
    for f in sorted(PATHS.raw_dir.glob("*.parquet")):
        pf = ParquetFile(f)
        out["tables"][f"raw/{f.name}"] = {
            "files": 1, "rows": pf.metadata.num_rows,
            "row_groups": pf.metadata.num_row_groups,
            "columns": [str(c) for c in pf.schema_arrow],
            "bytes": f.stat().st_size, "gb": round(f.stat().st_size / 2**30, 2),
        }
    fm = {f.name: {"bytes": f.stat().st_size} for f in sorted(PATHS.feature_map.glob("*.npy"))}
    out["feature_map"] = fm
    return out


def label_and_cardinality() -> dict:
    """Exact label balance + cardinality from the small ``*_samples`` tables."""
    con = _duck()
    out = {}
    for kind in ("train", "test"):
        p = PATHS.raw_dir / f"{kind}_samples.parquet"
        q = f"""
        SELECT count(*)                                        AS samples,
               sum(label_0[2])                                 AS positives,
               avg(label_0[2])                                 AS pos_rate,
               count(DISTINCT "129_1")                          AS users,
               count(DISTINCT "205")                            AS items
        FROM read_parquet('{p}')
        """
        r = con.execute(q).fetchone()
        out[kind] = {"samples": int(r[0]), "positives": int(r[1]),
                     "pos_rate": float(r[2]), "users": int(r[3]), "items": int(r[4])}
    # train/test user overlap -> leakage surface for the temporal split
    q = f"""
    SELECT count(*) FROM (
      SELECT DISTINCT "129_1" FROM read_parquet('{PATHS.raw_dir}/train_samples.parquet')
      INTERSECT
      SELECT DISTINCT "129_1" FROM read_parquet('{PATHS.raw_dir}/test_samples.parquet')
    )
    """
    out["user_overlap_train_test"] = int(con.execute(q).fetchone()[0])
    out["overlap_ratio_of_test_users"] = round(
        out["user_overlap_train_test"] / max(out["test"]["users"], 1), 4)
    con.close()
    return out


def popularity(top_k: int = 50) -> dict:
    """Item popularity long-tail from the exact train sample table.

    This is what makes the popularity baseline strong, so it must be measured, not assumed.
    """
    con = _duck()
    p = PATHS.raw_dir / "train_samples.parquet"
    con.execute(f"CREATE VIEW s AS SELECT \"205\" AS item, label_0[2] AS y FROM read_parquet('{p}')")
    hist = con.execute("SELECT item, count(*) c FROM s GROUP BY item ORDER BY c DESC").fetchnumpy()
    counts = np.asarray(hist["c"], dtype=np.int64)
    total = counts.sum()
    cum = np.cumsum(counts) / total
    top = [(int(i), int(c)) for i, c in zip(hist["item"][:top_k], counts[:top_k])]
    # concentration metrics
    res = {
        "distinct_items_in_samples": int(len(counts)),
        "total_interactions": int(total),
        "top10_share": float(cum[9]), "top100_share": float(cum[99]),
        "top1k_share": float(cum[999]), "top10k_share": float(cum[9999]),
        "items_for_50pct": int(np.searchsorted(cum, 0.5) + 1),
        "items_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
        "gini": float(_gini(counts)),
        "top50": top,
        "freq_histogram": _log_hist(counts),
    }
    # positive-only popularity (what the user actually clicked)
    pos = con.execute("SELECT item, count(*) c FROM s WHERE y=1 GROUP BY item ORDER BY c DESC").fetchnumpy()
    pc = np.asarray(pos["c"], dtype=np.int64)
    res["positives_distinct_items"] = int(len(pc))
    res["positives_top100_share"] = float(np.cumsum(pc)[:100].sum() / pc.sum())
    con.close()
    return res


def _gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    if n == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def _log_hist(counts: np.ndarray, bins: int = 30) -> dict:
    edges = np.logspace(0, np.log10(max(counts.max(), 2)), bins + 1)
    h, _ = np.histogram(counts, bins=edges)
    return {"edges": edges.tolist(), "counts": h.tolist()}


def sequence_stats(path: Path | None = None, row_groups: int = 1) -> dict:
    """Sequence-length distribution from one row-group of the 44.6 GB user-features table."""
    path = path or (PATHS.raw_dir / "train_user_features.parquet")
    lens, cats, demo = [], [], []
    seen_users = 0
    for d in iter_batches(path, columns=["150_2_180", "151_2_180", "130_1", "130_2", "130_5"],
                          batch_size=20000, row_groups=list(range(min(row_groups, 7)))):
        lens.extend(len(x) for x in d["150_2_180"])
        cats.extend(len(x) for x in d["151_2_180"])
        demo.extend(zip(d["130_1"], d["130_2"], d["130_5"]))
        seen_users += len(d["130_1"])
    L = np.asarray(lens); C = np.asarray(cats)
    return {
        "users_scanned": int(seen_users),
        "row_groups_read": row_groups,
        "seq_len": {"mean": float(L.mean()), "p10": float(np.percentile(L, 10)),
                    "p50": float(np.percentile(L, 50)), "p90": float(np.percentile(L, 90)),
                    "p99": float(np.percentile(L, 99)), "max": int(L.max()),
                    "frac_at_max": float((L == L.max()).mean())},
        "cat_len": {"mean": float(C.mean()), "max": int(C.max())},
        "histogram": _log_hist(L, bins=25) if L.max() > 0 else {},
        "distinct_age_buckets": int(len(set(d[0] for d in demo))),
        "distinct_gender_buckets": int(len(set(d[1] for d in demo))),
        "distinct_citylevel_buckets": int(len(set(d[2] for d in demo))),
    }


def embedding_stats() -> dict:
    """Quantisation range, per-row norms and the dequantisation scale.

    The scale is fixed at ``1/127`` (see :meth:`EmbeddingStore.quant_stats`). The decisive
    evidence is that the dequantised row norm is ~1.0, which means the shipped vectors are
    L2-normalised floats that were symmetrically quantised - so cosine similarity between them
    is a well-posed similarity, not an artefact of the int8 encoding.
    """
    store = EmbeddingStore.open()
    st = store.quant_stats()
    raw = np.asarray(store.values[:500_000], dtype=np.float32)
    norms_raw = np.linalg.norm(raw, axis=1)
    norms_deq = np.linalg.norm(raw * st["scale"], axis=1)
    per_dim = raw.mean(axis=0)
    return {
        "n_items": int(store.values.shape[0]),
        "dim": int(store.values.shape[1]),
        "dtype": "int8",
        "value_min": float(raw.min()), "value_max": float(raw.max()),
        "scale": st["scale"], "scale_source": st["scale_source"],
        "row_norm_raw_int8": {"mean": float(norms_raw.mean()), "std": float(norms_raw.std()),
                              "min": float(norms_raw.min()), "max": float(norms_raw.max())},
        "row_norm_dequantized": {"mean": float(norms_deq.mean()), "std": float(norms_deq.std()),
                                 "min": float(norms_deq.min()), "max": float(norms_deq.max())},
        "rows_are_l2_normalised_before_quantisation":
            bool((norms_deq.max() - norms_deq.min()) / norms_deq.mean() < 0.05
                 and abs(norms_deq.mean() - 1.0) < 0.05),
        "dim_means_absmax": float(np.abs(per_dim).max()),
        "effective_bits_per_item": 128 * 8,
        "bytes_if_float32": int(store.values.shape[0] * store.values.shape[1] * 4),
        "bytes_as_int8": int(store.values.shape[0] * store.values.shape[1]),
    }


def coverage(sample_items: int = 4_000_000) -> dict:
    """Do target items have an embedding? Do history items? (README claims 100 % / 90 %)."""
    con = _duck()
    ids = con.execute(
        f"SELECT DISTINCT \"205\" AS item FROM read_parquet('{PATHS.raw_dir}/train_samples.parquet')"
    ).fetchnumpy()["item"]
    idx = SortedKeyIndex.load(PATHS.emb_keys)
    rows = idx.find(np.asarray(ids, dtype=np.int64))
    target_cov = float((rows >= 0).mean())

    p90 = np.load(PATHS.feature_map / "150_2_180_sorted_map_p90.npy", mmap_mode="r")
    full = np.load(PATHS.feature_map / "150_2_180_sorted_map.npy", mmap_mode="r")
    hist_cov = len(p90) / len(full)

    con.close()
    return {"target_items_distinct": int(len(ids)),
            "target_items_with_embedding": int((rows >= 0).sum()),
            "target_embedding_coverage": target_cov,
            "history_vocab_full": int(len(full)),
            "history_vocab_p90": int(len(p90)),
            "history_embedding_coverage": float(hist_cov)}


def item_metadata() -> dict:
    """Category / geography cardinality from the 48 MB item_features table."""
    con = _duck()
    p = PATHS.raw_dir / "item_features.parquet"
    r = con.execute(f"""
        SELECT count(*) items, count(DISTINCT "206") cats,
               count(DISTINCT "213") cities, count(DISTINCT "214") provinces
        FROM read_parquet('{p}')""").fetchone()
    top = con.execute(f"""
        SELECT "206" AS cat, count(*) c FROM read_parquet('{p}')
        GROUP BY 1 ORDER BY c DESC LIMIT 20""").fetchall()
    con.close()
    return {"items": int(r[0]), "categories": int(r[1]), "cities": int(r[2]),
            "provinces": int(r[3]),
            "top_categories": [(int(a), int(b)) for a, b in top]}


def run_all(out_dir: Path | None = None, seq_row_groups: int = 1) -> dict:
    out_dir = (out_dir or PATHS.stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    steps = [
        ("inventory", inventory),
        ("labels", label_and_cardinality),
        ("popularity", popularity),
        ("sequence", lambda: sequence_stats(row_groups=seq_row_groups)),
        ("embedding", embedding_stats),
        ("coverage", coverage),
        ("item_metadata", item_metadata),
    ]
    for name, fn in steps:
        t = time.time()
        try:
            results[name] = fn()
            status = "ok"
        except Exception as exc:  # keep going; report which stage failed
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            status = "FAILED"
        results[name]["_seconds"] = round(time.time() - t, 2)
        print(f"[profile] {name:14s} {status:6s} {results[name]['_seconds']:>7.2f}s", flush=True)
    (out_dir / "profile.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":  # pragma: no cover
    run_all()
