"""T2 — turn the dataset into dense arrays a GPU can train on, without corrupting the labels.

How the two halves are sourced, and why
---------------------------------------
The joined shards carry the user's 1 000-item history inline. Measured structure of a shard
(``test-shard-000000``): 500 000 rows, **44 570 distinct users, 11.22 rows/user**, and for
**99.9 % of those users the entire candidate set lies inside that one shard**. In other words
the release *is* user-grouped at shard granularity, so a whole-shard scan does yield complete
candidate sets.

Even so, the two halves are sourced separately:

* **histories** <- streaming scan of the joined shards (projected columns only). A history is a
  property of the user and identical in every one of their rows, so this is cheap and safe.
* **candidate sets** <- ``raw/{split}_samples.parquet``, which is COMPLETE and tiny
  (0.714 GB train / 0.215 GB test for all 76 M / 23 M rows).

The reason is robustness, not correctness: relying on shard-level user locality is an
**undocumented** property of this particular release, it breaks the moment anyone reads a
subset of *row groups* rather than whole shards, and stopping a shard scan early at a user cap
can truncate the last user's candidates. Querying the complete sample table removes all three
failure modes and lets the user budget be applied exactly.

So the two halves are sourced separately:

* **histories** <- streaming scan of the joined shards (projected columns only)
* **candidate sets** <- ``raw/{split}_samples.parquet``, which is COMPLETE and tiny
  (0.714 GB train / 0.215 GB test for all 76 M / 23 M rows)

Other verified facts baked in here:

* ``label_0`` is a length-2 one-hot ``list<int8>``; ``label_0[2]`` (1-based) is the click flag.
* sequence length is ~1 000 for 96.3 % of users; truncation keeps the **newest** entries.
* there is no ``206_sorted_map.npy`` in the release, so the item-category vocabulary is built
  from ``raw/item_features.parquet`` (13 050 categories) with a reserved "unknown" slot.
* ``searchsorted`` may return ``len(map)``; every embedded column therefore has cardinality
  ``len(map) + 1``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import PATHS, SEED
from .io import HistoryStore, iter_batches, parquet_files
from .vocab import ItemVocab

HIST_COLS = ["129_1", "150_2_180"] + ["130_1", "130_2", "130_3", "130_4", "130_5"]
SAMPLE_KEEP = ["label_0", "129_1", "205"]

#: Demographics ship with a sorted map in ``feature_map/``; item category (``206``) does not.
USER_MAP_COLS = ("130_1", "130_2", "130_3", "130_4", "130_5")
ITEM_MAP_COLS = ("213", "214")
MAP_COLS = USER_MAP_COLS + ITEM_MAP_COLS + ("206",)

#: rows per user in the full dataset (76 015 123 / 6 929 606 = 10.97, and 11.1 for test)
ROWS_PER_USER = 11.0


def _safe_encode(values: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """``searchsorted`` -> index, with an explicit trailing "unknown" slot at ``len(mapping)``."""
    v = np.asarray(values, dtype=np.int64)
    idx = np.clip(np.searchsorted(mapping, v, side="left"), 0, len(mapping) - 1)
    hit = mapping[idx] == v
    return np.where(hit, idx, len(mapping)).astype(np.int64)


def category_vocab() -> np.ndarray:
    """Sorted vocabulary of item categories, built from ``item_features`` and cached."""
    cache = PATHS.data / "cat_vocab.npy"
    if cache.exists():
        return np.load(cache)
    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA threads=24")
    r = con.execute(
        f"SELECT DISTINCT \"206\" AS c FROM read_parquet('{PATHS.raw_dir}/item_features.parquet')"
    ).fetchnumpy()["c"]
    con.close()
    v = np.sort(np.asarray(r, dtype=np.int64))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, v)
    return v


def all_maps() -> dict[str, np.ndarray]:
    m = {c: np.asarray(np.load(PATHS.feature_map / f"{c}_sorted_map.npy"))
         for c in USER_MAP_COLS + ITEM_MAP_COLS}
    m["206"] = category_vocab()
    return m


def _encode_ragged(vocab: ItemVocab, seqs: list) -> list[np.ndarray]:
    """Encode a whole batch of variable-length id lists with one ``searchsorted`` call."""
    if not len(seqs):
        return []
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    if int(lens.sum()) == 0:
        return [np.empty(0, dtype=np.int32) for _ in seqs]
    flat = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
    return np.split(vocab.encode(flat), np.cumsum(lens)[:-1])


def scan_histories(kind: str, n_shards: int, seq_len: int, max_users: int,
                   vocab: ItemVocab, batch_size: int = 8192) -> dict:
    """Stream ``n_shards`` shards and keep one (complete) history + demographics per user.

    A user's history is identical in every one of their rows, so a partial shard scan still
    recovers a *complete* history. That is why histories can come from cheap shard slices
    while candidate sets cannot (see the module docstring).
    """
    files = parquet_files(kind)[:n_shards]
    hist: dict[int, np.ndarray] = {}
    demo: dict[int, tuple] = {}
    t0 = time.time()
    rows = 0
    for f in files:
        for b in iter_batches(f, columns=HIST_COLS, batch_size=batch_size):
            uu = np.asarray(b["129_1"], dtype=np.int64)
            rows += len(uu)
            need = np.array([int(u) not in hist for u in uu], dtype=bool)
            if not need.any():
                continue
            idxs = np.flatnonzero(need)
            enc = _encode_ragged(vocab, [b["150_2_180"][i] for i in idxs])
            for j, i in enumerate(idxs):
                h = enc[j]
                uid = int(uu[i])
                hist[uid] = h[-seq_len:] if seq_len and len(h) > seq_len else h
                demo[uid] = tuple(int(b[c][i]) for c in USER_MAP_COLS)
            if len(hist) >= max_users:
                break
        if len(hist) >= max_users:
            break
    return {"kind": kind, "shards": len(files), "rows_scanned": rows, "hist": hist,
            "demo": demo, "users": len(hist), "seconds": round(time.time() - t0, 2)}


def complete_candidates(kind: str, users: np.ndarray, vocab: ItemVocab,
                        maps: dict[str, np.ndarray], demo: dict[int, tuple]) -> dict:
    """Every candidate row for ``users``, from the COMPLETE ``raw/{kind}_samples`` table.

    ``raw/{kind}_samples.parquet`` holds only ``label_0``, ``129_1`` and ``205``; item
    metadata (206/213/214) is joined in from the 48 MB ``item_features`` table and user
    demographics come from the shard scan (they are per-user, so one lookup is enough).
    """
    import duckdb
    import pyarrow as pa

    con = duckdb.connect()
    for pragma in ("threads=24", "memory_limit='24GB'", "temp_directory='/tmp/tmm_duckdb'",
                   "preserve_insertion_order=false"):
        con.execute(f"PRAGMA {pragma}")
    con.register("wanted", pa.table({"u": np.asarray(users, dtype=np.int64)}))

    t0 = time.time()
    sql = f"""
    SELECT s."129_1" AS user_id, s."205" AS item_id, s.label_0[2]::TINYINT AS label,
           i."206" AS c206, i."213" AS c213, i."214" AS c214
    FROM read_parquet('{PATHS.raw_dir}/{kind}_samples.parquet') s
    JOIN wanted w ON s."129_1" = w.u
    LEFT JOIN read_parquet('{PATHS.raw_dir}/item_features.parquet') i ON i."205" = s."205"
    """
    tab = con.execute(sql).fetch_arrow_table()
    con.close()
    d = tab.to_pydict()
    uids = np.asarray(d["user_id"], dtype=np.int64)
    out = {
        "user": uids,
        "item": vocab.encode(np.asarray(d["item_id"], dtype=np.int64)),
        "label": np.asarray(d["label"], dtype=np.int8),
        "seconds": round(time.time() - t0, 2),
    }
    for c, key in zip(("206", "213", "214"), ("c206", "c213", "c214")):
        col = np.asarray([-1 if v is None else v for v in d[key]], dtype=np.int64)
        out[c] = _safe_encode(col, maps[c])
    for j, c in enumerate(USER_MAP_COLS):
        out[c] = _safe_encode(
            np.asarray([demo[int(u)][j] for u in uids], dtype=np.int64), maps[c])
    return out


def run(rows: int = 2_000_000, test_rows: int = 400_000, seq_len: int = 1000,
        train_shards: int | None = None, test_shards: int | None = None) -> dict:
    t0 = time.time()
    PATHS.data.mkdir(parents=True, exist_ok=True)
    vocab = ItemVocab.build()
    maps = all_maps()
    print(f"[prepare] vocab: {vocab.stats()}", flush=True)

    max_train_users = int(rows / ROWS_PER_USER)
    max_test_users = int(test_rows / ROWS_PER_USER)
    train_shards = train_shards or min(161, max(2, int(np.ceil(max_train_users / 45_000)) + 1))
    test_shards = test_shards or min(48, max(2, int(np.ceil(max_test_users / 45_000)) + 1))

    tr_h = scan_histories("train", train_shards, seq_len, max_train_users, vocab)
    print(f"[prepare] train histories: {tr_h['users']:,} users from {tr_h['shards']} shards "
          f"in {tr_h['seconds']}s", flush=True)
    te_h = scan_histories("test", test_shards, seq_len, max_test_users, vocab)
    print(f"[prepare] test  histories: {te_h['users']:,} users from {te_h['shards']} shards "
          f"in {te_h['seconds']}s", flush=True)

    tr_users = np.fromiter(sorted(tr_h["hist"].keys()), dtype=np.int64, count=len(tr_h["hist"]))
    te_users = np.fromiter(sorted(te_h["hist"].keys()), dtype=np.int64, count=len(te_h["hist"]))
    tr_c = complete_candidates("train", tr_users, vocab, maps, tr_h["demo"])
    print(f"[prepare] train candidates: {len(tr_c['label']):,} rows for "
          f"{len(np.unique(tr_c['user'])):,} users in {tr_c['seconds']}s", flush=True)
    te_c = complete_candidates("test", te_users, vocab, maps, te_h["demo"])
    print(f"[prepare] test  candidates: {len(te_c['label']):,} rows for "
          f"{len(np.unique(te_c['user'])):,} users in {te_c['seconds']}s", flush=True)

    # ---- one user vocabulary across both splits --------------------------------------
    merged: dict[int, np.ndarray] = {int(k): v for k, v in tr_h["hist"].items()}
    merged.update({int(k): v for k, v in te_h["hist"].items()})
    uid_of = {int(u): i for i, u in enumerate(sorted(merged.keys()))}
    user_ids = np.array(sorted(merged.keys()), dtype=np.int64)
    HistoryStore.build(np.arange(len(user_ids), dtype=np.int64),
                       [merged[int(u)] for u in user_ids]).save(PATHS.data / "hist")
    np.save(PATHS.data / "user_ids.npy", user_ids)
    del merged

    stats, cards = {}, {c: len(maps[c]) + 1 for c in MAP_COLS}
    for name, c in (("train", tr_c), ("test", te_c)):
        uidx = np.fromiter((uid_of[int(x)] for x in c["user"]), dtype=np.int32,
                           count=len(c["user"]))
        payload = {"user": uidx, "item": c["item"], "label": c["label"]}
        for col in MAP_COLS:
            payload[col] = c[col]
        np.savez_compressed(PATHS.data / f"{name}.npz", **payload)
        nun, npos = len(np.unique(uidx)), int(c["label"].sum())
        stats[name] = {
            "rows": int(len(c["label"])), "users": int(nun),
            "items": int(len(np.unique(c["item"]))),
            "positives": npos, "pos_rate": round(float(c["label"].mean()), 5),
            "rows_per_user": round(len(uidx) / max(nun, 1), 3),
            "pos_per_user": round(npos / max(nun, 1), 3),
            "users_with_a_positive": int(len(np.unique(uidx[c["label"] == 1]))),
            "users_with_a_positive_rate": round(
                len(np.unique(uidx[c["label"] == 1])) / max(nun, 1), 4),
        }

    hs = HistoryStore.load(PATHS.data / "hist")
    hl = np.diff(hs.offsets)
    res = {
        "seed": SEED, "seq_len": seq_len,
        "vocab": vocab.stats(), "splits": stats, "cardinalities": cards,
        "columns": list(MAP_COLS),
        "history": {
            "users": int(len(hs)), "items_total": int(len(hs.values)),
            "len_mean": round(float(hl.mean()), 1), "len_max": int(hl.max()),
            "bytes_int32": int(hs.values.nbytes),
            "bytes_if_int64": int(hs.values.nbytes * 2),
        },
        "method": {
            "histories": f"streamed {train_shards} train + {test_shards} test shards (projected cols)",
            "candidates": "raw/{split}_samples.parquet joined to the discovered users (COMPLETE)",
            "why": "shards ARE user-grouped (measured: 44 570 users / 500 k rows, 99.9% with a "
                   "complete candidate set inside one shard), but sourcing from the sample table "
                   "is exact, independent of that undocumented locality, and safe against "
                   "row-group-level reads and early stops",
        },
        "notes": [
            "151_2_180 (categories) skipped: item category already available as 206",
            "seq truncation keeps the newest seq_len entries",
            "206 has no shipped sorted map -> vocabulary built from item_features",
        ],
        "total_seconds": round(time.time() - t0, 2),
    }
    (PATHS.stats / "prepare.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps({k: v for k, v in run().items() if k != "notes"}, indent=2, default=str))
