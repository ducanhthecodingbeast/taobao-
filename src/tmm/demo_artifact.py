"""T6d — precomputed demo artifact: make the lightweight direction actually lightweight.

Why this module exists (a measured memory bug, not a guess)
----------------------------------------------------------
The Direction-1 service is supposed to fit a 2 vCPU / 4 GB VPS, and the reduced embedding table
is only 220 MB. Measured peak RSS of the real `Recommender` at startup was **7,025 MB**. The
cause was not the table size but *how* it was produced:

``load_embeddings_subset`` gathers ~900 k scattered rows out of the 4.5 GB
``scl_emb_int8_p90_values.npy`` memmap. Each row is 128 bytes, so neighbouring rows land on
different 4 KB pages; touching ~900 k pages makes the kernel fault in a large fraction of a
4.5 GB mapping, and **mmap-resident pages count toward RSS**. A 220 MB output therefore cost
~1.5 GB. `load_prepared` + the 35.46 M-entry item vocabulary accounted for most of the rest
(1.5 GB) because the service loaded both just to answer "which row is this item id?".

So the expensive work is moved **offline** into a single self-sufficient artifact:

| file | contents | size |
|---|---|---|
| ``emb_fp16.npy`` | ``[N+1, 128]`` float16, row ``N`` is the zero pad row | ~220 MB |
| ``vocab_ids.npy`` | sorted ``int64[N]`` raw anonymised id of each row (``searchsorted`` lookup) | ~7 MB |
| ``items.npy`` / ``item_ids.npy`` / ``item_cats.npy`` | demo catalogue: rows, raw ids, category indices | <1 MB |
| ``item_vecs.npy`` | ``[n_items, D]`` float32 L2-normalised item-tower vectors (the search space) | ~5 MB |
| ``user_tower.onnx`` | user tower exported from the **same** checkpoint as ``item_vecs`` | <1 MB |
| ``eval_*.npy`` | held-out test sessions + clicked targets inside the catalogue (quality guard) | ~3 MB |
| ``meta.json`` | version, sizes, checksums, build provenance | <1 KB |

The service then needs no memmap over the 4.5 GB table and no 35.46 M-entry vocabulary, and
no torch: serving is numpy + onnxruntime + FAISS.

Item vectors and the user tower are produced here, together, from one checkpoint. The service
used to index raw SCL vectors while querying with a trained user tower exported from a
different checkpoint; the two spaces were unrelated and recommendations were at random level
(HR@10 0.0015 vs 0.001 random on held-out clicks).

``build()`` is the *only* place allowed to touch the big memmap; ``load()`` is what the service
calls and it is memory-flat.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .config import BUDGET, PATHS

ARTIFACT_VERSION = "demo-v2"
DEMO_COLS = ("130_1", "130_2", "130_3", "130_4", "130_5")
EVAL_USERS = 2000
TOWER_SEQ = 200


def _dir(root: Path | None = None) -> Path:
    return (root or (PATHS.models / "demo"))


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(n_items: int = BUDGET.demo_items, root: Path | None = None,
          verbose: bool = True) -> dict:
    """Materialise the demo artifact. Run offline; never on the request path."""
    import torch

    from .config import SEED
    from .export_onnx import UserTowerExport, export, serving_checkpoint
    from .models import TwoTower
    from .train import load_prepared
    from .vocab import load_embeddings_subset

    t0 = time.time()
    out = _dir(root)
    out.mkdir(parents=True, exist_ok=True)

    d = load_prepared("cpu")
    vocab, cards = d["vocab"], d["cards"]

    # Catalogue = the most frequently *observed target* items in the training sample, not an
    # arbitrary slice of the vocabulary. Two reasons, both measured:
    #  1. `index.catalogue_items(mode="popular")` encodes the top-50 popularity ids and then pads
    #     with evenly spaced vocabulary indices. Those padded ids are essentially random items,
    #     so only 2.6 % of the catalogue had real item metadata and the rest fell back to a
    #     constant category -- the same class of bug that corrupted the retrieval experiment.
    #  2. Items that actually appear as targets are exactly the ones with known category,
    #     city and province, so category coverage becomes ~100 %.
    counts = np.bincount(d["train"]["item"].astype(np.int64), minlength=vocab.size)
    if (counts > 0).sum() < n_items:
        raise RuntimeError("not enough observed items for the requested catalogue size")
    items = np.sort(np.argpartition(counts, -n_items)[-n_items:]).astype(np.int64)
    extra = np.unique(np.concatenate([d["train"]["item"], d["test"]["item"]]))
    keep = np.unique(np.concatenate([items, extra]))             # rows the service can be asked for

    # real categories for the catalogue (never a constant feature)
    all_items = np.concatenate([d["train"]["item"], d["test"]["item"]])
    all_cats = np.concatenate([d["train"]["206"], d["test"]["206"]])
    uniq, first = np.unique(all_items, return_index=True)
    pos = np.clip(np.searchsorted(uniq, items), 0, len(uniq) - 1)
    hit = uniq[pos] == items
    item_cats = np.where(hit, all_cats[first[pos]], cards["206"] - 1).astype(np.int64)

    emb = load_embeddings_subset(vocab, keep)                    # [N+1, 128] fp16, +pad row
    vocab_ids = vocab.ids[keep].astype(np.int64)                 # sorted raw ids per row

    # rows of the catalogue inside the reduced table
    row_of_keep = np.full(len(vocab.ids) + 1, -1, dtype=np.int32)
    row_of_keep[keep] = np.arange(len(keep), dtype=np.int32)
    item_rows = row_of_keep[items].astype(np.int32)
    if (item_rows < 0).any():
        raise RuntimeError("catalogue item missing from the reduced table")

    # ---- the search space: item vectors + user tower from ONE checkpoint ---------------
    ck_path = serving_checkpoint()
    onnx_path = out / "user_tower.onnx"
    if ck_path is not None:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        tower = TwoTower(torch.from_numpy(emb).float(), cat_card=cards["206"],
                         demo_cards=[cards[c] for c in DEMO_COLS], dim=ck["dim"],
                         tower_seq_len=TOWER_SEQ, residual=bool(ck.get("residual", False)))
        tower.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            item_vecs = tower.eval().item_vec(torch.from_numpy(item_rows.astype(np.int64)),
                                              torch.from_numpy(item_cats)).numpy()
        export(UserTowerExport.load(ck), onnx_path)
        space = ck_path.name
    else:                                     # no trained tower: raw SCL on both sides
        item_vecs = emb[item_rows].astype(np.float32)
        item_vecs /= np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-12
        onnx_path.unlink(missing_ok=True)
        space = "scl"
    # demographics are unknown at serving time: use the most common training value per field
    default_demo = [int(np.bincount(d["train"][c].astype(np.int64)).argmax()) for c in DEMO_COLS]

    # ---- held-out quality guard: test clicks whose target is in the catalogue ---------
    te, hs = d["test"], d["hist"]
    cand = np.flatnonzero((te["label"] == 1) & np.isin(te["item"], items))
    cand = np.random.default_rng(SEED).choice(cand, min(EVAL_USERS, len(cand)), replace=False)
    sessions = []
    for r in cand:
        h = hs.get(int(te["user"][r]))[-TOWER_SEQ:]
        sessions.append(vocab.decode(h[h != vocab.pad_index]))
    np.save(out / "eval_session_values.npy", np.concatenate(sessions).astype(np.int64))
    np.save(out / "eval_session_offsets.npy",
            np.concatenate([[0], np.cumsum([len(x) for x in sessions])]).astype(np.int64))
    np.save(out / "eval_targets.npy", vocab.decode(te["item"][cand]))

    np.save(out / "item_vecs.npy", item_vecs.astype(np.float32))
    np.save(out / "emb_fp16.npy", emb)
    np.save(out / "vocab_ids.npy", vocab_ids)
    np.save(out / "items.npy", items.astype(np.int64))
    np.save(out / "item_ids.npy", vocab.decode(items))
    np.save(out / "item_cats.npy", item_cats)
    np.save(out / "item_rows.npy", item_rows)

    meta = {
        "version": ARTIFACT_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": int(len(keep)), "pad_row": int(len(keep)), "dim": int(emb.shape[1]),
        "n_items": int(len(items)),
        "emb_dtype": "float16", "emb_mb": round(emb.nbytes / 2**20, 1),
        "category_coverage": round(float(hit.mean()), 5),
        "space": space, "default_demo": default_demo, "eval_users": int(len(cand)),
        "files": {},
        "note": "vocab_ids is sorted, so raw item id -> row is np.searchsorted",
    }
    for f in ("emb_fp16.npy", "vocab_ids.npy", "items.npy", "item_ids.npy", "item_cats.npy",
              "item_rows.npy", "item_vecs.npy", "user_tower.onnx", "eval_session_values.npy",
              "eval_session_offsets.npy", "eval_targets.npy"):
        p = out / f
        if not p.exists():
            continue
        meta["files"][f] = {"bytes": p.stat().st_size, "sha1": _sha1(p)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    meta["build_seconds"] = round(time.time() - t0, 1)
    meta["total_mb"] = round(sum(v["bytes"] for v in meta["files"].values()) / 2**20, 1)
    if verbose:
        print(f"[demo-artifact] {meta['total_mb']} MB in {meta['build_seconds']}s -> {out}")
    return meta


class DemoArtifact:
    """Memory-flat loader used by the service. No 4.5 GB memap, no 35.46 M vocabulary."""

    def __init__(self, root: Path | None = None):
        self.root = _dir(root)
        meta_p = self.root / "meta.json"
        if not meta_p.exists():
            raise FileNotFoundError(
                f"demo artifact missing at {self.root}; run `python -m tmm.cli build-demo-artifact`")
        self.meta = json.loads(meta_p.read_text())
        if self.meta.get("version") != ARTIFACT_VERSION:
            raise RuntimeError(f"artifact version {self.meta.get('version')} != {ARTIFACT_VERSION}")
        # mmap so only the rows actually used are ever faulted in
        self.emb = np.load(self.root / "emb_fp16.npy", mmap_mode="r")
        self.vocab_ids = np.load(self.root / "vocab_ids.npy")
        self.items = np.load(self.root / "items.npy")
        self.item_ids = np.load(self.root / "item_ids.npy")
        self.item_cats = np.load(self.root / "item_cats.npy")
        self.item_rows = np.load(self.root / "item_rows.npy")
        self.item_vecs = np.load(self.root / "item_vecs.npy")
        self.pad_row = int(self.meta["pad_row"])
        self.dim = int(self.item_vecs.shape[1])
        self.space = self.meta["space"]
        self.default_demo = np.asarray(self.meta["default_demo"], dtype=np.int64)
        onnx_path = self.root / "user_tower.onnx"
        self.onnx_path = onnx_path if onnx_path.exists() else None
        if self.space != "scl" and self.onnx_path is None:
            raise RuntimeError(f"artifact at {self.root} has {self.space} item vectors but no "
                               "user_tower.onnx; rebuild it")

    def eval_sessions(self) -> tuple[list[list[int]], np.ndarray]:
        """Held-out ``(sessions, clicked_targets)`` with raw ids, for quality checks."""
        values = np.load(self.root / "eval_session_values.npy")
        offsets = np.load(self.root / "eval_session_offsets.npy")
        sessions = [values[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
        return sessions, np.load(self.root / "eval_targets.npy")

    def rows_for(self, item_ids: list[int] | np.ndarray) -> np.ndarray:
        """Raw anonymised item id -> row in :attr:`emb`; unknown ids map to the pad row."""
        ids = np.asarray(item_ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=np.int64)
        pos = np.clip(np.searchsorted(self.vocab_ids, ids), 0, len(self.vocab_ids) - 1)
        hit = self.vocab_ids[pos] == ids
        return np.where(hit, pos, self.pad_row).astype(np.int64)

    def category_cards(self) -> int:
        return int(self.item_cats.max()) + 2


def load(root: Path | None = None) -> DemoArtifact:
    return DemoArtifact(root)


def exists(root: Path | None = None) -> bool:
    return (_dir(root) / "meta.json").exists()


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(build(), indent=2, default=str))
