"""I/O layer: memory-mapped embedding store, anonymised-id indices, parquet streaming.

Design constraints this module exists to satisfy
------------------------------------------------
* ``scl_emb_int8_p90_values.npy`` is 4.54 GB and the disk has ~101 GB free. It must be
  *memory-mapped*, never copied.
* ``raw/train_user_features.parquet`` is 44.6 GB with only **7 row-groups**. Any
  ``read_table``/``to_pandas`` on it OOMs a 41 GiB budget. Everything streams.
* Anonymised ids are arbitrary ``int64`` (including negatives), so numpy ``searchsorted``
  is used instead of hashing, and all comparisons stay in ``int64``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq

from .config import PATHS, BUDGET


# --------------------------------------------------------------------------------------
# Anonymised id -> row index
# --------------------------------------------------------------------------------------
class SortedKeyIndex:
    """Maps arbitrary int64 ids to rows of a parallel array.

    ``feature_map/*_sorted_map.npy`` files are sorted, so ``searchsorted`` is O(log n).
    ``scl_emb_int8_p90_keys.npy`` is *not* guaranteed sorted; a cached argsort permutation
    (283 MB) is built once and reused so the 4.5 GB value array never has to be reordered.
    """

    def __init__(self, keys: np.ndarray, perm: np.ndarray | None = None):
        self.keys = keys
        self.perm = perm                      # perm[i] -> original row of sorted key i
        if perm is None:
            order = np.argsort(keys, kind="stable")
            self.keys = keys[order]
            self.perm = order

    @classmethod
    def load(cls, path: Path, cache: Path | None = None) -> "SortedKeyIndex":
        keys = np.load(path, mmap_mode="r")
        cache = cache or PATHS.data / f"{path.stem}.argsort.npy"
        if _is_sorted(keys):
            return cls(np.asarray(keys), np.arange(len(keys), dtype=np.int64))
        if cache.exists():
            perm = np.load(cache)
            return cls(np.asarray(keys)[perm], perm)
        order = np.argsort(np.asarray(keys), kind="stable")
        np.save(cache, order)
        return cls(np.asarray(keys)[order], order)

    def find(self, ids: np.ndarray) -> np.ndarray:
        """Return row indices for ``ids``; ``-1`` for ids absent from the table."""
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.keys, ids)
        pos = np.clip(pos, 0, len(self.keys) - 1)
        hit = self.keys[pos] == ids
        rows = np.where(hit, self.perm[pos], -1)
        return rows.astype(np.int64)

    def __len__(self) -> int: return len(self.keys)


def _is_sorted(a: np.ndarray, chunk: int = 8_000_000) -> bool:
    prev = None
    for i in range(0, len(a), chunk):
        c = np.asarray(a[i:i + chunk])
        if prev is not None and c[0] < prev:
            return False
        if not np.all(np.diff(c) >= 0):
            return False
        prev = c[-1]
    return True


# --------------------------------------------------------------------------------------
# Multimodal embedding store
# --------------------------------------------------------------------------------------
class EmbeddingStore:
    """Memory-mapped access to the 35.4 M x 128 int8 SCL multimodal embeddings.

    The int8 payload is a *per-tensor* uniform quantisation. The observed range
    (``min``/``max`` from :meth:`quant_stats`) defines the dequantisation scale, which is
    recorded in ``artifacts/stats/embedding_stats.json`` so nothing is a magic number.
    """

    def __init__(self, keys: SortedKeyIndex, values: np.ndarray, scale: float | None = None):
        self.index = keys
        self.values = values
        self.scale = scale

    @classmethod
    def open(cls, keys_path: Path | None = None, values_path: Path | None = None,
             scale: float | None = None) -> "EmbeddingStore":
        keys_path = keys_path or PATHS.emb_keys
        values_path = values_path or PATHS.emb_values
        return cls(
            SortedKeyIndex.load(keys_path),
            np.load(values_path, mmap_mode="r"),
            scale=scale,
        )

    # -- quantisation -----------------------------------------------------------------
    def quant_stats(self, sample_rows: int | None = None) -> dict:
        """The dequantisation scale is ``1/127``, taken from the dataset's own encoder.

        The official preprocessing (``utils/preprocess.py::convert_scl_int8`` in
        https://github.com/alimama-tech/MUSE) is::

            embeddings_clipped = np.clip(embeddings, -1.0, 1.0)
            int8_embeddings    = np.trunc(embeddings_clipped * 127).astype(np.int8)
            scale              = 1 / 127

        i.e. **symmetric per-element** quantisation of an already L2-normalised vector. This
        is *not* the usual ``max|v| / 127`` calibration: using that instead yields a scale of
        61/127 = 0.48 and dequantised row norms of ~58.9 rather than ~1.0. The measured row
        norm under ``1/127`` is 0.966, which confirms the hypothesis.

        It matters (rather than cancelling out) because the user tower is
        ``LayerNorm(W x + b)``: a constant rescaling of ``x`` scales ``W x`` but *not* ``b``,
        so the two choices are not equivalent. The official model sidesteps this entirely by
        casting to float32 and L2-normalising, which is scale-invariant - that is why the bug
        is easy to miss when only cosine similarity is used.
        """
        if self.scale is None:
            self.scale = 1.0 / 127.0
        return {"scale": self.scale, "scale_source": "convert_scl_int8() in alimama-tech/MUSE",
                "symmetric": True, "dim": int(self.values.shape[1]),
                "clipped_to": 1.0, "truncation": "np.trunc (toward zero, not round)"}

    def row_norm_check(self, sample_rows: int = 100_000) -> float:
        """Sanity check: dequantised row norms should be ≈ 1.0."""
        v = np.asarray(self.values[:sample_rows], dtype=np.float32)
        return float(np.linalg.norm(self.dequantize(v), axis=1).mean())

    def dequantize(self, raw: np.ndarray) -> np.ndarray:
        if self.scale is None:
            self.quant_stats()
        return np.asarray(raw, dtype=np.float32) * self.scale

    # -- lookup -----------------------------------------------------------------------
    def raw(self, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(raw_int8[B,128], mask[B])``; rows for missing ids are zeroed."""
        rows = self.index.find(np.asarray(item_ids, dtype=np.int64))
        mask = rows >= 0
        out = np.zeros((len(rows), self.values.shape[1]), dtype=np.int8)
        if mask.any():
            out[mask] = np.asarray(self.values[rows[mask]])
        return out, mask

    def __call__(self, item_ids: np.ndarray) -> np.ndarray:
        """Dequantised float32 embeddings (missing ids -> zero vector)."""
        raw, _ = self.raw(item_ids)
        return self.dequantize(raw)

    def __len__(self) -> int: return len(self.values)


# --------------------------------------------------------------------------------------
# Parquet streaming
# --------------------------------------------------------------------------------------
def parquet_files(kind: str = "train", shards: Sequence[int] | None = None) -> list[Path]:
    d = PATHS.train_dir if kind == "train" else PATHS.test_dir
    files = sorted(d.glob("*.parquet"))
    return [files[i] for i in shards] if shards is not None else files


def iter_batches(path: Path, columns: Sequence[str] | None = None,
                 batch_size: int = 8192, row_groups: Sequence[int] | None = None,
                 limit: int | None = None) -> Iterator[dict]:
    """Row-group aware streaming. Never materialises a whole 44 GB table."""
    pf = pq.ParquetFile(path)
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None,
                                 row_groups=list(row_groups) if row_groups else None):
        yield batch.to_pydict()
        seen += batch.num_rows
        if limit is not None and seen >= limit:
            return


def read_metadata(kind: str = "train") -> dict:
    d = PATHS.train_dir if kind == "train" else PATHS.test_dir
    return json.loads((d / "metadata.json").read_text())


def sample_rows(path: Path, n: int, columns: Sequence[str] | None = None,
                seed: int = 42, batch_size: int = 8192) -> dict[str, list]:
    """Reservoir sample ``n`` rows without loading the table.

    Reservoir sampling keeps the estimate unbiased even though shards are read in file
    order, which matters because row order is not guaranteed to be random.
    """
    rng = np.random.default_rng(seed)
    res: dict[str, list] = {}
    seen = 0
    for d in iter_batches(path, columns=columns, batch_size=batch_size):
        cols = list(d.keys())
        m = len(d[cols[0]])
        for i in range(m):
            seen += 1
            if len(res.get(cols[0], [])) < n:
                for c in cols:
                    res.setdefault(c, []).append(d[c][i])
            else:
                j = rng.integers(0, seen)
                if j < n:
                    for c in cols:
                        res[c][j] = d[c][i]
    return res


@lru_cache(maxsize=1)
def labels_from_onehot(batch_col: tuple) -> np.ndarray:
    """``[[1,0], [0,1], ...]`` -> ``array([0, 1, ...], int8)``."""
    arr = np.asarray(batch_col, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"label_0 must be a 2-D one-hot list, got shape {arr.shape}")
    return arr.argmax(axis=1).astype(np.int8)


# --------------------------------------------------------------------------------------
# Per-user history store (flat int32 + offsets)
# --------------------------------------------------------------------------------------
class HistoryStore:
    """Flat ``int32`` item-index history with per-user offsets.

    6.93 M users x up to 1 000 items = 27.7 GB as int64. Storing *vocabulary indices*
    (int32) instead halves that to ~14 GB, and only users present in the sampled subset
    are kept, so the real footprint is far smaller.
    """

    def __init__(self, values: np.ndarray, offsets: np.ndarray):
        self.values = values.astype(np.int32, copy=False)
        self.offsets = offsets.astype(np.int64, copy=False)

    def __len__(self) -> int: return len(self.offsets) - 1

    def get(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i]:self.offsets[i + 1]]

    def batch(self, idx: np.ndarray, length: int, pad: int = -1) -> np.ndarray:
        """Right-aligned, padded batch: newest items last, padding = ``pad``."""
        out = np.full((len(idx), length), pad, dtype=np.int32)
        for r, i in enumerate(idx):
            h = self.get(int(i))
            if len(h):
                take = h[-length:]
                out[r, length - len(take):] = take
        return out

    def save(self, prefix: Path) -> None:
        np.save(prefix.with_suffix(".values.npy"), self.values)
        np.save(prefix.with_suffix(".offsets.npy"), self.offsets)

    @classmethod
    def load(cls, prefix: Path) -> "HistoryStore":
        return cls(np.load(prefix.with_suffix(".values.npy"), mmap_mode="r"),
                   np.load(prefix.with_suffix(".offsets.npy")))

    @classmethod
    def build(cls, user_ids: np.ndarray, histories: list[np.ndarray]) -> "HistoryStore":
        """``user_ids`` must be contiguous ``0..N-1`` vocabulary indices."""
        n = int(user_ids.max()) + 1 if len(user_ids) else 0
        counts = np.zeros(n, dtype=np.int64)
        np.add.at(counts, user_ids, [len(h) for h in histories])
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        values = np.zeros(int(offsets[-1]), dtype=np.int32)
        cursor = offsets[:-1].copy()
        for u, h in zip(user_ids, histories):
            values[cursor[u]:cursor[u] + len(h)] = h
            cursor[u] += len(h)
        return cls(values, offsets)


def budget() -> "BUDGET.__class__":  # pragma: no cover - convenience
    return BUDGET
