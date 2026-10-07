"""Item vocabulary: anonymised int64 item id  <->  contiguous index  <->  embedding row.

Three different orderings exist in the dataset and mixing them up is the single easiest
way to silently train on garbage:

1. ``feature_map/150_2_180_sorted_map_p90.npy`` — the **sorted** list of the 35.46 M item
   ids that appear in users' histories (the 90 % coverage tier). This is our vocabulary
   order: it is sorted, dense, and cheap to ``searchsorted``.
2. ``feature_map/scl_emb_int8_p90_keys.npy`` — the item id of each **row** of the embedding
   array. Verified in :mod:`tmm.profile` to be only *block-wise* sorted (first order break
   at row 20 000 000), so it **cannot** be used with ``searchsorted`` directly.
3. ``raw/item_features.parquet`` — metadata for the 4.16 M items that appear as *targets*
   in the sample tables (a much smaller set than the history vocabulary).

:class:`ItemVocab` collapses (2) into a flat ``int32`` map ``vocab_index -> emb_row`` so
training code only ever deals with dense indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PATHS
from .io import SortedKeyIndex


class ItemVocab:
    """Dense item index space shared by every stage of the pipeline."""

    def __init__(self, ids: np.ndarray, emb_row: np.ndarray):
        self.ids = ids                      # sorted int64, vocab_index -> item_id
        self.emb_row = emb_row              # vocab_index -> row in scl values (-1 = none)
        self.pad_index = len(ids)           # reserved: padding / unknown
        self.unk_index = len(ids)           # same slot; kept as an alias for clarity

    # -- construction -----------------------------------------------------------------
    @classmethod
    def build(cls, cache: Path | None = None, force: bool = False) -> "ItemVocab":
        cache = cache or (PATHS.data / "item_vocab.npz")
        if cache.exists() and not force:
            z = np.load(cache)
            return cls(z["ids"], z["emb_row"])
        ids = np.asarray(np.load(PATHS.feature_map / "150_2_180_sorted_map_p90.npy"))
        if not np.all(np.diff(ids) >= 0):
            raise RuntimeError("expected the p90 history map to be sorted; it is not")
        keys = SortedKeyIndex.load(PATHS.emb_keys)
        # vectorised item_id -> embedding row
        emb_row = keys.find(ids).astype(np.int32)
        vocab = cls(ids, emb_row)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=ids, emb_row=emb_row)
        return vocab

    # -- encoding ---------------------------------------------------------------------
    def encode(self, item_ids: np.ndarray) -> np.ndarray:
        """item_id -> vocab index; unknown ids map to :attr:`pad_index`."""
        item_ids = np.asarray(item_ids, dtype=np.int64)
        pos = np.searchsorted(self.ids, item_ids)
        pos = np.clip(pos, 0, len(self.ids) - 1)
        hit = self.ids[pos] == item_ids
        return np.where(hit, pos, self.pad_index).astype(np.int32)

    def decode(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        out = np.full(idx.shape, -1, dtype=np.int64)
        ok = (idx >= 0) & (idx < len(self.ids))
        out[ok] = self.ids[idx[ok]]
        return out

    def encode_ragged(self, seqs: list[np.ndarray]) -> list[np.ndarray]:
        return [self.encode(s) for s in seqs]

    @property
    def size(self) -> int:
        """Vocabulary size *including* the padding slot."""
        return len(self.ids) + 1

    @property
    def embeddable(self) -> int:
        return int((self.emb_row >= 0).sum())

    def coverage_of(self, item_ids: np.ndarray) -> float:
        enc = self.encode(item_ids)
        return float((enc != self.pad_index).mean())

    def stats(self) -> dict:
        return {
            "vocab_size": int(len(self.ids)),
            "with_embedding": self.embeddable,
            "embedding_coverage": round(self.embeddable / len(self.ids), 6),
            "pad_index": int(self.pad_index),
            "id_min": int(self.ids.min()), "id_max": int(self.ids.max()),
        }


def load_embeddings_fp16(vocab: ItemVocab, dim: int = 128, limit: int | None = None,
                         scale: float | None = None, chunk: int = 1_000_000) -> np.ndarray:
    """Materialise ``[vocab_size, dim]`` float16 embeddings, dequantised from int8.

    ``[pad_index]`` is an all-zero row so padding contributes nothing to pooling.
    Memory: 35.46 M x 128 x 2 B = **8.45 GiB** for the full catalogue.

    The gather is **chunked on purpose**. A single ``store.values[rows]`` fancy-index over
    35.46 M rows materialises a 4.5 GB int8 array and then the ``astype(float32)`` copies it
    again as an **18 GB** fp32 array -- a 27 GB peak that does not fit beside a 9 GiB output on
    a 32 GiB budget. Chunking caps the transient at ``chunk x 128 x 4 B`` (512 MB at 1 M).
    """
    from .io import EmbeddingStore

    store = EmbeddingStore.open(scale=scale)
    qs = store.quant_stats()
    n = len(vocab.ids) if limit is None else min(limit, len(vocab.ids))
    out = np.zeros((n + 1, dim), dtype=np.float16)
    rows = vocab.emb_row[:n]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = rows[s:e]
        ok = block >= 0
        if not ok.any():
            continue
        raw = np.asarray(store.values[block[ok]], dtype=np.float32)
        raw *= qs["scale"]
        # place back into the vocab-aligned positions
        idx = np.flatnonzero(ok) + s
        out[idx] = raw.astype(np.float16)
    return out


def load_embeddings_subset(vocab: ItemVocab, keep: np.ndarray, dim: int = 128,
                           scale: float | None = None, chunk: int = 1_000_000) -> np.ndarray:
    """Dequantised embeddings for **only** ``keep`` (vocabulary indices) plus a pad row.

    The full table is 8.45 GiB in fp16, which is fine on the 48 GB GPU but absurd for the
    Direction-1 demo (a 2 vCPU / 4 GB VPS). The demo only ever needs vectors for the catalogue
    and for items that can appear in a session, so this reads just those rows straight out of
    the memmapped array. Output index ``len(keep)`` is the reserved all-zero padding row.
    """
    from .io import EmbeddingStore

    store = EmbeddingStore.open(scale=scale)
    qs = store.quant_stats()
    keep = np.asarray(keep, dtype=np.int64)
    out = np.zeros((len(keep) + 1, dim), dtype=np.float16)
    rows = vocab.emb_row[keep]
    for s in range(0, len(keep), chunk):
        e = min(s + chunk, len(keep))
        block = rows[s:e]
        ok = block >= 0
        if not ok.any():
            continue
        raw = np.asarray(store.values[block[ok]], dtype=np.float32) * qs["scale"]
        out[np.flatnonzero(ok) + s] = raw.astype(np.float16)
    return out
