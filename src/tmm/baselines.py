"""T4 — baselines that the neural models must actually beat.

Why this file is not optional
-----------------------------
Measured on the real data, item popularity has a **Gini of 0.815**, the top 100 items account
for **65.4 % of all positive interactions**, and **96 139 items** (of 3.78 M) cover 50 % of
interactions. On a candidate set of ~11 items per user, a global-popularity ranker is a very
strong baseline; a wrong-but-neural model often loses to it. Reporting the baselines first is
what makes the neural numbers interpretable.

Four baselines, all O(1) per candidate at serving time except the content-KNN one:

``global_pop``      interaction count per item over the training split
``positive_pop``    *positive* interaction count per item (the strongest cheap prior)
``category_pop``    category-conditional positive share
``content_knn``     max cosine between the candidate's SCL vector and the user's history --
                    a training-free version of what MUSE's search step does, and the fair
                    content baseline for a multimodal dataset

Item-CF on co-occurrence is deliberately **not** implemented: with a 35.46 M-item vocabulary a
co-occurrence matrix is not tractable on this machine, and the SCL content-KNN baseline
captures the same "users who touched similar items" signal without the 35 M x 35 M object.
"""

from __future__ import annotations

import numpy as np


def item_counts(user: np.ndarray, item: np.ndarray, label: np.ndarray,
                vocab_size: int) -> dict[str, np.ndarray]:
    """Frequency tables over the training split (the only place popularity may be estimated)."""
    glob = np.bincount(item, minlength=vocab_size).astype(np.float64)
    pos = np.bincount(item[label == 1], minlength=vocab_size).astype(np.float64)
    return {"global": glob, "positive": pos}


def category_counts(item: np.ndarray, label: np.ndarray, cat_of_item: np.ndarray,
                    n_cats: int) -> np.ndarray:
    cats = cat_of_item[item]
    return np.bincount(cats[label == 1], minlength=n_cats).astype(np.float64)


def global_pop(counts: dict, item: np.ndarray) -> np.ndarray:
    return counts["global"][item]


def positive_pop(counts: dict, item: np.ndarray) -> np.ndarray:
    return counts["positive"][item]


def category_pop(cat_counts: np.ndarray, cat: np.ndarray) -> np.ndarray:
    return cat_counts[cat]


def content_knn(emb_table, hist, user_idx: np.ndarray, item: np.ndarray,
                chunk: int = 1024, reduce: str = "max") -> np.ndarray:
    """``emb_table`` is the dense ``[vocab, dim]`` tensor (not an ``nn.Embedding``)."""
    """max/mean cosine(candidate item, history items).

    This is the training-free analogue of MUSE's search: the same similarity computation,
    without the learned attention on top. If a trained model cannot beat this, it has learned
    nothing beyond "the candidate is semantically close to something the user already viewed".
    """
    import torch

    n = len(item)
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        u = user_idx[s:e]
        # keep the padding mask on the same device as the embedding table, or masked_fill
        # fails with a CPU/CUDA mismatch
        h = torch.from_numpy(hist.batch(u, 200, pad=-1).astype(np.int64)).to(emb_table.device)
        mask = h >= 0
        hv = emb_table[h.clamp(min=0)].float()                   # [b, L, D]
        iv = emb_table[torch.from_numpy(item[s:e].astype(np.int64))].float()
        iv = torch.nn.functional.normalize(iv, dim=-1)
        hv = torch.nn.functional.normalize(hv, dim=-1)
        sim = torch.einsum("bd,bld->bl", iv, hv).masked_fill(~mask, float("-inf"))
        val = (sim.max(dim=1).values if reduce == "max" else
               sim.masked_fill(~mask, 0.0).sum(1) / mask.sum(1).clamp(min=1))
        out[s:e] = val.detach().float().cpu().numpy()
    return out


def random_baseline(n: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).random(n).astype(np.float32)
