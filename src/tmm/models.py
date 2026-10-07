"""T5 — the models. Deliberately small, because the difficulty is the data, not the net.

Three architectures, all sharing the **frozen** 128-d SCL multimodal embedding table:

``TwoTower``
    Direction 2 retrieval stage. User tower pools the (truncated) history of SCL vectors plus
    demographic embeddings; item tower is an MLP over the target item's SCL vector plus its
    category/city/province. Both outputs are L2-normalised, so ``dot == cosine`` and the same
    vectors can be dropped straight into FAISS. Trained point-wise with BCE **on the shipped
    hard negatives** (see :mod:`tmm.prepare`) — this is the honest objective for this dataset,
    and it avoids the sampled-softmax pitfalls catalogued in the report.

``DINRanker``
    Direction 2 ranking stage. Target item is the query; attention over the full 1 000-item
    history (Deep Interest Network). Attention over 1 000 items is the expensive part.

``MUSE``-style search
    Same ranker, but the 1 000-item history is first *searched* with the target embedding and
    only the top-``k`` most similar items go into the attention. This is the paper's core idea
    and it is the ablation that makes the latency/quality trade-off measurable rather than
    asserted: ``mode='din'`` vs ``mode='muse'`` differ only in ``k``.

A note on the frozen table: TAOBAO-MM ships embeddings produced by an SCL encoder trained on
the whole corpus (arXiv:2407.19467), so the table is a *feature*, not a parameter. Keeping it
frozen is what lets the 35.4 M x 128 table fit in 8.45 GiB of VRAM as bf16 instead of needing
an 18 GB fp32 optimiser state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenEmbedding(nn.Module):
    """Wrapper around a pre-trained table so it is excluded from optimizer state."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)

    @property
    def num_embeddings(self) -> int:
        return self.weight.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.weight.shape[1]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)


class MLP(nn.Module):
    def __init__(self, dims: list[int], out_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = dims[0]
        for h in dims[1:]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        if out_dim is not None:
            layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoTower(nn.Module):
    """Retrieval stage: independent user/item encoders + dot product."""

    def __init__(self, emb_table: torch.Tensor, cat_card: int, demo_cards: list[int],
                 dim: int = 128, hidden: int = 256, out_dim: int = 128,
                 tower_seq_len: int = 200, dropout: float = 0.1,
                 residual: bool = False):
        super().__init__()
        self.items = FrozenEmbedding(emb_table)
        # A plain LayerNorm MLP on top of an L2-normalised input *collapses* the geometry:
        # measured pairwise cosine of learned item vectors is 0.966 versus 0.022 for the raw
        # SCL vectors, which makes global top-K retrieval essentially arbitrary. A linear
        # residual from the frozen embedding into the output space preserves the original
        # metric while still letting the MLP add a learned correction.
        self.residual = residual
        self.item_cat = nn.Embedding(cat_card, 32)
        self.demo = nn.ModuleList([nn.Embedding(c, 16) for c in demo_cards])
        self.tower_seq_len = tower_seq_len

        self.user_tower = MLP([dim + 16 * len(demo_cards), hidden], out_dim=out_dim,
                              dropout=dropout)
        self.item_tower = MLP([dim + 32, hidden], out_dim=out_dim, dropout=dropout)
        self.user_res = nn.Linear(dim, out_dim, bias=False) if residual else None
        self.item_res = nn.Linear(dim, out_dim, bias=False) if residual else None
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    # -- towers ------------------------------------------------------------------------
    def user_vec(self, hist: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        """``hist`` [B, L] padded with ``pad_index``; ``demo`` [B, n_demo]."""
        if self.tower_seq_len and hist.shape[1] > self.tower_seq_len:
            hist = hist[:, -self.tower_seq_len:]
        # The frozen table may be bf16 (8.45 GiB on GPU) while the tower weights are fp32.
        # torch.cat happens to promote mixed dtypes but nn.Linear does not, so the cast is
        # made explicit here rather than relying on promotion rules.
        h = self.items(hist).float()                            # [B, L, D]
        mask = (hist != self.pad_index).unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        d = torch.cat([e(demo[:, i]) for i, e in enumerate(self.demo)], dim=-1)
        z = self.user_tower(torch.cat([pooled, d], dim=-1))
        if self.residual:
            z = z + self.user_res(pooled)
        return F.normalize(z, dim=-1)

    def item_vec(self, item: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        e = self.items(item).float()
        h = torch.cat([e, self.item_cat(cat)], dim=-1)
        z = self.item_tower(h)
        if self.residual:
            z = z + self.item_res(e)
        return F.normalize(z, dim=-1)

    # -- scoring -----------------------------------------------------------------------
    def forward(self, hist, demo, item, cat) -> torch.Tensor:
        u = self.user_vec(hist, demo)
        v = self.item_vec(item, cat)
        return (u * v).sum(-1) * self.logit_scale.exp()

    @torch.no_grad()
    def full_scores(self, u: torch.Tensor, item_chunk: torch.Tensor,
                    cat_chunk: torch.Tensor, chunk: int = 200_000) -> torch.Tensor:
        """[B_users, N_items] scores in chunks — used for full-catalogue evaluation/indexing."""
        outs = []
        for s in range(0, item_chunk.shape[0], chunk):
            v = self.item_vec(item_chunk[s:s + chunk], cat_chunk[s:s + chunk])
            outs.append(u @ v.T)
        return torch.cat(outs, dim=1)

    @property
    def pad_index(self) -> int:
        return self.items.num_embeddings - 1


class DINRanker(nn.Module):
    """Ranking stage. ``mode='din'`` = attention over the whole history;
    ``mode='muse'`` = search the history with the target first, then attend over top-k."""

    def __init__(self, emb_table: torch.Tensor, cat_card: int, demo_cards: list[int],
                 dim: int = 128, hidden: int = 256, mode: str = "din", k: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        assert mode in ("din", "muse")
        self.items = FrozenEmbedding(emb_table)
        self.mode, self.k = mode, k
        self.item_cat = nn.Embedding(cat_card, 32)
        self.demo = nn.ModuleList([nn.Embedding(c, 16) for c in demo_cards])

        self.att = nn.Sequential(nn.Linear(dim * 3, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        self.head = MLP([dim * 3 + 32 + 16 * len(demo_cards), hidden], out_dim=1,
                        dropout=dropout)

    def search(self, target: torch.Tensor, hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MUSE's search step: cosine(target, history item) -> top-k indices."""
        h = self.items(hist).float()                             # [B, L, D]
        mask = (hist != self.pad_index)
        sim = torch.einsum("bd,bld->bl", F.normalize(target, dim=-1),
                           F.normalize(h, dim=-1))
        sim = sim.masked_fill(~mask, float("-inf"))
        k = min(self.k, sim.shape[1])
        top = torch.topk(sim, k, dim=1).indices                 # [B, k]
        return hist.gather(1, top), top

    def forward(self, hist: torch.Tensor, demo: torch.Tensor, item: torch.Tensor,
                cat: torch.Tensor) -> torch.Tensor:
        target = self.items(item).float()                        # [B, D]
        if self.mode == "muse":
            hist, _ = self.search(target, hist)
        h = self.items(hist).float()                             # [B, L', D]
        mask = (hist != self.pad_index)
        q = target.unsqueeze(1).expand_as(h)
        logits = self.att(torch.cat([q, h, q * h], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        # a user with no history at all -> uniform is undefined; fall back to zeros
        bad = ~mask.any(dim=1)
        att = torch.softmax(torch.where(bad.unsqueeze(1), torch.zeros_like(logits), logits), dim=-1)
        att = torch.where(bad.unsqueeze(1), torch.zeros_like(att), att)
        interest = (att.unsqueeze(-1) * h).sum(1)                # [B, D]

        d = torch.cat([e(demo[:, i]) for i, e in enumerate(self.demo)], dim=-1)
        c = self.item_cat(cat)
        x = torch.cat([target, interest, target * interest, c, d], dim=-1)
        return self.head(x).squeeze(-1)

    @property
    def pad_index(self) -> int:
        """The reserved all-zero embedding row used for sequence padding."""
        return self.items.num_embeddings - 1


class PopularityModel:
    """Global / category-conditional popularity. Must be beaten before claiming progress."""

    def __init__(self, item_scores: torch.Tensor, cat_scores: dict | None = None,
                 alpha: float = 0.0):
        self.item_scores = item_scores
        self.cat_scores = cat_scores or {}
        self.alpha = alpha

    @torch.no_grad()
    def __call__(self, item, cat, **_):
        s = self.item_scores[item]
        if self.cat_scores:
            cs = torch.stack([self.cat_scores.get(int(c), self.item_scores.new_zeros(()))
                              for c in cat])
            s = s + self.alpha * cs
        return s


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen(model: nn.Module) -> int:
    return sum(b.numel() for b in model.buffers())
