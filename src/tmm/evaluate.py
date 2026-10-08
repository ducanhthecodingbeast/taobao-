"""T7 — evaluation: baselines + trained models under one honest protocol.

Protocol (chosen after measuring the data, see ``TASKS.md`` R5)
--------------------------------------------------------------
* The shipped test split is used as-is. **There is no timestamp column in the released
  schema**, so re-deriving a temporal split is impossible and any random re-split would be
  *worse* (it would ignore time entirely). This is stated as a known limitation, not hidden.
* Every model is scored on exactly the same rows and the same per-user candidate sets, so
  the candidate-set size is constant across models and the comparison is legal under Rendle
  (arXiv:1912.02263). ``metrics.compare`` asserts this.
* Both metric families are reported because they answer different questions:
  ``HR@K/NDCG@K/MRR`` (per-user ranking of the shipped hard negatives) and
  ``AUC/LogLoss/PR-AUC/GAUC`` (point-wise discrimination at a 13.7 % positive rate).
* ``GAUC`` is included because popularity is extremely skewed (Gini 0.815); a global AUC can
  be dominated by whether the model ranks popular items well.
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import baselines as B
from . import metrics as M
from .config import PATHS, SEED, device as resolve_device


def _torch():
    import torch
    return torch


def score_all(device_str: str | None = None, models: tuple[str, ...] = ("two_tower", "din", "muse"),
              max_rows: int | None = None) -> dict:
    """Return ``{model_name: scores}`` over the prepared test rows, plus the shared context."""
    torch = _torch()
    from .train import build_emb_table, load_prepared
    from .models import DINRanker, TwoTower

    device_str = device_str or resolve_device()
    d = load_prepared(device_str)
    te, tr, hist, cards, vocab = d["test"], d["train"], d["hist"], d["cards"], d["vocab"]

    if max_rows:
        te = {k: v[:max_rows] for k, v in te.items()}

    emb = build_emb_table(vocab, device_str)
    demo_cols = ["130_1", "130_2", "130_3", "130_4", "130_5"]
    u = te["user"].astype(np.int64)
    it = te["item"].astype(np.int64)
    y = te["label"].astype(np.int8)

    scores: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    # ---- baselines -------------------------------------------------------------------
    # Popularity is estimated on the TRAIN split only. An earlier version passed the
    # arguments shifted by one (item=labels), so every test item scored 0 and the baseline
    # reported AUC = 0.5000 exactly; another counted categories over the TEST split.
    counts = B.item_counts(tr["user"], tr["item"].astype(np.int64), tr["label"], vocab.size)
    cat_pos = np.bincount(tr["206"][tr["label"] == 1].astype(np.int64),
                          minlength=cards["206"]).astype(np.float64)
    scores["random"] = B.random_baseline(len(y), SEED)
    # log1p keeps the counts usable as logits for LogLoss; ranks are unchanged
    scores["global_pop"] = np.log1p(B.global_pop(counts, it))
    scores["positive_pop"] = np.log1p(B.positive_pop(counts, it))
    scores["category_pop"] = np.log1p(B.category_pop(cat_pos, te["206"]))
    t = time.time()
    scores["content_knn"] = B.content_knn(emb, hist, u, it)
    timings["content_knn"] = round(time.time() - t, 2)

    # ---- two-tower -------------------------------------------------------------------
    p = PATHS.models / "two_tower.pt"
    if "two_tower" in models and p.exists():
        ck = torch.load(p, map_location=device_str, weights_only=False)
        model = TwoTower(emb, cat_card=cards["206"],
                         demo_cards=[cards[c] for c in demo_cols],
                         dim=ck["dim"], tower_seq_len=200).to(device_str).eval()
        model.load_state_dict(ck["state_dict"])
        out = np.zeros(len(y), dtype=np.float32)
        t, bs = time.time(), 2048
        with torch.no_grad():
            for s in range(0, len(y), bs):
                e = min(s + bs, len(y))
                h = torch.from_numpy(hist.batch(u[s:e], 1000, pad=model.pad_index).astype(np.int64)) \
                    .to(device_str)
                demo = torch.stack([torch.from_numpy(te[c][s:e].astype(np.int64))
                                    for c in demo_cols], 1).to(device_str)
                ii = torch.from_numpy(it[s:e]).to(device_str)
                cc = torch.from_numpy(te["206"][s:e].astype(np.int64)).to(device_str)
                out[s:e] = model(h, demo, ii, cc).float().cpu().numpy()
        scores["two_tower"] = out
        timings["two_tower"] = round(time.time() - t, 2)

    # ---- rankers ---------------------------------------------------------------------
    demo_t = torch.stack([torch.from_numpy(te[c].astype(np.int64)) for c in demo_cols], 1) \
        .to(device_str)
    for mode in ("din", "muse"):
        p = PATHS.models / f"ranker_{mode}.pt"
        if mode not in models or not p.exists():
            continue
        ck = torch.load(p, map_location=device_str, weights_only=False)
        model = DINRanker(emb, cat_card=cards["206"],
                          demo_cards=[cards[c] for c in demo_cols], dim=128,
                          mode=mode, k=ck["k"]).to(device_str).eval()
        model.load_state_dict(ck["state_dict"])
        out = np.zeros(len(y), dtype=np.float32)
        t, bs = time.time(), 128
        with torch.no_grad():
            for s in range(0, len(y), bs):
                e = min(s + bs, len(y))
                h = torch.from_numpy(hist.batch(u[s:e], 1000, pad=model.pad_index).astype(np.int64)) \
                    .to(device_str)
                ii = torch.from_numpy(it[s:e]).to(device_str)
                cc = torch.from_numpy(te["206"][s:e].astype(np.int64)).to(device_str)
                out[s:e] = model(h, demo_t[s:e], ii, cc).float().cpu().numpy()
        scores[mode] = out
        timings[mode] = round(time.time() - t, 2)

    return {"scores": scores, "y": y, "user": u, "item": it, "cat": te["206"],
            "timings": timings, "n_rows": len(y), "device": device_str,
            "n_users": int(len(np.unique(u)))}


def evaluate(scores: dict, y: np.ndarray, user: np.ndarray) -> dict:
    """Ranking + classification metrics for every model, all on identical candidate sets."""
    uidx, _ = M.factorize(user)
    out = {}
    for name, s in scores.items():
        r = M.per_user_ranking(y, np.asarray(s, dtype=np.float64), uidx)
        r.update(M.classification(y, np.asarray(s, dtype=np.float64)))
        r["GAUC"] = M.gauc(y, np.asarray(s, dtype=np.float64), uidx)
        out[name] = r
    return out


def full_catalogue_recall(device_str: str | None = None, n_users: int = 4000,
                          catalogue: str = "test_targets",
                          ckpt: str = "two_tower.pt") -> dict:
    """Retrieval-stage Recall@K over a catalogue of millions of items.

    The catalogue is the set of items observed in the sampled splits (up to ~2.5 M for test),
    which is stated explicitly: scoring all 35.46 M catalogue items is a cluster-scale job,
    and the *relative* ordering of retrieval methods is preserved on a 2 M-item catalogue
    while the absolute Recall@K is optimistic relative to the full catalogue.
    """
    torch = _torch()
    from .train import build_emb_table, load_prepared
    from .models import TwoTower
    import torch.nn.functional as F

    device_str = device_str or resolve_device()
    if device_str.startswith("cuda"):
        # score_all has already allocated and released its own 8.45 GiB embedding table; the
        # caching allocator would otherwise hold that block while this function asks for a
        # second one.
        torch.cuda.empty_cache()
    d = load_prepared(device_str)
    te, hist, cards, vocab = d["test"], d["hist"], d["cards"], d["vocab"]
    demo_cols = ["130_1", "130_2", "130_3", "130_4", "130_5"]

    cat_items = np.unique(np.concatenate([d["train"]["item"], te["item"]]))
    pos_rows = np.flatnonzero(te["label"] == 1)
    if len(pos_rows) > n_users:
        pos_rows = np.random.default_rng(SEED).choice(pos_rows, n_users, replace=False)
    users = te["user"][pos_rows].astype(np.int64)
    targets = te["item"][pos_rows].astype(np.int64)
    keep = np.isin(targets, cat_items)
    users, targets = users[keep], targets[keep]

    emb = build_emb_table(vocab, device_str)
    ck = torch.load(PATHS.models / ckpt, map_location=device_str, weights_only=False)
    # ``residual`` must be read back from the checkpoint: constructing the tower without it
    # silently drops the residual branch and, worse, the state_dict load then fails or is
    # skipped. This function previously omitted load_state_dict entirely, so the first
    # retrieval ablation reported numbers for a RANDOMLY INITIALISED model.
    model = TwoTower(emb, cat_card=cards["206"], demo_cards=[cards[c] for c in demo_cols],
                     dim=ck["dim"], tower_seq_len=200,
                     residual=bool(ck.get("residual", False))).to(device_str).eval()
    model.load_state_dict(ck["state_dict"])

    # Real categories for the catalogue. The first version of this function passed a constant
    # cat=0 for every catalogue item -- a feature the model never saw at training time, which
    # silently corrupts the whole item tower. Categories are available for every item that
    # appeared as a candidate, which is exactly the catalogue used here.
    all_items = np.concatenate([d["train"]["item"], te["item"]])
    all_cats = np.concatenate([d["train"]["206"], te["206"]])
    uniq, first = np.unique(all_items, return_index=True)
    pos = np.clip(np.searchsorted(uniq, cat_items), 0, len(uniq) - 1)
    hit = uniq[pos] == cat_items
    cat_of_item = np.where(hit, all_cats[first[pos]], len(vocab.ids) + 1).astype(np.int64)
    cat_coverage = float(hit.mean())

    with torch.no_grad():
        h = torch.from_numpy(hist.batch(users, 1000, pad=model.pad_index).astype(np.int64)) \
            .to(device_str)
        demo = torch.stack([torch.from_numpy(te[c][pos_rows][keep].astype(np.int64))
                            for c in demo_cols], 1).to(device_str)
        uv = model.user_vec(h, demo)                                  # [U, D]

        item_t = torch.from_numpy(cat_items).to(device_str)
        cat_t = torch.from_numpy(np.minimum(cat_of_item, cards["206"] - 1)).to(device_str)
        vs = []
        for s in range(0, len(cat_items), 200_000):
            vs.append(model.item_vec(item_t[s:s + 200_000], cat_t[s:s + 200_000]))
        v = torch.cat(vs, 0)                                          # [C, D]

        # Training-free retrieval baseline: cosine between the pooled history vector and the
        # raw SCL item vectors. This is the closest thing to MUSE's GSU, which is a pure
        # similarity search over frozen embeddings rather than a learned tower.
        h_raw = emb[h.clamp(min=0)].float()
        hmask = (h >= 0).unsqueeze(-1).float()
        pooled = torch.nn.functional.normalize((h_raw * hmask).sum(1) / hmask.sum(1).clamp(min=1),
                                              dim=-1)
        v_scl = None

        def _topk_blocked(q: torch.Tensor, kk: torch.Tensor | None = None,
                          block: int = 128, maxk: int = 500) -> np.ndarray:
            """Top-K over the catalogue without ever materialising [U, C]."""
            outs = []
            for s in range(0, q.shape[0], block):
                e = min(s + block, q.shape[0])
                sim = q[s:e] @ kk.T if kk is not None else q[s:e] @ v.float().T
                outs.append(torch.topk(sim, min(maxk, sim.shape[1]), dim=1).indices.cpu().numpy())
                del sim
            return np.concatenate(outs, 0)

        tgt_pos = torch.from_numpy(np.searchsorted(cat_items, targets)).to(device_str)
        top = _topk_blocked(uv)

        scl_blocks = []
        for s in range(0, len(cat_items), 200_000):
            scl_blocks.append(torch.nn.functional.normalize(emb[item_t[s:s + 200_000]].float(),
                                                            dim=-1))
        v_scl = torch.cat(scl_blocks, 0)
        top_scl = _topk_blocked(pooled, v_scl)

    tgt_np = tgt_pos.cpu().numpy()
    res = M.recall_from_topk(top, tgt_np, ks=(10, 50, 100, 200, 500))
    res["content_knn_pooled"] = M.recall_from_topk(top_scl, tgt_np,
                                                   ks=(10, 50, 100, 200, 500))
    res["category_coverage_of_catalogue"] = round(cat_coverage, 5)
    res["checkpoint"] = ckpt
    res["random_recall@500"] = round(500 / len(cat_items), 8)
    res.update({"n_users": int(len(users)), "catalogue_size": int(len(cat_items)),
                "catalogue_note": "items observed in the sampled train+test splits, not the "
                                  "full 35.46M vocabulary",
                "random_recall@10": round(10 / len(cat_items), 8)})
    return res


def run(n_candidates: int = 500, protocol: str = "both", max_rows: int | None = None,
        device_str: str | None = None) -> dict:
    t0 = time.time()
    ctx = score_all(device_str=device_str, max_rows=max_rows)
    table = evaluate(ctx["scores"], ctx["y"], ctx["user"])
    res = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": ctx["device"],
        "rows": ctx["n_rows"], "users": ctx["n_users"],
        "rows_per_user": round(ctx["n_rows"] / max(ctx["n_users"], 1), 3),
        "score_seconds": ctx["timings"],
        "metrics": table,
        "comparison": M.compare(table, "AUC"),
        "ranking_comparison": M.compare(table, "NDCG@10"),
        "total_seconds": round(time.time() - t0, 1),
    }
    if protocol in ("full", "both"):
        try:
            res["full_catalogue_recall"] = full_catalogue_recall(device_str=ctx["device"])
        except Exception as exc:
            res["full_catalogue_recall"] = {"error": f"{type(exc).__name__}: {exc}"}
    (PATHS.stats / "evaluate.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def retrieval_ablation(device_str: str | None = None, n_users: int = 4000) -> dict:
    """Does the training objective fix global retrieval?

    A pointwise BCE objective only ever compares a user against their own ~11 shipped
    candidates, so it never has to separate an item from the other 860 902. In-batch sampled
    softmax forces exactly that separation. This runs the same evaluation for both
    checkpoints plus the training-free SCL cosine baseline, so the claim is measured.
    """
    out: dict = {}
    for name, ck in (("two_tower_bce_pointwise", "two_tower.pt"),
                     ("two_tower_sampled_softmax", "two_tower_ss.pt"),
                     ("two_tower_sampled_softmax_residual", "two_tower_res.pt")):
        if (PATHS.models / ck).exists():
            out[name] = full_catalogue_recall(device_str=device_str, n_users=n_users, ckpt=ck)
        else:
            out[name] = {"error": f"{ck} not found"}
    PATHS.bench.mkdir(parents=True, exist_ok=True)
    (PATHS.bench / "retrieval_ablation.json").write_text(json.dumps(out, indent=2, default=str))
    return out
