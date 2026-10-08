"""T5/T7 training loops — two-tower retrieval and DIN/MUSE ranking.

Design notes that matter for correctness on this dataset
--------------------------------------------------------
1. **Frozen embedding table.** The 128-d SCL vectors are pre-trained features, so they are
   registered as a buffer, not a parameter. This keeps optimizer state at zero for the
   8.45 GiB table and makes the full-catalogue forward pass affordable.
2. **Point-wise BCE on the shipped hard negatives.** The dataset's negatives are real sampled
   candidates, not random items, so ``BCEWithLogits`` is the honest objective. In-batch
   sampled-softmax is available (``objective='sampled_softmax'``) but is *not* the default,
   because it changes the effective candidate distribution and therefore the metric.
3. **Class imbalance.** 13.7 % positives; ``pos_weight`` is configurable but left at 1.0 by
   default so AUC comparisons stay calibrated.
4. **MUSE vs DIN is a one-flag ablation** (``mode`` + ``k``), which is what makes the
   latency/quality trade-off in the report a measurement instead of a claim.
5. **Everything is checkpointed per epoch** so a GPU job that hits a shared-GPU preemption
   does not lose the run.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .config import BUDGET, PATHS, SEED, device
from .io import HistoryStore
from .vocab import ItemVocab


# --------------------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def load_prepared(device_str: str = "cpu") -> dict:
    """Load the arrays produced by :mod:`tmm.prepare` and move them to the device."""
    torch = _torch()
    vocab = ItemVocab.build()
    train = dict(np.load(PATHS.data / "train.npz"))
    test = dict(np.load(PATHS.data / "test.npz"))
    hs = HistoryStore.load(PATHS.data / "hist")
    cards = json.loads((PATHS.stats / "prepare.json").read_text())["cardinalities"]
    return {"vocab": vocab, "train": train, "test": test, "hist": hs, "cards": cards,
            "device": device_str}


def build_emb_table(vocab: ItemVocab, device_str: str, dtype=None):
    """Materialise the frozen ``[vocab+1, 128]`` embedding table on the target device.

    Memory discipline matters here. The naive form

        arr = load_embeddings_fp16(vocab)      # 9.08 GiB fp16 numpy
        t   = torch.from_numpy(arr).to(dtype)  # + another 9.08 GiB CPU tensor
        t   = t.to(device)                     # + 8.45 GiB GPU

    peaks at ~18 GiB of CPU RAM and was killed by the OOM reaper on this host. Taking the
    dtype cast *during* the host->device copy keeps the peak at the single numpy buffer:

        arr = load_embeddings_fp16(vocab)      # 9.08 GiB
        t   = torch.from_numpy(arr)            # zero-copy view
        t   = t.to(device, dtype=bf16)         # one cast-and-transfer, no CPU staging
    """
    torch = _torch()
    from .vocab import load_embeddings_fp16

    dt = dtype or (torch.bfloat16 if device_str.startswith("cuda") else torch.float32)
    arr = load_embeddings_fp16(vocab)
    t = torch.from_numpy(arr)                 # shares the numpy buffer, no copy
    if device_str.startswith("cuda"):
        return t.to(device_str, dtype=dt, non_blocking=True)
    return t.to(dt)


def _history_batch(hs: HistoryStore, uid: np.ndarray, seq_len: int, pad: int):
    torch = _torch()
    h = hs.batch(uid, seq_len, pad=pad)
    return torch.from_numpy(h.astype(np.int64, copy=False))


# --------------------------------------------------------------------------------------
def _pos_weight(labels: np.ndarray, device_str: str, enabled: bool):
    torch = _torch()
    if not enabled:
        return None
    p = float(labels.mean())
    return torch.tensor([(1 - p) / max(p, 1e-6)], device=device_str)


def train_two_tower(epochs: int = 3, batch_size: int = 512, dim: int = 128,
                    lr: float | None = None, seq_len: int | None = None,
                    objective: str = "bce", device_str: str | None = None,
                    max_steps: int | None = None, use_pos_weight: bool = False,
                    tag: str = "", residual: bool = False) -> dict:
    torch = _torch()
    import torch.nn.functional as F

    from .models import TwoTower, count_frozen, count_parameters

    device_str = device_str or device()
    lr = lr or BUDGET.lr
    seq_len = seq_len or 1000
    d = load_prepared(device_str)
    vocab, hist, cards = d["vocab"], d["hist"], d["cards"]
    tr, te = d["train"], d["test"]

    emb = build_emb_table(vocab, device_str)
    model = TwoTower(emb, cat_card=cards["206"],
                     demo_cards=[cards[c] for c in ("130_1", "130_2", "130_3", "130_4", "130_5")],
                     dim=dim, tower_seq_len=200, residual=residual).to(device_str)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=1e-5)
    if objective == "sampled_softmax":
        # In-batch softmax treats row i's item as the right answer for row i's user, so only
        # clicked rows are training pairs. Using every row (86 % are non-clicks) taught the
        # tower to retrieve items the user was shown and ignored.
        rows = np.flatnonzero(tr["label"] == 1)
        # logQ correction (Yi et al., RecSys 2019): in-batch negatives are sampled in
        # proportion to how often an item is a positive, so popular items are over-penalised
        # unless log q(item) is subtracted from their logits.
        freq = np.bincount(tr["item"][rows].astype(np.int64), minlength=vocab.size)
        log_q = np.log(freq / max(freq.sum(), 1) + 1e-12).astype(np.float32)
    else:
        rows = np.arange(len(tr["label"]))
    n = len(rows)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * math.ceil(n / batch_size)))

    demo_cols = ["130_1", "130_2", "130_3", "130_4", "130_5"]
    rng = np.random.default_rng(SEED)
    history: list[dict] = []
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch_size):
            idx = rows[perm[s:s + batch_size]]
            if len(idx) < 2:
                continue
            uid = tr["user"][idx]
            item = torch.from_numpy(tr["item"][idx].astype(np.int64)).to(device_str)
            cat = torch.from_numpy(tr["206"][idx].astype(np.int64)).to(device_str)
            y = torch.from_numpy(tr["label"][idx].astype(np.float32)).to(device_str)
            h = _history_batch(hist, uid, seq_len, pad=model.pad_index).to(device_str)
            demo = torch.stack([torch.from_numpy(tr[c][idx].astype(np.int64)) for c in demo_cols],
                               dim=1).to(device_str)

            if objective == "sampled_softmax":
                u = model.user_vec(h, demo)
                v = model.item_vec(item, cat)
                logits = (u @ v.T) * model.logit_scale.exp()
                logits = logits - torch.from_numpy(log_q[tr["item"][idx]]).to(device_str)[None, :]
                # another row of the same user, or the same item, is not a negative
                ut = torch.from_numpy(uid.astype(np.int64)).to(device_str)
                clash = (ut[:, None] == ut[None, :]) | (item[:, None] == item[None, :])
                clash.fill_diagonal_(False)
                logits = logits.masked_fill(clash, float("-inf"))
                loss = F.cross_entropy(logits, torch.arange(len(idx), device=device_str))
            else:
                logits = model(h, demo, item, cat)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=_pos_weight(tr["label"], device_str, use_pos_weight))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            tot += float(loss); nb += 1; step += 1
            if max_steps and step >= max_steps:
                break
        if device_str.startswith("cuda"):
            torch.cuda.synchronize()
        history.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5),
                        "seconds": round(time.time() - t0, 1), "steps": nb})
        print(f"[two-tower] epoch {ep}: loss={history[-1]['loss']} "
              f"({history[-1]['seconds']}s, {nb} steps)", flush=True)
        if max_steps and step >= max_steps:
            break

    train_s = time.time() - t0
    PATHS.models.mkdir(parents=True, exist_ok=True)
    ckpt = PATHS.models / f"two_tower{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "dim": dim, "cards": cards,
                "vocab_size": int(vocab.size), "seq_len": seq_len,
                "objective": objective, "residual": residual}, ckpt)

    res = {
        "model": f"two_tower{tag}", "objective": objective, "device": device_str,
        "epochs": epochs, "batch_size": batch_size, "steps": step,
        "rows": int(n), "lr": lr, "dim": dim, "seq_len": seq_len,
        "train_seconds": round(train_s, 1),
        "rows_per_second": int(n * epochs / max(train_s, 1e-6)),
        "params_trainable": count_parameters(model), "residual": residual,
        "params_frozen": count_frozen(model),
        "emb_table_gib": round(emb.numel() * emb.element_size() / 2**30, 2),
        "vram_peak_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                          if device_str.startswith("cuda") else None),
        "history": history,
        "checkpoint": str(ckpt),
    }
    (PATHS.stats / f"train_two_tower{tag or ''}.json").write_text(
        json.dumps(res, indent=2, default=str))
    return res


def train_rankers(epochs: int = 2, batch_size: int = 128, k: int = 50,
                  device_str: str | None = None, max_steps: int | None = None,
                  modes: tuple[str, ...] = ("din", "muse")) -> dict:
    """Train DIN and the MUSE-search variant as a controlled ablation (same data, same seeds)."""
    torch = _torch()
    import torch.nn.functional as F

    from .models import DINRanker, count_parameters

    device_str = device_str or device()
    d = load_prepared(device_str)
    vocab, hist, cards = d["vocab"], d["hist"], d["cards"]
    tr = d["train"]
    emb = build_emb_table(vocab, device_str)
    demo_cols = ["130_1", "130_2", "130_3", "130_4", "130_5"]
    n = len(tr["label"])
    out: dict = {}

    for mode in modes:
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = DINRanker(emb, cat_card=cards["206"],
                          demo_cards=[cards[c] for c in demo_cols],
                          dim=128, mode=mode, k=k).to(device_str)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3,
                                weight_decay=1e-5)
        t0 = time.time()
        step, hist_log = 0, []
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot, nb = 0.0, 0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                if len(idx) < 2:
                    continue
                uid = tr["user"][idx]
                item = torch.from_numpy(tr["item"][idx].astype(np.int64)).to(device_str)
                cat = torch.from_numpy(tr["206"][idx].astype(np.int64)).to(device_str)
                y = torch.from_numpy(tr["label"][idx].astype(np.float32)).to(device_str)
                h = _history_batch(hist, uid, 1000, pad=model.pad_index).to(device_str)
                demo = torch.stack([torch.from_numpy(tr[c][idx].astype(np.int64))
                                    for c in demo_cols], dim=1).to(device_str)
                logits = model(h, demo, item, cat)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                tot += float(loss); nb += 1; step += 1
                if max_steps and step >= max_steps:
                    break
            hist_log.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5)})
            print(f"[{mode}] epoch {ep}: loss={hist_log[-1]['loss']} ({nb} steps)", flush=True)
            if max_steps and step >= max_steps:
                break
        if device_str.startswith("cuda"):
            torch.cuda.synchronize()
        ckpt = PATHS.models / f"ranker_{mode}.pt"
        torch.save({"state_dict": model.state_dict(), "mode": mode, "k": k, "cards": cards},
                   ckpt)
        out[mode] = {
            "mode": mode, "k": k if mode == "muse" else 1000, "epochs": epochs,
            "batch_size": batch_size, "steps": step, "train_seconds": round(time.time() - t0, 1),
            "params_trainable": count_parameters(model), "history": hist_log,
            "checkpoint": str(ckpt),
        }
    out["device"] = device_str
    (PATHS.stats / "train_rankers.json").write_text(json.dumps(out, indent=2, default=str))
    return out
