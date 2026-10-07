"""T7 — evaluation metrics, implemented so that the *unsound* variants are hard to use.

Two protocols are supported and are never mixed silently:

``ranking``
    The shipped test set is already a **hard-negative candidate set**: ~11.5 candidate rows
    per user of which 13.9 % are positive. So the honest task is *per-user re-ranking of the
    given candidates*: group by user, sort by score, compute HR@K / NDCG@K / MRR.

``classify``
    Global AUC / LogLoss / PR-AUC over the same rows, which is the natural metric for a
    ~1:6.3 positive:negative problem.

Two traps this module makes explicit (see ``Sources`` in the report):

* Rendle, *Evaluation Metrics for Item Recommendation under Sampling* (arXiv:1912.02263):
  sampled metrics do **not** preserve model rankings, not even in expectation. Therefore
  ``compare()`` refuses to compare two models evaluated with different candidate-set sizes.
* Per-*row* averaging of NDCG inflates users that have many candidate rows. Everything is
  computed per user and macro-averaged.
"""

from __future__ import annotations

import numpy as np


def _dcg(rank_positions: np.ndarray, k: int) -> np.ndarray:
    """DCG of the positive items given their 0-based ranks within each user."""
    gains = 1.0 / np.log2(rank_positions + 2.0)
    return np.where(rank_positions < k, gains, 0.0)


def per_user_ranking(y_true: np.ndarray, scores: np.ndarray, user_idx: np.ndarray,
                     ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Group by user, rank candidates by score, return macro-averaged ranking metrics.

    ``user_idx`` must map every row to a contiguous user id (call :func:`factorize`).
    """
    order = np.lexsort((-scores, user_idx))
    u_sorted = user_idx[order]
    y_sorted = y_true[order].astype(bool)
    starts = np.searchsorted(u_sorted, np.arange(u_sorted[-1] + 1), side="left") \
        if len(u_sorted) else np.zeros(0, dtype=int)

    n_users = len(starts)
    # rank of each row inside its user block
    rank_in_user = np.arange(len(u_sorted)) - starts[u_sorted]

    # first positive rank per user (np.inf when a user has no positive)
    pos_rank = np.full(n_users, np.inf)
    sel = y_sorted
    if sel.any():
        np.minimum.at(pos_rank, u_sorted[sel], rank_in_user[sel])

    hit = np.isfinite(pos_rank)
    out: dict = {
        "n_users": int(n_users),
        "n_rows": int(len(y_true)),
        "rows_per_user_mean": round(len(y_true) / max(n_users, 1), 3),
        "users_with_a_positive": int(hit.sum()),
        "users_with_a_positive_rate": round(float(hit.mean()), 4) if n_users else 0.0,
        "candidate_set_size": round(len(y_true) / max(n_users, 1), 3),
    }
    for k in ks:
        out[f"HR@{k}"] = round(float(np.mean(hit & (pos_rank < k))), 5)
        dcg = _dcg(pos_rank, k)
        # ideal DCG: the best k candidates are positive (IDCG = sum over min(k, #pos) hits)
        n_pos = np.bincount(u_sorted[y_sorted], minlength=n_users) if sel.any() else np.zeros(n_users)
        m = np.minimum(n_pos, k)
        idcg = np.array([sum(1.0 / np.log2(j + 2.0) for j in range(int(mm))) for mm in m])
        with np.errstate(divide="ignore", invalid="ignore"):
            ndcg = np.where(idcg > 0, dcg / idcg, 0.0)
        out[f"NDCG@{k}"] = round(float(np.mean(ndcg)), 5)
    # MRR over users that have a positive
    mrr = np.where(hit, 1.0 / (pos_rank + 1.0), 0.0)
    out["MRR"] = round(float(np.mean(mrr)), 5)
    out["MRR_hit_users_only"] = round(float(mrr[hit].mean()), 5) if hit.any() else 0.0
    return out


def classification(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Global AUC / LogLoss / PR-AUC (natural for the ~1:6.3 label imbalance)."""
    y = y_true.astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-np.clip(scores, -60, 60)))
    eps = 1e-9
    logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    # AUC via rank statistic (Mann-Whitney U), no sklearn dependency at this layer
    r = _rankdata(scores)
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    auc = float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) \
        if n_pos > 0 and n_neg > 0 else float("nan")

    o = np.argsort(-scores)
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / max(n_pos, 1e-9)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return {"AUC": round(auc, 5), "LogLoss": round(logloss, 5),
            "PR_AUC": round(ap, 5), "pos_rate": round(float(y.mean()), 4),
            "n": int(len(y))}


def gauc(y_true: np.ndarray, scores: np.ndarray, user_idx: np.ndarray) -> float:
    """Group AUC: average of per-user AUCs, which is the honest global number under
    extreme item-popularity skew."""
    vals, weights = [], []
    order = np.argsort(user_idx, kind="stable")
    u = user_idx[order]
    bounds = np.searchsorted(u, np.unique(u), side="left").tolist() + [len(u)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sl = order[a:b]
        y = y_true[sl]
        if 0 < y.sum() < len(y):
            vals.append(classification(y, scores[sl])["AUC"])
            weights.append(len(y))
    return round(float(np.average(vals, weights=weights)), 5) if vals else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    s = a[order]
    obs = np.r_[True, s[1:] != s[:-1], True]
    starts = np.flatnonzero(obs)
    for a_, b_ in zip(starts[:-1], starts[1:]):
        if b_ - a_ > 1:
            r[order[a_:b_]] = r[order[a_:b_]].mean()
    return r


def factorize(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic id -> contiguous index (sorted, so it is reproducible)."""
    uniq, inv = np.unique(ids, return_inverse=True)
    return inv.astype(np.int64), uniq


def compare(results: dict[str, dict], metric: str = "AUC") -> dict:
    """Guard against the sampled-metric trap when comparing models."""
    sizes = {k: v.get("candidate_set_size") or v.get("n") for k, v in results.items()}
    distinct = set(sizes.values())
    ranked = sorted(((k, v.get(metric)) for k, v in results.items()),
                    key=lambda kv: (kv[1] is None, -(kv[1] or 0)))
    return {
        "metric": metric,
        "comparable": len(distinct) <= 1,
        "candidate_set_sizes": sizes,
        "ranking": [(k, v) for k, v in ranked],
        "warning": None if len(distinct) <= 1 else
        "Models were evaluated on different candidate-set sizes; per Rendle "
        "(arXiv:1912.02263) sampled metrics do not preserve model order. Not comparable.",
    }


def recall_at_k_full_catalogue(scores_matrix: np.ndarray, target_positions: np.ndarray,
                               ks: tuple[int, ...] = (10, 50, 100, 200)) -> dict:
    """Full-catalogue Recall@K from a *materialised* ``[n_users, n_items]`` score matrix.

    Kept for small catalogues and unit tests. For the real job this shape is wrong:
    ``[4000 users x 1e6 items] x float32`` is **16 GB**, which OOM-killed the evaluation run
    before it was split into blocks. Use :func:`recall_from_topk`.
    """
    out = {}
    for k in ks:
        topk = np.argpartition(-scores_matrix, kth=min(k, scores_matrix.shape[1] - 1),
                               axis=1)[:, :k]
        out[f"Recall@{k}"] = round(float((topk == target_positions[:, None]).any(axis=1).mean()), 5)
        hit = (topk == target_positions[:, None])
        ranks = np.where(hit.any(axis=1), hit.argmax(axis=1), np.inf)
        out[f"NDCG@{k}"] = round(float(np.mean(1.0 / np.log2(ranks + 2.0))), 5)
    return out


def recall_from_topk(top_indices: np.ndarray, target_positions: np.ndarray,
                     ks: tuple[int, ...] = (10, 50, 100, 200, 500)) -> dict:
    """Recall@K / NDCG@K from already-reduced top-N catalogue indices.

    ``top_indices`` is ``[n_users, max_k]``. Reducing *before* accumulating is what keeps the
    peak at one user-block of scores instead of the whole 4000 x 1e6 cross-product.
    """
    max_k = top_indices.shape[1]
    match = top_indices == np.asarray(target_positions)[:, None]
    out = {"n_users": int(len(target_positions)), "top_k_computed": int(max_k)}
    for k in ks:
        if k > max_k:
            continue
        m = match[:, :k]
        hit = m.any(axis=1)
        out[f"Recall@{k}"] = round(float(hit.mean()), 5)
        # 0-based first-match position; +2 => 1-based rank +1 in the log denominator
        first = np.where(hit, m.argmax(axis=1), np.inf)
        out[f"NDCG@{k}"] = round(float(np.mean(1.0 / np.log2(first + 2.0))), 5)
    return out
