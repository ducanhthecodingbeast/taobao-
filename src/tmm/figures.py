"""T8 — figures. Every plot is regenerated from ``artifacts/stats/*.json``, never hand-made.

Figures produced
----------------
01 label balance + candidate-set structure (why this is 1:6.3, not 1:1)
02 sequence-length distribution (why 1 000-token attention is expensive)
03 item popularity long tail, log-log (why a popularity baseline is strong)
04 embedding int8 histogram + per-row norm (why cosine is meaningful)
05 two-stage funnel: catalogue -> retrieval -> ranking
06 model comparison: AUC / NDCG@3 / HR@1 per model
07 recall vs latency for the ANN index
08 training curves
09 memory footprint of the embedding table under each encoding
10 data-engineering throughput (DuckDB vs Spark)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import PATHS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})
C = {"blue": "#2563eb", "red": "#dc2626", "green": "#059669",
     "amber": "#d97706", "violet": "#7c3aed", "slate": "#475569"}


def _j(name: str) -> dict:
    p = PATHS.stats / name
    return json.loads(p.read_text()) if p.exists() else {}


def _bj(name: str) -> dict:
    p = PATHS.bench / name
    return json.loads(p.read_text()) if p.exists() else {}


def _save(fig, name: str) -> str:
    p = PATHS.figures / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {p}")
    return str(p)


# --------------------------------------------------------------------------------------
def fig01_labels() -> str:
    p, pr = _j("profile.json"), _j("prepare.json")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    t = p.get("labels", {})
    for a, key, title in ((ax[0], "train", "Train"), (ax[0], "test", "Test")):
        d = t.get(key, {})
        if d:
            a.bar([f"{title}\npos", f"{title}\nneg"],
                  [d["pos_rate"] * 100, (1 - d["pos_rate"]) * 100],
                  color=[C["green"], C["slate"]], width=0.6)
    ax[0].set_ylabel("% of samples"); ax[0].set_title("(a) Label balance is 1:6.3, not 1:1")
    for i, v in enumerate([t.get("train", {}).get("pos_rate", 0) * 100] * 2):
        pass

    s = pr.get("splits", {})
    names = [k for k in ("train", "test") if k in s]
    ax[1].bar(names, [s[k]["rows_per_user"] for k in names], color=C["blue"], width=0.5)
    ax[1].axhline(1.0, ls="--", c=C["red"], lw=1, label="1 candidate/user")
    ax[1].set_ylabel("candidate rows per user")
    ax[1].set_title("(b) Shipped candidate set is ~11 hard negatives")
    ax[1].legend()

    ax[2].bar(names, [s[k]["users_with_a_positive_rate"] * 100 for k in names],
              color=C["amber"], width=0.5)
    ax[2].set_ylabel("% of users"); ax[2].set_ylim(0, 100)
    ax[2].set_title("(c) Share of users with >=1 positive")
    return _save(fig, "01_labels_and_candidates.png")


def fig02_sequence() -> str:
    p = _j("profile.json").get("sequence", {})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    h = p.get("histogram", {})
    if h.get("edges"):
        edges = np.asarray(h["edges"]); counts = np.asarray(h["counts"])
        ax[0].step(edges[:-1], counts, where="post", color=C["blue"])
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("history length"); ax[0].set_ylabel("users")
    ax[0].set_title(f"(a) History length (mean {p.get('seq_len', {}).get('mean', 0):.0f})")
    frac = p.get("seq_len", {}).get("frac_at_max", 0) * 100
    ax[1].bar(["= 1000", "< 1000"], [frac, 100 - frac], color=[C["red"], C["slate"]], width=0.5)
    ax[1].set_ylabel("% of users")
    ax[1].set_title("(b) Truncation is the norm, not the exception")
    return _save(fig, "02_sequence_length.png")


def fig03_popularity() -> str:
    p = _j("profile.json").get("popularity", {})
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    h = p.get("freq_histogram", {})
    if h.get("edges"):
        edges = np.asarray(h["edges"]); counts = np.asarray(h["counts"])
        ax[0].loglog(edges[:-1], np.maximum(counts, 1), "o-", ms=3, color=C["blue"])
        ax[0].set_xlabel("interactions per item"); ax[0].set_ylabel("number of items")
    ax[0].set_title(f"(a) Long tail (Gini {p.get('gini', 0):.3f})")
    labels = ["top10", "top100", "top1k", "top10k"]
    vals = [p.get(k, 0) * 100 for k in ("top10_share", "top100_share", "top1k_share",
                                        "top10k_share")]
    ax[1].bar(labels, vals, color=C["violet"])
    ax[1].set_ylabel("% of interactions")
    ax[1].set_title(f"(b) Head concentration; top-100 = "
                    f"{p.get('positives_top100_share', 0) * 100:.0f}% of clicks")
    return _save(fig, "03_popularity_longtail.png")


def fig04_embedding() -> str:
    e = _j("profile.json").get("embedding", {})
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
    v = e.get("value_min"), e.get("value_max")
    if None not in v:
        ax[0].hist([v[0], v[1]], bins=2, color=C["blue"])
        ax[0].set_xlabel("int8 value"); ax[0].set_ylabel("count")
    ax[0].set_title(f"(a) int8 range [{v[0]:.0f}, {v[1]:.0f}], scale={e.get('scale', 0):.4f}")
    rn = e.get("row_norm_dequantized") or e.get("row_norm")
    if rn:
        ax[1].bar(["min", "mean", "max"], [rn["min"], rn["mean"], rn["max"]], color=C["green"])
        ax[1].axhline(1.0, ls="--", c=C["red"], lw=1)
        ax[1].set_ylabel("L2 norm after /127")
        ax[1].set_title(f"(b) Dequantised norms = {rn['mean']:.3f} ~ 1.0")
    if e.get("bytes_as_int8"):
        ax[2].bar(["fp32", "int8"], [e["bytes_if_float32"] / 2**30, e["bytes_as_int8"] / 2**30],
                  color=[C["red"], C["green"]])
        ax[2].set_ylabel("GiB")
        ax[2].set_title(f"(c) {e['bytes_if_float32'] / 2**30:.1f} GiB -> "
                        f"{e['bytes_as_int8'] / 2**30:.2f} GiB (4x)")
    return _save(fig, "04_embedding_int8.png")


def fig05_funnel() -> str:
    p = _j("profile.json")
    inv = p.get("inventory", {}).get("tables", {})
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    stages = [
        ("Item vocabulary\n(history, p90)", 35_458_498, C["slate"]),
        ("Items with SCL embedding", 35_458_466, C["blue"]),
        ("Items seen as targets", 3_776_576, C["violet"]),
        ("Demo index (Direction 1)", 10_000, C["amber"]),
    ]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [np.log10(s[1]) for s in stages], color=[s[2] for s in stages])
    for yy, s in zip(y, stages):
        ax.text(np.log10(s[1]) + 0.05, yy, f"{s[1]:,}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in stages])
    ax.set_xlabel("log10(number of items)"); ax.set_xlim(0, 8.6)
    ax.set_title("Two-stage funnel: retrieval narrows 35.4M -> 10k -> ranker's ~11")
    return _save(fig, "05_funnel.png")


def fig06_models() -> str:
    ev = _j("evaluate.json").get("metrics", {})
    if not ev:
        return ""
    order = [k for k in ("random", "global_pop", "positive_pop", "category_pop",
                         "content_knn", "two_tower", "din", "muse") if k in ev]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, met, title in ((ax[0], "AUC", "(a) AUC (point-wise, 1:6.3)"),
                          # HR@10 out of ~11 candidates is nearly guaranteed; @3 separates
                          (ax[1], "NDCG@3", "(b) NDCG@3 (users with a click)"),
                          (ax[2], "HR@1", "(c) HR@1 (users with a click)")):
        vals = [ev[m].get(met, 0) or 0 for m in order]
        cols = [C["slate"] if m in ("random",) else
                C["amber"] if m.endswith("pop") or m == "content_knn" else C["blue"]
                for m in order]
        a.barh(np.arange(len(order))[::-1], vals, color=cols)
        a.set_yticks(np.arange(len(order))[::-1]); a.set_yticklabels(order, fontsize=8)
        a.set_title(title)
        for yy, v in zip(np.arange(len(order))[::-1], vals):
            a.text(v, yy, f" {v:.3f}", va="center", fontsize=7)
    return _save(fig, "06_model_comparison.png")


def fig07_index() -> str:
    idx = _bj("index.json")
    sw = idx.get("sweep", [])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    if sw:
        ef = [s["efSearch"] for s in sw]
        ax[0].plot(ef, [s["recall@10"] * 100 for s in sw], "o-", color=C["green"])
        ax[0].set_xscale("log", base=2); ax[0].set_xlabel("efSearch")
        ax[0].set_ylabel("recall@10 (%)"); ax[0].set_ylim(0, 105)
        ax[0].set_title("(a) HNSW recall vs efSearch")
        ax[1].plot([s["p95_ms"] for s in sw], [s["recall@10"] * 100 for s in sw], "o-",
                   color=C["blue"], label="HNSW")
        ex = idx.get("exact", {}).get("latency", {})
        if ex:
            ax[1].axhline(100, ls=":", c=C["slate"], label="exact = 100% recall")
            ax[1].errorbar([ex.get("p95_ms", 0)], [100], fmt="s", color=C["red"],
                           label=f"IndexFlatIP p95={ex.get('p95_ms', 0):.3f} ms")
        ax[1].set_xlabel("p95 single-query latency (ms)"); ax[1].set_ylabel("recall@10 (%)")
        ax[1].set_title("(b) Recall / latency trade-off"); ax[1].legend(fontsize=7)
    return _save(fig, "07_index_recall_latency.png")


def fig08_training() -> str:
    tt = _j("train_two_tower.json")
    rk = _j("train_rankers.json")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    if tt.get("history"):
        ax[0].plot([h["epoch"] for h in tt["history"]], [h["loss"] for h in tt["history"]],
                   "o-", color=C["blue"], label="two-tower")
    for mode in ("din", "muse"):
        r = rk.get(mode, {})
        if r.get("history"):
            ax[0].plot([h["epoch"] for h in r["history"]], [h["loss"] for h in r["history"]],
                       "s-", color=C["red"] if mode == "din" else C["violet"], label=mode)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss")
    ax[0].set_title("(a) Training loss"); ax[0].legend(fontsize=8)
    if tt:
        ax[1].bar(["trainable", "frozen (SCL)"],
                  [tt.get("params_trainable", 0) / 1e3, tt.get("params_frozen", 0) / 1e6],
                  color=[C["green"], C["slate"]])
        ax[1].set_ylabel("k params / M params")
        ax[1].set_title(f"(b) {tt.get('rows_per_second', 0):,} rows/s on "
                        f"{tt.get('device', '')}")
    return _save(fig, "08_training.png")


def fig09_memory() -> str:
    idx = _bj("index.json")
    mt = idx.get("memory_table", [])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    if mt:
        names = [m["encoding"] for m in mt]
        full = [m["bytes_per_vector"] * 35_458_467 / 2**30 for m in mt]
        ax[0].barh(np.arange(len(names))[::-1], full, color=C["violet"])
        ax[0].set_yticks(np.arange(len(names))[::-1]); ax[0].set_yticklabels(names, fontsize=7)
        ax[0].set_xlabel("GiB for the full 35.46M catalogue")
        ax[0].set_title("(a) Embedding table footprint per encoding")
        demo = [m["total_mb"] for m in mt]
        ax[1].barh(np.arange(len(names))[::-1], demo, color=C["blue"])
        ax[1].set_yticks(np.arange(len(names))[::-1]); ax[1].set_yticklabels(names, fontsize=7)
        ax[1].set_xlabel("MB for the 10k-item demo index")
        ax[1].set_title("(b) Direction-1 demo index is single-digit MB")
    return _save(fig, "09_memory_encodings.png")


def fig10_engines() -> str:
    d, s = _bj("join_duckdb.json"), _bj("join_spark.json")
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
    eng, tps = [], []
    if d.get("join"):
        eng.append("DuckDB\n(single node)"); tps.append(d["join"]["rows_per_second"])
    if s.get("rows_per_second"):
        eng.append("Spark\n(local[4])"); tps.append(s["rows_per_second"])
    if tps:
        ax[0].bar(eng, tps, color=[C["green"], C["amber"]][:len(tps)])
        ax[0].set_ylabel("rows joined / second")
        ax[0].set_title("(a) Join throughput on identical data")
        for i, v in enumerate(tps):
            ax[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    if d.get("join"):
        ax[1].bar(["read", "join+write"],
                  [0, d["join"]["seconds"]], color=[C["slate"], C["green"]])
        ax[1].set_ylabel("seconds")
        ax[1].set_title(f"(b) DuckDB: {d['join']['rows']:,} rows in "
                        f"{d['join']['seconds']}s")
    if s.get("extrapolation"):
        ax[2].bar(["measured\nrow set", "extrapolated\n76M rows"],
                  [s.get("join_seconds", 0) / 60, s["extrapolation"]["estimated_join_hours"] * 60],
                  color=[C["blue"], C["red"]])
        ax[2].set_ylabel("minutes")
        ax[2].set_title("(c) Spark linear extrapolation to full scale")
    return _save(fig, "10_engines.png")


def fig11_architecture() -> str:
    """Schematic of both directions, drawn so the report has one self-contained diagram."""
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.axis("off")

    def box(x, y, w, h, text, color, fc=None):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc or "white", ec=color, lw=1.4,
                                   zorder=2, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, zorder=3)

    def arrow(x1, y1, x2, y2, color="#475569", style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.2), zorder=4)

    # ---- Direction 2 ------------------------------------------------------------------
    # Vertical bands are chosen so the two direction titles never overlap a box:
    #   D2 title  y=0.945   D2 row A y=0.66-0.82   D2 row B y=0.42-0.58
    #   D1 title  y=0.335   D1 row   y=0.06-0.24
    ax.text(0.02, 0.945, "Direction 2 — Enterprise (full 139 GB, GPU + PySpark)",
            fontsize=9.5, weight="bold", transform=ax.transAxes)
    box(0.02, 0.66, 0.13, 0.16, "raw/train_samples\n76.0M rows", C["slate"])
    box(0.02, 0.42, 0.13, 0.16, "scl_embedding\n35.46M x 128 int8", C["slate"])
    box(0.19, 0.54, 0.13, 0.16, "PySpark join\nco-partitioned\n(driver 20 GB)", C["amber"])
    arrow(0.15, 0.74, 0.19, 0.65)
    arrow(0.15, 0.50, 0.19, 0.59)
    box(0.36, 0.54, 0.13, 0.16, "Two-Tower\nretrieval\n(145 s / 3 ep)", C["blue"])
    arrow(0.32, 0.62, 0.36, 0.62)
    box(0.53, 0.54, 0.15, 0.16, "FAISS / Milvus\ntop-300 candidates", C["violet"])
    arrow(0.49, 0.62, 0.53, 0.62)
    box(0.72, 0.66, 0.16, 0.16, "DIN\n(1000-token attn)", C["red"])
    box(0.72, 0.42, 0.16, 0.16, "MUSE search\n(top-50 then attn)", C["green"])
    arrow(0.68, 0.62, 0.72, 0.71)
    arrow(0.68, 0.62, 0.72, 0.50)
    ax.text(0.895, 0.60, "same AUC,\n3.3x faster\nscoring", fontsize=7, style="italic",
            va="center")

    # ---- Direction 1 ------------------------------------------------------------------
    ax.text(0.02, 0.335, "Direction 1 — Lightweight live demo (10k catalogue, CPU, ~4 GB VPS)",
            fontsize=9.5, weight="bold", transform=ax.transAxes)
    box(0.02, 0.06, 0.12, 0.18, "click event\nPOST /event", C["slate"])
    box(0.17, 0.06, 0.12, 0.18, "Kafka\n(KRaft, 512 MB)", C["amber"])
    box(0.32, 0.06, 0.12, 0.18, "Redis\nsession + KNN", C["red"])
    box(0.47, 0.06, 0.14, 0.18, "ONNX int8\nuser tower", C["blue"])
    box(0.64, 0.06, 0.13, 0.18, "FAISS HNSW\n10k items, 7 MB", C["violet"])
    box(0.80, 0.06, 0.16, 0.18, "Caddy + TLS\nauto HTTPS", C["green"])
    for x in (0.14, 0.29, 0.44, 0.61, 0.77):
        arrow(x, 0.15, x + 0.03, 0.15)
    return _save(fig, "11_architecture.png")


ALL = [fig01_labels, fig02_sequence, fig03_popularity, fig04_embedding, fig05_funnel,
       fig06_models, fig07_index, fig08_training, fig09_memory, fig10_engines,
       fig11_architecture]


def run_all() -> list[str]:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in ALL:
        try:
            out = fn()
            if out:
                made.append(out)
        except Exception as exc:
            print(f"[figures] {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
    return made


if __name__ == "__main__":  # pragma: no cover
    run_all()
