"""Unified CLI: ``python -m tmm.cli <stage>``.

Stages map 1:1 onto the tasks in ``TASKS.md`` so a reviewer can rerun any single one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import ARTIFACTS, BUDGET, DATA_ROOT, PATHS, PROJECT_ROOT, SEED, device


def _banner(name: str) -> float:
    print(f"\n{'=' * 78}\n[stage] {name}\n{'=' * 78}", flush=True)
    return time.time()


def _done(name: str, t0: float) -> None:
    print(f"[stage] {name} finished in {time.time() - t0:.1f}s", flush=True)


# --------------------------------------------------------------------------------------
def cmd_doctor(_args) -> int:
    t0 = _banner("doctor")
    import numpy as np
    import pyarrow
    import torch

    info = {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "data_root_exists": DATA_ROOT.exists(),
        "python": sys.version.split()[0],
        "seed": SEED,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "pyarrow": pyarrow.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        info["gpu"] = {"name": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
                       "vram_total_gib": round(total / 2**30, 1),
                       "vram_free_gib": round(free / 2**30, 1)}
    try:
        import duckdb
        info["duckdb"] = duckdb.__version__
    except Exception:
        pass
    try:
        import faiss
        info["faiss"] = faiss.__version__
    except Exception:
        pass
    for d in ("stats", "figures", "models", "bench", "data", "reports"):
        getattr(PATHS, d).mkdir(parents=True, exist_ok=True)

    files = sorted(PATHS.data_root.glob("**/*.parquet"))
    info["parquet_files"] = len(files)
    info["dataset_gb"] = round(sum(f.stat().st_size for f in files) / 2**30, 1)
    print(json.dumps(info, indent=2))
    (PATHS.stats / "env.json").write_text(json.dumps(info, indent=2))
    _done("doctor", t0)
    return 0


def cmd_profile(args) -> int:
    t0 = _banner("profile")
    from . import profile as P

    res = P.run_all(seq_row_groups=args.seq_row_groups)
    for k, v in res.items():
        print(f"  {k:16s} {v.get('_seconds', '?')}s")
    _done("profile", t0)
    return 0


def cmd_join_duckdb(args) -> int:
    t0 = _banner("join-duckdb")
    from . import join_duckdb as J

    res = J.run(rows=args.rows)
    print(json.dumps(res, indent=2, default=str))
    _done("join-duckdb", t0)
    return 0


def cmd_join_spark(args) -> int:
    t0 = _banner("join-spark")
    from . import join_spark as S

    res = S.run(rows=args.rows, buckets=args.buckets)
    print(json.dumps(res, indent=2, default=str))
    _done("join-spark", t0)
    return 0


def cmd_prepare(args) -> int:
    t0 = _banner("prepare")
    from . import prepare as P

    res = P.run(rows=args.rows, test_rows=args.test_rows, seq_len=args.seq_len)
    print(json.dumps({k: v for k, v in res.items() if k != "notes"}, indent=2, default=str))
    _done("prepare", t0)
    return 0


def cmd_train_retrieval(args) -> int:
    t0 = _banner("train-retrieval (two-tower)")
    from . import train as T

    res = T.train_two_tower(epochs=args.epochs, batch_size=args.batch_size, dim=args.dim,
                            device_str=args.device, objective=args.objective, tag=args.tag,
                            residual=args.residual)
    print(json.dumps(res, indent=2, default=str))
    _done("train-retrieval", t0)
    return 0


def cmd_train_ranker(args) -> int:
    t0 = _banner("train-ranker (DIN vs MUSE)")
    from . import train as T

    res = T.train_rankers(epochs=args.epochs, batch_size=args.batch_size, k=args.k,
                          device_str=args.device)
    print(json.dumps(res, indent=2, default=str))
    _done("train-ranker", t0)
    return 0


def cmd_evaluate(args) -> int:
    t0 = _banner("evaluate")
    from . import evaluate as E

    res = E.run(n_candidates=args.candidates, protocol=args.protocol, device_str=args.device)
    print(json.dumps(res, indent=2, default=str))
    _done("evaluate", t0)
    return 0


def cmd_retrieval_ablation(args) -> int:
    t0 = _banner("retrieval-ablation")
    from . import evaluate as E

    res = E.retrieval_ablation(n_users=args.users)
    print(json.dumps(res, indent=2, default=str))
    _done("retrieval-ablation", t0)
    return 0


def cmd_index(args) -> int:
    t0 = _banner("index")
    from . import index as I

    res = I.run(n_items=args.items, exact=args.exact)
    print(json.dumps(res, indent=2, default=str))
    _done("index", t0)
    return 0


def cmd_serve_check(args) -> int:
    t0 = _banner("serve-check")
    from . import servecheck

    res = servecheck.run(concurrency=args.concurrency, requests=args.requests,
                         workers=args.workers)
    print(json.dumps(res, indent=2, default=str))
    _done("serve-check", t0)
    return 0


def cmd_build_demo_artifact(args) -> int:
    """T6d: precompute the self-sufficient demo artifact (keeps the service memory-flat)."""
    t0 = _banner("build-demo-artifact")
    from . import demo_artifact

    res = demo_artifact.build(n_items=args.items)
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=2, default=str))
    _done("build-demo-artifact", t0)
    return 0


def cmd_measure_memory(args) -> int:
    """T6d: measure real startup RSS. Guards the 2 vCPU / 4 GB VPS sizing claim."""
    t0 = _banner("measure-memory")
    import resource
    import sys as _sys
    # NOTE: `subprocess` is already imported at module scope (line 11); re-importing it here
    # shadowed the module-level name and tripped ruff F811.

    mode = args.mode
    code = (
        "import resource,sys,time\n"
        "from tmm.serve.app import Recommender\n"
        f"r = Recommender(n_items={args.items}, use_demo_artifact={mode == 'artifact'})\n"
        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)\n"
    )
    env = {"PYTHONPATH": "src", "MPLCONFIGDIR": "/tmp/mpl", "PATH": __import__("os").environ["PATH"]}
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True, env=env)
    peak = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "ERROR"
    print(json.dumps({"mode": mode, "peak_rss_mb": peak, "stderr": out.stderr[-500:]},
                     indent=2))
    _done("measure-memory", t0)
    return 0


def cmd_figures(_args) -> int:
    t0 = _banner("figures")
    from . import figures

    made = figures.run_all()
    print("wrote:", *made, sep="\n  ")
    _done("figures", t0)
    return 0


def cmd_export_onnx(_args) -> int:
    t0 = _banner("export-onnx")
    from . import export_onnx

    res = export_onnx.run()
    print(json.dumps(res, indent=2, default=str))
    _done("export-onnx", t0)
    return 0


def cmd_all(args) -> int:
    """Run the full CPU-only pipeline end to end."""
    seq = [
        ("doctor", cmd_doctor, {}),
        ("profile", cmd_profile, {"seq_row_groups": 1}),
        ("join-duckdb", cmd_join_duckdb, {"rows": args.rows}),
        ("prepare", cmd_prepare, {"rows": args.rows, "test_rows": args.test_rows,
                                  "seq_len": args.seq_len}),
        ("evaluate", cmd_evaluate, {"candidates": 500, "protocol": "both"}),
        ("index", cmd_index, {"items": 10000, "exact": True}),
        ("serve-check", cmd_serve_check, {"concurrency": 32, "requests": 2000}),
        ("figures", cmd_figures, {}),
    ]
    for name, fn, kw in seq:
        ns = argparse.Namespace(**kw)
        try:
            fn(ns)
        except Exception as exc:
            print(f"[stage] {name} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tmm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    s = sub.add_parser("profile"); s.add_argument("--seq-row-groups", type=int, default=1); s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("join-duckdb"); s.add_argument("--rows", type=int, default=BUDGET.train_rows); s.set_defaults(fn=cmd_join_duckdb)
    s = sub.add_parser("join-spark")
    s.add_argument("--rows", type=int, default=BUDGET.train_rows)
    s.add_argument("--buckets", type=int, default=256)
    s.set_defaults(fn=cmd_join_spark)

    s = sub.add_parser("prepare")
    s.add_argument("--rows", type=int, default=BUDGET.train_rows)
    s.add_argument("--test-rows", type=int, default=BUDGET.test_rows)
    s.add_argument("--seq-len", type=int, default=BUDGET.seq_len)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("train-retrieval")
    s.add_argument("--epochs", type=int, default=BUDGET.epochs)
    s.add_argument("--batch-size", type=int, default=BUDGET.batch_size)
    s.add_argument("--dim", type=int, default=128)
    s.add_argument("--device", default=None)
    s.add_argument("--objective", choices=["bce", "sampled_softmax"], default="bce")
    s.add_argument("--tag", default="")
    s.add_argument("--residual", action="store_true")
    s.set_defaults(fn=cmd_train_retrieval)

    s = sub.add_parser("train-ranker")
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--k", type=int, default=BUDGET.muse_k)
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_train_ranker)

    s = sub.add_parser("evaluate")
    s.add_argument("--candidates", type=int, default=BUDGET.n_negatives)
    s.add_argument("--protocol", choices=["sampled", "full", "both"], default="both")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("build-demo-artifact")
    s.add_argument("--items", type=int, default=BUDGET.demo_items)
    s.set_defaults(fn=cmd_build_demo_artifact)

    s = sub.add_parser("measure-memory")
    s.add_argument("--mode", choices=["legacy", "artifact"], default="artifact")
    s.add_argument("--items", type=int, default=BUDGET.demo_items)
    s.set_defaults(fn=cmd_measure_memory)

    s = sub.add_parser("retrieval-ablation")
    s.add_argument("--users", type=int, default=4000)
    s.set_defaults(fn=cmd_retrieval_ablation)

    s = sub.add_parser("index")
    s.add_argument("--items", type=int, default=BUDGET.demo_items)
    s.add_argument("--exact", action="store_true")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("serve-check")
    s.add_argument("--concurrency", type=int, default=32)
    s.add_argument("--requests", type=int, default=2000)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=cmd_serve_check)

    sub.add_parser("figures").set_defaults(fn=cmd_figures)
    sub.add_parser("export-onnx").set_defaults(fn=cmd_export_onnx)

    s = sub.add_parser("all")
    s.add_argument("--rows", type=int, default=BUDGET.train_rows)
    s.add_argument("--test-rows", type=int, default=BUDGET.test_rows)
    s.add_argument("--seq-len", type=int, default=BUDGET.seq_len)
    s.set_defaults(fn=cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not DATA_ROOT.exists():
        print(f"data root not found: {DATA_ROOT}", file=sys.stderr)
        return 2
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
