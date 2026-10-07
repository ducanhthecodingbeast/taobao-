"""T6a/T6b — ONNX export + INT8 quantisation of the Direction-1 user tower.

Why the exported graph does **not** contain the embedding table
--------------------------------------------------------------
The frozen SCL table is 35.46 M x 128 x 2 B = 8.45 GiB. Emitting it as an ONNX initializer
would produce a multi-gigabyte ``.onnx`` file, and ONNX Runtime's INT8 quantiser is known to
balloon memory on multi-gigabyte tables (microsoft/onnxruntime#21979). So the demo splits the
tower at the natural seam:

* **history embedding gather** happens outside the graph (Redis/array lookup, already int8)
* the graph takes pooled history vectors + demographics and emits the user vector

That keeps the exported model at a few MB, makes INT8 quantisation meaningful (it now
quantises real matmuls rather than a lookup), and is exactly how such a tower is deployed in
practice: embeddings in a KV store, MLP in the model server.

Post-export we benchmark FP32 vs dynamic-INT8 latency and report the measured winner rather
than assuming quantisation always wins (for small MLPs it frequently does not).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import PATHS

DEMO_COLS = ("130_1", "130_2", "130_3", "130_4", "130_5")


class UserTowerExport:
    """Thin nn.Module wrapper so the ONNX signature is explicit and stable."""

    @staticmethod
    def build(ckpt: dict):
        import torch
        import torch.nn as nn

        class Wrap(nn.Module):
            def __init__(self, dim, demo_dims):
                super().__init__()
                self.user_tower = nn.Sequential(
                    nn.Linear(128 + 16 * len(demo_dims), 256), nn.LayerNorm(256),
                    nn.SiLU(), nn.Dropout(0.0), nn.Linear(256, dim))
                self.demo_emb = nn.ModuleList([nn.Embedding(c, 16) for c in demo_dims])
                self.pool_dim = 128

            def forward(self, hist_vec, demo):
                d = torch.cat([e(demo[:, i]) for i, e in enumerate(self.demo_emb)], dim=-1)
                x = torch.cat([hist_vec, d], dim=-1)
                y = self.user_tower(x)
                return torch.nn.functional.normalize(y, dim=-1)

        return Wrap(ckpt["dim"], [ckpt["cards"][c] for c in DEMO_COLS])


def run(batch: int = 256, seq: int = 200, repeats: int = 50) -> dict:
    import torch

    out_dir = PATHS.models / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path = PATHS.models / "two_tower.pt"
    if not ck_path.exists():
        return {"error": "two_tower.pt not found; run train-retrieval first"}
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    model = UserTowerExport.build(ck).eval()
    # TwoTower stores demographics as ``demo.N.weight``; the export wrapper names the same
    # module ``demo_emb`` so the checkpoint has to be key-remapped or the demographic
    # embeddings silently stay at their random init.
    # Real keys are ``user_tower.net.<i>.weight`` because ``MLP`` wraps its layers in
    # ``nn.Sequential``; the export wrapper uses a bare Sequential. Without this remap the
    # tower silently keeps its random init and the exported graph is a *latency* benchmark
    # of the right shape but not the trained model.
    sd = {}
    for k, v in ck["state_dict"].items():
        if k.startswith("user_tower.net."):
            sd["user_tower." + k[len("user_tower.net."):]] = v
        elif k.startswith("demo."):
            sd["demo_emb." + k[len("demo."):]] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mapping incomplete: missing={list(missing)} "
                           f"unexpected={list(unexpected)}")

    hist = torch.randn(batch, model.pool_dim)
    demo = torch.zeros(batch, len(DEMO_COLS), dtype=torch.long)

    fp32_path = out_dir / "user_tower_fp32.onnx"
    # torch >= 2.6 defaults to the dynamo-based exporter, which requires the extra
    # `onnxscript` package. The legacy TorchScript exporter produces a graph that
    # onnxruntime's INT8 quantiser handles identically here, with one fewer dependency.
    try:
        torch.onnx.export(model, (hist, demo), str(fp32_path),
                          input_names=["hist_vec", "demo"], output_names=["user_vec"],
                          dynamic_axes={"hist_vec": {0: "batch"}, "demo": {0: "batch"},
                                        "user_vec": {0: "batch"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
    except TypeError:
        torch.onnx.export(model, (hist, demo), str(fp32_path),
                          input_names=["hist_vec", "demo"], output_names=["user_vec"],
                          dynamic_axes={"hist_vec": {0: "batch"}, "demo": {0: "batch"},
                                        "user_vec": {0: "batch"}},
                          opset_version=17, do_constant_folding=True)

    res = {
        "graph_inputs": ["hist_vec[B,128]", "demo[B,5] int64"],
        "graph_outputs": ["user_vec[B,128] L2-normalised"],
        "why_no_embedding_table": "35.46M x 128 fp16 = 8.45 GiB initializer; ORT dynamic "
                                  "quantisation is known to blow up on multi-GB tables "
                                  "(onnxruntime#21979). Gather happens outside the graph.",
        "weights_loaded": len(sd), "weights_missing": len(missing),
        "fp32": {"path": str(fp32_path), "bytes": fp32_path.stat().st_size},
    }

    # ---- INT8 dynamic quantisation ---------------------------------------------------
    int8_path = out_dir / "user_tower_int8.onnx"
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        res["int8"] = {"path": str(int8_path), "bytes": int8_path.stat().st_size,
                       "compression": round(fp32_path.stat().st_size / int8_path.stat().st_size, 2)}
    except Exception as exc:
        res["int8"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- benchmark -------------------------------------------------------------------
    res["benchmark"] = {"batch": batch, "repeats": repeats}
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for name, path in (("fp32", fp32_path),
                           ("int8", int8_path) if int8_path.exists() else ("skip", None)):
            if path is None:
                continue
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            feed = {"hist_vec": hist.numpy(), "demo": demo.numpy()}
            for _ in range(3):
                sess.run(None, feed)
            t0 = time.perf_counter()
            for _ in range(repeats):
                sess.run(None, feed)
            dt = (time.perf_counter() - t0) / repeats
            res["benchmark"][name] = {
                "ms_per_batch": round(dt * 1000, 4),
                "us_per_user": round(dt * 1e6 / batch, 2),
                "users_per_second": int(batch / dt),
            }

        # torch reference
        with torch.no_grad():
            for _ in range(3):
                model(hist, demo)
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(hist, demo)
            dt = (time.perf_counter() - t0) / repeats
        res["benchmark"]["torch_fp32"] = {"ms_per_batch": round(dt * 1000, 4),
                                          "us_per_user": round(dt * 1e6 / batch, 2)}
    except Exception as exc:
        res["benchmark"]["error"] = f"{type(exc).__name__}: {exc}"

    # ---- numerical equivalence: ONNX must reproduce the trained torch tower -------------
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(fp32_path), so, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(hist, demo).numpy()
        got = sess.run(["user_vec"], {"hist_vec": hist.numpy(), "demo": demo.numpy()})[0]
        res["equivalence"] = {
            "max_abs_diff": float(np.abs(ref - got).max()),
            "cosine_mean": float(np.mean(np.sum(ref * got, axis=1) /
                                         (np.linalg.norm(ref, axis=1) *
                                          np.linalg.norm(got, axis=1) + 1e-12))),
            "check": "ONNX fp32 vs trained torch tower",
        }
    except Exception as exc:
        res["equivalence"] = {"error": f"{type(exc).__name__}: {exc}"}

    (PATHS.bench / "onnx.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, default=str))
