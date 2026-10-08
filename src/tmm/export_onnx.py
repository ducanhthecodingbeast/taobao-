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


def serving_checkpoint() -> Path | None:
    """The one checkpoint every serving artifact (ONNX graph, item vectors) is built from.

    The residual sampled-softmax tower is preferred: the plain pointwise one has a collapsed
    item space (mean pairwise cosine 0.966), so its index is useless.
    """
    return next((p for p in (PATHS.models / "two_tower_res.pt", PATHS.models / "two_tower.pt")
                 if p.exists()), None)


class UserTowerExport:
    """Thin nn.Module wrapper so the ONNX signature is explicit and stable.

    Input ``hist_vec`` is the **un-normalised** masked mean of the history embeddings, exactly
    what :meth:`tmm.models.TwoTower.user_vec` pools before its MLP. ``LayerNorm(Wx + b)`` is
    not scale-invariant, so feeding an L2-normalised mean silently changes the output.
    """

    @staticmethod
    def build(ckpt: dict):
        import torch
        import torch.nn as nn

        class Wrap(nn.Module):
            def __init__(self, dim, demo_dims, residual):
                super().__init__()
                self.user_tower = nn.Sequential(
                    nn.Linear(128 + 16 * len(demo_dims), 256), nn.LayerNorm(256),
                    nn.SiLU(), nn.Dropout(0.0), nn.Linear(256, dim))
                self.demo_emb = nn.ModuleList([nn.Embedding(c, 16) for c in demo_dims])
                self.user_res = nn.Linear(128, dim, bias=False) if residual else None
                self.pool_dim = 128

            def forward(self, hist_vec, demo):
                d = torch.cat([e(demo[:, i]) for i, e in enumerate(self.demo_emb)], dim=-1)
                y = self.user_tower(torch.cat([hist_vec, d], dim=-1))
                if self.user_res is not None:
                    y = y + self.user_res(hist_vec)
                return torch.nn.functional.normalize(y, dim=-1)

        return Wrap(ckpt["dim"], [ckpt["cards"][c] for c in DEMO_COLS],
                    bool(ckpt.get("residual", False)))

    @staticmethod
    def load(ckpt: dict):
        """Wrapper with the trained weights; raises if any user-tower weight is unmapped."""
        model = UserTowerExport.build(ckpt).eval()
        # TwoTower stores the MLP as ``user_tower.net.<i>`` and demographics as ``demo.<i>``;
        # without this remap the exported graph silently keeps its random init.
        sd = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("user_tower.net."):
                sd["user_tower." + k[len("user_tower.net."):]] = v
            elif k.startswith("demo."):
                sd["demo_emb." + k[len("demo."):]] = v
            elif k.startswith("user_res."):
                sd[k] = v
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mapping incomplete: missing={list(missing)} "
                               f"unexpected={list(unexpected)}")
        return model


def export(model, path: Path) -> None:
    """Write ``model`` (a :class:`UserTowerExport` wrapper) as a dynamic-batch ONNX graph."""
    import torch

    hist = torch.randn(2, model.pool_dim)
    demo = torch.zeros(2, len(DEMO_COLS), dtype=torch.long)
    kw = dict(input_names=["hist_vec", "demo"], output_names=["user_vec"],
              dynamic_axes={"hist_vec": {0: "batch"}, "demo": {0: "batch"},
                            "user_vec": {0: "batch"}},
              opset_version=17, do_constant_folding=True)
    # torch >= 2.6 defaults to the dynamo exporter, which needs the extra `onnxscript`
    # package; the TorchScript exporter gives the same graph with one fewer dependency.
    try:
        torch.onnx.export(model, (hist, demo), str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, (hist, demo), str(path), **kw)


def check_equivalence(ck: dict, onnx_path: Path, n: int = 64, seq: int = 200) -> dict:
    """End-to-end check: masked-mean pooling + ONNX graph == ``TwoTower.user_vec``.

    Uses a random unit-norm stand-in embedding table (the table is not a checkpoint weight),
    with padding mixed into the histories so the masking is exercised too.
    """
    import onnxruntime as ort
    import torch

    from .models import TwoTower

    g = torch.Generator().manual_seed(0)
    n_rows = 5000
    table = torch.nn.functional.normalize(torch.randn(n_rows + 1, 128, generator=g), dim=-1)
    table[n_rows] = 0.0                                          # pad row
    tower = TwoTower(table, cat_card=ck["cards"]["206"],
                     demo_cards=[ck["cards"][c] for c in DEMO_COLS], dim=ck["dim"],
                     tower_seq_len=seq, residual=bool(ck.get("residual", False))).eval()
    tower.load_state_dict(ck["state_dict"])
    hist = torch.randint(0, n_rows, (n, seq), generator=g)
    hist[:, : seq // 4] = n_rows                                 # left padding
    demo = torch.stack([torch.randint(0, ck["cards"][c], (n,), generator=g)
                        for c in DEMO_COLS], 1)
    with torch.no_grad():
        ref = tower.user_vec(hist, demo).numpy()
    mask = (hist != n_rows).numpy()
    pooled = (table[hist].numpy() * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["user_vec"], {"hist_vec": pooled.astype(np.float32),
                                  "demo": demo.numpy()})[0]
    return {"max_abs_diff": float(np.abs(ref - got).max()),
            "cosine_min": float(np.min(np.sum(ref * got, axis=1))),
            "check": "masked mean + ONNX fp32 vs TwoTower.user_vec"}


def run(batch: int = 256, seq: int = 200, repeats: int = 50) -> dict:
    import torch

    out_dir = PATHS.models / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path = serving_checkpoint()
    if ck_path is None:
        return {"error": "no two-tower checkpoint found; run train-retrieval first"}
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = UserTowerExport.load(ck)

    # realistic inputs: pooled means of unit-norm vectors are short, not N(0, 1)
    hist = torch.nn.functional.normalize(torch.randn(batch, model.pool_dim), dim=-1) * 0.5
    demo = torch.zeros(batch, len(DEMO_COLS), dtype=torch.long)

    fp32_path = out_dir / "user_tower_fp32.onnx"
    export(model, fp32_path)

    res = {
        "graph_inputs": ["hist_vec[B,128]", "demo[B,5] int64"],
        "graph_outputs": ["user_vec[B,128] L2-normalised"],
        "why_no_embedding_table": "35.46M x 128 fp16 = 8.45 GiB initializer; ORT dynamic "
                                  "quantisation is known to blow up on multi-GB tables "
                                  "(onnxruntime#21979). Gather happens outside the graph.",
        "checkpoint": ck_path.name, "residual": bool(ck.get("residual", False)),
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
            # one request = one user: this is the latency the service actually pays
            one = {"hist_vec": hist.numpy()[:1], "demo": demo.numpy()[:1]}
            lat = []
            for _ in range(1000):
                t1 = time.perf_counter()
                sess.run(None, one)
                lat.append((time.perf_counter() - t1) * 1e6)
            res["benchmark"][name] = {
                "ms_per_batch": round(dt * 1000, 4),
                "us_per_user_amortized": round(dt * 1e6 / batch, 2),
                "users_per_second": int(batch / dt),
                "single_request_us": {"p50": round(float(np.percentile(lat, 50)), 2),
                                      "p95": round(float(np.percentile(lat, 95)), 2)},
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
                                          "us_per_user_amortized": round(dt * 1e6 / batch, 2)}
    except Exception as exc:
        res["benchmark"]["error"] = f"{type(exc).__name__}: {exc}"

    # ---- numerical equivalence: ONNX vs the real TwoTower.user_vec, not vs the wrapper -----
    try:
        res["equivalence"] = check_equivalence(ck, fp32_path)
    except Exception as exc:
        res["equivalence"] = {"error": f"{type(exc).__name__}: {exc}"}

    (PATHS.bench / "onnx.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, default=str))
