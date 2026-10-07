"""T6c — load test for the live demo.

Starts the real FastAPI app under uvicorn, fires concurrent traffic with httpx, and reports
p50/p95/p99 for both endpoints. The numbers are measured, never asserted: the point is to
publish the tested ceiling, not to claim one.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import sys
import time

import numpy as np

from .config import PATHS


def _wait_health(base: str, timeout: float = 420.0) -> dict:
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{base}/health", timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"server at {base} did not become healthy in {timeout}s")


def run(concurrency: int = 32, requests: int = 2000, k: int = 10,
        port: int = 8123, n_users: int = 512, workers: int = 1) -> dict:
    import httpx

    base = f"http://127.0.0.1:{port}"
    env = {"PYTHONPATH": "src", "MPLCONFIGDIR": "/tmp/mpl"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tmm.serve.app:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning", "--workers", str(workers)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**__import__("os").environ, **env})
    try:
        health = _wait_health(base)

        # real anonymised item ids: posting fabricated integers would be silently dropped by
        # the vocab encoder and would measure an empty-history code path
        with httpx.Client(timeout=30.0) as c:
            pool = np.asarray(c.get(f"{base}/items/sample", params={"n": 512}).json()["item_ids"],
                              dtype=np.int64)
        rng = np.random.default_rng(0)

        async def worker(client, uid, out: list, n: int):
            for _ in range(n):
                iid = int(pool[rng.integers(0, len(pool))])
                t0 = time.perf_counter()
                r = await client.post(f"{base}/event", json={"user_id": uid, "item_id": iid})
                out.append((time.perf_counter() - t0) * 1000)
                if r.status_code != 200:
                    raise RuntimeError(f"event {r.status_code}")

        async def bench(endpoint: str, total: int) -> dict:
            per = max(1, total // concurrency)
            lat: list[float] = []
            async with httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(
                    max_connections=concurrency * 2)) as client:
                if endpoint == "health":
                    async def h_worker(n):
                        for _ in range(n):
                            t0 = time.perf_counter()
                            r = await client.get(f"{base}/health")
                            lat.append((time.perf_counter() - t0) * 1000)
                            if r.status_code != 200:
                                raise RuntimeError(f"health {r.status_code}")
                    tasks = [asyncio.create_task(h_worker(per)) for _ in range(concurrency)]
                elif endpoint == "event":
                    tasks = [asyncio.create_task(worker(client, i % n_users, lat, per))
                             for i in range(concurrency)]
                else:
                    async def rec_worker(uid, n):
                        for _ in range(n):
                            t0 = time.perf_counter()
                            r = await client.get(f"{base}/recommend",
                                                 params={"user_id": uid, "k": k})
                            lat.append((time.perf_counter() - t0) * 1000)
                            if r.status_code != 200:
                                raise RuntimeError(f"recommend {r.status_code}: {r.text[:200]}")
                    tasks = [asyncio.create_task(rec_worker(i % n_users, per))
                             for i in range(concurrency)]
                await asyncio.gather(*tasks)
            a = np.asarray(lat)
            return {
                "requests": len(a), "concurrency": concurrency,
                "p50_ms": round(float(np.percentile(a, 50)), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3),
                "p99_ms": round(float(np.percentile(a, 99)), 3),
                "max_ms": round(float(a.max()), 3),
                "mean_ms": round(float(a.mean()), 3),
                "rps": round(len(a) / (a.sum() / 1000 / concurrency), 1),
            }

        # warm the paths (first call includes lazy imports / JIT)
        with httpx.Client(timeout=30.0) as c:
            c.post(f"{base}/event", json={"user_id": 1, "item_id": int(pool[0])})
            c.get(f"{base}/recommend", params={"user_id": 1, "k": k})

        res = {
            "health": health,
            "workers": workers,
            # control: a trivial endpoint measuring the raw HTTP/JSON/framework ceiling. Without
            # it, any p95 measured here would be wrongly attributed to ONNX+FAISS inference.
            "control_health": asyncio.run(bench("health", requests)),
            "event": asyncio.run(bench("event", requests)),
            "recommend": asyncio.run(bench("recommend", requests)),
            "k": k,
        }
        rec, ctl = res["recommend"], res["control_health"]
        res["verdict"] = {
            "p95_ms": rec["p95_ms"],
            "control_p95_ms": ctl["p95_ms"],
            "inference_overhead_p50_ms": round(rec["p50_ms"] - ctl["p50_ms"], 3),
            "p95_under_20ms": rec["p95_ms"] < 20,
            "note": "the /health control isolates framework overhead from model+index work; "
                    "run with --workers >1 to see whether the ceiling is the GIL",
        }
        with httpx.Client(timeout=30.0) as c:
            res["metrics"] = c.get(f"{base}/metrics").json()
        PATHS.bench.mkdir(parents=True, exist_ok=True)
        (PATHS.bench / "serve.json").write_text(json.dumps(res, indent=2, default=str))
        return res
    except Exception:
        try:
            proc.terminate()
            out = proc.stdout.read().decode(errors="replace")[-3000:]
        except Exception:
            out = ""
        raise RuntimeError(f"load test failed; server output tail:\n{out}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, default=str))
