"""Regression guard for the *existing* serving path with the **real** model artifacts.

No services needed (memory session store, memory publisher), but this does build the real
:class:`tmm.serve.app.Recommender`.  It exists so the backend hardening cannot silently regress
the behaviour the project had before B1–B5 (contract §0.5/§0.6) and so the *measured* startup
memory of the real service stays honest.

Two artifact layouts are covered:

* **demo artifact** (``artifacts/models/demo/``) — the precomputed, self-sufficient layout the
  service prefers. Startup must stay well under 1.5 GB; the legacy layout measured ~6.9 GB.
* **legacy layout** (``artifacts/data`` + ``TMM_DATA/feature_map``) — kept as a fallback.

The memory assertion runs in a **subprocess**, because ``ru_maxrss`` is a high-water mark of the
whole process and would otherwise include pytest, collected test data and the fixtures of every
earlier test module.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from tests.backend import _harness as H

DEMO_DIR = H.ROOT / "artifacts" / "models" / "demo"
HAS_DEMO = (DEMO_DIR / "meta.json").exists()
HAS_LEGACY = (H.ROOT / "artifacts" / "data" / "item_vocab.npz").exists()

pytestmark = pytest.mark.skipif(
    not (HAS_DEMO or HAS_LEGACY),
    reason="no model artifacts (neither artifacts/models/demo nor artifacts/data) in this checkout",
)

#: The documented sizing claim is a 2 vCPU / 4 GB VPS; 1.5 GB leaves room for sessions,
#: torch/onnxruntime overhead and the API process itself.
MAX_ARTIFACT_PEAK_RSS_MB = 1536.0


def _real_app():
    from tmm.serve.app import create_app

    return create_app()


@pytest.fixture(scope="module")
def real_app():
    return _real_app()


def _measure_recommender_startup(env_extra: dict[str, str] | None = None) -> dict:
    """Build the real Recommender in a subprocess and return its peak RSS and metadata."""
    code = (
        "import json, resource\n"
        "from tmm.serve.app import Recommender\n"
        "r = Recommender(n_items=10000, use_demo_artifact=True)\n"
        "print('RESULT' + json.dumps({\n"
        "    'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,\n"
        "    'emb_mb': getattr(r, 'emb_mb', None),\n"
        "    'n_items': getattr(r, 'n_items', None),\n"
        "    'item_ids': [int(x) for x in list(r.item_ids)[:5]],\n"
        "    'use_demo_artifact': getattr(r, 'use_demo_artifact', None),\n"
        "}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(H.ROOT_SRC)
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env.update(env_extra or {})
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=600)
    match = re.search(r"^RESULT(\{.*\})$", proc.stdout, re.MULTILINE)
    assert match, f"memory probe failed:\nstdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
    return json.loads(match.group(1))


# ======================================================================================
# measured startup memory (the 7 GB -> sub-GB fix)
# ======================================================================================
@pytest.mark.skipif(not HAS_DEMO, reason="artifacts/models/demo is not built in this checkout")
def test_artifact_mode_startup_peak_rss_is_bounded():
    result = _measure_recommender_startup()
    peak = float(result["peak_rss_mb"])
    assert result["n_items"] == 10_000, result
    assert result["item_ids"] and all(isinstance(i, int) for i in result["item_ids"]), result
    assert peak < MAX_ARTIFACT_PEAK_RSS_MB, (
        f"artifact-mode Recommender startup peaked at {peak:.0f} MB RSS, over the "
        f"{MAX_ARTIFACT_PEAK_RSS_MB:.0f} MB bound (2 vCPU / 4 GB VPS sizing claim); "
        f"full result: {result}"
    )
    print(f"[measured] artifact-mode startup peak RSS = {peak:.1f} MB, emb_mb={result['emb_mb']}")


@pytest.mark.skipif(not HAS_DEMO, reason="artifacts/models/demo is not built in this checkout")
def test_artifact_mode_service_uses_the_artifact_catalogue_and_reports_its_size(real_app):
    import numpy as np

    artifact_ids = {int(x) for x in np.load(DEMO_DIR / "item_ids.npy")}
    file_mb = (DEMO_DIR / "emb_fp16.npy").stat().st_size / 2**20

    with TestClient(real_app) as client:
        health = client.get("/health").json()
        sampled = client.get("/items/sample", params={"n": 50}).json()["item_ids"]
        recs = client.get("/recommend", params={"user_id": 12345, "k": 5}).json()

    assert health["catalogue"] == 10_000, health
    # §"honest reporting": the table size must match the artifact actually loaded, not a constant
    reported = float(health["embedding_table_mb"])
    assert abs(reported - file_mb) <= 3.0, (
        f"/health reports embedding_table_mb={reported} but artifacts/models/demo/emb_fp16.npy "
        f"is {file_mb:.1f} MiB"
    )
    # real anonymised item ids from the artifact catalogue, not synthetic placeholders
    assert set(sampled) <= artifact_ids, (
        f"/items/sample returned ids outside the artifact catalogue: "
        f"{sorted(set(sampled) - artifact_ids)[:5]}"
    )
    assert recs, "artifact-mode /recommend returned nothing"
    assert {r["item_id"] for r in recs} <= artifact_ids, recs


# ======================================================================================
# behaviour that predates the hardening (contract §0.5/§0.6)
# ======================================================================================
def test_health_reports_the_reduced_embedding_table(real_app):
    with TestClient(real_app) as client:
        body = client.get("/health").json()
    assert body["session_backend"] == "memory"
    assert body["catalogue"] == 10_000, body
    # the reduced/demo table (~210 MB) must be in place, not the full 8.45 GiB fp16 table
    assert 50 < body["embedding_table_mb"] < 400, body
    assert body["user_tower"] in ("onnx", "torch", "identity"), body
    assert body["uptime_s"] >= 0


def test_items_sample_returns_real_ids(real_app):
    with TestClient(real_app) as client:
        ids = client.get("/items/sample", params={"n": 5}).json()["item_ids"]
    assert len(ids) == 5
    assert all(isinstance(i, int) for i in ids)


def test_recommend_returns_ranked_items_and_excludes_seen(real_app):
    with TestClient(real_app) as client:
        sample = client.get("/items/sample", params={"n": 3}).json()["item_ids"]
        for item in sample:
            assert client.post("/events", json={"user_id": 42, "item_id": item,
                                                "event_type": "view"}).status_code == 202
        r = client.get("/recommend", params={"user_id": 42, "k": 5})
        assert r.status_code == 200, r.text
        recs = r.json()
    assert recs, "the recommend handler returned nothing"
    for rec in recs:
        assert set(rec) >= {"item_id", "score", "cold_start"}, rec
        assert isinstance(rec["item_id"], int)
    returned = {rec["item_id"] for rec in recs}
    assert not (returned & set(sample)), f"/recommend returned already-seen items: {returned}"


def test_cold_start_user_still_gets_recommendations(real_app):
    with TestClient(real_app) as client:
        r = client.get("/recommend", params={"user_id": 999_999_999, "k": 5})
    assert r.status_code == 200, r.text
    assert len(r.json()) == 5
