"""Contract §0.1 — *zero hard dependencies*, including the model artifacts.

The contract requires that "if the broker or cache is absent the app must still start and serve
in a documented degraded mode". The same must hold when the model artifacts are absent (a fresh
checkout, a CI runner, a half-populated volume): the service must come up and answer its
operational endpoints instead of crashing.

Checked in a **subprocess** because ``tmm.config`` resolves ``TMM_ARTIFACTS``/``TMM_DATA`` at
import time; the subprocess points both at empty directories and imports the ASGI app the same
way uvicorn does.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from tests.backend import _harness as H


def _boot_without_artifacts(tmp_path) -> dict:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "data").mkdir()
    code = (
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "from tmm.serve.app import app\n"
        "with TestClient(app) as c:\n"
        "    out = {\n"
        "        'livez': c.get('/livez').status_code,\n"
        "        'health': c.get('/health').status_code,\n"
        "        'mode': c.get('/health').json().get('mode'),\n"
        "        'readyz': c.get('/readyz').status_code,\n"
        "        'metrics': c.get('/metrics').status_code,\n"
        "    }\n"
        "print('RESULT' + json.dumps(out))\n"
    )
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(H.ROOT_SRC),
        "MPLCONFIGDIR": "/tmp/mpl",
        "TMM_ARTIFACTS": str(tmp_path / "artifacts"),
        "TMM_DATA": str(tmp_path / "data"),
    })
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=300)
    match = re.search(r"^RESULT(\{.*\})$", proc.stdout, re.MULTILINE)
    assert match, (
        f"the app did not boot without model artifacts.\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
    )
    return json.loads(match.group(1))


def test_app_starts_and_serves_with_no_model_artifacts(tmp_path):
    result = _boot_without_artifacts(tmp_path)
    assert result["livez"] == 200, result
    assert result["health"] == 200, result
    # no artifacts is a *minimal* deployment: it serves, but it is not replica-safe
    assert result["mode"] == "minimal", result
    assert result["readyz"] == 503, result
    assert result["metrics"] == 200, result
