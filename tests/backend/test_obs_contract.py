"""Contract §3.6 — observability: JSON logs, metric names/labels, stage timings."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys

from tests.backend import _harness as H


def _json_formatter() -> logging.Formatter:
    """The JSON formatter the contract mandates (``obs.JsonFormatter``)."""
    formatter_cls = getattr(H.module("tmm.serve.obs"), "JsonFormatter")
    return formatter_cls()


def _capture_logs(level: int = logging.DEBUG) -> tuple[io.StringIO, logging.Logger]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_json_formatter())
    logger = logging.getLogger("tmm.verify")
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
    return buf, logger


def test_configure_logging_installs_a_json_formatter_on_root():
    """Checked in a subprocess so pytest's own root handlers cannot mask the result."""
    code = (
        "import logging\n"
        "import tmm.serve.obs as o\n"
        "o.configure_logging('DEBUG')\n"
        "assert logging.getLogger().level == logging.DEBUG, logging.getLogger().level\n"
        "logging.getLogger('tmm.probe').warning('hello %s', 'world')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(H.ROOT_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    assert lines, f"configure_logging produced no stderr output: {proc.stderr!r}"
    parsed = None
    for line in lines:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "hello world" in json.dumps(candidate):
            parsed = candidate
    assert parsed is not None, (
        f"root-logger output is not JSON: {lines!r} (contract §3.6: JSON formatter on the "
        "root logger)"
    )
    assert parsed.get("level") == "WARNING", parsed


def test_log_event_emits_one_valid_json_line():
    buf, logger = _capture_logs()
    H.module("tmm.serve.obs").log_event(logger, "ingest_ok", user_id=11, stage="decode")
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one JSON line, got {lines!r}"
    payload = json.loads(lines[0])
    assert "ingest_ok" in json.dumps(payload), payload
    assert payload.get("user_id") == 11, payload
    assert payload.get("stage") == "decode", payload


def test_log_event_output_survives_embedded_newlines():
    """One JSON line means newlines in values must be escaped, not split across records."""
    buf, logger = _capture_logs()
    H.module("tmm.serve.obs").log_event(logger, "weird", detail="line1\nline2")
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1, lines
    assert json.loads(lines[0])["detail"] == "line1\nline2"


def test_stage_timer_records_named_stage_histogram():
    mod = H.module("tmm.serve.obs")
    StageTimer = getattr(mod, "StageTimer")

    def run() -> None:
        with StageTimer("tower"):
            pass

    before, after, _ = H.delta("tmm_stage_duration_seconds", {"stage": "tower"}, run, suffix="_count")
    assert after > before, (
        "StageTimer('tower') did not record tmm_stage_duration_seconds{stage='tower'}"
    )


# ======================================================================================
# metric registry contract
# ======================================================================================
def test_every_contract_metric_exists_with_the_right_type_and_labels():
    missing = []
    problems = []
    for name, (kind, labels) in H.REQUIRED_METRICS.items():
        metric = H.find_metric(name)
        if metric is None:
            missing.append(name)
            continue
        if getattr(metric, "_type", None) != kind:
            problems.append(f"{name}: type {getattr(metric, '_type', None)!r} != {kind!r}")
        got = set(metric._labelnames)
        if got != labels:
            problems.append(f"{name}: labels {sorted(got)} != {sorted(labels)}")
    assert not missing, f"metrics named in contract §3.6 are missing: {missing}"
    assert not problems, problems


def test_stage_labels_used_by_the_recommend_path_are_the_documented_ones():
    """Contract §3.6: stage names are exactly decode, session_read, tower, search, total."""
    metric = H.find_metric("tmm_stage_duration_seconds")
    assert metric is not None
    observed = set()
    for family in metric.collect():
        for sample in family.samples:
            if sample.labels.get("stage"):
                observed.add(sample.labels["stage"])
    # Not all stages need to have fired yet, but none outside the documented set may exist.
    undocumented = observed - {"decode", "session_read", "tower", "search", "total"}
    assert not undocumented, f"undocumented stage labels recorded: {undocumented}"


def test_module_reimport_does_not_duplicate_registration():
    """Contract §3.6: a dedicated CollectorRegistry per app instance, else re-import explodes."""
    code = (
        "import importlib, tmm.serve.obs as o\n"
        "importlib.reload(o)\n"
        "importlib.reload(o)\n"
        "print('reload-ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(H.ROOT_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=120)
    assert proc.returncode == 0, f"re-import failed:\n{proc.stdout}\n{proc.stderr}"
    assert "reload-ok" in proc.stdout
