"""Contract §1 — ``src/tmm/serve/settings.py``.

Expected values come from the frozen contract table, not from the implementation.
"""

from __future__ import annotations

import pytest

from tests.backend import _harness as H


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for key in list(__import__("os").environ):
        if key.startswith("TMM_") or key in ("REDIS_URL", "KAFKA_BOOTSTRAP"):
            monkeypatch.delenv(key, raising=False)


def test_load_settings_defaults_match_contract():
    s = H.module("tmm.serve.settings").load_settings()
    assert s.api_keys == ()
    assert s.auth_required is False
    assert s.rate_limit_rps == 20.0
    assert s.rate_limit_burst == 40
    assert s.redis_url is None
    assert s.kafka_bootstrap is None
    assert s.kafka_topic == "tmm.clickstream"
    assert s.kafka_dlq_topic == "tmm.clickstream.dlq"
    assert s.idempotency_ttl_s == 3600
    assert s.session_max == 50
    assert s.cache_ttl_s == 5
    assert s.stage_timeout_s == 0.5
    assert s.circuit_fail_threshold == 5
    assert s.circuit_reset_s == 10.0


def test_api_keys_parsed_from_comma_separated_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TMM_API_KEYS", "alpha,beta , gamma")
    s = H.module("tmm.serve.settings").load_settings()
    assert s.api_keys == ("alpha", "beta", "gamma")
    # `auth_required` defaults to bool(api_keys)
    assert s.auth_required is True


def test_auth_required_explicit_false_without_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TMM_AUTH_REQUIRED", "false")
    assert H.module("tmm.serve.settings").load_settings().auth_required is False
    monkeypatch.setenv("TMM_AUTH_REQUIRED", "0")
    assert H.module("tmm.serve.settings").load_settings().auth_required is False


def test_auth_required_explicit_true_without_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TMM_AUTH_REQUIRED", "1")
    assert H.module("tmm.serve.settings").load_settings().auth_required is True


def test_numeric_and_url_env_parsing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TMM_RATE_LIMIT_RPS", "5.5")
    monkeypatch.setenv("TMM_RATE_LIMIT_BURST", "7")
    monkeypatch.setenv("TMM_SESSION_MAX", "11")
    monkeypatch.setenv("TMM_CACHE_TTL_S", "0")
    monkeypatch.setenv("TMM_STAGE_TIMEOUT_S", "0.25")
    monkeypatch.setenv("TMM_CIRCUIT_FAILS", "3")
    monkeypatch.setenv("TMM_CIRCUIT_RESET_S", "2.5")
    monkeypatch.setenv("TMM_IDEMPOTENCY_TTL_S", "60")
    monkeypatch.setenv("TMM_KAFKA_TOPIC", "t")
    monkeypatch.setenv("TMM_KAFKA_DLQ_TOPIC", "t.dlq")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6399/0")
    monkeypatch.setenv("KAFKA_BOOTSTRAP", "127.0.0.1:9199")

    s = H.module("tmm.serve.settings").load_settings()
    assert s.rate_limit_rps == 5.5
    assert s.rate_limit_burst == 7
    assert s.session_max == 11
    assert s.cache_ttl_s == 0
    assert s.stage_timeout_s == 0.25
    assert s.circuit_fail_threshold == 3
    assert s.circuit_reset_s == 2.5
    assert s.idempotency_ttl_s == 60
    assert s.kafka_topic == "t"
    assert s.kafka_dlq_topic == "t.dlq"
    assert s.redis_url == "redis://127.0.0.1:6399/0"
    assert s.kafka_bootstrap == "127.0.0.1:9199"


def test_load_settings_is_not_cached(monkeypatch: pytest.MonkeyPatch):
    """Contract §1: tests monkeypatch the environment and call it repeatedly."""
    mod = H.module("tmm.serve.settings")
    monkeypatch.setenv("TMM_RATE_LIMIT_BURST", "1")
    first = mod.load_settings()
    monkeypatch.setenv("TMM_RATE_LIMIT_BURST", "2")
    second = mod.load_settings()
    assert first.rate_limit_burst == 1
    assert second.rate_limit_burst == 2
    assert first is not second


def test_settings_is_frozen():
    import dataclasses

    s = H.contract_settings()
    assert dataclasses.is_dataclass(s)
    with pytest.raises(Exception):
        s.rate_limit_rps = 1.0  # type: ignore[misc]
