"""Fixtures shared by the backend verification suite.

Two layers, one directory:

* **unit / contract tests** (default) -- use the in-memory implementations and must pass with no
  Redis and no Kafka anywhere on the host;
* **integration tests** (``@pytest.mark.integration``) -- need the isolated stack from
  ``docker/docker-compose.test.yml`` (Redis on 127.0.0.1:6399, Kafka on 127.0.0.1:9199).  They
  *skip* with a loud reason when that stack is not running, so the default command never hangs
  on a missing broker and never silently passes because of one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (str(SRC), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from tests.backend._harness import (  # noqa: E402
    kafka_bootstrap,
    redis_url,
    tcp_open,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires real Redis + Kafka (docker-compose.test.yml)")
    config.addinivalue_line("markers", "adversarial: adversarial case required by contract §7")


def _services_up() -> tuple[bool, str]:
    r_host, r_port = _split(redis_url())
    k_host, k_port = _split(kafka_bootstrap())
    if not tcp_open(r_host, r_port):
        return False, f"redis not listening on {r_host}:{r_port} (start docker/docker-compose.test.yml)"
    if not tcp_open(k_host, k_port):
        return False, f"kafka not listening on {k_host}:{k_port} (start docker/docker-compose.test.yml)"
    return True, ""


def _split(url: str) -> tuple[str, int]:
    from urllib.parse import urlsplit

    parsed = urlsplit(url if "//" in url else f"//{url}")
    return parsed.hostname or "127.0.0.1", parsed.port or 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    up, reason = _services_up()
    if up:
        return
    skip = pytest.mark.skip(reason=f"integration stack unavailable: {reason}")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def redis_test_url() -> str:
    return redis_url()


@pytest.fixture(scope="session")
def kafka_test_bootstrap() -> str:
    return kafka_bootstrap()


@pytest.fixture
def no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure an unrelated proxy in the host environment cannot intercept test sockets."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
