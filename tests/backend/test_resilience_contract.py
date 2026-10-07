"""Contract §3.5 — circuit breaker + timeout helper.

The clock is injected, so the whole closed -> open -> half_open -> closed cycle is exercised
without a single :func:`time.sleep`.
"""

from __future__ import annotations

import time

import pytest

from tests.backend import _harness as H

THRESHOLD = 3
RESET_S = 10.0


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def breaker(clock=None, fail_threshold: int = THRESHOLD, reset_s: float = RESET_S):
    mod = H.module("tmm.serve.resilience")
    clock = clock or FakeClock()
    cb = H.construct(mod.CircuitBreaker, {
        "fail_threshold": fail_threshold, "threshold": fail_threshold,
        "max_failures": fail_threshold, "fails": fail_threshold,
        "reset_s": reset_s, "reset_timeout_s": reset_s, "recovery_s": reset_s,
        "clock": clock,
    }, fallback_args=(fail_threshold, reset_s, clock))
    return cb, clock


def boom(*_a, **_k):
    raise RuntimeError("backend down")


def test_initial_state_is_closed_and_calls_pass_through():
    cb, _ = breaker()
    assert cb.state == "closed"
    assert cb.call(lambda x: x + 1, 1) == 2


def test_first_and_second_failure_keep_the_circuit_closed():
    cb, _ = breaker()
    for _ in range(THRESHOLD - 1):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state == "closed"


def test_threshold_consecutive_failures_open_the_circuit():
    cb, _ = breaker()
    for _ in range(THRESHOLD):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state == "open"


def test_open_circuit_short_circuits_without_invoking_the_function():
    cb, _ = breaker()
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        raise RuntimeError("down")

    for _ in range(THRESHOLD):
        with pytest.raises(RuntimeError):
            cb.call(counted)
    assert calls["n"] == THRESHOLD

    with pytest.raises(H.module("tmm.serve.resilience").CircuitOpen):
        cb.call(counted)
    assert calls["n"] == THRESHOLD, "the open breaker still invoked the failing function"


def test_success_resets_the_consecutive_failure_count():
    cb, _ = breaker()
    for _ in range(THRESHOLD - 1):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.call(lambda: "ok") == "ok"
    for _ in range(THRESHOLD - 1):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state == "closed"


# --------------------------------------------------------------------------------------
# adversarial: full cycle on an injected clock, no sleep()
# --------------------------------------------------------------------------------------
def test_clock_advance_through_closed_open_half_open_closed():
    cb, clock = breaker()
    assert cb.state == "closed"

    for _ in range(THRESHOLD):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state == "open"

    clock.advance(RESET_S - 0.001)
    with pytest.raises(H.module("tmm.serve.resilience").CircuitOpen):
        cb.call(lambda: "too early")

    clock.advance(0.002)  # now strictly past reset_s
    assert cb.call(lambda: "probe") == "probe", "half-open probe was not allowed"
    assert cb.state == "closed"

    # closed again -> normal service
    assert cb.call(lambda: "after") == "after"


def test_failed_half_open_probe_reopens_and_waits_again():
    cb, clock = breaker()
    for _ in range(THRESHOLD):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state == "open"

    clock.advance(RESET_S + 1)
    with pytest.raises(RuntimeError):
        cb.call(boom)
    assert cb.state == "open", "a failed half-open probe must re-open the circuit"

    clock.advance(RESET_S - 0.001)
    with pytest.raises(H.module("tmm.serve.resilience").CircuitOpen):
        cb.call(lambda: "early")
    clock.advance(0.002)
    assert cb.call(lambda: "late") == "late"
    assert cb.state == "closed"


def test_circuit_state_gauge_is_defined_with_backend_label():
    metric = H.find_metric("tmm_circuit_state")
    assert metric is not None, "tmm_circuit_state is missing (contract §3.6)"
    assert set(metric._labelnames) == {"backend"}, metric._labelnames


# ======================================================================================
# call_with_timeout
# ======================================================================================
def test_call_with_timeout_returns_fast_result():
    mod = H.module("tmm.serve.resilience")
    assert mod.call_with_timeout(lambda a, b: a + b, 1.0, 2, 3) == 5


def test_call_with_timeout_passes_kwargs():
    mod = H.module("tmm.serve.resilience")
    assert mod.call_with_timeout(lambda *, x: x * 2, 1.0, x=21) == 42


def test_call_with_timeout_raises_timeout_error():
    mod = H.module("tmm.serve.resilience")
    with pytest.raises(TimeoutError):
        mod.call_with_timeout(time.sleep, 0.05, 1.0)


def test_call_with_timeout_propagates_exceptions():
    mod = H.module("tmm.serve.resilience")

    def boom2():
        raise ValueError("inner")

    with pytest.raises(ValueError, match="inner"):
        mod.call_with_timeout(boom2, 1.0)
