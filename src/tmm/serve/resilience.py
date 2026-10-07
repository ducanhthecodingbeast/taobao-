"""Resilience primitives: a hand-rolled circuit breaker and a timeout helper (contract §3.5).

No third-party breaker library. ``clock`` is injectable so tests can walk the full
closed -> open -> half-open -> closed cycle without ever sleeping.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

__all__ = ["STATE_CODES", "CircuitBreaker", "CircuitOpen", "call_with_timeout"]

#: Gauge encoding required by contract §3.6.
STATE_CODES: dict[str, int] = {"closed": 0, "half_open": 1, "open": 2}


class CircuitOpen(RuntimeError):
    """Raised when a call is refused because the circuit is open."""

    def __init__(self, name: str = "circuit", message: str | None = None) -> None:
        self.name = name
        super().__init__(message or f"circuit '{name}' is open")


class CircuitBreaker:
    """N consecutive failures open the circuit for ``reset_s``, then one probe decides."""

    def __init__(
        self,
        fail_threshold: int,
        reset_s: float,
        clock: Callable[[], float] = time.monotonic,
        name: str = "circuit",
        on_state_change: Callable[[str], None] | None = None,
    ) -> None:
        if fail_threshold < 1:
            raise ValueError("fail_threshold must be >= 1")
        self.fail_threshold = int(fail_threshold)
        self.reset_s = float(reset_s)
        self.clock = clock
        self.name = name
        self._on_state_change = on_state_change
        self._lock = threading.RLock()
        self._state = "closed"
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False

    # -- introspection -----------------------------------------------------------------
    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def state_code(self) -> int:
        return STATE_CODES[self.state]

    # -- transitions -------------------------------------------------------------------
    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
        if self._on_state_change is not None:
            try:
                self._on_state_change(state)
            except Exception:  # pragma: no cover - observers must never break the call
                pass

    def _maybe_half_open(self) -> None:
        """Caller must hold the lock."""
        if self._state == "open" and self._opened_at is not None:
            if self.clock() - self._opened_at >= self.reset_s:
                self._half_open_in_flight = False
                self._set_state("half_open")

    def _open(self) -> None:
        self._opened_at = self.clock()
        self._half_open_in_flight = False
        self._set_state("open")

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._half_open_in_flight = False
            self._opened_at = None
            if self._state != "closed":
                self._set_state("closed")

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._half_open_in_flight = False
            if self._state == "half_open":
                self._open()
            elif self._state == "closed" and self._failures >= self.fail_threshold:
                self._open()

    #: Alias kept because app/consumer code reads well with it.
    reset = record_success

    # -- the call path -----------------------------------------------------------------
    def call[T](self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        with self._lock:
            self._maybe_half_open()
            if self._state == "open":
                raise CircuitOpen(self.name)
            if self._state == "half_open":
                if self._half_open_in_flight:
                    raise CircuitOpen(self.name, f"circuit '{self.name}' probe in flight")
                self._half_open_in_flight = True
        try:
            result = fn(*args, **kwargs)
        except BaseException:
            with self._lock:
                self._half_open_in_flight = False
            self.record_failure()
            raise
        self.record_success()
        return result


def call_with_timeout[T](fn: Callable[..., T], timeout_s: float, *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` in a worker thread and raise :class:`TimeoutError` if it overruns.

    The worker is a daemon: a timed-out call cannot be killed, but it can no longer block
    the request path.
    """
    out: dict[str, Any] = {}

    def run() -> None:
        try:
            out["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            out["error"] = exc

    thread = threading.Thread(target=run, daemon=True, name="tmm-timeout")
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', 'call')} exceeded {timeout_s}s")
    if "error" in out:
        raise out["error"]
    return out.get("value")  # type: ignore[return-value]
