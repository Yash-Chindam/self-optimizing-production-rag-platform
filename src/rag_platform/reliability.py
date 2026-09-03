"""Circuit breakers and declared degradation for optional dependencies (specification section 17).

A dependency wrapped in a `CircuitBreaker` is never allowed to crash a query: a failing call is
caught, counted and turned into the fallback value the caller supplies; the caller's job is only
to say what "no answer from this dependency" means for it — an empty candidate list for graph
expansion, the unranked candidates for reranking, an empty synthesis for generation. After
`failure_threshold` consecutive failures the breaker opens and stops calling the dependency at
all for `reset_after_seconds`, so a dependency that is down stays fast-failed instead of being
retried on every single query; after that cooldown one call is let through to test recovery.

This is cooperative rather than preemptive: a slow call is only detected once it returns, because
the deterministic in-process adapters this platform ships are guaranteed to return. A real
network adapter must enforce its own request timeout; `timeout_seconds` here turns "took too
long" into the same counted failure a raised exception is, so a live adapter's timeout still
trips the breaker instead of being invisible to it.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    """Calls the dependency normally."""
    OPEN = "open"
    """Fails fast without calling the dependency."""
    HALF_OPEN = "half_open"
    """The cooldown has elapsed; the next call is a trial that can close or reopen the circuit."""


@dataclass(frozen=True, slots=True)
class DegradedCall:
    """What happened the last time a breaker's dependency was asked for a result."""

    ok: bool
    detail: str


@dataclass(slots=True)
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    reset_after_seconds: float = 30.0
    timeout_seconds: float = 1.0
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self.clock() - self._opened_at >= self.reset_after_seconds:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def call(self, operation: Callable[[], T], *, fallback: T) -> tuple[T, DegradedCall]:
        """Run `operation`; return `fallback` and a degradation note if it fails or is skipped."""
        if self.state is CircuitState.OPEN:
            return fallback, DegradedCall(ok=False, detail=f"{self.name}_circuit_open")

        start = self.clock()
        try:
            result = operation()
        except Exception as error:  # any dependency failure degrades a query, it never crashes it
            self._record_failure()
            return fallback, DegradedCall(ok=False, detail=f"{self.name}_failed:{error}")

        elapsed = self.clock() - start
        if elapsed > self.timeout_seconds:
            self._record_failure()
            return fallback, DegradedCall(ok=False, detail=f"{self.name}_timeout:{elapsed:.3f}s")

        self._record_success()
        return result, DegradedCall(ok=True, detail=f"{self.name}_ok")

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = self.clock()

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None
