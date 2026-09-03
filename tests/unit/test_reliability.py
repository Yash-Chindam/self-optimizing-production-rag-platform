from rag_platform.reliability import CircuitBreaker, CircuitState


class FakeClock:
    """A controllable clock: advance() moves it forward without a real sleep."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def failing() -> int:
    raise RuntimeError("dependency unavailable")


def test_a_healthy_call_returns_its_own_result() -> None:
    breaker = CircuitBreaker(name="test")
    result, note = breaker.call(lambda: 42, fallback=0)
    assert result == 42
    assert note.ok is True
    assert breaker.state is CircuitState.CLOSED


def test_a_failing_call_returns_the_fallback_and_a_degradation_note() -> None:
    breaker = CircuitBreaker(name="test", failure_threshold=5)
    result, note = breaker.call(failing, fallback=-1)
    assert result == -1
    assert note.ok is False
    assert "test_failed" in note.detail
    assert breaker.state is CircuitState.CLOSED


def test_the_circuit_opens_after_the_failure_threshold() -> None:
    breaker = CircuitBreaker(name="test", failure_threshold=2)
    breaker.call(failing, fallback=-1)
    assert breaker.state is CircuitState.CLOSED
    breaker.call(failing, fallback=-1)
    assert breaker.state is CircuitState.OPEN


def test_an_open_circuit_fails_fast_without_calling_the_dependency() -> None:
    breaker = CircuitBreaker(name="test", failure_threshold=1)
    breaker.call(failing, fallback=-1)
    assert breaker.state is CircuitState.OPEN

    calls = 0

    def tracked() -> int:
        nonlocal calls
        calls += 1
        return 1

    result, note = breaker.call(tracked, fallback=-1)
    assert calls == 0
    assert result == -1
    assert note.detail == "test_circuit_open"


def test_a_success_resets_the_failure_count() -> None:
    breaker = CircuitBreaker(name="test", failure_threshold=2)
    breaker.call(failing, fallback=-1)
    breaker.call(lambda: 1, fallback=-1)
    breaker.call(failing, fallback=-1)
    # Two failures, but a success reset the streak in between: still closed.
    assert breaker.state is CircuitState.CLOSED


def test_the_circuit_half_opens_after_the_cooldown_and_allows_one_trial() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(
        name="test", failure_threshold=1, reset_after_seconds=10.0, clock=clock
    )
    breaker.call(failing, fallback=-1)
    assert breaker.state is CircuitState.OPEN

    clock.advance(10.0)
    assert breaker.state is CircuitState.HALF_OPEN

    result, note = breaker.call(lambda: 7, fallback=-1)
    assert result == 7
    assert note.ok is True
    assert breaker.state is CircuitState.CLOSED


def test_a_failed_trial_during_half_open_reopens_the_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(
        name="test", failure_threshold=1, reset_after_seconds=10.0, clock=clock
    )
    breaker.call(failing, fallback=-1)
    clock.advance(10.0)
    assert breaker.state is CircuitState.HALF_OPEN

    breaker.call(failing, fallback=-1)
    assert breaker.state is CircuitState.OPEN


def test_a_call_slower_than_the_timeout_counts_as_a_failure() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(name="test", timeout_seconds=1.0, clock=clock)

    def slow() -> int:
        clock.advance(2.0)
        return 1

    result, note = breaker.call(slow, fallback=-1)
    assert result == -1
    assert note.ok is False
    assert "timeout" in note.detail
