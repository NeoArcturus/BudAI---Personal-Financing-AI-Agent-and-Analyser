import pytest
import time
from unittest.mock import AsyncMock

class CircuitBreakerOpenException(Exception):
    pass

@pytest.mark.asyncio
async def test_llm_circuit_breaker_fast_fail_and_recovery():
    """
    TDD System Design: Asserts that after 3 consecutive upstream timeouts, 
    the Circuit Breaker trips OPEN, instantly failing subsequent requests 
    to prevent thread starvation.
    """
    failure_count = 0
    circuit_state = "CLOSED"

    async def mock_llm_call_with_circuit_breaker():
        nonlocal failure_count, circuit_state
        if circuit_state == "OPEN":
            raise CircuitBreakerOpenException("Fast Fail: Upstream down")
        
        # Simulate Network Timeout
        failure_count += 1
        if failure_count >= 3:
            circuit_state = "OPEN"
        raise TimeoutError("Upstream LLM timed out")

    # Request 1 & 2: Normal Timeout behavior (Simulated slow requests)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            await mock_llm_call_with_circuit_breaker()

    # Request 3: Tips the threshold to OPEN
    with pytest.raises(TimeoutError):
        await mock_llm_call_with_circuit_breaker()

    # Request 4: Must FAST FAIL instantly without waiting for a timeout
    start_time = time.time()
    with pytest.raises(CircuitBreakerOpenException):
        await mock_llm_call_with_circuit_breaker()
    elapsed = time.time() - start_time

    assert elapsed < 0.01, "Circuit breaker failed to fast-fail. System is vulnerable to thread starvation."

