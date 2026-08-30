import pytest
import time

@pytest.mark.asyncio
async def test_api_idempotency_blocks_duplicate_webhooks(async_client):
    """
    TDD Test for System Design: API Idempotency.
    Fires two identical POST payloads with the same Idempotency-Key.
    Asserts that the second request returns a cached 200 OK without hitting the DB.
    """
    payload = {"event_id": "evt_sandbox_123", "type": "Data.Refresh.Successful", "client_id": "sandbox_user"}
    headers = {"Idempotency-Key": "idem_12345"}
    
    # First request should be processed normally
    response_1 = await async_client.post("/api/webhooks/truelayer", json=payload, headers=headers)
    assert response_1.status_code in [200, 202]
    
    # Second request should be caught by the API Gateway / Caching layer
    response_2 = await async_client.post("/api/webhooks/truelayer", json=payload, headers=headers)
    assert response_2.status_code in [200, 202]
    assert "X-Idempotent-Cache-Hit" in response_2.headers, "Idempotency layer is missing"

@pytest.mark.asyncio
async def test_llm_circuit_breaker_timeout(async_client):
    """
    TDD Test for System Design: Circuit Breaker Pattern.
    Simulates a heavy categorization request. If the LLM hangs, the API must fail fast (e.g. 503)
    instead of blocking the Uvicorn worker indefinitely.
    """
    start_time = time.time()
    
    # Simulating a call to a chat/categorization endpoint that might hang
    # If the circuit breaker is missing, this will hang. We expect it to trip and return 503 quickly.
    response = await async_client.post("/api/chat", json={"message": "Categorize my last 50 transactions"})
    
    elapsed = time.time() - start_time
    
    # We expect the system to shed the load or fallback within 3 seconds
    assert elapsed < 3.0, "Circuit breaker failed to trip, endpoint hung for too long"
    assert response.status_code != 500, "Endpoint threw an unhandled 500 instead of a graceful degradation"
