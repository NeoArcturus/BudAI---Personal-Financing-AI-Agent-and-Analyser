import pytest

@pytest.mark.asyncio
async def test_truelayer_webhook_route_valid_payload(async_client):
    """
    TDD: Hits the APIRouter. Asserts that a structurally valid TrueLayer webhook 
    returns a 200 OK or 202 Accepted.
    """
    valid_payload = {
        "type": "Data.Refresh.Successful",
        "client_id": "test_client_id",
        "event_id": "evt_123"
    }
    
    response = await async_client.post("/api/webhooks/truelayer", json=valid_payload)
    
    assert response.status_code in [200, 202], f"Webhook route failed. Expected 200/202, got {response.status_code}"

@pytest.mark.asyncio
async def test_truelayer_webhook_route_schema_validation(async_client):
    """
    TDD: Asserts that FastAPI's Pydantic validation intercepts malformed payloads 
    (missing 'type' field) and returns exactly 422 Unprocessable Entity.
    """
    malformed_payload = {
        "client_id": "test_client_id"
        # Missing 'type' and 'event_id'
    }
    
    response = await async_client.post("/api/webhooks/truelayer", json=malformed_payload)
    
    assert response.status_code == 422, "FastAPI failed to reject malformed webhook payload at the routing layer."

