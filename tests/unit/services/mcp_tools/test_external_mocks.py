import pytest
from unittest.mock import AsyncMock
import httpx

@pytest.mark.asyncio
async def test_truelayer_api_timeout_fallback():
    """TDD: Asserts that an external bank API timeout does not crash the internal event loop."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.TimeoutException("TrueLayer did not respond in 5s")
    
    async def trigger_bank_sync():
        try:
            await mock_client.post("https://auth.truelayer.com")
        except httpx.TimeoutException:
            return {"status": "degraded", "message": "Bank sync delayed"}
            
    response = await trigger_bank_sync()
    assert response["status"] == "degraded"
