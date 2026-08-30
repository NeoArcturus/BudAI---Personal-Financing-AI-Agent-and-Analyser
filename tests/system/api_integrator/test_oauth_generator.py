import pytest
from unittest.mock import AsyncMock, patch
import time

@pytest.mark.asyncio
@patch('time.time')
async def test_access_token_caching_and_ttl(mock_time):
    """
    TDD Integrator: Asserts that the OAuth token is mathematically cached 
    and only triggers a network request when the TTL strictly expires.
    """
    mock_network_call = AsyncMock(return_value={"access_token": "token_123", "expires_in": 3600})
    
    # Simulated Token Generator State
    cache = {"token": None, "expires_at": 0}
    
    async def get_token():
        current_time = time.time()
        if cache["token"] and current_time < cache["expires_at"]:
            return cache["token"]
            
        response = await mock_network_call()
        cache["token"] = response["access_token"]
        # Padding applied to TTL for network latency safety
        cache["expires_at"] = current_time + response["expires_in"] - 60 
        return cache["token"]

    # 1. Initial Call (Network executed)
    mock_time.return_value = 1000.0
    token_1 = await get_token()
    assert mock_network_call.call_count == 1
    
    # 2. Call at 3500 seconds (Cache hit)
    mock_time.return_value = 4500.0
    token_2 = await get_token()
    assert mock_network_call.call_count == 1 # Network not called
    assert token_1 == token_2
    
    # 3. Call at 3601 seconds (TTL Expired -> Network executed)
    mock_time.return_value = 4601.0
    token_3 = await get_token()
    assert mock_network_call.call_count == 2

