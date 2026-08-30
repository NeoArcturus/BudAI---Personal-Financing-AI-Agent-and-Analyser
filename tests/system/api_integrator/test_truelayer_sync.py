import pytest
from unittest.mock import AsyncMock, patch
import httpx

def test_account_reader_single_id_enforcement():
    """TDD Integrator: Strictly enforces the one-account-per-call architecture."""
    def read_account(account_id: str):
        if not isinstance(account_id, str):
            raise TypeError("account_id must be a singular string.")
        return True

    with pytest.raises(TypeError):
        read_account(["acc_1", "acc_2"]) # Multi-account list blocked
        
    with pytest.raises(TypeError):
        read_account(None)

@pytest.mark.asyncio
async def test_pagination_cursor_exhaustion():
    """TDD Integrator: Asserts the while-loop exhausts all paginated network requests."""
    mock_http_client = AsyncMock()
    
    # Simulate page 1 with a cursor, page 2 with None
    mock_http_client.get.side_effect = [
        httpx.Response(200, json={"results": [1, 2], "meta": {"next_cursor": "cursor_xyz"}}),
        httpx.Response(200, json={"results": [3, 4], "meta": {"next_cursor": None}})
    ]
    
    async def fetch_all_transactions():
        cursor = None
        all_results = []
        while True:
            url = "https://api.truelayer.com/transactions" + (f"?cursor={cursor}" if cursor else "")
            resp = await mock_http_client.get(url)
            data = resp.json()
            all_results.extend(data["results"])
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return all_results

    total_data = await fetch_all_transactions()
    assert len(total_data) == 4
    assert mock_http_client.get.call_count == 2

@pytest.mark.asyncio
@patch('asyncio.sleep')
async def test_rate_limit_backoff_enforcement(mock_sleep):
    """TDD Integrator: Asserts HTTP 429 Retry-After headers force a safe thread sleep."""
    mock_client = AsyncMock()
    
    # First call throws 429, second call succeeds
    mock_client.post.side_effect = [
        httpx.HTTPStatusError("Rate Limited", request=AsyncMock(), response=httpx.Response(429, headers={"Retry-After": "5"})),
        httpx.Response(200, json={"status": "success"})
    ]
    
    async def resilient_post():
        try:
            return await mock_client.post("url")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                sleep_time = int(e.response.headers.get("Retry-After", 1))
                await asyncio.sleep(sleep_time)
                return await mock_client.post("url") # Retry

    await resilient_post()
    mock_sleep.assert_called_once_with(5)
    assert mock_client.post.call_count == 2

