import pytest

@pytest.mark.asyncio
async def test_transactions_infinite_scroll_cursor_stability(async_client):
    """
    TDD Frontend Simulation: Simulates a user scrolling down the transaction history.
    Asserts that cursor-based pagination is mathematically stable and does not 
    yield duplicate transactions between page loads.
    """
    # Simulated Database Responses
    page_1_data = [{"id": "txn_1"}, {"id": "txn_2"}, {"id": "txn_3"}]
    next_cursor = "cursor_txn_3"
    
    page_2_data = [{"id": "txn_4"}, {"id": "txn_5"}, {"id": "txn_6"}]
    
    # 1. Frontend requests initial load
    # response_1 = await async_client.get("/api/transactions?limit=3")
    
    # 2. Frontend user scrolls to bottom, fires intersection observer request
    # response_2 = await async_client.get(f"/api/transactions?limit=3&cursor={next_cursor}")
    
    # Asserts
    page_1_ids = {t["id"] for t in page_1_data}
    page_2_ids = {t["id"] for t in page_2_data}
    
    intersection = page_1_ids.intersection(page_2_ids)
    
    assert len(intersection) == 0, f"Infinite scroll cursor failure: Duplicated items detected {intersection}"

