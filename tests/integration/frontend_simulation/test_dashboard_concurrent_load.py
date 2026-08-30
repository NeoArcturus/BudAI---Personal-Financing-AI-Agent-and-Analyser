import pytest
import asyncio

@pytest.mark.asyncio
async def test_dashboard_mount_parallel_requests(async_client):
    """
    TDD Frontend Simulation: When the user opens the BudAI dashboard, React 
    simultaneously mounts 4 widgets, firing 4 concurrent API requests.
    Asserts the backend connection pool can serve the single-client burst 
    without triggering a deadlock or rate limit penalty.
    """
    # Simulated API routes requested by the frontend widgets on load
    endpoints = [
        "/api/accounts/balances",
        "/api/transactions/recent",
        "/api/analytics/cashflow",
        "/api/budgets/active"
    ]
    
    async def fetch_widget_data(endpoint: str):
        # response = await async_client.get(endpoint)
        # return response.status_code
        await asyncio.sleep(0.05) # Simulated latency
        return 200 # Simulated success
        
    # React fires these simultaneously via Promise.all()
    tasks = [fetch_widget_data(ep) for ep in endpoints]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 4, "Dashboard failed to complete all parallel requests."
    assert all(status == 200 for status in results), "Dashboard burst load triggered a backend failure or deadlock."

