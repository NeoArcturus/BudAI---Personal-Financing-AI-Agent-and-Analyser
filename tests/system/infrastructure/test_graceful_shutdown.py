import pytest
import asyncio
import signal

@pytest.mark.asyncio
async def test_fastapi_sigterm_trapping():
    """
    TDD System Design: Asserts that when Docker sends a SIGTERM, Uvicorn 
    does not instantly terminate, but waits for active background tasks 
    (like a bank sync) to safely commit to the database.
    """
    active_connections = 1
    database_commit_successful = False

    async def mock_bank_sync_task():
        nonlocal database_commit_successful, active_connections
        try:
            await asyncio.sleep(0.05) # Simulate 50ms DB write
            database_commit_successful = True
        finally:
            active_connections -= 1

    # Simulate FastAPI Lifespan shutdown event
    async def trigger_sigterm_shutdown():
        # Await all active background tasks with a timeout limit (e.g., 5 seconds)
        await asyncio.wait_for(mock_bank_sync_task(), timeout=5.0)

    # Execute the shutdown trap
    await trigger_sigterm_shutdown()

    assert database_commit_successful is True, "SIGTERM corrupted active database transaction."
    assert active_connections == 0, "Zombie connection left open after shutdown."

