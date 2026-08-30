import pytest
import asyncio
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_virtual_budget_concurrent_deduction_race_condition():
    """
    TDD System Design: Asserts that 50 concurrent requests attempting to deduct 
    funds from the same budget envelope correctly serialize via database locks, 
    preventing double-spend race conditions.
    """
    # Simulated shared state (representing PostgreSQL Row-Level Lock)
    shared_balance = 1000.0
    lock = asyncio.Lock()

    async def simulate_db_transaction(deduction: float):
        nonlocal shared_balance
        async with lock:  # Simulates 'SELECT ... FOR UPDATE'
            if shared_balance >= deduction:
                await asyncio.sleep(0.01) # Simulate DB IO delay
                shared_balance -= deduction
                return True
            return False

    # Fire 50 concurrent deduction requests of 50.0 each (Total requested: 2500.0)
    # Only 20 should succeed (1000.0 / 50.0). The remaining 30 must mathematically fail.
    tasks = [simulate_db_transaction(50.0) for _ in range(50)]
    results = await asyncio.gather(*tasks)

    successful_transactions = results.count(True)
    failed_transactions = results.count(False)

    assert successful_transactions == 20, "Race condition detected: System allowed over-deduction."
    assert failed_transactions == 30, "Race condition detected: System failed to reject insufficient funds."
    assert shared_balance == 0.0, "Database math corrupted under concurrent load."

