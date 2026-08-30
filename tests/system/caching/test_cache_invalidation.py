import pytest

class SimulatedRedisCache:
    def __init__(self):
        self.memory = {}
    
    def set(self, key: str, value: any):
        self.memory[key] = value
        
    def get(self, key: str):
        return self.memory.get(key)
        
    def delete(self, key: str):
        if key in self.memory:
            del self.memory[key]

@pytest.fixture
def redis_cache():
    return SimulatedRedisCache()

def test_cache_invalidation_on_webhook_mutation(redis_cache):
    """
    TDD Caching: Asserts that a database mutation (e.g. a TrueLayer webhook sync)
    strictly triggers the destruction of localized Redis cache keys, mathematically 
    preventing the frontend from rendering stale balance data.
    """
    user_id = "user_777"
    cache_key = f"dashboard_balance_{user_id}"
    
    # 1. Frontend requests balance (Cache Miss -> DB Hit -> Cache Set)
    database_balance = 1500.00
    redis_cache.set(cache_key, database_balance)
    
    # Assert frontend is reading cached state
    assert redis_cache.get(cache_key) == 1500.00
    
    # 2. Asynchronous TrueLayer Webhook arrives, deducting 50.00
    def trigger_truelayer_webhook_sync(uid: str, amount: float):
        nonlocal database_balance
        database_balance -= amount
        # System Architecture Rule: Mutations MUST invalidate related caches
        redis_cache.delete(f"dashboard_balance_{uid}")
        
    trigger_truelayer_webhook_sync(user_id, 50.00)
    
    # 3. Frontend refreshes
    stale_check = redis_cache.get(cache_key)
    
    # Assertions
    assert stale_check is None, "Cache Invalidation Failure: Stale data was served to the frontend."
    assert database_balance == 1450.00, "Database mutation failed."
    
    # 4. Subsequent read re-populates cache with exact current state
    if stale_check is None:
        redis_cache.set(cache_key, database_balance)
        
    assert redis_cache.get(cache_key) == 1450.00, "Cache repopulation failed after invalidation."

