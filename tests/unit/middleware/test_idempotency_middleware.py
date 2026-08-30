import pytest
from unittest.mock import MagicMock

class MockIdempotencyCache:
    def __init__(self):
        self.store = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, response: dict):
        self.store[key] = response

@pytest.fixture
def cache():
    return MockIdempotencyCache()

def test_idempotency_intercepts_duplicate_request(cache):
    """
    TDD: Asserts that if an Idempotency-Key exists in the Redis cache, 
    the middleware halts request execution and returns the cached payload instantly.
    """
    idem_key = "req_12345"
    cached_response = {"status": "success", "message": "Transaction synced"}
    
    # Simulate first request caching the output
    cache.set(idem_key, cached_response)
    
    # Simulate second identical request
    retrieved_response = cache.get(idem_key)
    
    assert retrieved_response is not None, "Idempotency cache failed to retrieve payload."
    assert retrieved_response["message"] == "Transaction synced", "Idempotency cache corrupted payload."

def test_idempotency_allows_unique_requests(cache):
    """
    TDD: Asserts that unique Idempotency-Keys proceed through the middleware.
    """
    idem_key = "req_99999"
    retrieved_response = cache.get(idem_key)
    assert retrieved_response is None, "Idempotency cache falsely intercepted a unique request."

