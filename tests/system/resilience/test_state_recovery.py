import pytest

def test_network_partition_replay_attack():
    """
    TDD System Design: A client submits a payload, the DB updates, but the network 
    drops before the 200 OK reaches the client. The client retries 5 minutes later.
    Asserts the Idempotency engine prevents a duplicate database insertion.
    """
    db_insert_count = 0
    idempotency_cache = set()

    def process_webhook(idempotency_key: str, payload: dict):
        nonlocal db_insert_count
        if idempotency_key in idempotency_cache:
            return 202 # Cached Response, DB untouched
        
        # New Request
        db_insert_count += 1
        idempotency_cache.add(idempotency_key)
        return 200

    idem_key = "hash_xyz123"
    payload = {"amount": 500.0}

    # First request arrives
    status_1 = process_webhook(idem_key, payload)
    
    # Network dies. Client retries exact same payload 5 minutes later.
    status_2 = process_webhook(idem_key, payload)

    assert status_1 == 200, "First request failed to process."
    assert status_2 == 202, "Idempotency engine failed to trap replay attack."
    assert db_insert_count == 1, "Replay attack bypassed cache, corrupting database state."

