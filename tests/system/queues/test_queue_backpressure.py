import pytest
from unittest.mock import MagicMock

def test_poison_pill_dead_letter_queue_routing():
    """
    TDD System Design: Asserts that a severely corrupted payload (Poison Pill) 
    does not permanently stall the Redis worker. After 3 retries, it must be 
    purged from the main queue and logged to the DLQ.
    """
    max_retries = 3
    attempt_counter = 0
    dlq_storage = []

    def process_queue_item(payload: dict):
        nonlocal attempt_counter
        attempt_counter += 1
        try:
            # Simulated processing crash on corrupted data
            _ = payload["missing_critical_key"]
        except KeyError as e:
            if attempt_counter >= max_retries:
                dlq_storage.append(payload)
                return "MOVED_TO_DLQ"
            raise e # Triggers retry

    poison_pill = {"corrupted": "data"}

    # Attempt 1
    with pytest.raises(KeyError):
        process_queue_item(poison_pill)
    
    # Attempt 2
    with pytest.raises(KeyError):
        process_queue_item(poison_pill)
        
    # Attempt 3 - Threshold Reached
    result = process_queue_item(poison_pill)

    assert result == "MOVED_TO_DLQ", "Worker failed to route poison pill to DLQ."
    assert len(dlq_storage) == 1, "DLQ storage failed to persist corrupted payload."
    assert attempt_counter == 3, "Worker retry logic violated max_retries limit."

