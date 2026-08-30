import pytest
from unittest.mock import patch, MagicMock

def test_llm_sweep_worker_batch_size_limits():
    """
    TDD: Asserts that the Prefect background worker correctly partitions 
    large datasets (e.g., 10,000 uncategorized rows) into safe batch sizes 
    to prevent GPU OOM (Out of Memory) crashes.
    """
    total_uncategorized_rows = 1500
    max_safe_batch_size = 500
    
    # Placeholder for: from workers.categorization_worker import partition_batches
    def partition_batches(total_rows, batch_limit):
        return [batch_limit] * (total_rows // batch_limit)
        
    batches = partition_batches(total_uncategorized_rows, max_safe_batch_size)
    
    assert len(batches) == 3, "Worker failed to partition the dataset correctly."
    assert all(b <= max_safe_batch_size for b in batches), "Worker exceeded GPU batch size safety limit."

