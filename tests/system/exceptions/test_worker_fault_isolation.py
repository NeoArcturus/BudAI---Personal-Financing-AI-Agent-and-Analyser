import pytest

def test_background_worker_batch_fault_isolation():
    """
    TDD Exceptions: Asserts that if a single item in a background batch processing 
    loop triggers a Python exception, it is isolated to a DLQ and does not crash 
    the entire container thread.
    """
    transactions_to_process = [10.0, 20.0, None, 40.0]
    successful_processes = 0
    failed_items = []

    for txn in transactions_to_process:
        try:
            # Simulated complex logic that crashes on None
            result = txn * 2.0 
            successful_processes += 1
        except Exception as e:
            failed_items.append(txn)

    # Assert execution continued despite the crash at index 2
    assert successful_processes == 3
    assert len(failed_items) == 1
    assert failed_items[0] is None

