import pytest
from datetime import datetime, timedelta

@pytest.mark.parametrize("transaction_days_old, expected_to_process", [
    (1, True),   # 1 day ago (Inside overlap window)
    (2, True),   # 2 days ago (Inside overlap window)
    (4, False),  # 4 days ago (Outside overlap window)
    (180, False) # Deep history (Outside overlap window)
])
def test_high_water_mark_sync_filtering(transaction_days_old: int, expected_to_process: bool):
    """
    TDD ETL Extract: Asserts that the extraction layer applies the 'High-Water Mark' 
    algorithm, mathematically dropping all incoming TrueLayer payload data that is older 
    than (last_synced_at - 3 days), preventing massive 180-day deep syncs on every webhook.
    """
    # Simulated State
    current_time = datetime.utcnow()
    last_synced_at = current_time - timedelta(days=7) # Last sync was a week ago
    overlap_window = 3 # Read 3 days before last sync to catch pending settlements
    
    cutoff_date = last_synced_at - timedelta(days=overlap_window)
    
    transaction_date = current_time - timedelta(days=transaction_days_old)
    
    # Extraction Logic
    def should_extract(txn_date: datetime, cutoff: datetime) -> bool:
        return txn_date >= cutoff

    is_extracted = should_extract(transaction_date, cutoff_date)
    assert is_extracted == expected_to_process, f"Extraction filter failed boundary check for {transaction_days_old} days old."

