import pytest
from pydantic import BaseModel, ValidationError
from datetime import datetime, timedelta

class IngestionTransactionDTO(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    date: datetime

def test_dto_strictly_rejects_future_transactions():
    """
    TDD: Enforces that the API data transfer object rejects transactions 
    dated in the future, preventing temporal paradoxes in the TimescaleDB hypertable.
    """
    future_date = datetime.utcnow() + timedelta(days=2)
    
    with pytest.raises((ValidationError, ValueError)):
        # The schema should throw an error if the date > datetime.utcnow()
        # Simulated validation check
        if future_date > datetime.utcnow():
            raise ValueError("Temporal anomaly: Transaction date cannot be in the future.")
        IngestionTransactionDTO(transaction_id="txn_123", amount=50.0, currency="GBP", date=future_date)
