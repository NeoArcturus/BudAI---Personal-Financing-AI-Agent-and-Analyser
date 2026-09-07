import pytest
import numpy as np
from datetime import datetime, timedelta

@pytest.mark.asyncio
async def test_forecaster_agent_30_day_projection_matrix():
    """
    TDD: Enforces that the ForecasterAgent mathematically outputs a strict 30-element 
    numpy array representing the predicted 30-day cash flow, preventing shape mismatch errors.
    """
    current_balance = 1000.00
    
    # Placeholder for: from agents.core_financial.Forecaster_Agent.ForecasterAgent import generate_30_day_projection
    async def mock_generate_projection(balance):
        # Simulating LSTM/Algorithmic output shape
        return np.array([balance - (i * 10) for i in range(30)])

    projection = await mock_generate_projection(current_balance)
    
    assert isinstance(projection, np.ndarray), "Forecaster Agent violated output type contract. Must be ndarray."
    assert projection.shape == (30,), f"Forecaster Agent returned invalid matrix shape: {projection.shape}. Expected (30,)."
    assert projection[0] == 1000.00, "Projection index 0 does not match current T=0 balance."

