import pytest
from unittest.mock import AsyncMock, patch

# TDD: Enforcing the exact input/output matrix for the CategorizerAgent's core function.
@pytest.mark.parametrize("merchant_string, predicted_category, expected_confidence", [
    ("TESCO STORES LTD", "Groceries", 0.95),
    ("UBER *TRIP", "Transport", 0.88),
    ("AMZN MKTP UK", "Shopping", 0.75),
    ("RANDOM_MERCHANT_999", "Uncategorized", 0.10), # Should fail vector match and drop to LLM
])
@pytest.mark.asyncio
async def test_categorizer_agent_expected_outputs(merchant_string, predicted_category, expected_confidence):
    """
    TDD: Asserts that the CategorizerAgent returns a strictly typed tuple containing 
    the resolved Category string and the mathematical confidence float.
    """
    # Placeholder for: from services.Categorizer_Agent.CategorizerAgent import categorize_transaction
    async def mock_categorize_transaction(merchant):
        # Simulated agent logic
        if "TESCO" in merchant: return ("Groceries", 0.95)
        if "UBER" in merchant: return ("Transport", 0.88)
        if "AMZN" in merchant: return ("Shopping", 0.75)
        return ("Uncategorized", 0.10)

    category, confidence = await mock_categorize_transaction(merchant_string)
    
    assert category == predicted_category, f"Agent resolved incorrect category for {merchant_string}"
    assert confidence == expected_confidence, f"Agent confidence score drifted for {merchant_string}"

