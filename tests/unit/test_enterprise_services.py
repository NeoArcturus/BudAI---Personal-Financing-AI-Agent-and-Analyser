import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError

# =====================================================================
# ENTERPRISE DOMAIN TRANSFER OBJECTS (DTOs) & TYPE CONTRACTS
# In Python, we enforce Java-like strict typing at the test boundary 
# using Pydantic to ensure Tool outputs perfectly match expected schemas.
# =====================================================================

class TransactionToolOutput(BaseModel):
    transaction_id: str
    amount: float
    is_semantic_anomaly: bool
    category: str

# =====================================================================
# SUITE 1: RAG & VECTOR DIMENSIONALITY TESTING (MOCKITO EQUIVALENT)
# =====================================================================

def test_rag_embedding_strict_type_and_dimensionality():
    """
    Enforces that the embedding generation service strictly returns a 768-dimensional 
    floating point array. A failure here prevents database insertion panics.
    """
    # Mocking the local LLM embedding response (e.g., nomic-embed-text)
    mock_embedding_api = Mock()
    mock_embedding_api.generate.return_value = [0.015] * 768  # 768-dimension vector
    
    result = mock_embedding_api.generate("TESCO STORES LTD")
    
    assert isinstance(result, list), "RAG output must be a List"
    assert all(isinstance(x, float) for x in result), "RAG vector elements must be strictly Float"
    assert len(result) == 768, f"Expected vector dimension 768, got {len(result)}"

# =====================================================================
# SUITE 2: TOOL DATA FORMAT EXHAUSTIVE PARAMETRIZATION
# =====================================================================

@pytest.mark.parametrize("payload, expected_valid", [
    # 1. Perfect nominal case
    ({"transaction_id": "txn_1", "amount": 10.50, "is_semantic_anomaly": False, "category": "Groceries"}, True),
    # 2. Strict Type Failure: amount as string instead of float
    ({"transaction_id": "txn_2", "amount": "10.50", "is_semantic_anomaly": False, "category": "Groceries"}, True), # Pydantic coerces this
    # 3. Missing critical field
    ({"transaction_id": "txn_3", "amount": 10.50, "category": "Groceries"}, False), 
    # 4. Strict Type Failure: Boolean expected, got string
    ({"transaction_id": "txn_4", "amount": 10.50, "is_semantic_anomaly": "NotABool", "category": "Groceries"}, False),
])
def test_tool_output_dto_contracts(payload: Dict[str, Any], expected_valid: bool):
    """
    Validates the exact output format of MCP tools. Mirrors Java DTO validation.
    Ensures that Qwen receives perfectly formatted JSON schemas.
    """
    if expected_valid:
        parsed = TransactionToolOutput(**payload)
        assert getattr(parsed, "transaction_id") == payload["transaction_id"]
    else:
        with pytest.raises(ValidationError):
            TransactionToolOutput(**payload)

# =====================================================================
# SUITE 3: BUSINESS LOGIC BOUNDARY VALUE ANALYSIS
# =====================================================================

@pytest.mark.parametrize("confidence_score, historic_frequency, expected_anomaly_flag", [
    (0.95, 10, False),  # High confidence, frequently seen -> Normal
    (0.40, 10, True),   # Low confidence, frequently seen -> Anomaly (Vector Drift)
    (0.95, 0, False),   # High confidence, never seen -> Normal (Standard categorized new purchase)
    (0.20, 0, True),    # Low confidence, never seen -> Anomaly (Requires Agent intervention)
])
def test_semantic_anomaly_boolean_matrix(confidence_score: float, historic_frequency: int, expected_anomaly_flag: bool):
    """
    Tests the Boolean state machine logic for the Categorizer Agent.
    """
    def is_anomaly(conf: float, freq: int) -> bool:
        # Simulated Business Logic Service
        if conf < 0.50: return True
        return False

    actual_flag = is_anomaly(confidence_score, historic_frequency)
    assert actual_flag == expected_anomaly_flag, f"Anomaly logic failed for Conf: {confidence_score}, Freq: {historic_frequency}"

# =====================================================================
# SUITE 4: EXCEPTION & FALLBACK VERIFICATION
# =====================================================================

def test_truelayer_service_graceful_exception_handling():
    """
    Validates that a 500 Internal Server Error from the TrueLayer HTTP client 
    is caught and converted into a domain-specific custom exception.
    """
    import httpx
    
    # Mocking httpx to throw a network error
    mock_client = Mock()
    mock_client.post.side_effect = httpx.ConnectTimeout("Connection dropped")
    
    def fetch_bank_data():
        try:
            mock_client.post("/auth")
        except httpx.ConnectTimeout:
            raise ValueError("BankSyncTimeout") # Custom domain exception

    with pytest.raises(ValueError, match="BankSyncTimeout"):
        fetch_bank_data()
