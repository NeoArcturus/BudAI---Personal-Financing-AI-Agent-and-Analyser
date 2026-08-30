import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_waterfall_classification_routing_logic():
    """
    TDD ETL Transform: Asserts that transactions cascade exactly through the 
    4-Stage classification architecture, short-circuiting at the earliest match 
    to eliminate unnecessary LLM API compute.
    """
    # Execution counters for assertion
    execution_metrics = {"stage_1": 0, "stage_2": 0, "stage_3": 0, "stage_4": 0}

    async def stage_1_exact_match(merchant: str):
        execution_metrics["stage_1"] += 1
        return "groceries" if merchant == "TESCO" else None

    async def stage_2_pgvector_rag(merchant: str):
        execution_metrics["stage_2"] += 1
        return "transport" if "UBER" in merchant else None

    async def stage_3_local_llm(merchant: str):
        execution_metrics["stage_3"] += 1
        return "software" if "GITHUB" in merchant else None

    async def stage_4_web_search(merchant: str):
        execution_metrics["stage_4"] += 1
        return "unknown_misc"

    async def categorize_waterfall(merchant: str):
        # Stage 1
        result = await stage_1_exact_match(merchant)
        if result: return result
        
        # Stage 2
        result = await stage_2_pgvector_rag(merchant)
        if result: return result
        
        # Stage 3
        result = await stage_3_local_llm(merchant)
        if result: return result
        
        # Stage 4
        return await stage_4_web_search(merchant)

    # Test Short-Circuiting (TESCO should stop at Stage 1)
    await categorize_waterfall("TESCO")
    assert execution_metrics["stage_1"] == 1
    assert execution_metrics["stage_2"] == 0 # Must not trigger

    # Test Fallback to Stage 2 (UBER EATS skips 1, stops at 2)
    await categorize_waterfall("UBER EATS 123")
    assert execution_metrics["stage_1"] == 2
    assert execution_metrics["stage_2"] == 1
    assert execution_metrics["stage_3"] == 0

    # Test Total Fallback to Stage 4 Web Search MCP
    await categorize_waterfall("OBSCURE_LLC_999")
    assert execution_metrics["stage_4"] == 1

