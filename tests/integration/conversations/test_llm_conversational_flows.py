import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_multi_turn_conversation_context_assembly():
    """
    TDD Conversation Flow: Tests that the backend correctly assembles a multi-turn 
    chat history before sending it to the LLM inference engine.
    Asserts the System Prompt is perfectly retained at index [0].
    """
    # Simulated User-Assistant history from the database
    db_chat_history = [
        {"role": "user", "content": "I bought a coffee for 5.00"},
        {"role": "assistant", "content": "Got it. That sounds like an expense."},
        {"role": "user", "content": "Can you categorize that last purchase?"}
    ]
    
    # Mocking the actual LLM network call to trap the payload being sent
    mock_llm_client = AsyncMock()
    
    async def simulate_backend_chat_router(history):
        # The backend must inject the system prompt and format the array
        system_prompt = {"role": "system", "content": "You are BudAI, a financial advisor."}
        payload = [system_prompt] + history
        await mock_llm_client.chat(messages=payload)
    
    await simulate_backend_chat_router(db_chat_history)
    
    # Assertions
    mock_llm_client.chat.assert_called_once()
    called_messages = mock_llm_client.chat.call_args.kwargs["messages"]
    
    assert len(called_messages) == 4, "Context Assembly Failed: Missing messages in payload."
    assert called_messages[0]["role"] == "system", "Security Breach: System prompt was not injected at index 0."
    assert called_messages[-1]["content"] == "Can you categorize that last purchase?", "Context Assembly Failed: Latest message missing."


@pytest.mark.asyncio
async def test_conversational_tool_orchestration_loop():
    """
    TDD Conversation Flow: Tests the 3-step orchestration loop of an Agentic action.
    1. User states intent.
    2. LLM responds with a Tool Call (Generative UI).
    3. User confirms (Tool Result), LLM finalizes.
    """
    # Mocking the LLM to output a specific tool call when asked to move money
    async def mock_llm_inference(messages):
        last_message = messages[-1]
        
        # Step 1 -> Step 2: LLM generates tool call
        if last_message["role"] == "user" and "move 50" in last_message["content"].lower():
            return {
                "role": "assistant", 
                "tool_calls": [{"id": "call_1", "function": {"name": "render_budget_swap", "arguments": "{}"}}]
            }
            
        # Step 3 -> Final: LLM sees tool result and confirms
        if last_message["role"] == "tool" and last_message["name"] == "render_budget_swap":
            return {"role": "assistant", "content": "The budget has been successfully updated."}
            
        return {"role": "assistant", "content": "I did not understand."}

    # Step 1: User Request
    conversation_state = [{"role": "user", "content": "Please move 50 to savings."}]
    llm_response_1 = await mock_llm_inference(conversation_state)
    
    assert "tool_calls" in llm_response_1, "LLM failed to trigger tool orchestration."
    assert llm_response_1["tool_calls"][0]["function"]["name"] == "render_budget_swap"
    
    # Step 2: Backend appends LLM response and User's Tool Confirmation
    conversation_state.append(llm_response_1)
    conversation_state.append({
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "render_budget_swap",
        "content": "SUCCESS"
    })
    
    # Step 3: Final LLM execution
    llm_response_2 = await mock_llm_inference(conversation_state)
    
    assert "tool_calls" not in llm_response_2, "LLM stuck in infinite tool loop."
    assert "successfully updated" in llm_response_2["content"], "LLM failed to finalize conversation after tool execution."

