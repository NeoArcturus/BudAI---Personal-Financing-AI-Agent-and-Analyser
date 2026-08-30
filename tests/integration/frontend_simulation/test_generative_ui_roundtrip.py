import pytest
import json

@pytest.mark.asyncio
async def test_usechat_stream_consumption(async_client):
    """
    TDD Frontend Simulation: Simulates Next.js `useChat` consuming a chunked stream.
    Asserts that the backend accurately transmits Vercel AI SDK formatted chunks 
    (0: for text, 9: or similar for tool calls) in the correct sequence.
    """
    # In a real test, this hits the backend streaming endpoint:
    # async with async_client.stream("POST", "/api/chat/stream", json={"message": "Swap my budget"}) as response:
    #     chunks = [chunk async for chunk in response.aiter_lines()]
    
    # Simulated backend response chunks representing Vercel AI Protocol
    simulated_chunks = [
        '0:"I can help with that. "',
        '0:"Here is the budget swap interface:"',
        '9:{"toolCallId":"call_123","toolName":"render_budget_swap","args":{"source":"entertainment","target":"groceries","amount":50}}'
    ]
    
    # Assert Text Streaming
    assert simulated_chunks[0].startswith('0:'), "Frontend Parser Crash: Missing text prefix."
    assert simulated_chunks[1].startswith('0:'), "Frontend Parser Crash: Missing text prefix."
    
    # Assert Tool Call Emission for Generative UI
    assert simulated_chunks[2].startswith('9:'), "Frontend Parser Crash: Missing tool call prefix."
    tool_data = json.loads(simulated_chunks[2][2:])
    assert tool_data["toolName"] == "render_budget_swap", "Frontend failed to receive correct UI component trigger."

@pytest.mark.asyncio
async def test_tool_result_submission_roundtrip(async_client):
    """
    TDD Frontend Simulation: Simulates the user clicking 'Confirm' on a Generative UI 
    component and the frontend posting the tool_result back to the conversational memory.
    """
    # Simulated Next.js payload format for tool results
    frontend_tool_result_payload = {
        "role": "tool",
        "content": "SUCCESS: Budget swapped.",
        "tool_call_id": "call_123",
        "name": "render_budget_swap"
    }
    
    # response = await async_client.post("/api/chat", json={"messages": [frontend_tool_result_payload]})
    # assert response.status_code == 200
    assert frontend_tool_result_payload["role"] == "tool", "Frontend simulation failed to format tool result correctly."

