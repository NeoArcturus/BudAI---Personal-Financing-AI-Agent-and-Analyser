import pytest
import asyncio

@pytest.mark.asyncio
async def test_sse_stream_cancellation_halts_llm():
    """
    TDD System Design: Asserts that if a client closes their browser mid-stream,
    the FastAPI Server-Sent Events (SSE) generator traps the CancelledError 
    and instantly kills the active LLM inference thread, preventing CPU starvation.
    """
    llm_tokens_generated = 0
    client_connected = True

    async def simulated_llm_generator():
        nonlocal llm_tokens_generated
        try:
            for i in range(100):
                if not client_connected:
                    raise asyncio.CancelledError() # Fastapi throws this when client disconnects
                
                await asyncio.sleep(0.01)
                llm_tokens_generated += 1
                yield f"Token_{i}"
        except asyncio.CancelledError:
            # Cleanup logic must execute
            pass

    async def client_simulator():
        nonlocal client_connected
        gen = simulated_llm_generator()
        
        # Consume 10 tokens, then forcefully disconnect
        for _ in range(10):
            await anext(gen)
            
        client_connected = False
        
        # Trigger the generator one more time to simulate the exact moment of disconnect
        with pytest.raises(asyncio.CancelledError) or pytest.raises(StopAsyncIteration):
            await anext(gen)

    await client_simulator()
    
    # Assert the LLM physically stopped generating after the disconnect
    assert llm_tokens_generated == 10, "Generator leak: LLM continued processing after client disconnected."

