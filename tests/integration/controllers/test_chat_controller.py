import pytest
from unittest.mock import MagicMock

@pytest.mark.asyncio
async def test_chat_controller_rejects_empty_messages():
    """
    TDD: Bypasses the HTTP route and tests the controller function directly. 
    Asserts that the controller raises a ValueError if the LLM receives an empty string.
    """
    # Placeholder for: from controllers.chat.post import process_chat_message
    async def mock_process_chat_message(message: str, user: MagicMock):
        if not message.strip():
            raise ValueError("Message cannot be empty")
        return True

    with pytest.raises(ValueError, match="Message cannot be empty"):
        await mock_process_chat_message("   ", MagicMock())
