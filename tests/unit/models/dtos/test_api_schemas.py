import pytest
from pydantic import BaseModel, Field, ValidationError

# Simulated Schemas
class ChatRequestSchema(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str

@pytest.mark.parametrize("payload, expected_valid", [
    ({"message": "Hello", "session_id": "sess_123"}, True),
    ({"message": "", "session_id": "sess_123"}, False), # Violates min_length
    ({"message": "A" * 1001, "session_id": "sess_123"}, False), # Violates max_length
    ({"message": "Hello"}, False), # Missing required session_id
    ({"message": None, "session_id": "sess_123"}, False), # Null message
])
def test_chat_request_schema_boundaries(payload, expected_valid):
    if expected_valid:
        parsed = ChatRequestSchema(**payload)
        assert parsed.session_id == payload["session_id"]
    else:
        with pytest.raises(ValidationError):
            ChatRequestSchema(**payload)
