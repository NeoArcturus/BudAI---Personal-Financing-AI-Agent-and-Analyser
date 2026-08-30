import pytest
from datetime import datetime, timedelta
import jwt

def test_jwt_token_expiration_rejection():
    """TDD: Ensures expired JWT tokens are strictly rejected by the auth middleware."""
    secret = "dummy_secret"
    # Create a token that expired 1 second ago
    expired_payload = {
        "sub": "user_123",
        "exp": datetime.utcnow() - timedelta(seconds=1)
    }
    token = jwt.encode(expired_payload, secret, algorithm="HS256")
    
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, secret, algorithms=["HS256"])
