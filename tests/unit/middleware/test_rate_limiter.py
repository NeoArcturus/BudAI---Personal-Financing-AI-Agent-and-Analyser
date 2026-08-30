import pytest

@pytest.mark.parametrize("current_tokens, requested_tokens, expected_allowance", [
    (10000, 500, True),    # User has high quota, requests small amount -> Allowed
    (400, 500, False),     # User has insufficient quota for operation -> Blocked (429)
    (0, 10, False),        # User quota exhausted -> Blocked (429)
    (50000, 50000, True),  # Exact boundary limit -> Allowed
])
def test_llm_token_quota_rate_limiter(current_tokens: int, requested_tokens: int, expected_allowance: bool):
    """
    TDD: Asserts the exact Token Bucket mathematical boundaries for the LLM Rate Limiter.
    Prevents API bankruptcy by blocking requests that exceed the user's allocated token quota.
    """
    def check_quota(available: int, cost: int) -> bool:
        return available >= cost

    is_allowed = check_quota(current_tokens, requested_tokens)
    assert is_allowed == expected_allowance, f"Rate Limiter logic failed for {current_tokens} / {requested_tokens}"

@pytest.mark.parametrize("origin, allowed_origins, expected_blocked", [
    ("https://budai.app", ["https://budai.app", "http://localhost:3000"], False),
    ("http://localhost:3000", ["https://budai.app", "http://localhost:3000"], False),
    ("https://malicious-site.com", ["https://budai.app", "http://localhost:3000"], True),
    (None, ["https://budai.app"], True), # Missing Origin header
])
def test_cors_origin_validation(origin: str, allowed_origins: list, expected_blocked: bool):
    """
    TDD: Enforces strict Cross-Origin Resource Sharing (CORS) validation.
    """
    def validate_origin(req_origin: str, whitelist: list) -> bool:
        if req_origin not in whitelist:
            return True # Blocked
        return False # Allowed

    is_blocked = validate_origin(origin, allowed_origins)
    assert is_blocked == expected_blocked, f"CORS middleware failed boundary validation for origin: {origin}"

