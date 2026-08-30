import pytest
import re

# Simulated Output Sanitizer Middleware
class LLMOutputSanitizer:
    @staticmethod
    def strip_emojis(text: str) -> str:
        # Basic regex pattern catching common emojis to enforce the NO EMOJI rule
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def detect_uuid_leakage(text: str) -> bool:
        # Standard UUIDv4 Regex
        uuid_pattern = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE)
        return bool(uuid_pattern.search(text))

    @staticmethod
    def detect_system_terms(text: str) -> bool:
        forbidden_terms = ["CategorizerAgent", "ForecasterAgent", "pgvector", "SQLAlchemy", "TimescaleDB"]
        return any(term.lower() in text.lower() for term in forbidden_terms)


@pytest.mark.parametrize("llm_response, expected_leak", [
    ("I have updated your budget.", False),
    ("I modified record f47ac10b-58cc-4372-a567-0e02b2c3d479 in the ledger.", True),
    ("The transaction UUID is 123e4567-e89b-12d3-a456-426614174000.", True),
])
def test_output_sanitizer_prevents_database_id_leakage(llm_response: str, expected_leak: bool):
    """
    TDD Security: Asserts that if the LLM hallucinates or attempts to display 
    raw internal Database UUIDs to the frontend client, the output sanitizer flags it.
    """
    is_leaking = LLMOutputSanitizer.detect_uuid_leakage(llm_response)
    assert is_leaking == expected_leak, f"Sanitizer failed UUID boundary check for payload: {llm_response}"


@pytest.mark.parametrize("llm_response, expected_leak", [
    ("I will analyze this transaction.", False),
    ("I am passing this to the CategorizerAgent for processing.", True),
    ("Let me search pgvector for similar merchants.", True),
])
def test_output_sanitizer_prevents_system_term_leakage(llm_response: str, expected_leak: bool):
    """
    TDD Security: Asserts that internal architectural terms (Classes, Agents, DB engines) 
    are strictly forbidden from being leaked to the client UI.
    """
    is_leaking = LLMOutputSanitizer.detect_system_terms(llm_response)
    assert is_leaking == expected_leak, f"Sanitizer failed system term check for payload: {llm_response}"


@pytest.mark.parametrize("llm_response, expected_cleaned", [
    ("Your budget is on track.", "Your budget is on track."),
    ("You saved £50 today! 🎉", "You saved £50 today! "),
    ("Warning: Overdraft imminent ⚠️📉", "Warning: Overdraft imminent "),
])
def test_output_sanitizer_enforces_strict_formatting_rules(llm_response: str, expected_cleaned: str):
    """
    TDD Security: Enforces Master Rule 2 (NO EMOJIS). Asserts that the sanitizer 
    mathematically strips forbidden characters from the LLM output before it hits the frontend.
    """
    cleaned_response = LLMOutputSanitizer.strip_emojis(llm_response)
    assert cleaned_response == expected_cleaned, "Sanitizer failed to enforce strict character formatting rules."

