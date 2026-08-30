import pytest
import json
import re

def robust_json_parse(raw_string: str) -> dict:
    """Simulated recovery parser."""
    try:
        return json.loads(raw_string)
    except json.JSONDecodeError:
        # Regex to salvage partial JSON
        match = re.search(r'\{.*\}', raw_string.replace('\n', ''))
        if match:
            try:
                # Attempt to append missing bracket if broken
                repaired = match.group(0)
                if repaired.count('{') > repaired.count('}'):
                    repaired += '}'
                return json.loads(repaired)
            except:
                pass
        raise ValueError("Unrecoverable LLM hallucination")

@pytest.mark.parametrize("llm_output, expected_key", [
    ('{"tool": "swap", "amount": 50}', "tool"), # Perfect JSON
    ('Here is the json: {"tool": "swap", "amount": 50}', "tool"), # Prefix conversational text
    ('{"tool": "swap", "amount": 50', "tool"), # Missing trailing bracket
])
def test_llm_json_recovery_regex(llm_output, expected_key):
    parsed = robust_json_parse(llm_output)
    assert expected_key in parsed

def test_llm_unrecoverable_hallucination():
    with pytest.raises(ValueError, match="Unrecoverable LLM hallucination"):
        robust_json_parse("I cannot perform this action.")
