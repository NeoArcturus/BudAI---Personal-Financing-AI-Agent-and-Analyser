import pytest
import json

def test_llm_tool_call_parser_handles_hallucinated_tools():
    """TDD: Tests the defensive boundary around the Qwen model. If the LLM hallucinates a tool that does not exist, the parser must catch it gracefully."""
    hallucinated_payload = {
        "name": "delete_all_user_data",
        "arguments": "{}"
    }
    
    valid_tools = ["propose_budget_swap", "get_safe_balance"]
    
    # from services.llm.parser import parse_tool_call
    # with pytest.raises(ValueError):
    #     parse_tool_call(hallucinated_payload, valid_tools)
    
    assert hallucinated_payload["name"] in valid_tools, "LLM Parser Security Breach: Allowed execution of hallucinated/unregistered tool"

def test_llm_json_recovery():
    """TDD: Tests if the system can recover when the local LLM forgets a trailing bracket in its JSON output."""
    malformed_json = '{"arguments": {"category": "Dining"' # Missing closing brackets
    
    # from services.llm.parser import robust_json_parse
    # result = robust_json_parse(malformed_json)
    
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed_json)
    # The actual test will assert that our custom parser fixes this and DOES NOT raise the decode error.
