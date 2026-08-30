import pytest

def calculate_variance(budgeted: float, actual: float) -> float:
    if budgeted == 0:
        if actual > 0: return -100.0 # Infinite overrun mathematically capped
        return 0.0
    return ((budgeted - actual) / budgeted) * 100.0

@pytest.mark.parametrize("budgeted, actual, expected_variance", [
    (100.0, 50.0, 50.0),    # Under budget by 50%
    (100.0, 150.0, -50.0),  # Over budget by 50%
    (100.0, 100.0, 0.0),    # Exactly on budget
    (0.0, 50.0, -100.0),    # Zero-division trap (Budget 0, spent 50)
    (0.0, 0.0, 0.0),        # Zero-division trap (Budget 0, spent 0)
])
def test_budget_variance_calculation_and_zero_division(budgeted, actual, expected_variance):
    variance = calculate_variance(budgeted, actual)
    assert variance == expected_variance

def test_context_window_truncation():
    """Asserts sliding window drops oldest messages while keeping system prompt."""
    history = [{"role": "system", "content": "You are BudAI"}]
    for i in range(100):
        history.append({"role": "user", "content": f"Message {i}"})
    
    def truncate_history(hist, max_len=10):
        system_prompt = [msg for msg in hist if msg["role"] == "system"]
        recent = hist[-(max_len-1):] if len(hist) > max_len else hist[1:]
        return system_prompt + recent
        
    truncated = truncate_history(history, 10)
    assert len(truncated) == 10
    assert truncated[0]["role"] == "system"
    assert truncated[1]["content"] == "Message 91"
