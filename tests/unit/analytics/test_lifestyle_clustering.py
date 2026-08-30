import pytest

@pytest.mark.parametrize("income, expenses, expected_score, expected_layout", [
    (4000.0, 2000.0, 50.0, "GROWTH"),
    (4000.0, 3000.0, 25.0, "SURVIVAL"), # Below 50 is survival
    (2000.0, 6000.0, 0.0, "SURVIVAL"),  # Extreme deficit must clamp to 0.0
    (5000.0, 0.0, 100.0, "GROWTH"),     # Extreme surplus must clamp to 100.0
])
def test_financial_health_score_clamping_and_layout(income, expenses, expected_score, expected_layout):
    """
    TDD Analytics: Asserts mathematical bounding (clamping) of the Financial Health Score
    and strict mapping to UI layout states.
    """
    def calculate_health(inc: float, exp: float):
        if inc <= 0: return 0.0, "SURVIVAL"
        score = ((inc - exp) / inc) * 100
        
        # Mathematical Clamping
        score = max(0.0, min(100.0, score))
        layout = "GROWTH" if score >= 50.0 else "SURVIVAL"
        return score, layout

    score, layout = calculate_health(income, expenses)
    assert score == pytest.approx(expected_score, 0.1), "Health score mathematical clamp failed."
    assert layout == expected_layout, "Layout toggle state mapped incorrectly."

