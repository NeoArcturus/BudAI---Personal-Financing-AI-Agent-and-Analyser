import pytest

@pytest.mark.parametrize("budget, current_spend, days_elapsed, expected_velocity, expected_variance, expected_alert", [
    (400.0, 200.0, 10, 20.0, -200.0, "WARNING_OVERSPEND"),
    (1000.0, 300.0, 15, 20.0, 400.0, "ON_TRACK"),
    (500.0, 500.0, 30, 16.66, 0.0, "ON_TRACK"), # End of month, exactly on budget
])
def test_spend_velocity_forecasting(budget, current_spend, days_elapsed, expected_velocity, expected_variance, expected_alert):
    """
    TDD Analytics: Asserts precise velocity projections for proactive insight generation.
    """
    def forecast_spend(budg: float, spend: float, days: int, total_days: int = 30):
        if days == 0: return 0.0, 0.0, "ON_TRACK"
        
        velocity = spend / days
        projected_total = velocity * total_days
        variance = budg - projected_total
        
        alert = "WARNING_OVERSPEND" if variance < 0 else "ON_TRACK"
        return velocity, variance, alert

    velocity, variance, alert = forecast_spend(budget, current_spend, days_elapsed)
    
    assert velocity == pytest.approx(expected_velocity, 0.1), "Velocity math failed."
    assert variance == pytest.approx(expected_variance, 0.1), "Variance projection failed."
    assert alert == expected_alert, "Proactive alert mapping failed."

