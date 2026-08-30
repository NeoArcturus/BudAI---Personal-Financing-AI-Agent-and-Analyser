import pytest
from datetime import datetime
import statistics

@pytest.mark.parametrize("date_strings, expected_is_subscription", [
    (["2023-01-01", "2023-02-01", "2023-03-01"], True), # StdDev 1.5 <= 2.0
    (["2023-01-01", "2023-01-06", "2023-02-17"], False), # Highly erratic
])
def test_subscription_interval_standard_deviation(date_strings, expected_is_subscription):
    """
    TDD Analytics: Asserts exact standard deviation calculations to detect recurring payments.
    """
    def is_recurring(dates: list[str]) -> bool:
        dt_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        intervals = [(dt_objs[i+1] - dt_objs[i]).days for i in range(len(dt_objs)-1)]
        std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0
        return std_dev <= 2.0

    result = is_recurring(date_strings)
    assert result == expected_is_subscription, "Statistical variance model failed detection boundary."

@pytest.mark.parametrize("amounts, threshold, expected_flag, expected_delta_pct, expected_raw_delta", [
    ([10.00, 10.00, 10.00, 15.00], 10.0, True, 50.0, 5.00),
    ([20.00, 20.00, 20.00, 20.00], 10.0, False, 0.0, 0.0),
    ([50.00, 50.00, 50.00, 52.00], 10.0, False, 4.0, 2.00), # Hike is below 10% threshold
])
def test_subscription_price_hike_moving_average(amounts, threshold, expected_flag, expected_delta_pct, expected_raw_delta):
    """
    TDD Analytics: Asserts mathematical detection of anomalous rate hikes based on 
    trailing moving averages.
    """
    def detect_price_hike(history: list[float], limit_pct: float):
        tma = sum(history[:-1]) / len(history[:-1])
        final = history[-1]
        raw_delta = final - tma
        delta_pct = (raw_delta / tma) * 100
        return delta_pct > limit_pct, delta_pct, raw_delta

    flag, delta_pct, raw_delta = detect_price_hike(amounts, threshold)
    assert flag == expected_flag
    assert delta_pct == pytest.approx(expected_delta_pct, 0.1)
    assert raw_delta == pytest.approx(expected_raw_delta, 0.1)

