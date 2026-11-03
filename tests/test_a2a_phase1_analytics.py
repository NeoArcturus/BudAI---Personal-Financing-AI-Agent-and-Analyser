import asyncio
import json
import pytest
from services.a2a.bootstrap import register_all_a2a_workers
from services.a2a.bus import get_a2a_bus
from services.a2a.protocol import A2AMessage, A2AResponse
from services.a2a.registry import get_agent_registry
from models.status_codes import AgentStatus, PipelineStatus

TEST_USER = "8e7d5972-0074-477f-b99e-e15d27c63704"

@pytest.mark.asyncio
async def test_analyser_subsistence_baseline():
    """Verify AnalyserAgent calculates empirical subsistence floor from database."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="AnalyserAgent",
        message_type="QUERY",
        intent="GET_SUBSISTENCE_BASELINE",
        user_uuid=TEST_USER
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert resp.status_code == AgentStatus.AGENT_RESPONSE_READY.value
    assert "monthly_subsistence_floor" in resp.payload
    assert resp.payload["monthly_subsistence_floor"] >= 0.0
    assert "daily_subsistence_floor" in resp.payload
    assert resp.payload["observation_window_days"] == 90

@pytest.mark.asyncio
async def test_analyser_historical_volatility():
    """Verify AnalyserAgent computes empirical daily mean drift and standard deviation."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="AnalyserAgent",
        message_type="QUERY",
        intent="GET_HISTORICAL_VOLATILITY",
        user_uuid=TEST_USER,
        payload={"window_days": 180}
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert resp.status_code == AgentStatus.AGENT_RESPONSE_READY.value
    assert "daily_mean_drift" in resp.payload
    assert "daily_std_dev" in resp.payload
    assert "observation_count" in resp.payload
    assert isinstance(resp.payload["daily_std_dev"], (int, float))

@pytest.mark.asyncio
async def test_analyser_net_cashflow_surplus():
    """Verify AnalyserAgent computes empirical income vs expense surplus."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="AnalyserAgent",
        message_type="QUERY",
        intent="GET_NET_CASHFLOW_SURPLUS",
        user_uuid=TEST_USER
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert resp.status_code == AgentStatus.AGENT_RESPONSE_READY.value
    assert "monthly_income" in resp.payload
    assert "monthly_expenses" in resp.payload
    assert "net_surplus" in resp.payload
    assert "savings_ratio" in resp.payload

@pytest.mark.asyncio
async def test_analyser_category_baseline():
    """Verify AnalyserAgent computes category spending stats and anomaly check."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="AnalyserAgent",
        message_type="QUERY",
        intent="GET_CATEGORY_BASELINE",
        user_uuid=TEST_USER,
        payload={"category": "Food & Dining", "test_amount": 1500.0}
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert "category_mean" in resp.payload
    assert "category_std_dev" in resp.payload
    assert "is_anomaly" in resp.payload

@pytest.mark.asyncio
async def test_health_worker_a2a_analyser_integration():
    """Verify HealthWorker queries AnalyserAgent dynamically rather than using hardcoded burn rate."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="HealthWorker",
        message_type="QUERY",
        intent="AUDIT_HEALTH_SCORE",
        user_uuid=TEST_USER
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert "health_score" in resp.payload
    assert "monthly_burn_rate" in resp.payload
    assert "subsistence_floor_used" in resp.payload

@pytest.mark.asyncio
async def test_forecaster_worker_a2a_analyser_integration():
    """Verify ForecasterWorker queries AnalyserAgent dynamically for volatility parameters."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="ForecasterWorker",
        message_type="QUERY",
        intent="SIMULATE_STOCHASTIC_BALANCE",
        user_uuid=TEST_USER,
        payload={"days": 45}
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert "mean_projected_balance" in resp.payload
    assert "empirical_std_dev" in resp.payload
    assert resp.payload["horizon_days"] == 45

@pytest.mark.asyncio
async def test_scenario_worker_expense_shock_handoff():
    """Verify ScenarioWorker orchestrates A2A calls across Analyser and Health to simulate shock."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="ScenarioWorker",
        message_type="QUERY",
        intent="SIMULATE_EXPENSE_SHOCK",
        user_uuid=TEST_USER,
        payload={"shock_amount": 400.0, "category": "Automotive & Transport"}
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert "can_absorb_shock" in resp.payload
    assert "post_shock_liquid_cash" in resp.payload
    assert "post_shock_runway_months" in resp.payload

@pytest.mark.asyncio
async def test_anomaly_worker_multihop_handoff():
    """Verify AnomalyWorker orchestrates A2A calls across Analyser and Market to evaluate anomaly."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="AnomalyWorker",
        message_type="QUERY",
        intent="EVALUATE_TRANSACTION_ANOMALY",
        user_uuid=TEST_USER,
        payload={"amount": 350.0, "category": "Food & Dining", "merchant": "Whole Foods"}
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert "is_anomaly" in resp.payload
    assert "z_score" in resp.payload
    assert "inflation_adjusted_deviation" in resp.payload

if __name__ == "__main__":
    asyncio.run(test_analyser_subsistence_baseline())
    asyncio.run(test_analyser_historical_volatility())
    asyncio.run(test_analyser_net_cashflow_surplus())
    asyncio.run(test_analyser_category_baseline())
    asyncio.run(test_health_worker_a2a_analyser_integration())
    asyncio.run(test_forecaster_worker_a2a_analyser_integration())
    asyncio.run(test_scenario_worker_expense_shock_handoff())
    asyncio.run(test_anomaly_worker_multihop_handoff())
    print("All Phase 1 tests passed successfully.")
