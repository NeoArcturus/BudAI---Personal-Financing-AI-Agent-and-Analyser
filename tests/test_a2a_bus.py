import asyncio
import json
import uuid
import sys
from services.a2a.bootstrap import register_all_a2a_workers
from services.a2a.bus import get_a2a_bus
from services.a2a.protocol import A2AMessage, A2AResponse
from services.a2a.registry import get_agent_registry
from models.status_codes import AgentStatus, MacroStatus, PipelineStatus

async def run_a2a_tests():
    print("=== Testing A2A Agent-to-Agent Architecture ===")
    
    # 1. Register workers
    register_all_a2a_workers()
    registry = get_agent_registry()
    agents = registry.list_agents()
    print(f"Registered {len(agents)} agents: {[a.name for a in agents]}")
    assert len(agents) == 6, f"Expected 6 agents, found {len(agents)}"

    bus = get_a2a_bus()
    test_user = "ea0e5c07-ab5b-4c14-9ad9-95a036b24637"

    # 2. Test MarketWorker (ONS Macro Telemetry)
    print("\n--- 1. Testing MarketWorker (FETCH_MACRO_INFLATION) ---")
    market_msg = A2AMessage(
        sender="TestRunner",
        recipient="MarketWorker",
        message_type="QUERY",
        intent="FETCH_MACRO_INFLATION",
        user_uuid=test_user,
        payload={"category": "Food & Dining"}
    )
    market_resp: A2AResponse = await bus.request_reply(market_msg)
    print(f"MarketWorker Response: status={market_resp.status}, code={market_resp.status_code}")
    print(f"Payload: {json.dumps(market_resp.payload, indent=2)}")
    assert market_resp.status == "SUCCESS"
    assert market_resp.status_code == MacroStatus.TELEMETRY_SYNCED.value
    assert "delta_pct_yoy" in market_resp.payload

    # 3. Test HealthWorker (Runway & Health Score)
    print("\n--- 2. Testing HealthWorker (AUDIT_HEALTH_SCORE) ---")
    health_msg = A2AMessage(
        sender="TestRunner",
        recipient="HealthWorker",
        message_type="QUERY",
        intent="AUDIT_HEALTH_SCORE",
        user_uuid=test_user
    )
    health_resp: A2AResponse = await bus.request_reply(health_msg)
    print(f"HealthWorker Response: status={health_resp.status}, code={health_resp.status_code}")
    print(f"Payload: {json.dumps(health_resp.payload, indent=2)}")
    assert health_resp.status == "SUCCESS"
    assert health_resp.status_code == AgentStatus.AGENT_RESPONSE_READY.value
    assert "runway_months" in health_resp.payload
    assert "liquidity_status" in health_resp.payload

    # 4. Test ForecasterWorker (Stochastic Corridor)
    print("\n--- 3. Testing ForecasterWorker (SIMULATE_STOCHASTIC_BALANCE) ---")
    forecast_msg = A2AMessage(
        sender="TestRunner",
        recipient="ForecasterWorker",
        message_type="QUERY",
        intent="SIMULATE_STOCHASTIC_BALANCE",
        user_uuid=test_user,
        payload={"days": 60}
    )
    forecast_resp: A2AResponse = await bus.request_reply(forecast_msg)
    print(f"ForecasterWorker Response: status={forecast_resp.status}, code={forecast_resp.status_code}")
    print(f"Payload: {json.dumps(forecast_resp.payload, indent=2)}")
    assert forecast_resp.status == "SUCCESS"
    assert "mean_projected_balance" in forecast_resp.payload
    assert "careless_bound" in forecast_resp.payload

    # 5. Test CategorizerAgent (Taxonomy & ONS Tags)
    print("\n--- 4. Testing CategorizerAgent (CLASSIFY_TRANSACTION) ---")
    cat_msg = A2AMessage(
        sender="TestRunner",
        recipient="CategorizerAgent",
        message_type="QUERY",
        intent="CLASSIFY_TRANSACTION",
        user_uuid=test_user,
        payload={"description": "Tesco Superstore milk and sourdough bread", "merchant": "Tesco"}
    )
    cat_resp: A2AResponse = await bus.request_reply(cat_msg)
    print(f"CategorizerAgent Response: status={cat_resp.status}, code={cat_resp.status_code}")
    print(f"Payload: {json.dumps(cat_resp.payload, indent=2)}")
    assert cat_resp.status == "SUCCESS"
    assert "Food" in cat_resp.payload.get("category", "")
    assert any("staple" in t for t in cat_resp.payload.get("tags", []))

    # 6. Test AnalyserAgent (Spending Surge Check)
    print("\n--- 5. Testing AnalyserAgent (ANALYZE_SPENDING_SURGE) ---")
    analyser_msg = A2AMessage(
        sender="TestRunner",
        recipient="AnalyserAgent",
        message_type="QUERY",
        intent="ANALYZE_SPENDING_SURGE",
        user_uuid=test_user,
        payload={"amount": 450.0, "category": "Food & Dining"}
    )
    analyser_resp: A2AResponse = await bus.request_reply(analyser_msg)
    print(f"AnalyserAgent Response: status={analyser_resp.status}, code={analyser_resp.status_code}")
    print(f"Payload: {json.dumps(analyser_resp.payload, indent=2)}")
    assert analyser_resp.status == "SUCCESS"
    assert "is_surge" in analyser_resp.payload

    # 7. Test BucketTransferAgent (Solvency Guardrail on negative/invalid amount)
    print("\n--- 6. Testing BucketTransferAgent Solvency Guardrail ---")
    invalid_transfer_msg = A2AMessage(
        sender="HealthWorker",
        recipient="BucketTransferAgent",
        message_type="PROPOSAL",
        intent="PROPOSE_REBALANCE",
        user_uuid=test_user,
        payload={
            "source_bucket_id": "non-existent-source",
            "target_bucket_id": "non-existent-target",
            "amount": -50.0
        }
    )
    transfer_resp: A2AResponse = await bus.request_reply(invalid_transfer_msg)
    print(f"Transfer Response on negative amount: status={transfer_resp.status}, code={transfer_resp.status_code}")
    assert transfer_resp.status == "REJECTED"
    assert transfer_resp.status_code == AgentStatus.AGENT_PROPOSAL_REJECTED.value

    print("\n=== ALL A2A TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(run_a2a_tests())
