import pytest
from fastapi.testclient import TestClient
from main import app
from services.a2a.bootstrap import register_all_a2a_workers

register_all_a2a_workers()
client = TestClient(app)
TEST_USER = "8e7d5972-0074-477f-b99e-e15d27c63704"

def test_api_list_a2a_agents():
    """Verify GET /api/a2a/agents returns registered agents."""
    response = client.get("/api/a2a/agents")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["agent_count"] == 9
    agent_names = [a["name"] for a in data["agents"]]
    assert "MarketWorker" in agent_names
    assert "HealthWorker" in agent_names
    assert "ForecasterWorker" in agent_names
    assert "BucketTransferAgent" in agent_names
    assert "CategorizerAgent" in agent_names
    assert "AnalyserAgent" in agent_names
    assert "ScenarioWorker" in agent_names
    assert "AnomalyWorker" in agent_names
    assert "CascadeWorker" in agent_names

def test_api_dispatch_categorizer_handoff():
    """Verify POST /api/a2a/dispatch classifies a transaction via A2A."""
    payload = {
        "recipient": "CategorizerAgent",
        "intent": "CLASSIFY_TRANSACTION",
        "message_type": "QUERY",
        "user_uuid": TEST_USER,
        "payload": {
            "description": "Sainsbury's groceries sourdough loaf and butter",
            "merchant": "Sainsbury's"
        }
    }
    response = client.post("/api/a2a/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "Food" in data["payload"].get("category", "")
    assert data["payload"].get("ons_code") == "CP01"

def test_api_dispatch_analyser_subsistence():
    """Verify POST /api/a2a/dispatch queries AnalyserAgent subsistence baseline."""
    payload = {
        "recipient": "AnalyserAgent",
        "intent": "GET_SUBSISTENCE_BASELINE",
        "message_type": "QUERY",
        "user_uuid": TEST_USER
    }
    response = client.post("/api/a2a/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "monthly_subsistence_floor" in data["payload"]

def test_api_dispatch_scenario_expense_shock():
    """Verify POST /api/a2a/dispatch runs ScenarioWorker multi-hop handoff."""
    payload = {
        "recipient": "ScenarioWorker",
        "intent": "SIMULATE_EXPENSE_SHOCK",
        "message_type": "QUERY",
        "user_uuid": TEST_USER,
        "payload": {"shock_amount": 300.0}
    }
    response = client.post("/api/a2a/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "can_absorb_shock" in data["payload"]
    assert "post_shock_runway_months" in data["payload"]

if __name__ == "__main__":
    test_api_list_a2a_agents()
    test_api_dispatch_categorizer_handoff()
    test_api_dispatch_analyser_subsistence()
    test_api_dispatch_scenario_expense_shock()
    print("All Phase 4 API tests passed successfully.")
