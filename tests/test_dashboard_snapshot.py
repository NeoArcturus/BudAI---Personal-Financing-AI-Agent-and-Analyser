import pytest
from fastapi.testclient import TestClient
from main import app
from middleware.auth_middleware import get_current_user
from models.database_models import User

client = TestClient(app)
TEST_USER_UUID = "8e7d5972-0074-477f-b99e-e15d27c63704"

@pytest.fixture(autouse=True)
def override_auth():
    def mock_user():
        return User(
            user_uuid=TEST_USER_UUID,
            email="test.user@budai.local",
            name="Test User"
        )
    app.dependency_overrides[get_current_user] = mock_user
    yield
    app.dependency_overrides.pop(get_current_user, None)

def test_get_dashboard_snapshot_schema():
    """Verify GET /api/dashboard/snapshot returns complete structured dashboard data."""
    response = client.get("/api/dashboard/snapshot?range_days=60")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    
    # 1. Reconciliation
    assert "reconciliation" in data
    rec = data["reconciliation"]
    assert "physical_balance" in rec
    assert "virtual_balance" in rec
    assert "delta" in rec
    assert "is_reconciled" in rec
    assert "bank_synced" in rec
    assert isinstance(rec["physical_balance"], (int, float))
    assert isinstance(rec["virtual_balance"], (int, float))

    # 2. Buckets
    assert "buckets" in data
    assert isinstance(data["buckets"], list)
    for b in data["buckets"]:
        assert "id" in b
        assert "name" in b
        assert "type" in b
        assert "cached_balance" in b
        assert "priority_index" in b
        assert "allocation_percentage" in b
        assert b["cached_balance"] >= 0.0

    # 3. Forecast
    assert "forecast" in data
    fc = data["forecast"]
    assert fc["horizon_days"] == 60
    assert "safety_buffer_floor" in fc
    assert "trajectory" in fc
    assert len(fc["trajectory"]) > 0
    first_pt = fc["trajectory"][0]
    assert "day" in first_pt
    assert "date" in first_pt
    assert "median" in first_pt
    assert "lower_bound" in first_pt
    assert "upper_bound" in first_pt

    # 4. Transactions
    assert "transactions" in data
    assert isinstance(data["transactions"], list)

    # 5. Decisions
    assert "decisions" in data
    assert isinstance(data["decisions"], list)

    # 6. Guardrails
    assert "guardrails" in data
    gr = data["guardrails"]
    assert "safety_buffer" in gr
    assert "autonomous_limit" in gr
    assert "guardrails_active" in gr
    assert gr["guardrails_active"] is True
