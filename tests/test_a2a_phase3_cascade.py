import asyncio
import uuid
import pytest
from sqlalchemy import text
from config import SessionLocal
from services.a2a.bootstrap import register_all_a2a_workers
from services.a2a.bus import get_a2a_bus
from services.a2a.protocol import A2AMessage, A2AResponse
from models.status_codes import AgentStatus

TEST_USER = "8e7d5972-0074-477f-b99e-e15d27c63704"

@pytest.fixture
def setup_cascade_buckets():
    """Sets up a DEFAULT bucket and two auxiliary buckets for priority cascade testing."""
    def_id = f"test-def-{uuid.uuid4().hex[:8]}"
    low_prio_id = f"test-low-{uuid.uuid4().hex[:8]}"
    high_prio_id = f"test-high-{uuid.uuid4().hex[:8]}"

    with SessionLocal() as session:
        session.execute(
            text("""
                INSERT INTO buckets (id, user_id, name, type, priority_index, cached_balance, is_active, updated_at)
                VALUES (:def_id, :u, 'Default Main', 'DEFAULT', 1.0, 10.0, TRUE, NOW()),
                       (:low_id, :u, 'Holiday Savings', 'TARGET_DATE', 10.0, 50.0, TRUE, NOW()),
                       (:high_id, :u, 'Emergency Fund', 'ACCUMULATING', 5.0, 100.0, TRUE, NOW())
            """),
            {"def_id": def_id, "low_id": low_prio_id, "high_id": high_prio_id, "u": TEST_USER}
        )
        session.commit()

    yield def_id, low_prio_id, high_prio_id

    # Cleanup
    with SessionLocal() as session:
        session.execute(
            text("DELETE FROM virtual_transfers WHERE source_bucket_id IN (:d, :l, :h) OR target_bucket_id IN (:d, :l, :h)"),
            {"d": def_id, "l": low_prio_id, "h": high_prio_id}
        )
        session.execute(
            text("DELETE FROM buckets WHERE id IN (:d, :l, :h)"),
            {"d": def_id, "l": low_prio_id, "h": high_prio_id}
        )
        session.commit()

@pytest.mark.asyncio
async def test_cascade_worker_covers_deficit(setup_cascade_buckets):
    """Verify CascadeWorker drains lowest priority bucket first and replenishes default."""
    def_id, low_id, high_id = setup_cascade_buckets
    register_all_a2a_workers()
    bus = get_a2a_bus()

    # Need £80 deficit covered. Lowest priority (Holiday, £50) should be wiped first, then Emergency (£30).
    msg = A2AMessage(
        sender="TestRunner",
        recipient="CascadeWorker",
        message_type="COMMAND",
        intent="EXECUTE_PRIORITY_CASCADE",
        user_uuid=TEST_USER,
        payload={"deficit": 80.0, "target_default_bucket_id": def_id}
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert resp.payload["deficit_resolved"] is True
    assert resp.payload["total_recovered"] == 80.0
    assert resp.payload["uncovered_deficit"] == 0.0
    assert len(resp.payload["transfers_executed"]) == 2

    # Verify order of drain
    transfers = resp.payload["transfers_executed"]
    assert transfers[0]["source_bucket_id"] == low_id  # priority 10.0
    assert transfers[0]["amount"] == 50.0
    assert transfers[1]["source_bucket_id"] == high_id # priority 5.0
    assert transfers[1]["amount"] == 30.0

@pytest.mark.asyncio
async def test_cascade_worker_partial_exhaustion(setup_cascade_buckets):
    """Verify CascadeWorker gracefully reports uncovered deficit when reserves are insufficient."""
    def_id, low_id, high_id = setup_cascade_buckets
    register_all_a2a_workers()
    bus = get_a2a_bus()

    # Total available in auxiliary is 50 + 100 = 150. Requesting 250 deficit.
    msg = A2AMessage(
        sender="TestRunner",
        recipient="CascadeWorker",
        message_type="COMMAND",
        intent="EXECUTE_PRIORITY_CASCADE",
        user_uuid=TEST_USER,
        payload={"deficit": 250.0, "target_default_bucket_id": def_id}
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert resp.payload["deficit_resolved"] is False
    assert resp.payload["total_recovered"] == 150.0
    assert resp.payload["uncovered_deficit"] == 100.0

@pytest.mark.asyncio
async def test_anomaly_remediation_workflow():
    """Verify AnomalyWorker evaluates severe surge and produces remediation actions."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="AnomalyWorker",
        message_type="COMMAND",
        intent="REMEDIATE_ANOMALY",
        user_uuid=TEST_USER,
        payload={
            "amount": 950.0,
            "category": "Food & Dining",
            "merchant": "Luxury Catering Ltd"
        }
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert "anomaly_evaluation" in resp.payload
    assert resp.payload["anomaly_evaluation"]["is_anomaly"] is True
    assert "remediation_action" in resp.payload
    assert "alert_dispatched" in resp.payload

if __name__ == "__main__":
    def_id = f"test-def-{uuid.uuid4().hex[:8]}"
    low_id = f"test-low-{uuid.uuid4().hex[:8]}"
    high_id = f"test-high-{uuid.uuid4().hex[:8]}"
    with SessionLocal() as session:
        session.execute(
            text("""
                INSERT INTO buckets (id, user_id, name, type, priority_index, cached_balance, is_active, updated_at)
                VALUES (:def_id, :u, 'Default Main', 'DEFAULT', 1.0, 10.0, TRUE, NOW()),
                       (:low_id, :u, 'Holiday Savings', 'TARGET_DATE', 10.0, 50.0, TRUE, NOW()),
                       (:high_id, :u, 'Emergency Fund', 'ACCUMULATING', 5.0, 100.0, TRUE, NOW())
            """),
            {"def_id": def_id, "low_id": low_id, "high_id": high_prio_id if 'high_prio_id' in locals() else high_id, "u": TEST_USER}
        )
        session.commit()
    try:
        asyncio.run(test_cascade_worker_covers_deficit((def_id, low_id, high_id)))
        # Reset balances for second cascade test
        with SessionLocal() as session:
            session.execute(text("UPDATE buckets SET cached_balance = 50.0 WHERE id = :id"), {"id": low_id})
            session.execute(text("UPDATE buckets SET cached_balance = 100.0 WHERE id = :id"), {"id": high_id})
            session.commit()
        asyncio.run(test_cascade_worker_partial_exhaustion((def_id, low_id, high_id)))
        asyncio.run(test_anomaly_remediation_workflow())
    finally:
        with SessionLocal() as session:
            session.execute(
                text("DELETE FROM virtual_transfers WHERE source_bucket_id IN (:d, :l, :h) OR target_bucket_id IN (:d, :l, :h)"),
                {"d": def_id, "l": low_id, "h": high_id}
            )
            session.execute(
                text("DELETE FROM buckets WHERE id IN (:d, :l, :h)"),
                {"d": def_id, "l": low_id, "h": high_id}
            )
            session.commit()
    print("All Phase 3 tests passed successfully.")
