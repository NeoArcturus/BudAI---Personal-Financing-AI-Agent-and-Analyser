import asyncio
import uuid
import pytest
from sqlalchemy import text
from config import SessionLocal
from services.a2a.bootstrap import register_all_a2a_workers
from services.a2a.bus import get_a2a_bus
from services.a2a.protocol import A2AMessage, A2AResponse
from services.a2a.negotiator import A2ANegotiator
from models.status_codes import AgentStatus

TEST_USER = "8e7d5972-0074-477f-b99e-e15d27c63704"

@pytest.fixture
def setup_test_buckets():
    """Sets up a test source bucket and target bucket for consensus testing."""
    src_id = f"test-src-{uuid.uuid4().hex[:8]}"
    tgt_id = f"test-tgt-{uuid.uuid4().hex[:8]}"

    with SessionLocal() as session:
        # Create a funded source bucket and empty target bucket
        session.execute(
            text("""
                INSERT INTO buckets (id, user_id, name, type, priority_index, cached_balance, is_active, updated_at)
                VALUES (:src, :u, 'Test Liquidity', 'DEFAULT', 1.0, 500.0, TRUE, NOW()),
                       (:tgt, :u, 'Test Recurring Bill', 'RECURRING', 2.0, 0.0, TRUE, NOW())
            """),
            {"src": src_id, "tgt": tgt_id, "u": TEST_USER}
        )
        session.commit()

    yield src_id, tgt_id

    # Cleanup
    with SessionLocal() as session:
        session.execute(text("DELETE FROM virtual_transfers WHERE source_bucket_id = :src OR target_bucket_id = :tgt"), {"src": src_id, "tgt": tgt_id})
        session.execute(text("DELETE FROM buckets WHERE id IN (:src, :tgt)"), {"src": src_id, "tgt": tgt_id})
        session.commit()

@pytest.mark.asyncio
async def test_forecaster_solvency_impact_intent():
    """Verify ForecasterWorker computes solvency impact and risk bounds."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    # Small deduction should be safe
    safe_msg = A2AMessage(
        sender="TestRunner",
        recipient="ForecasterWorker",
        message_type="QUERY",
        intent="EVALUATE_SOLVENCY_IMPACT",
        user_uuid=TEST_USER,
        payload={"deduction_amount": 1.0}
    )
    safe_resp: A2AResponse = await bus.request_reply(safe_msg)
    assert safe_resp.status == "SUCCESS"
    assert "is_safe" in safe_resp.payload
    assert "projected_balance_after" in safe_resp.payload
    assert "shortfall_probability" in safe_resp.payload

    # Massive deduction should violate solvency
    risky_msg = A2AMessage(
        sender="TestRunner",
        recipient="ForecasterWorker",
        message_type="QUERY",
        intent="EVALUATE_SOLVENCY_IMPACT",
        user_uuid=TEST_USER,
        payload={"deduction_amount": 50000.0}
    )
    risky_resp: A2AResponse = await bus.request_reply(risky_msg)
    assert risky_resp.status == "SUCCESS"
    assert risky_resp.payload["is_safe"] is False
    assert risky_resp.payload["shortfall_probability"] >= 0.8

@pytest.mark.asyncio
async def test_negotiate_rebalance_insufficient_source_balance():
    """Verify A2A consensus rejects transfer when source bucket has insufficient funds."""
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="BucketTransferAgent",
        message_type="COMMAND",
        intent="CONSENSUS_REBALANCE",
        user_uuid=TEST_USER,
        payload={
            "source_bucket_id": "9b462a24-1a45-44cb-83c7-8260be1db8ca", # only £3.15
            "target_bucket_id": "non-existent-target",
            "amount": 1000.0,
            "category": "Utilities",
            "reason": "Test massive rebalance"
        }
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "REJECTED"
    assert resp.status_code == AgentStatus.AGENT_SOLVENCY_VIOLATION.value
    assert "dossier" in resp.payload
    assert resp.payload["dossier"]["BucketTransferAgent"]["vote"] == "REJECT"

@pytest.mark.asyncio
async def test_negotiate_rebalance_successful_consensus(setup_test_buckets):
    """Verify full multi-agent quorum approval and automatic virtual transfer execution."""
    src_id, tgt_id = setup_test_buckets
    register_all_a2a_workers()
    bus = get_a2a_bus()

    msg = A2AMessage(
        sender="TestRunner",
        recipient="BucketTransferAgent",
        message_type="COMMAND",
        intent="CONSENSUS_REBALANCE",
        user_uuid=TEST_USER,
        payload={
            "source_bucket_id": src_id,
            "target_bucket_id": tgt_id,
            "amount": 50.0,
            "category": "Food & Dining",
            "reason": "Monthly grocery pre-allocation"
        }
    )
    resp: A2AResponse = await bus.request_reply(msg)
    assert resp.status == "SUCCESS"
    assert resp.payload["consensus"] == "CONSENSUS_REACHED"
    assert "transfer_id" in resp.payload
    assert "dossier" in resp.payload

    dossier = resp.payload["dossier"]
    assert dossier["BucketTransferAgent"]["vote"] == "ACCEPT"
    assert dossier["HealthWorker"]["vote"] == "ACCEPT"
    assert dossier["ForecasterWorker"]["vote"] == "ACCEPT"
    assert "MarketWorker" in dossier

    # Verify virtual transfer was executed into DB
    with SessionLocal() as session:
        v_tx = session.execute(
            text("SELECT amount FROM virtual_transfers WHERE id = :id"),
            {"id": resp.payload["transfer_id"]}
        ).fetchone()
        assert v_tx is not None
        assert float(v_tx[0]) == 50.0

@pytest.mark.asyncio
async def test_negotiate_rebalance_counter_proposal(setup_test_buckets):
    """Verify that when proposed amount is too large for remaining safety cushion, a counter-proposal is issued."""
    src_id, tgt_id = setup_test_buckets
    register_all_a2a_workers()
    bus = get_a2a_bus()

    # Available is 450 (after previous 50 transfer). Proposing 440 leaves only 10, breaching minimum cushion of 25
    msg = A2AMessage(
        sender="TestRunner",
        recipient="BucketTransferAgent",
        message_type="COMMAND",
        intent="CONSENSUS_REBALANCE",
        user_uuid=TEST_USER,
        payload={
            "source_bucket_id": src_id,
            "target_bucket_id": tgt_id,
            "amount": 440.0,
            "category": "Housing",
            "reason": "Aggressive emergency rent shift"
        }
    )
    resp: A2AResponse = await bus.request_reply(msg)
    print(f"Counter proposal resp: status={resp.status}, code={resp.status_code}, error={resp.error_message}, payload={resp.payload}")
    assert resp.status == "COUNTER_PROPOSED"
    assert resp.status_code == AgentStatus.AGENT_PROPOSAL_ACCEPTED.value
    assert "recommended_safe_amount" in resp.payload
    assert resp.payload["recommended_safe_amount"] < 440.0
    assert resp.payload["recommended_safe_amount"] > 0.0

if __name__ == "__main__":
    asyncio.run(test_forecaster_solvency_impact_intent())
    
    # Run bucket-dependent tests
    src = f"test-src-{uuid.uuid4().hex[:8]}"
    tgt = f"test-tgt-{uuid.uuid4().hex[:8]}"
    with SessionLocal() as session:
        session.execute(
            text("""
                INSERT INTO buckets (id, user_id, name, type, priority_index, cached_balance, is_active, updated_at)
                VALUES (:src, :u, 'Test Liquidity', 'DEFAULT', 1.0, 500.0, TRUE, NOW()),
                       (:tgt, :u, 'Test Recurring Bill', 'RECURRING', 2.0, 0.0, TRUE, NOW())
            """),
            {"src": src, "tgt": tgt, "u": TEST_USER}
        )
        session.commit()
    try:
        asyncio.run(test_negotiate_rebalance_insufficient_source_balance())
        asyncio.run(test_negotiate_rebalance_successful_consensus((src, tgt)))
        asyncio.run(test_negotiate_rebalance_counter_proposal((src, tgt)))
    finally:
        with SessionLocal() as session:
            session.execute(text("DELETE FROM virtual_transfers WHERE source_bucket_id = :src OR target_bucket_id = :tgt"), {"src": src, "tgt": tgt})
            session.execute(text("DELETE FROM buckets WHERE id IN (:src, :tgt)"), {"src": src, "tgt": tgt})
            session.commit()
    print("All Phase 2 tests passed successfully.")
