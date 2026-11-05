import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field

from services.a2a.bus import get_a2a_bus
from services.a2a.protocol import A2AMessage, A2AResponse
from services.a2a.registry import get_agent_registry
from services.logger_setup import get_core_logger
from middleware.auth_middleware import get_optional_current_user
from models.database_models import User
from services.workers.worker_utils import resolve_user_uuid

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/a2a", tags=["A2A Autonomous Multi-Agent"])

class DispatchRequest(BaseModel):
    recipient: str = Field(description="Target specialist agent, e.g. 'CategorizerAgent', 'AnalyserAgent'")
    intent: str = Field(description="Action intent, e.g. 'CLASSIFY_TRANSACTION', 'GET_SUBSISTENCE_BASELINE'")
    message_type: str = Field(default="QUERY", description="QUERY, COMMAND, PROPOSAL")
    payload: Dict[str, Any] = Field(default_factory=dict)
    user_uuid: Optional[str] = None
    sender: str = Field(default="APIClient")

class ConsensusRebalanceRequest(BaseModel):
    source_bucket_id: str
    target_bucket_id: str
    amount: float
    category: str = "General & Discretionary"
    reason: str = "API automated consensus rebalance"
    user_uuid: Optional[str] = None

class CascadeRequest(BaseModel):
    deficit: float
    user_uuid: Optional[str] = None
    target_default_bucket_id: Optional[str] = None

class RemediateAnomalyRequest(BaseModel):
    amount: float
    category: str = "General & Discretionary"
    merchant: str = "Unknown"
    user_uuid: Optional[str] = None

@router.get("/agents")
async def list_a2a_agents():
    """
    Returns live directory of all registered specialist workers and capability cards.
    """
    registry = get_agent_registry()
    agents = registry.list_agents()
    return {
        "status": "SUCCESS",
        "agent_count": len(agents),
        "agents": [
            {
                "name": a.name,
                "description": a.description,
                "intents_handled": a.intents_handled,
                "avg_latency_ms": a.avg_latency_ms
            }
            for a in agents
        ]
    }

@router.post("/dispatch")
async def dispatch_a2a_message(req: DispatchRequest, current_user: Optional[User] = Depends(get_optional_current_user)):
    """
    Dispatches a typed message across the in-process A2A event bus to any specialist agent.
    """
    target_user_uuid = getattr(current_user, "user_uuid", None) or req.user_uuid
    if not target_user_uuid:
        target_user_uuid = resolve_user_uuid({})

    bus = get_a2a_bus()
    msg = A2AMessage(
        sender=req.sender,
        recipient=req.recipient,
        message_type=req.message_type, # type: ignore
        intent=req.intent,
        user_uuid=target_user_uuid,
        payload=req.payload
    )
    resp: A2AResponse = await bus.request_reply(msg)
    return {
        "reply_to_id": resp.reply_to_id,
        "correlation_id": resp.correlation_id,
        "sender": resp.sender,
        "recipient": resp.recipient,
        "status": resp.status,
        "status_code": resp.status_code,
        "payload": resp.payload,
        "error_message": resp.error_message
    }

@router.post("/consensus-rebalance")
async def execute_consensus_rebalance(req: ConsensusRebalanceRequest, current_user: Optional[User] = Depends(get_optional_current_user)):
    """
    Initiates multi-agent quorum consensus rebalance across Health, Forecaster, Market, and Ledger.
    """
    target_user_uuid = getattr(current_user, "user_uuid", None) or req.user_uuid
    if not target_user_uuid:
        target_user_uuid = resolve_user_uuid({})

    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="APIClient",
        recipient="BucketTransferAgent",
        message_type="COMMAND",
        intent="CONSENSUS_REBALANCE",
        user_uuid=target_user_uuid,
        payload={
            "source_bucket_id": req.source_bucket_id,
            "target_bucket_id": req.target_bucket_id,
            "amount": req.amount,
            "category": req.category,
            "reason": req.reason
        }
    )
    resp: A2AResponse = await bus.request_reply(msg)
    return {
        "status": resp.status,
        "status_code": resp.status_code,
        "payload": resp.payload,
        "error_message": resp.error_message
    }

@router.post("/cascade")
async def execute_priority_cascade(req: CascadeRequest, current_user: Optional[User] = Depends(get_optional_current_user)):
    """
    Initiates autonomous priority cascade to safely replenish the DEFAULT bucket.
    """
    target_user_uuid = getattr(current_user, "user_uuid", None) or req.user_uuid
    if not target_user_uuid:
        target_user_uuid = resolve_user_uuid({})

    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="APIClient",
        recipient="CascadeWorker",
        message_type="COMMAND",
        intent="EXECUTE_PRIORITY_CASCADE",
        user_uuid=target_user_uuid,
        payload={
            "deficit": req.deficit,
            "target_default_bucket_id": req.target_default_bucket_id
        }
    )
    resp: A2AResponse = await bus.request_reply(msg)
    return {
        "status": resp.status,
        "status_code": resp.status_code,
        "payload": resp.payload,
        "error_message": resp.error_message
    }

@router.post("/remediate-anomaly")
async def remediate_spending_anomaly(req: RemediateAnomalyRequest, current_user: Optional[User] = Depends(get_optional_current_user)):
    """
    Evaluates spending anomaly with peers and triggers autonomous remediation.
    """
    target_user_uuid = getattr(current_user, "user_uuid", None) or req.user_uuid
    if not target_user_uuid:
        target_user_uuid = resolve_user_uuid({})

    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="APIClient",
        recipient="AnomalyWorker",
        message_type="COMMAND",
        intent="REMEDIATE_ANOMALY",
        user_uuid=target_user_uuid,
        payload={
            "amount": req.amount,
            "category": req.category,
            "merchant": req.merchant
        }
    )
    resp: A2AResponse = await bus.request_reply(msg)
    return {
        "status": resp.status,
        "status_code": resp.status_code,
        "payload": resp.payload,
        "error_message": resp.error_message
    }
