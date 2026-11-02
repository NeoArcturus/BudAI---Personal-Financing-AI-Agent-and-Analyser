import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from models.status_codes import AgentStatus, TaskStatus

class A2AMessage(BaseModel):
    """
    Standardized Agent-to-Agent (A2A) protocol communication envelope.
    Enforces type safety, multi-hop correlation tracing, and anti-recursion limits.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_message_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Routing
    sender: str = Field(description="Originating agent name, e.g. 'HealthAgent'")
    recipient: str = Field(description="Target specialist agent, e.g. 'BucketTransferAgent'")

    # Protocol State & Action
    message_type: Literal["QUERY", "PROPOSAL", "COMMAND", "RESPONSE", "ERROR", "ACK"]
    intent: str = Field(description="Machine-readable action code, e.g. 'SIMULATE_CASCADE', 'REALLOCATE_FUNDS'")

    # Data Payload
    payload: Dict[str, Any] = Field(default_factory=dict)
    user_uuid: str

    # Guardrails
    timeout_seconds: float = Field(default=30.0)
    recursion_depth: int = Field(default=0, ge=0, le=5)
    requires_zero_sum_lock: bool = Field(default=False)

class A2AResponse(BaseModel):
    """
    Standardized reply payload returned from a recipient agent to the sender.
    """
    reply_to_id: str
    correlation_id: str
    sender: str
    recipient: str
    status: Literal["SUCCESS", "REJECTED", "ERROR"]
    status_code: str = Field(default=AgentStatus.AGENT_RESPONSE_READY.value, description="Custom BudAI status code e.g. 500-200, 500-409")
    payload: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
