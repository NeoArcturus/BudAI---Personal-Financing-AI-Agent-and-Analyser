from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    input: str
    active_account_id: Optional[str] = None
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    context_data: Optional[Dict[str, Any]] = None


class VercelMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = ""
    parts: Optional[List[Dict[str, Any]]] = None
    type: Optional[str] = "text"
    payload: Optional[Dict[str, Any]] = None


class StreamChatRequest(BaseModel):
    messages: List[VercelMessage]
    session_id: Optional[str] = None
    active_account_id: Optional[str] = None
    htil_response: Optional[Dict[str, Any]] = None
    messageId: Optional[str] = None


class ExplanationRequest(BaseModel):
    user_uuid: str
    chart_type: str | None = None
    raw_data: Any


class MediaExecuteRequest(BaseModel):
    tool_name: str
    parameters: dict


class ExtendConnectionRequest(BaseModel):
    provider_ids: List[str]


class RevokeConnectionRequest(BaseModel):
    provider_id: str


class TransactionLabelCorrectionRequest(BaseModel):
    transaction_uuid: str
    corrected_label: str
    retrain_model: bool = True


class RetrainCategorizerRequest(BaseModel):
    force: bool = True


class ChatMessageResponse(BaseModel):
    role: str
    content: str
    reasoning_content: Optional[str] = None
    timestamp: datetime


class ChatSessionRenameRequest(BaseModel):
    title: str


class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    last_updated: datetime
    context_data: Optional[Dict[str, Any]] = None
    messages: List[ChatMessageResponse] = []
