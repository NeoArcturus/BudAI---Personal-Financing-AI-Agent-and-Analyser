from pydantic import BaseModel
from typing import List, Optional


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    input: str
    active_account_id: Optional[str] = None
    user_id: str


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
