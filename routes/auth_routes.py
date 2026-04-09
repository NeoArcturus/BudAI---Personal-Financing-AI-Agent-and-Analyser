from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.user_service import UserService
from middleware.auth_middleware import get_current_user

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
callback_router = APIRouter(tags=["callback"])


class LoginRequest(BaseModel):
    email: str
    password: str


class ExtendConnectionsRequest(BaseModel):
    provider_ids: List[str] = []


class RevokeConnectionRequest(BaseModel):
    provider_id: Optional[str] = None


@auth_router.post('/login')
def login(request_data: LoginRequest):
    user_service = UserService()
    user_uuid = user_service.authenticate_or_create_user(
        request_data.email, request_data.password)
    return {"token": user_uuid, "status": "success"}


@auth_router.get('/truelayer/status')
def truelayer_status(current_user=Depends(get_current_user)):
    token_gen = AccessTokenGenerator()
    return {"auth_url": token_gen.get_auth_link(current_user.user_uuid)}


@callback_router.get('/callback')
def truelayer_callback(code: str, state: str):
    token_gen = AccessTokenGenerator()
    if token_gen.validate_callback(code, state):
        return RedirectResponse(url="http://localhost:3000/home")
    raise HTTPException(
        status_code=400, detail="Authentication failed or session expired")


@auth_router.post('/connections/extend')
def extend_user_connections(request_data: ExtendConnectionsRequest, current_user=Depends(get_current_user)):
    token_gen = AccessTokenGenerator()
    results = token_gen.extend_providers(
        request_data.provider_ids, current_user.user_uuid)
    return {"results": results}


@auth_router.post('/connections/revoke')
def revoke_truelayer_access(request_data: RevokeConnectionRequest, current_user=Depends(get_current_user)):
    token_gen = AccessTokenGenerator()
    results = token_gen.revoke_provider(
        request_data.provider_id, current_user.user_uuid)
    return {"results": results}
