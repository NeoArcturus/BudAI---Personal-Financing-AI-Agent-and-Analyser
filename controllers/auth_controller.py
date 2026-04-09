from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from config import get_db, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, FRONTEND_URL
from schemas.api_schema import LoginRequest, RegisterRequest, ExtendConnectionRequest, RevokeConnectionRequest
from middleware.auth_middleware import get_current_user
from models.database_models import User
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.user_service import UserService
from datetime import datetime, timedelta, timezone
from jose import jwt

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
callback_router = APIRouter(tags=["callback"])


@auth_router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user_service = UserService()
    try:
        user_uuid = user_service.authenticate_user(
            request.email, request.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {"sub": user_uuid, "iat": int(
            now.timestamp()), "exp": int(expires_at.timestamp())},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {"token": token, "status": "success", "expires_in_minutes": ACCESS_TOKEN_EXPIRE_MINUTES}


@auth_router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    user_service = UserService()
    try:
        user_uuid = user_service.register_user(
            request.email, request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "user_uuid": user_uuid}


@auth_router.get("/truelayer/status")
def truelayer_status(current_user: User = Depends(get_current_user)):
    token_gen = AccessTokenGenerator()
    return {"auth_url": token_gen.get_auth_link(current_user.user_uuid)}


@callback_router.get("/callback")
def truelayer_callback(code: str, state: str):
    token_gen = AccessTokenGenerator()
    if token_gen.validate_callback(code, state):
        return RedirectResponse(f"{FRONTEND_URL}/home")
    raise HTTPException(
        status_code=400, detail="Authentication failed or session expired")


@auth_router.post("/connections/extend")
def extend_user_connections(request: ExtendConnectionRequest, current_user: User = Depends(get_current_user)):
    token_gen = AccessTokenGenerator()
    results = token_gen.extend_providers(
        request.provider_ids, current_user.user_uuid)
    return {"results": results}


@auth_router.post("/connections/revoke")
def revoke_truelayer_access(request: RevokeConnectionRequest, current_user: User = Depends(get_current_user)):
    token_gen = AccessTokenGenerator()
    results = token_gen.revoke_provider(
        request.provider_id, current_user.user_uuid)
    return {"results": results}
