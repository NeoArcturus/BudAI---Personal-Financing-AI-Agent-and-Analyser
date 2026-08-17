from fastapi import APIRouter, Depends, Request, Response, Cookie
from sqlalchemy.orm import Session
from config import get_db, FRONTEND_URL
from schemas.api_schema import LoginRequest, RegisterRequest, ExtendConnectionRequest, RevokeConnectionRequest, RefreshRequest
from middleware.auth_middleware import get_current_user
from models.database_models import User

from controllers.auth.get import get_user_profile, get_truelayer_status, handle_truelayer_callback
from controllers.auth.post import login_user, refresh_user_token, register_user, extend_connections, revoke_access, generate_reauth_link_controller

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
callback_router = APIRouter(tags=["callback"])

@auth_router.post("/login")
async def login_route(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    return await login_user(request, response)

@auth_router.post("/refresh")
async def refresh_token_route(request: Request, response: Response, db: Session = Depends(get_db), refresh_cookie: str | None = Cookie(alias="refresh_token", default=None)):
    try:
        body = await request.json()
        token_from_body = body.get("refresh_token")
    except Exception:
        token_from_body = None
        
    final_token = token_from_body or refresh_cookie
    return await refresh_user_token(response, db, final_token)

@auth_router.get("/me")
async def get_current_user_profile_route(current_user: User = Depends(get_current_user)):
    return await get_user_profile(current_user)

@auth_router.post("/register")
async def register_route(request: RegisterRequest, db: Session = Depends(get_db)):
    return await register_user(request)

@auth_router.get("/truelayer/status")
async def truelayer_status_route(request: Request, current_user: User = Depends(get_current_user)):
    origin = request.headers.get("origin", FRONTEND_URL)
    return await get_truelayer_status(current_user, origin)

@callback_router.get("/callback")
async def truelayer_callback_route(code: str, state: str):
    return await handle_truelayer_callback(code, state)

@auth_router.post("/connections/extend")
async def extend_user_connections_route(request: ExtendConnectionRequest, current_user: User = Depends(get_current_user)):
    return await extend_connections(request.provider_ids, current_user.user_uuid)

@auth_router.post("/connections/revoke")
async def revoke_truelayer_access_route(request: RevokeConnectionRequest, current_user: User = Depends(get_current_user)):
    return await revoke_access(request.provider_id, current_user.user_uuid)

@auth_router.post("/banks/{bank_uuid}/reauth")
async def generate_bank_reauth_link_route(bank_uuid: str, current_user: User = Depends(get_current_user)):
    return await generate_reauth_link_controller(bank_uuid, current_user.user_uuid)
