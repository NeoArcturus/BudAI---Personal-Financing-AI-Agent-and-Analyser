from fastapi import APIRouter, Depends, HTTPException, Request, Response, Cookie
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from config import get_db, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, FRONTEND_URL
from schemas.api_schema import LoginRequest, RegisterRequest, ExtendConnectionRequest, RevokeConnectionRequest
from middleware.auth_middleware import get_current_user
from models.database_models import User
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.user_service import UserService
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi_cache import FastAPICache

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
callback_router = APIRouter(tags=["callback"])


def create_access_token(user_uuid: str, username: str = ""):
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": user_uuid, "username": username, "token_type": "access",
            "iat": int(now.timestamp()), "exp": int(expires_at.timestamp())},
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def create_refresh_token(user_uuid: str):
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=7)
    return jwt.encode(
        {"sub": user_uuid, "token_type": "refresh", "iat": int(
            now.timestamp()), "exp": int(expires_at.timestamp())},
        SECRET_KEY,
        algorithm=ALGORITHM
    )


@auth_router.post("/login")
def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user_service = UserService()
    try:
        user_uuid = user_service.authenticate_user(
            request.email, request.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    # Extract username from email
    username = request.email.split('@')[0]

    access_token = create_access_token(user_uuid, username)
    refresh_token = create_refresh_token(user_uuid)

    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=7 * 24 * 60 * 60
    )

    return {
        "token": access_token,
        "status": "success",
        "expires_in_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
        "username": username,
        "email": request.email,
        "user_id": user_uuid
    }


@auth_router.post("/refresh")
async def refresh_token(response: Response, db: Session = Depends(get_db), refresh_token: str | None = Cookie(None)):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("token_type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")

        user_uuid = payload.get("sub")
        if not user_uuid:
            raise HTTPException(status_code=401, detail="Invalid payload")

        await FastAPICache.clear(namespace="transactions", key=str(user_uuid))
        await FastAPICache.clear(namespace="categorizer", key=str(user_uuid))
        await FastAPICache.clear(namespace="accounts", key=str(user_uuid))

        # Fetch user to get username for the new token
        user = db.query(User).filter(User.user_uuid == user_uuid).first()
        username = user.email.split(
            '@')[0] if user and hasattr(user, 'email') else "User"

        new_access_token = create_access_token(user_uuid, username)

        return {
            "token": new_access_token,
            "status": "success",
            "username": username
        }

    except JWTError:
        raise HTTPException(
            status_code=401, detail="Invalid or expired refresh token")


@auth_router.get("/me")
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    username = current_user.email.split(
        '@')[0] if hasattr(current_user, 'email') else "User"
    return {
        "user_uuid": current_user.user_uuid,
        "username": username,
        "email": getattr(current_user, 'email', '')
    }


@auth_router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    user_service = UserService()
    try:
        user_uuid = user_service.register_user(request.email, request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "user_uuid": user_uuid}


@auth_router.get("/truelayer/status")
def truelayer_status(current_user: User = Depends(get_current_user)):
    token_gen = AccessTokenGenerator()
    return {"auth_url": token_gen.get_auth_link(current_user.user_uuid)}


@callback_router.get("/callback")
async def truelayer_callback(code: str, state: str):
    token_gen = AccessTokenGenerator()
    if await token_gen.validate_callback(code, state):
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
