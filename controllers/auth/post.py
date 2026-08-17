import json
from fastapi import HTTPException, Response
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from models.database_models import User
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from schemas.api_schema import LoginRequest, RegisterRequest
from services.user_service import UserService
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def create_access_token(user_uuid: str, username: str = ""):
    """
    Generates a short-lived JWT access token for API authentication.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        username (str, optional): The user's username. Defaults to "".
        
    Returns:
        str: The encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {"sub": user_uuid, "username": username, "token_type": "access",
            "iat": int(now.timestamp()), "exp": int(expires_at.timestamp())},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return token

def create_refresh_token(user_uuid: str):
    """
    Generates a long-lived JWT refresh token for session sliding.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        
    Returns:
        str: The encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=7)
    token = jwt.encode(
        {"sub": user_uuid, "token_type": "refresh", "iat": int(
            now.timestamp()), "exp": int(expires_at.timestamp())},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return token

async def login_user(request: LoginRequest, response: Response):
    """
    Authenticates a user via email and password, and issues JWT tokens.
    Sets the refresh token in a secure HTTP-only cookie.
    
    Args:
        request (LoginRequest): The login credentials payload.
        response (Response): The FastAPI response object for setting cookies.
        
    Returns:
        dict: The access token and basic user metadata.
        
    Raises:
        HTTPException (401): If authentication fails.
    """
    user_service = UserService()
    try:
        user_uuid = user_service.authenticate_user(request.email, request.password)
    except ValueError as e:
        logger.warning(json.dumps({"message": f"Authentication failed for {request.email}: {e}", "status_code": 400}))
        raise HTTPException(status_code=401, detail=str(e))
    
    username = request.email.split('@')[0]
    access_token = create_access_token(user_uuid, username)
    refresh_token = create_refresh_token(user_uuid)
    
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=7 * 24 * 60 * 60
    )
    return {
        "token": access_token,
        "refresh_token": refresh_token,
        "status": "success",
        "expires_in_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
        "username": username,
        "email": request.email,
        "user_id": user_uuid
    }

async def refresh_user_token(response: Response, db: Session, refresh_token: str | None):
    """
    Validates a refresh token and issues a new access token and refresh token (sliding session).
    
    Args:
        response (Response): The FastAPI response object for setting new cookies.
        db (Session): The database session dependency.
        refresh_token (str | None): The refresh token retrieved from cookies.
        
    Returns:
        dict: The new access token and username.
        
    Raises:
        HTTPException (401): If the token is missing, invalid, or expired.
    """
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("token_type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_uuid = payload.get("sub")
        if not user_uuid:
            raise HTTPException(status_code=401, detail="Invalid payload")
        
        from utils.cache_utils import clear_user_cache
        clear_user_cache(str(user_uuid), namespace="transactions")
        clear_user_cache(str(user_uuid), namespace="categorizer")
        clear_user_cache(str(user_uuid), namespace="accounts")
        
        user = db.query(User).filter(User.user_uuid == user_uuid).first()
        username = user.email.split('@')[0] if user and hasattr(user, 'email') else "User"
        new_access_token = create_access_token(user_uuid, username)
        new_refresh_token = create_refresh_token(user_uuid)
        
        response.set_cookie(
            key="refresh_token",
            value=new_refresh_token,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=7 * 24 * 60 * 60
        )
        
        return {
            "token": new_access_token,
            "status": "success",
            "username": username
        }
    except JWTError as e:
        logger.error(json.dumps({"message": f"JWT Error during token refresh: {e}", "status_code": 500}))
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

async def register_user(request: RegisterRequest):
    """
    Registers a new user account with the provided email and password.
    
    Args:
        request (RegisterRequest): The registration credentials.
        
    Returns:
        dict: A success payload containing the new user_uuid.
        
    Raises:
        HTTPException (400): If registration fails (e.g., user already exists).
    """
    user_service = UserService()
    try:
        user_uuid = user_service.register_user(
            email=request.email, 
            password=request.password,
            name=request.name,
            date_of_birth=request.date_of_birth,
            employment_status=request.employment_status,
            country_of_tax_residence=request.country_of_tax_residence
        )
    except ValueError as e:
        logger.warning(json.dumps({"message": f"Registration failed for {request.email}: {e}", "status_code": 400}))
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "user_uuid": user_uuid}

async def extend_connections(provider_ids: list, user_uuid: str):
    """
    Extends the access window for linked TrueLayer bank providers.
    
    Args:
        provider_ids (list): List of TrueLayer provider IDs to extend.
        user_uuid (str): The UUID of the requesting user.
        
    Returns:
        dict: The results of the extension requests per provider.
    """
    token_gen = AccessTokenGenerator()
    results = token_gen.extend_providers(provider_ids, user_uuid)
    return {"results": results}

async def revoke_access(provider_id: str, user_uuid: str):
    """
    Revokes access to a specific TrueLayer bank provider and deletes its credentials.
    
    Args:
        provider_id (str): The TrueLayer provider ID to revoke.
        user_uuid (str): The UUID of the requesting user.
        
    Returns:
        dict: The results of the revocation request.
    """
    token_gen = AccessTokenGenerator()
    results = token_gen.revoke_provider(provider_id, user_uuid)
    return {"results": results}

async def generate_reauth_link_controller(bank_uuid: str, user_uuid: str):
    """
    Generates a TrueLayer reauthentication link for a specific bank connection.
    
    Args:
        bank_uuid (str): The UUID of the bank connection.
        user_uuid (str): The UUID of the user.
        
    Returns:
        dict: The reauthentication URI.
    """
    from config import SessionLocal
    from models.database_models import Bank
    token_gen = AccessTokenGenerator()
    with SessionLocal() as session:
        bank = session.query(Bank).filter_by(bank_uuid=bank_uuid, user_uuid=user_uuid).first()
        if not bank:
            raise HTTPException(status_code=404, detail="Bank not found")
        
        enc_refresh = bank.refresh_token
        if isinstance(enc_refresh, memoryview):
            enc_refresh = enc_refresh.tobytes()
        
        refresh_token = token_gen.cipher_suite.decrypt(enc_refresh).decode()
        reauth_url = token_gen.get_reauth_link(refresh_token, user_uuid)
        
        if not reauth_url:
            raise HTTPException(status_code=500, detail="Failed to generate reauth link")
            
        return {"reauth_url": reauth_url}
