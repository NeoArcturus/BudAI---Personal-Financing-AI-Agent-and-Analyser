from fastapi import HTTPException, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from models.database_models import User
from config import FRONTEND_URL
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def get_user_profile(current_user: User):
    """
    Retrieves the basic profile information for the authenticated user.
    
    Args:
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: The user's UUID, username, and email.
    """
    username = current_user.email.split('@')[0] if hasattr(current_user, 'email') else "User"
    return {
        "user_uuid": current_user.user_uuid,
        "username": username,
        "email": getattr(current_user, 'email', '')
    }

async def get_truelayer_status(current_user: User, origin: str):
    """
    Generates a TrueLayer authentication link to connect a new bank account.
    
    Args:
        current_user (User): The authenticated user making the request.
        origin (str): The frontend origin URL for redirection after auth.
        
    Returns:
        dict: A payload containing the generated TrueLayer auth_url.
    """
    token_gen = AccessTokenGenerator()
    auth_url = token_gen.get_auth_link(current_user.user_uuid, origin)
    return {"auth_url": auth_url}

async def handle_truelayer_callback(code: str, state: str):
    """
    Handles the OAuth callback from TrueLayer after a user links a bank account.
    Validates the authorization code, exchanges it for tokens, clears cache,
    and redirects the user back to the frontend.
    
    Args:
        code (str): The authorization code returned by TrueLayer.
        state (str): The state parameter containing user_uuid and target_frontend.
        
    Returns:
        RedirectResponse: Redirects the user back to the application home.
        
    Raises:
        HTTPException (400): If validation fails or the session is expired.
    """
    parts = state.split("::", 1)
    user_uuid = parts[0]
    target_frontend = parts[1] if len(parts) > 1 and parts[1] else FRONTEND_URL

    token_gen = AccessTokenGenerator()
    if await token_gen.validate_callback(code, user_uuid):
        from utils.cache_utils import clear_user_cache
        clear_user_cache(str(user_uuid), namespace="accounts")
        return RedirectResponse(f"{target_frontend}/home")
    
    logger.warning("TrueLayer callback validation failed")
    raise HTTPException(status_code=400, detail="Authentication failed or session expired")
