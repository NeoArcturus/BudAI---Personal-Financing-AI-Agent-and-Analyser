import json
import asyncio
from fastapi import HTTPException
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def revoke_connection(user_uuid: str, provider_id: str):
    """
    Revokes the OAuth connection for a specific TrueLayer provider and clears
    the user's account and transaction caches.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        provider_id (str): The TrueLayer provider ID to disconnect.
        
    Returns:
        dict: The success status and results payload from the revocation API.
        
    Raises:
        HTTPException (500): If revocation fails at the API or database level.
    """
    try:
        token_gen = AccessTokenGenerator()
        results = await asyncio.to_thread(token_gen.revoke_provider, provider_id, user_uuid)
        
        from utils.cache_utils import clear_user_cache
        clear_user_cache(str(user_uuid), namespace="accounts")
        clear_user_cache(str(user_uuid), namespace="transactions")
        
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(json.dumps({"message": f"Error revoking connection for user {user_uuid}, provider {provider_id}: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail=str(e))
